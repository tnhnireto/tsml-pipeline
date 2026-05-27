"""
Tests for eToro API verification.

Core logic (``verify()``, ``format_report()``) is imported directly from
``tsml.broker.verify`` — no importlib needed.

``TestMain`` tests the CLI entry point in ``scripts/verify_etoro_api.py``
via ``importlib`` for the auth-error / mode-error code paths where
``EtoroClient`` construction itself fails.

No network calls, no real API keys, no orders.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tsml.broker.base import (
    AccountInfo,
    BrokerAuthError,
    BrokerError,
    BrokerModeError,
    InstrumentInfo,
    PositionInfo,
)
from tsml.broker.verify import CheckResult, VerifyResult, format_report, verify


# ---------------------------------------------------------------------------
# Stub broker client
# ---------------------------------------------------------------------------

class FakeBrokerClient:
    """
    Minimal ``BrokerClient`` stub with per-method error injection.

    Parameters
    ----------
    cash, balance, equity:
        Account values returned by ``get_account()``.
    n_positions:
        Number of open positions returned by ``get_positions()``.
    instrument_name:
        Name returned by ``get_instrument()``.
    raise_on_account, raise_on_positions, raise_on_instrument:
        If not None, raise this exception from the corresponding method.
    """

    def __init__(
        self,
        *,
        cash: float = 10_000.0,
        balance: float = 10_000.0,
        equity: float = 0.0,
        n_positions: int = 2,
        instrument_name: str = "Apple Inc.",
        raise_on_account: Exception | None = None,
        raise_on_positions: Exception | None = None,
        raise_on_instrument: Exception | None = None,
    ) -> None:
        self._cash               = cash
        self._balance            = balance
        self._equity             = equity
        self._n_positions        = n_positions
        self._instrument_name    = instrument_name
        self._raise_account      = raise_on_account
        self._raise_positions    = raise_on_positions
        self._raise_instrument   = raise_on_instrument

    def get_account(self) -> AccountInfo:
        if self._raise_account is not None:
            raise self._raise_account
        return AccountInfo(
            account_id="test-account",
            mode="demo",
            balance=self._balance,
            equity=self._equity,
            cash=self._cash,
        )

    def get_positions(self) -> list[PositionInfo]:
        if self._raise_positions is not None:
            raise self._raise_positions
        return [
            PositionInfo(symbol=f"SYM{i}", quantity=1.0, market_value=100.0, open_price=100.0)
            for i in range(self._n_positions)
        ]

    def get_instrument(self, symbol: str) -> InstrumentInfo:
        if self._raise_instrument is not None:
            raise self._raise_instrument
        return InstrumentInfo(
            symbol=symbol,
            name=self._instrument_name,
            tradeable=True,
            leverage_max=1.0,
        )

    def place_order(self, symbol, side, amount, dry_run=True):  # pragma: no cover
        raise NotImplementedError


# ===========================================================================
# verify()
# ===========================================================================

class TestVerify:

    def test_all_checks_pass_when_client_healthy(self):
        client = FakeBrokerClient()
        result = verify(client)
        assert result.all_ok is True
        assert result.n_passed == 3
        assert len(result.checks) == 3

    def test_all_checks_have_names(self):
        result = verify(FakeBrokerClient())
        names  = [c.name for c in result.checks]
        assert any("account" in n.lower() for n in names)
        assert any("position" in n.lower() for n in names)
        assert any("instrument" in n.lower() for n in names)

    def test_all_checks_ok_true(self):
        result = verify(FakeBrokerClient())
        assert all(c.ok for c in result.checks)

    def test_account_detail_contains_balance(self):
        client = FakeBrokerClient(balance=9_876.54)
        result = verify(client)
        account_check = result.checks[0]
        assert "9,876.54" in account_check.detail

    def test_account_detail_contains_cash(self):
        client = FakeBrokerClient(cash=5_432.10)
        result = verify(client)
        assert "5,432.10" in result.checks[0].detail

    def test_account_detail_contains_mode(self):
        result = verify(FakeBrokerClient())
        assert "demo" in result.checks[0].detail

    def test_positions_detail_contains_count(self):
        client = FakeBrokerClient(n_positions=3)
        result = verify(client)
        pos_check = result.checks[1]
        assert "3" in pos_check.detail

    def test_positions_zero_is_ok(self):
        client = FakeBrokerClient(n_positions=0)
        result = verify(client)
        assert result.checks[1].ok is True
        assert "0" in result.checks[1].detail

    def test_instrument_detail_contains_name(self):
        client = FakeBrokerClient(instrument_name="Apple Inc.")
        result = verify(client)
        instr_check = result.checks[2]
        assert "Apple Inc." in instr_check.detail

    def test_instrument_check_uses_aapl(self):
        """Instrument-lookup must use AAPL as the probe symbol."""
        lookups: list[str] = []
        original = FakeBrokerClient.get_instrument
        def recording_get_instrument(self, symbol):
            lookups.append(symbol)
            return original(self, symbol)
        client = FakeBrokerClient()
        client.get_instrument = lambda s: (lookups.append(s), InstrumentInfo(s, "", True))[1]
        verify(client)
        assert "AAPL" in lookups

    # ── Per-check failure scenarios ───────────────────────────────────────

    def test_account_failure_captured(self):
        client = FakeBrokerClient(raise_on_account=BrokerError("account unavailable"))
        result = verify(client)
        assert result.checks[0].ok is False
        assert "account unavailable" in result.checks[0].error

    def test_account_failure_makes_all_ok_false(self):
        client = FakeBrokerClient(raise_on_account=BrokerError("err"))
        assert verify(client).all_ok is False

    def test_account_failure_does_not_stop_other_checks(self):
        """Positions and instrument checks must still run even if account fails."""
        client = FakeBrokerClient(raise_on_account=BrokerError("err"))
        result = verify(client)
        assert len(result.checks) == 3          # all three attempted
        assert result.checks[1].ok is True      # positions OK
        assert result.checks[2].ok is True      # instrument OK

    def test_positions_failure_captured(self):
        client = FakeBrokerClient(raise_on_positions=BrokerError("positions unavailable"))
        result = verify(client)
        assert result.checks[1].ok is False
        assert "positions unavailable" in result.checks[1].error

    def test_positions_failure_makes_all_ok_false(self):
        assert verify(FakeBrokerClient(raise_on_positions=BrokerError("err"))).all_ok is False

    def test_positions_failure_does_not_stop_instrument_check(self):
        client = FakeBrokerClient(raise_on_positions=BrokerError("err"))
        result = verify(client)
        assert result.checks[2].ok is True

    def test_instrument_failure_captured(self):
        client = FakeBrokerClient(raise_on_instrument=BrokerError("lookup failed"))
        result = verify(client)
        assert result.checks[2].ok is False
        assert "lookup failed" in result.checks[2].error

    def test_instrument_failure_makes_all_ok_false(self):
        assert verify(FakeBrokerClient(raise_on_instrument=BrokerError("err"))).all_ok is False

    def test_n_passed_counts_correctly(self):
        client = FakeBrokerClient(raise_on_positions=BrokerError("err"))
        result = verify(client)
        assert result.n_passed == 2   # account + instrument pass

    def test_all_three_fail(self):
        client = FakeBrokerClient(
            raise_on_account=BrokerError("a"),
            raise_on_positions=BrokerError("b"),
            raise_on_instrument=BrokerError("c"),
        )
        result = verify(client)
        assert result.all_ok is False
        assert result.n_passed == 0
        assert len(result.checks) == 3


# ===========================================================================
# format_report()
# ===========================================================================

class TestFormatReport:

    def _report(self, **kwargs) -> str:
        return format_report(verify(FakeBrokerClient(**kwargs)))

    def test_report_is_ascii_safe(self):
        self._report().encode("ascii")   # must not raise

    def test_passed_banner_on_all_ok(self):
        assert "ALL CHECKS PASSED" in self._report()

    def test_failed_banner_on_any_failure(self):
        report = format_report(verify(FakeBrokerClient(raise_on_positions=BrokerError("x"))))
        assert "VERIFICATION FAILED" in report

    def test_ok_label_per_passing_check(self):
        assert self._report().count("[OK]") == 3

    def test_failed_label_on_failing_check(self):
        report = format_report(verify(FakeBrokerClient(raise_on_account=BrokerError("x"))))
        assert "[FAILED]" in report

    def test_passed_count_shown(self):
        assert "3/3" in self._report()

    def test_partial_pass_count_shown(self):
        report = format_report(verify(FakeBrokerClient(raise_on_instrument=BrokerError("x"))))
        assert "2/3" in report

    def test_balance_in_report(self):
        report = self._report(balance=12_345.67)
        assert "12,345.67" in report

    def test_position_count_in_report(self):
        report = self._report(n_positions=5)
        assert "5" in report

    def test_error_message_in_report_on_failure(self):
        report = format_report(verify(FakeBrokerClient(raise_on_account=BrokerError("auth denied"))))
        assert "auth denied" in report

    def test_no_unicode_characters(self):
        report = self._report()
        for ch in ("\u2500", "\u2502", "\u2192", "\u2714", "\u2718", "\u2713"):
            assert ch not in report, f"Unicode char {ch!r} found in report"


# ===========================================================================
# TestMain — CLI entry point in scripts/verify_etoro_api.py
# ===========================================================================

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_etoro_api.py"


@pytest.fixture(scope="module")
def vea():
    """Load scripts/verify_etoro_api.py as a module."""
    spec = importlib.util.spec_from_file_location("verify_etoro_api", _SCRIPT_PATH)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


class TestMain:
    """Tests for the CLI main() function."""

    def test_all_checks_pass_returns_zero(self, vea):
        """main(client=stub) returns 0 when all checks pass."""
        assert vea.main(client=FakeBrokerClient()) == 0

    def test_account_failure_returns_one(self, vea):
        client = FakeBrokerClient(raise_on_account=BrokerError("account error"))
        assert vea.main(client=client) == 1

    def test_positions_failure_returns_one(self, vea):
        client = FakeBrokerClient(raise_on_positions=BrokerError("positions error"))
        assert vea.main(client=client) == 1

    def test_instrument_failure_returns_one(self, vea):
        client = FakeBrokerClient(raise_on_instrument=BrokerError("instrument error"))
        assert vea.main(client=client) == 1

    def test_auth_failure_returns_one(self, vea, monkeypatch):
        """
        main() without a pre-built client and no ETORO_API_KEY must exit 1.

        EtoroClient raises BrokerAuthError when the env var is absent.
        Monkeypatching EtoroClient in the module triggers the same code path
        that end-users hit when they forget to set their API key.
        """
        monkeypatch.setattr(
            vea, "BrokerAuthError", BrokerAuthError,
        )
        # Patch EtoroClient at the tsml.broker.etoro_client level so main()'s
        # `from tsml.broker.etoro_client import EtoroClient` gets the mock.
        import tsml.broker.etoro_client as _ec_mod
        monkeypatch.setattr(
            _ec_mod, "EtoroClient",
            type("_FakeAuth", (), {"__init__": lambda s: (_ for _ in ()).throw(BrokerAuthError("no key"))}),
        )
        assert vea.main() == 1

    def test_mode_error_returns_one(self, vea, monkeypatch):
        """main() with ETORO_ACCOUNT_MODE=real must exit 1."""
        import tsml.broker.etoro_client as _ec_mod
        monkeypatch.setattr(
            _ec_mod, "EtoroClient",
            type("_FakeMode", (), {"__init__": lambda s: (_ for _ in ()).throw(BrokerModeError("real not allowed"))}),
        )
        assert vea.main() == 1

    def test_main_prints_report(self, vea, capsys):
        """main() must print something to stdout."""
        vea.main(client=FakeBrokerClient())
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_main_output_is_ascii_safe(self, vea, capsys):
        vea.main(client=FakeBrokerClient())
        captured = capsys.readouterr().out
        captured.encode("ascii")   # must not raise

    def test_main_does_not_place_orders(self, vea):
        """place_order must never be called by verify."""
        order_calls: list = []
        client = FakeBrokerClient()
        client.place_order = lambda *a, **kw: order_calls.append(a)
        vea.main(client=client)
        assert order_calls == []
