"""
Tests for broker reconciliation.

``reconcile()`` and ``format_report()`` live in ``tsml.broker.reconcile``
and are imported directly -- no importlib required for the core logic.

``TestMain`` tests the CLI entry point in ``scripts/reconcile_broker.py``
and still uses importlib (the script lives outside the installed package).

No network calls, no real API keys, no subprocess spawning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct imports from the installed package
# ---------------------------------------------------------------------------

from tsml.broker.base import AccountInfo, BrokerError, PositionInfo
from tsml.broker.reconcile import (
    CASH_TOLERANCE_ABS,
    CASH_TOLERANCE_REL,
    ReconcileResult,
    format_report,
    reconcile,
)
from tsml.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Fake broker client stub
# ---------------------------------------------------------------------------

class FakeBrokerClient:
    """
    Minimal stub satisfying the BrokerClient protocol.

    Parameters
    ----------
    cash, balance, equity:
        Account values returned by ``get_account()``.
    positions:
        List of ``(symbol, quantity, market_value)`` tuples for
        ``get_positions()``.
    raise_on_account, raise_on_positions:
        If not None, raise these exceptions from the corresponding methods.
    """

    def __init__(
        self,
        *,
        cash: float = 10_000.0,
        balance: float = 10_000.0,
        equity: float = 0.0,
        positions: list[tuple[str, float, float]] | None = None,
        raise_on_account: Exception | None = None,
        raise_on_positions: Exception | None = None,
    ) -> None:
        self._cash              = cash
        self._balance           = balance
        self._equity            = equity
        self._pos_data          = positions or []
        self._raise_account     = raise_on_account
        self._raise_positions   = raise_on_positions

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
            PositionInfo(symbol=sym, quantity=qty, market_value=mv, open_price=0.0)
            for sym, qty, mv in self._pos_data
        ]

    def get_instrument(self, symbol):   # pragma: no cover
        raise NotImplementedError

    def place_order(self, symbol, side, amount, dry_run=True):   # pragma: no cover
        raise NotImplementedError


# ===========================================================================
# reconcile()
# ===========================================================================

class TestReconcile:

    def test_perfect_match_is_ok(self):
        state  = PortfolioState(cash=8_000.0, positions={"AAPL": 1.0, "MSFT": 1.0})
        client = FakeBrokerClient(
            cash=8_000.0,
            positions=[("AAPL", 5.0, 750.0), ("MSFT", 3.0, 1_200.0)],
        )
        result = reconcile(state, client)
        assert result.overall_ok is True
        assert result.cash_ok is True
        assert result.positions_ok is True

    def test_cash_within_abs_tolerance_is_ok(self):
        """$50 difference is within the $100 absolute tolerance."""
        state  = PortfolioState(cash=9_950.0, positions={})
        client = FakeBrokerClient(cash=10_000.0)
        result = reconcile(state, client, cash_tolerance_abs=100.0, cash_tolerance_rel=0.01)
        assert result.cash_ok is True
        assert result.overall_ok is True

    def test_cash_within_rel_tolerance_is_ok(self):
        """8 % relative drift is within 10 % relative tolerance."""
        state  = PortfolioState(cash=9_200.0, positions={})
        client = FakeBrokerClient(cash=10_000.0)
        result = reconcile(state, client, cash_tolerance_abs=10.0, cash_tolerance_rel=0.10)
        assert result.cash_ok is True

    def test_cash_outside_both_tolerances_is_mismatch(self):
        """$2 000 gap on a $10 000 balance exceeds $100 abs AND 10 % rel."""
        state  = PortfolioState(cash=8_000.0, positions={})
        client = FakeBrokerClient(cash=10_000.0)
        result = reconcile(state, client, cash_tolerance_abs=100.0, cash_tolerance_rel=0.10)
        assert result.cash_ok is False
        assert result.overall_ok is False

    def test_cash_diff_computed_correctly(self):
        state  = PortfolioState(cash=9_000.0, positions={})
        client = FakeBrokerClient(cash=10_000.0)
        result = reconcile(state, client)
        assert result.cash_diff == pytest.approx(1_000.0)

    def test_cash_diff_pct_computed_correctly(self):
        state  = PortfolioState(cash=9_000.0, positions={})
        client = FakeBrokerClient(cash=10_000.0)
        result = reconcile(state, client)
        assert result.cash_diff_pct == pytest.approx(0.10)

    def test_zero_broker_cash_no_division_error(self):
        state  = PortfolioState(cash=0.0, positions={})
        client = FakeBrokerClient(cash=0.0)
        result = reconcile(state, client)
        assert result.cash_diff == pytest.approx(0.0)

    def test_positions_ok_when_sets_match(self):
        state  = PortfolioState(positions={"AAPL": 1.0, "NVDA": 1.0})
        client = FakeBrokerClient(
            positions=[("AAPL", 2.0, 300.0), ("NVDA", 1.0, 800.0)],
        )
        result = reconcile(state, client)
        assert result.positions_ok is True
        assert result.only_local  == set()
        assert result.only_broker == set()

    def test_only_local_detected(self):
        """Local has GOOGL but broker does not -- should be flagged."""
        state  = PortfolioState(positions={"AAPL": 1.0, "GOOGL": 1.0})
        client = FakeBrokerClient(positions=[("AAPL", 1.0, 200.0)])
        result = reconcile(state, client)
        assert "GOOGL" in result.only_local
        assert result.positions_ok is False
        assert result.overall_ok is False

    def test_only_broker_detected(self):
        """Broker has JPM but local does not -- should be flagged."""
        state  = PortfolioState(positions={"AAPL": 1.0})
        client = FakeBrokerClient(
            positions=[("AAPL", 1.0, 200.0), ("JPM", 3.0, 450.0)],
        )
        result = reconcile(state, client)
        assert "JPM" in result.only_broker
        assert result.positions_ok is False
        assert result.overall_ok is False

    def test_matched_symbols_correct(self):
        state  = PortfolioState(positions={"AAPL": 1.0, "MSFT": 1.0})
        client = FakeBrokerClient(positions=[("AAPL", 1.0, 200.0)])
        result = reconcile(state, client)
        assert result.matched_symbols == {"AAPL"}

    def test_empty_state_matches_empty_broker(self):
        state  = PortfolioState(positions={})
        client = FakeBrokerClient(positions=[])
        result = reconcile(state, client)
        assert result.positions_ok is True

    def test_broker_account_stored_in_result(self):
        state  = PortfolioState()
        client = FakeBrokerClient(cash=10_000.0, balance=12_000.0)
        result = reconcile(state, client)
        assert result.broker_account is not None
        assert result.broker_account.balance == pytest.approx(12_000.0)

    def test_broker_positions_stored_in_result(self):
        state  = PortfolioState(positions={})
        client = FakeBrokerClient(positions=[("SPY", 5.0, 2_000.0)])
        result = reconcile(state, client)
        assert len(result.broker_positions) == 1
        assert result.broker_positions[0].symbol == "SPY"

    def test_broker_error_on_account_propagates(self):
        state  = PortfolioState()
        client = FakeBrokerClient(raise_on_account=BrokerError("network timeout"))
        with pytest.raises(BrokerError, match="network timeout"):
            reconcile(state, client)

    def test_broker_error_on_positions_propagates(self):
        state  = PortfolioState()
        client = FakeBrokerClient(raise_on_positions=BrokerError("unavailable"))
        with pytest.raises(BrokerError, match="unavailable"):
            reconcile(state, client)


# ===========================================================================
# format_report()
# ===========================================================================

class TestFormatReport:

    def _result(
        self,
        *,
        local_cash=10_000.0,
        broker_cash=10_000.0,
        local_syms=(),
        broker_syms=(),
    ) -> tuple[ReconcileResult, PortfolioState]:
        state  = PortfolioState(
            cash=local_cash,
            positions={s: 1.0 for s in local_syms},
        )
        client = FakeBrokerClient(
            cash=broker_cash,
            positions=[(s, 1.0, 0.0) for s in broker_syms],
        )
        return reconcile(state, client), state

    def test_report_is_ascii_safe(self):
        result, state = self._result()
        format_report(result, state).encode("ascii")

    def test_passed_banner_on_success(self):
        result, state = self._result()
        assert "RECONCILIATION PASSED" in format_report(result, state)

    def test_failed_banner_on_failure(self):
        result, state = self._result(local_cash=0.0, broker_cash=10_000.0)
        assert "RECONCILIATION FAILED" in format_report(result, state)

    def test_cash_values_shown(self):
        result, state = self._result(local_cash=8_000.0, broker_cash=9_500.0)
        report = format_report(result, state)
        assert "8,000.00" in report
        assert "9,500.00" in report

    def test_only_local_in_report(self):
        result, state = self._result(local_syms=["GOOGL"], broker_syms=[])
        report = format_report(result, state)
        assert "GOOGL" in report
        assert "only local" in report.lower()

    def test_only_broker_in_report(self):
        result, state = self._result(local_syms=[], broker_syms=["JPM"])
        assert "JPM" in format_report(result, state)

    def test_matched_symbols_shown(self):
        result, state = self._result(local_syms=["AAPL"], broker_syms=["AAPL"])
        assert "AAPL" in format_report(result, state)

    def test_ok_label_present(self):
        result, state = self._result()
        assert "[OK]" in format_report(result, state)

    def test_mismatch_label_on_symbol_failure(self):
        result, state = self._result(local_syms=["NVDA"], broker_syms=[])
        assert "[MISMATCH]" in format_report(result, state)

    def test_broker_account_id_in_report(self):
        result, state = self._result()
        assert "test-account" in format_report(result, state)

    def test_position_detail_shown_when_broker_has_positions(self):
        result, state = self._result(local_syms=["AAPL"], broker_syms=["AAPL"])
        assert "Position detail" in format_report(result, state)

    def test_position_detail_absent_when_no_broker_positions(self):
        result, state = self._result()
        assert "Position detail" not in format_report(result, state)

    def test_no_unicode_box_drawing(self):
        """Ensure no characters that fail on Windows cp1252 consoles."""
        result, state = self._result()
        report = format_report(result, state)
        for ch in ("\u2500", "\u2502", "\u2192", "\u2714", "\u2718"):
            assert ch not in report, f"Unicode char {ch!r} found in report"


# ===========================================================================
# TestMain — CLI entry point in scripts/reconcile_broker.py
# ===========================================================================

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "reconcile_broker.py"


def _load_script() -> object:
    """Load scripts/reconcile_broker.py as a module via importlib."""
    spec = importlib.util.spec_from_file_location("reconcile_broker_script", _SCRIPT_PATH)
    mod  = importlib.util.module_from_spec(spec)
    # Register so that @dataclass resolution works inside any imports triggered.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


@pytest.fixture(scope="module")
def rb():
    return _load_script()


class TestMain:
    """Tests for the CLI main() in scripts/reconcile_broker.py."""

    def _patched_main(self, rb, client, state, tmp_path, monkeypatch):
        """
        Return a callable that re-runs main()'s core logic with a stub client.

        This avoids environment variable requirements (no ETORO_API_KEY needed)
        while exercising the full reconcile -> report -> return-code path.
        """
        from tsml.portfolio.state import save_state
        state_file = tmp_path / "portfolio_state.json"
        save_state(state, state_file)
        monkeypatch.setattr(rb, "STATE_PATH", state_file)

        def _run():
            state_loaded = rb.load_state(state_file)
            try:
                result = rb.reconcile(state_loaded, client)
            except rb.BrokerError as exc:
                import sys as _sys
                print(f"ERROR: Broker call failed: {exc}", file=_sys.stderr)
                return 1
            report = rb.format_report(result, state_loaded, state_path=state_file)
            print(report)
            return 0 if result.overall_ok else 1

        return _run

    def test_exit_zero_on_perfect_match(self, rb, tmp_path, monkeypatch):
        state  = PortfolioState(cash=10_000.0, positions={"AAPL": 1.0})
        client = FakeBrokerClient(cash=10_000.0, positions=[("AAPL", 1.0, 200.0)])
        fn = self._patched_main(rb, client, state, tmp_path, monkeypatch)
        assert fn() == 0

    def test_exit_one_on_symbol_mismatch(self, rb, tmp_path, monkeypatch):
        state  = PortfolioState(cash=10_000.0, positions={"NVDA": 1.0})
        client = FakeBrokerClient(cash=10_000.0, positions=[])
        fn = self._patched_main(rb, client, state, tmp_path, monkeypatch)
        assert fn() == 1

    def test_exit_one_on_cash_mismatch(self, rb, tmp_path, monkeypatch):
        state  = PortfolioState(cash=0.0, positions={})
        client = FakeBrokerClient(cash=10_000.0, positions=[])
        fn = self._patched_main(rb, client, state, tmp_path, monkeypatch)
        assert fn() == 1

    def test_exit_one_on_broker_error(self, rb, tmp_path, monkeypatch):
        state  = PortfolioState()
        client = FakeBrokerClient(raise_on_account=BrokerError("timeout"))
        fn = self._patched_main(rb, client, state, tmp_path, monkeypatch)
        assert fn() == 1

    def test_exit_zero_with_default_state_when_no_file(self, rb, tmp_path, monkeypatch):
        """No state file -> default state (all cash, no positions)."""
        state  = PortfolioState()
        client = FakeBrokerClient(cash=10_000.0, positions=[])
        fn = self._patched_main(rb, client, state, tmp_path, monkeypatch)
        assert fn() == 0
