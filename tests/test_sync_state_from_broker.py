"""
Tests for broker-to-local state sync.

Core logic is in ``tsml.broker.sync``; CLI is ``scripts/sync_state_from_broker.py``.
No network calls, no real API keys, no orders.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tsml.broker.base import AccountInfo, BrokerAuthError, BrokerError, PositionInfo
from tsml.broker.sync import build_state_from_broker, format_sync_report
from tsml.portfolio.state import PortfolioState, load_state, save_state


# ---------------------------------------------------------------------------
# Stub broker
# ---------------------------------------------------------------------------

class FakeBrokerClient:
    def __init__(
        self,
        *,
        cash: float = 10_000.0,
        positions: list[tuple[str, float, float]] | None = None,
        raise_on_account: Exception | None = None,
        raise_on_positions: Exception | None = None,
    ) -> None:
        self._cash = cash
        self._positions = positions or []
        self._raise_account = raise_on_account
        self._raise_positions = raise_on_positions

    def get_account(self) -> AccountInfo:
        if self._raise_account is not None:
            raise self._raise_account
        return AccountInfo(
            account_id="demo-1",
            mode="demo",
            balance=self._cash,
            equity=0.0,
            cash=self._cash,
        )

    def get_positions(self) -> list[PositionInfo]:
        if self._raise_positions is not None:
            raise self._raise_positions
        return [
            PositionInfo(symbol=sym, quantity=qty, market_value=mv, open_price=1.0)
            for sym, qty, mv in self._positions
        ]

    def get_instrument(self, symbol):  # pragma: no cover
        raise NotImplementedError

    def place_order(self, symbol, side, amount, dry_run=True):  # pragma: no cover
        raise NotImplementedError


# ===========================================================================
# build_state_from_broker()
# ===========================================================================

class TestBuildStateFromBroker:

    def test_cash_copied_from_broker(self):
        client = FakeBrokerClient(cash=99_307.54)
        result = build_state_from_broker(client)
        assert result.state.cash == pytest.approx(99_307.54)
        assert result.broker_cash == pytest.approx(99_307.54)

    def test_positions_copied_with_quantities(self):
        client = FakeBrokerClient(
            cash=5_000.0,
            positions=[("AAPL", 0.5, 100.0), ("MSFT", 2.0, 800.0)],
        )
        result = build_state_from_broker(client)
        assert result.state.positions == {"AAPL": 0.5, "MSFT": 2.0}
        assert result.symbols == ["AAPL", "MSFT"]

    def test_empty_broker_positions_produce_empty_dict(self):
        client = FakeBrokerClient(cash=10_000.0, positions=[])
        result = build_state_from_broker(client)
        assert result.state.positions == {}
        assert result.broker_position_count == 0

    def test_dates_cleared(self):
        result = build_state_from_broker(FakeBrokerClient())
        assert result.state.last_signal_date is None
        assert result.state.last_rebalance_date is None

    def test_broker_error_propagates(self):
        client = FakeBrokerClient(raise_on_account=BrokerError("timeout"))
        with pytest.raises(BrokerError, match="timeout"):
            build_state_from_broker(client)


# ===========================================================================
# format_sync_report()
# ===========================================================================

class TestFormatSyncReport:

    def test_report_is_ascii_safe(self):
        result = build_state_from_broker(
            FakeBrokerClient(cash=12_345.67, positions=[("AAPL", 1.0, 100.0)])
        )
        report = format_sync_report(result, confirmed=False, written=False)
        report.encode("ascii")

    def test_dry_run_banner(self):
        result = build_state_from_broker(FakeBrokerClient())
        report = format_sync_report(result, confirmed=False, written=False)
        assert "DRY-RUN" in report
        assert "NO CHANGES MADE" in report

    def test_written_banner(self):
        result = build_state_from_broker(FakeBrokerClient())
        report = format_sync_report(result, confirmed=True, written=True)
        assert "STATE FILE UPDATED" in report


# ===========================================================================
# CLI — scripts/sync_state_from_broker.py
# ===========================================================================

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sync_state_from_broker.py"


@pytest.fixture(scope="module")
def sync_mod():
    spec = importlib.util.spec_from_file_location("sync_state_from_broker", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


class TestMain:

    def test_dry_run_does_not_write_state(self, sync_mod, tmp_path, monkeypatch, capsys):
        state_file = tmp_path / "portfolio_state.json"
        save_state(PortfolioState(cash=1.0, positions={"OLD": 1.0}), state_file)
        client = FakeBrokerClient(cash=9_000.0, positions=[("AAPL", 1.5, 200.0)])

        rc = sync_mod.main(client=client, confirm=False, state_path=state_file)
        out = capsys.readouterr().out

        assert rc == 0
        assert "DRY-RUN" in out
        assert "NO CHANGES MADE" in out
        loaded = load_state(state_file)
        assert loaded.cash == 1.0
        assert "OLD" in loaded.positions

    def test_confirm_writes_state(self, sync_mod, tmp_path, capsys):
        state_file = tmp_path / "portfolio_state.json"
        client = FakeBrokerClient(
            cash=99_307.54,
            positions=[("AAPL", 0.049485, 100.0), ("MSFT", 3.0, 900.0)],
        )

        rc = sync_mod.main(client=client, confirm=True, state_path=state_file)
        out = capsys.readouterr().out

        assert rc == 0
        assert "STATE FILE UPDATED" in out
        loaded = load_state(state_file)
        assert loaded.cash == pytest.approx(99_307.54)
        assert loaded.positions == {"AAPL": pytest.approx(0.049485), "MSFT": 3.0}
        assert loaded.last_signal_date is None
        assert loaded.last_rebalance_date is None

    def test_confirm_empty_positions_writes_empty_dict(self, sync_mod, tmp_path):
        state_file = tmp_path / "portfolio_state.json"
        client = FakeBrokerClient(cash=10_000.0, positions=[])

        assert sync_mod.main(client=client, confirm=True, state_path=state_file) == 0
        loaded = load_state(state_file)
        assert loaded.positions == {}

    def test_broker_error_returns_one(self, sync_mod, tmp_path, capsys):
        state_file = tmp_path / "portfolio_state.json"
        client = FakeBrokerClient(raise_on_positions=BrokerError("positions unavailable"))

        rc = sync_mod.main(client=client, confirm=False, state_path=state_file)
        err = capsys.readouterr().err

        assert rc == 1
        assert "positions unavailable" in err
        assert not state_file.exists()

    def test_auth_error_returns_one(self, sync_mod, tmp_path, monkeypatch, capsys):
        import tsml.broker.etoro_client as ec_mod

        monkeypatch.setattr(
            ec_mod,
            "EtoroClient",
            type(
                "_FailAuth",
                (),
                {"__init__": lambda s: (_ for _ in ()).throw(BrokerAuthError("no key"))},
            ),
        )
        state_file = tmp_path / "portfolio_state.json"

        rc = sync_mod.main(confirm=False, state_path=state_file)
        err = capsys.readouterr().err

        assert rc == 1
        assert "no key" in err

    def test_written_state_has_no_secrets(self, sync_mod, tmp_path, monkeypatch):
        state_file = tmp_path / "portfolio_state.json"
        monkeypatch.setenv("ETORO_API_KEY", "secret-api-key")
        monkeypatch.setenv("ETORO_USER_KEY", "secret-user-key")
        client = FakeBrokerClient(cash=1_000.0, positions=[])

        sync_mod.main(client=client, confirm=True, state_path=state_file)
        content = state_file.read_text(encoding="utf-8")
        assert "secret-api-key" not in content
        assert "secret-user-key" not in content
        assert "ETORO" not in content

    def test_dry_run_exit_zero(self, sync_mod, tmp_path):
        state_file = tmp_path / "portfolio_state.json"
        client = FakeBrokerClient(cash=500.0, positions=[])
        assert sync_mod.main(client=client, confirm=False, state_path=state_file) == 0
