"""
Tests for src/tsml/portfolio/state.py

Covers:
- load_state: default state when file is absent, JSON load, graceful fallback
- save_state: file creation, contents, parent-dir creation
- round-trip: save then load returns identical data
- apply_orders: BUY adds positions, SELL removes positions, cash changes,
                does not mutate input state, SELL with amount=0 handled
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from tsml.portfolio.state import PortfolioState, apply_orders, load_state, save_state


# ---------------------------------------------------------------------------
# Minimal fake OrderRecord so tests do not import the broker package
# ---------------------------------------------------------------------------

class _FakeOrder:
    def __init__(self, symbol: str, side: str, amount: float = 2_000.0):
        self.symbol = symbol
        self.side   = side
        self.amount = amount


class _FakeRecord:
    def __init__(self, symbol: str, side: str, amount: float = 2_000.0):
        self.order = _FakeOrder(symbol, side, amount)


# ===========================================================================
# load_state
# ===========================================================================

class TestLoadState:
    def test_default_state_when_file_absent(self, tmp_path):
        state = load_state(tmp_path / "missing.json")
        assert isinstance(state, PortfolioState)
        assert state.cash == 10_000.0
        assert state.positions == {}
        assert state.last_signal_date is None
        assert state.last_rebalance_date is None

    def test_returns_portfolio_state_type(self, tmp_path):
        state = load_state(tmp_path / "nope.json")
        assert type(state) is PortfolioState

    def test_graceful_on_corrupt_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json !!!", encoding="utf-8")
        state = load_state(bad)
        assert state.cash == 10_000.0
        assert state.positions == {}

    def test_loads_cash(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"cash": 7_500.0, "positions": {}}), encoding="utf-8")
        assert load_state(path).cash == pytest.approx(7_500.0)

    def test_loads_positions(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"cash": 5_000.0, "positions": {"AAPL": 1.0, "MSFT": 1.0}}), encoding="utf-8")
        state = load_state(path)
        assert set(state.positions.keys()) == {"AAPL", "MSFT"}

    def test_loads_dates(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(
            json.dumps({
                "cash": 10_000.0,
                "positions": {},
                "last_signal_date": "2026-05-07",
                "last_rebalance_date": "2026-05-07",
            }),
            encoding="utf-8",
        )
        state = load_state(path)
        assert state.last_signal_date    == "2026-05-07"
        assert state.last_rebalance_date == "2026-05-07"

    def test_missing_keys_get_defaults(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{}", encoding="utf-8")
        state = load_state(path)
        assert state.cash == 10_000.0
        assert state.positions == {}


# ===========================================================================
# save_state
# ===========================================================================

class TestSaveState:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "state.json"
        save_state(PortfolioState(), path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "deep" / "state.json"
        save_state(PortfolioState(), path)
        assert path.exists()

    def test_file_is_valid_json(self, tmp_path):
        path = tmp_path / "state.json"
        save_state(PortfolioState(), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "cash" in data
        assert "positions" in data

    def test_no_secrets_in_file(self, tmp_path):
        path = tmp_path / "state.json"
        save_state(PortfolioState(cash=9_000.0, positions={"SPY": 1.0}), path)
        text = path.read_text(encoding="utf-8")
        for secret_keyword in ("api_key", "password", "secret", "token"):
            assert secret_keyword not in text.lower()


# ===========================================================================
# Round-trip
# ===========================================================================

class TestRoundTrip:
    def test_roundtrip(self, tmp_path):
        original = PortfolioState(
            cash=8_200.0,
            positions={"NVDA": 1.0, "AAPL": 1.0},
            last_signal_date="2026-05-14",
            last_rebalance_date="2026-05-14",
        )
        path = tmp_path / "state.json"
        save_state(original, path)
        restored = load_state(path)

        assert restored.cash                == pytest.approx(original.cash)
        assert restored.positions           == original.positions
        assert restored.last_signal_date    == original.last_signal_date
        assert restored.last_rebalance_date == original.last_rebalance_date

    def test_multiple_loads_are_independent(self, tmp_path):
        path = tmp_path / "state.json"
        save_state(PortfolioState(positions={"MSFT": 1.0}), path)
        a = load_state(path)
        b = load_state(path)
        a.positions["NEW"] = 1.0
        assert "NEW" not in b.positions


# ===========================================================================
# apply_orders
# ===========================================================================

class TestApplyOrders:
    def test_buy_adds_to_positions(self):
        state  = PortfolioState()
        record = _FakeRecord("AAPL", "BUY", 2_000.0)
        new    = apply_orders(state, [record], signal_date="2026-05-14")
        assert "AAPL" in new.positions

    def test_buy_decrements_cash(self):
        state  = PortfolioState(cash=10_000.0)
        record = _FakeRecord("AAPL", "BUY", 2_000.0)
        new    = apply_orders(state, [record], signal_date="2026-05-14")
        assert new.cash == pytest.approx(8_000.0)

    def test_sell_removes_from_positions(self):
        state  = PortfolioState(positions={"MSFT": 1.0})
        record = _FakeRecord("MSFT", "SELL", 0.0)
        new    = apply_orders(state, [record], signal_date="2026-05-14")
        assert "MSFT" not in new.positions

    def test_sell_amount_zero_does_not_alter_cash(self):
        state  = PortfolioState(cash=10_000.0, positions={"MSFT": 1.0})
        record = _FakeRecord("MSFT", "SELL", 0.0)
        new    = apply_orders(state, [record], signal_date="2026-05-14")
        assert new.cash == pytest.approx(10_000.0)

    def test_sell_unknown_symbol_is_harmless(self):
        state  = PortfolioState(positions={})
        record = _FakeRecord("UNKNOWN", "SELL", 0.0)
        new    = apply_orders(state, [record], signal_date="2026-05-14")
        assert new.positions == {}

    def test_cash_floored_at_zero(self):
        state  = PortfolioState(cash=500.0)
        record = _FakeRecord("NVDA", "BUY", 10_000.0)
        new    = apply_orders(state, [record], signal_date="2026-05-14")
        assert new.cash == pytest.approx(0.0)

    def test_signal_date_stored(self):
        state = PortfolioState()
        new   = apply_orders(state, [], signal_date="2026-05-14")
        assert new.last_signal_date    == "2026-05-14"
        assert new.last_rebalance_date == "2026-05-14"

    def test_does_not_mutate_input(self):
        state  = PortfolioState(cash=10_000.0, positions={"AAPL": 1.0})
        record = _FakeRecord("NVDA", "BUY", 2_000.0)
        _      = apply_orders(state, [record], signal_date="2026-05-14")
        # input unchanged
        assert state.cash == pytest.approx(10_000.0)
        assert "NVDA" not in state.positions

    def test_empty_records_updates_date_only(self):
        state = PortfolioState(cash=10_000.0, positions={"AAPL": 1.0})
        new   = apply_orders(state, [], signal_date="2026-05-14")
        assert new.cash == pytest.approx(10_000.0)
        assert new.positions == {"AAPL": 1.0}

    def test_multiple_buys(self):
        state   = PortfolioState(cash=10_000.0)
        records = [_FakeRecord("AAPL", "BUY", 2_000.0), _FakeRecord("MSFT", "BUY", 2_000.0)]
        new     = apply_orders(state, records, signal_date="2026-05-14")
        assert "AAPL" in new.positions
        assert "MSFT" in new.positions
        assert new.cash == pytest.approx(6_000.0)
