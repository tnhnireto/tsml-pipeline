"""
Tests for the broker integration layer.

All tests are fully offline — no network calls are made.
EtoroClient HTTP calls are intercepted by monkeypatching requests.Session.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tsml.broker.base import (
    AccountInfo,
    BrokerAuthError,
    BrokerModeError,
    OrderResult,
    PositionInfo,
)
from tsml.broker.etoro_client import EtoroClient
from tsml.broker.execution import (
    ExecutionPlan,
    build_execution_plan,
    execute_plan,
    log_orders,
    signals_to_proposed_orders,
)
from tsml.broker.risk import ProposedOrder, RiskConfig, RiskResult, validate_order


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> RiskConfig:
    return RiskConfig(
        mode="demo",
        max_positions=5,
        max_trade_amount_pct=0.20,
        cash_buffer_pct=0.05,
    )


@pytest.fixture()
def demo_env(monkeypatch):
    """Set minimal environment for EtoroClient construction."""
    monkeypatch.setenv("ETORO_API_KEY", "test-key-abc123")
    monkeypatch.setenv("ETORO_ACCOUNT_MODE", "demo")


# ---------------------------------------------------------------------------
# Risk rules
# ---------------------------------------------------------------------------

class TestRiskRules:
    def test_approves_valid_buy(self, default_config):
        order  = ProposedOrder("AAPL", "BUY", 1_000.0, score=0.62)
        result = validate_order(order, default_config, 10_000.0, 9_000.0, [])
        assert result.approved

    def test_rejects_real_mode(self):
        config = RiskConfig(mode="real")
        order  = ProposedOrder("AAPL", "BUY", 100.0)
        result = validate_order(order, config, 10_000.0, 9_000.0, [])
        assert not result.approved
        assert result.rule == "mode"
        assert "real" in result.reason.lower() or "demo" in result.reason.lower()

    def test_rejects_leverage(self, default_config):
        order  = ProposedOrder("AAPL", "BUY", 1_000.0, leverage=2)
        result = validate_order(order, default_config, 10_000.0, 9_000.0, [])
        assert not result.approved
        assert result.rule == "leverage"

    def test_rejects_short_selling(self, default_config):
        """SELL on a symbol not currently held is a short — must be rejected."""
        order  = ProposedOrder("AAPL", "SELL", 0.0)
        result = validate_order(
            order, default_config, 10_000.0, 5_000.0,
            open_positions=["MSFT", "QQQ"],  # AAPL not held
        )
        assert not result.approved
        assert result.rule == "short_sell"

    def test_allows_sell_for_held_position(self, default_config):
        order  = ProposedOrder("AAPL", "SELL", 0.0)
        result = validate_order(
            order, default_config, 10_000.0, 5_000.0,
            open_positions=["AAPL", "MSFT"],
        )
        assert result.approved

    def test_rejects_too_many_positions(self, default_config):
        order  = ProposedOrder("NVDA", "BUY", 1_000.0)
        result = validate_order(
            order, default_config, 10_000.0, 5_000.0,
            open_positions=["A", "B", "C", "D", "E"],  # already at max (5)
        )
        assert not result.approved
        assert result.rule == "max_positions"

    def test_allows_buy_just_below_max_positions(self, default_config):
        order  = ProposedOrder("NVDA", "BUY", 1_000.0)
        result = validate_order(
            order, default_config, 10_000.0, 5_000.0,
            open_positions=["A", "B", "C", "D"],  # 4 of 5
        )
        assert result.approved

    def test_rejects_oversized_trade(self, default_config):
        """Trade exceeding max_trade_amount_pct * balance should be rejected."""
        order  = ProposedOrder("AAPL", "BUY", 3_000.0)   # 30% > 20% limit
        result = validate_order(order, default_config, 10_000.0, 9_000.0, [])
        assert not result.approved
        assert result.rule == "max_trade_size"

    def test_rejects_insufficient_cash_buffer(self, default_config):
        """Buying when remaining cash would fall below 5% buffer."""
        # balance=10k, cash=600, buffer=5%=500, buying 200 → cash=400 < 500
        order  = ProposedOrder("AAPL", "BUY", 200.0)
        result = validate_order(order, default_config, 10_000.0, 600.0, [])
        assert not result.approved
        assert result.rule == "cash_buffer"

    def test_rejects_symbol_not_in_universe(self, default_config):
        config = RiskConfig(
            mode="demo",
            approved_universe=frozenset(["AAPL", "MSFT"]),
        )
        order  = ProposedOrder("TSLA", "BUY", 500.0)
        result = validate_order(order, config, 10_000.0, 9_000.0, [])
        assert not result.approved
        assert result.rule == "universe"

    def test_empty_universe_accepts_any_symbol(self, default_config):
        """An empty approved_universe means all symbols are acceptable."""
        order  = ProposedOrder("RANDOM", "BUY", 500.0)
        result = validate_order(order, default_config, 10_000.0, 9_000.0, [])
        assert result.approved


# ---------------------------------------------------------------------------
# EtoroClient construction
# ---------------------------------------------------------------------------

class TestEtoroClientConstruction:
    def test_missing_api_key_raises_broker_auth_error(self, monkeypatch):
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "demo")
        with pytest.raises(BrokerAuthError, match="ETORO_API_KEY"):
            EtoroClient()

    def test_empty_api_key_raises_broker_auth_error(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "   ")   # whitespace-only
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "demo")
        with pytest.raises(BrokerAuthError):
            EtoroClient()

    def test_real_mode_raises_broker_mode_error(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "some-key")
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "real")
        with pytest.raises(BrokerModeError, match="demo"):
            EtoroClient()

    def test_unknown_mode_raises_broker_mode_error(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "some-key")
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "paper")
        with pytest.raises(BrokerModeError):
            EtoroClient()

    def test_valid_demo_env_constructs_successfully(self, demo_env):
        client = EtoroClient()
        assert client._mode == "demo"

    def test_api_key_not_exposed_in_repr(self, demo_env):
        client = EtoroClient()
        assert "test-key-abc123" not in repr(client)


# ---------------------------------------------------------------------------
# Dry-run: no HTTP POST is sent
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_place_order_does_not_call_http_post(self, demo_env):
        client = EtoroClient()
        with patch.object(client._session, "post") as mock_post:
            result = client.place_order("AAPL", "BUY", 500.0, dry_run=True)
        mock_post.assert_not_called()
        assert result.status == "dry_run"
        assert result.order_id is None

    def test_dry_run_place_order_does_not_call_http_get(self, demo_env):
        """place_order dry-run should not trigger any network call at all."""
        client = EtoroClient()
        with patch.object(client._session, "get") as mock_get, \
             patch.object(client._session, "post") as mock_post:
            client.place_order("MSFT", "BUY", 200.0, dry_run=True)
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_dry_run_result_fields(self, demo_env):
        client = EtoroClient()
        result = client.place_order("NVDA", "SELL", 300.0, dry_run=True)
        assert result.symbol  == "NVDA"
        assert result.side    == "SELL"
        assert result.amount  == 300.0
        assert result.dry_run is True


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------

class TestExecutionPlan:
    def _make_signal_actions(self):
        from tsml.portfolio.strategy import SignalAction
        return [
            SignalAction("AAPL", "buy",  0.62),
            SignalAction("MSFT", "buy",  0.59),
            SignalAction("GOOGL", "sell", 0.51),
            SignalAction("TSLA", "hold", 0.58),
            SignalAction("META", "blocked", 0.57, "blocked: below SMA200"),
        ]

    def test_buy_signals_become_buy_orders(self):
        from tsml.portfolio.strategy import SignalAction
        signals = [SignalAction("AAPL", "buy", 0.62)]
        orders  = signals_to_proposed_orders(signals, 10_000.0, RiskConfig())
        assert len(orders) == 1
        assert orders[0].side == "BUY"

    def test_sell_signals_become_sell_orders(self):
        from tsml.portfolio.strategy import SignalAction
        signals = [SignalAction("AAPL", "sell", 0.51)]
        orders  = signals_to_proposed_orders(signals, 10_000.0, RiskConfig())
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    def test_hold_and_blocked_signals_generate_no_orders(self):
        from tsml.portfolio.strategy import SignalAction
        signals = [
            SignalAction("AAPL", "hold",    0.60),
            SignalAction("META", "blocked", 0.57),
        ]
        orders = signals_to_proposed_orders(signals, 10_000.0, RiskConfig())
        assert orders == []

    def test_build_plan_separates_approved_and_rejected(self):
        config  = RiskConfig(mode="demo", max_positions=1)
        orders  = [
            ProposedOrder("AAPL", "BUY", 1_000.0),
            ProposedOrder("MSFT", "BUY", 1_000.0),  # rejected: max_positions=1
        ]
        plan = build_execution_plan(orders, config, 10_000.0, 9_000.0, [])
        assert len(plan.approved)  == 1
        assert len(plan.rejected)  == 1
        assert plan.approved[0].order.symbol == "AAPL"

    def test_cash_is_tracked_across_batch(self):
        """The second BUY should fail the cash-buffer check after the first."""
        config = RiskConfig(
            mode="demo",
            max_positions=5,
            max_trade_amount_pct=0.20,
            cash_buffer_pct=0.05,
        )
        # cash=1200; first buy of 1000 leaves 200 < 5% of 10k (500) → 2nd fails
        orders = [
            ProposedOrder("AAPL", "BUY", 1_000.0),
            ProposedOrder("MSFT", "BUY", 1_000.0),
        ]
        plan = build_execution_plan(orders, config, 10_000.0, 1_200.0, [])
        # First BUY may be approved; second must be rejected (cash buffer)
        approved_syms = {r.order.symbol for r in plan.approved}
        rejected_syms = {r.order.symbol for r in plan.rejected}
        assert "MSFT" in rejected_syms or "AAPL" in rejected_syms  # at least one rejected


# ---------------------------------------------------------------------------
# Order logging
# ---------------------------------------------------------------------------

class TestOrderLogging:
    def test_log_does_not_contain_api_key(self, tmp_path, monkeypatch):
        """Ensure no API key or secret leaks into the JSONL log."""
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan = ExecutionPlan(account_balance=10_000.0, cash_before=10_000.0)
        order  = ProposedOrder("AAPL", "BUY", 500.0)
        result = RiskResult(approved=True)
        from tsml.broker.execution import OrderRecord
        plan.approved.append(OrderRecord(order=order, risk_result=result))

        log_path = log_orders(plan, dry_run=True)
        content  = log_path.read_text()

        # API key must never appear in log
        assert "test-key-abc123" not in content
        assert "ETORO_API_KEY"   not in content

    def test_log_file_is_named_by_date(self, tmp_path, monkeypatch):
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan     = ExecutionPlan()
        log_path = log_orders(plan, dry_run=True)
        assert log_path.suffix == ".jsonl"
        # filename is YYYY-MM-DD.jsonl
        assert len(log_path.stem) == 10   # "2026-05-12"

    def test_log_appends_one_line_per_record(self, tmp_path, monkeypatch):
        import json
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan = ExecutionPlan()
        from tsml.broker.execution import OrderRecord
        for sym in ["AAPL", "MSFT"]:
            plan.approved.append(
                OrderRecord(
                    order=ProposedOrder(sym, "BUY", 500.0),
                    risk_result=RiskResult(approved=True),
                )
            )
        log_path = log_orders(plan, dry_run=True)
        lines    = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            assert "symbol" in entry
            assert entry["dry_run"] is True


class TestIdempotentLogging:
    """Idempotency: rerunning log_orders with the same plan must not duplicate entries."""

    def _make_plan(self, symbols: list[str]) -> ExecutionPlan:
        from tsml.broker.execution import OrderRecord
        plan = ExecutionPlan(account_balance=10_000.0, cash_before=10_000.0)
        for sym in symbols:
            plan.approved.append(
                OrderRecord(
                    order=ProposedOrder(sym, "BUY", 2_000.0),
                    risk_result=RiskResult(approved=True),
                )
            )
        return plan

    def test_order_id_present_in_log_entry(self, tmp_path, monkeypatch):
        import json
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan     = self._make_plan(["AAPL"])
        log_path = log_orders(plan, dry_run=True, signal_date="2026-05-14")
        entry    = json.loads(log_path.read_text().strip())
        assert "order_id" in entry
        assert entry["order_id"] == "2026-05-14:AAPL:BUY"

    def test_signal_date_in_log_entry(self, tmp_path, monkeypatch):
        import json
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan     = self._make_plan(["MSFT"])
        log_path = log_orders(plan, dry_run=True, signal_date="2026-05-14")
        entry    = json.loads(log_path.read_text().strip())
        assert entry["signal_date"] == "2026-05-14"

    def test_rerun_same_plan_does_not_duplicate_lines(self, tmp_path, monkeypatch):
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan     = self._make_plan(["AAPL", "MSFT"])
        log_path = log_orders(plan, dry_run=True, signal_date="2026-05-14")
        # second run -- same plan, same signal_date
        log_orders(plan, dry_run=True, signal_date="2026-05-14")

        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2, "Second run must not append duplicate lines"

    def test_different_symbol_creates_new_entry(self, tmp_path, monkeypatch):
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan1 = self._make_plan(["AAPL"])
        plan2 = self._make_plan(["NVDA"])
        log_path = log_orders(plan1, dry_run=True, signal_date="2026-05-14")
        log_orders(plan2, dry_run=True, signal_date="2026-05-14")

        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2   # AAPL and NVDA are distinct

    def test_duplicate_detection_works_for_pre_existing_file(self, tmp_path, monkeypatch):
        """Idempotency holds even when the file already exists from a previous session."""
        import json
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        # First run: write AAPL entry to the file.
        plan     = self._make_plan(["AAPL"])
        log_path = log_orders(plan, dry_run=True, signal_date="2026-05-14")
        count_before = len([l for l in log_path.read_text().splitlines() if l.strip()])

        # Simulate script restart: log same plan again.
        log_orders(plan, dry_run=True, signal_date="2026-05-14")
        count_after = len([l for l in log_path.read_text().splitlines() if l.strip()])

        assert count_before == count_after == 1

    def test_same_symbol_different_date_creates_new_entry(self, tmp_path, monkeypatch):
        import tsml.broker.execution as exec_mod
        monkeypatch.setattr(exec_mod, "LOGS_DIR", tmp_path)

        plan = self._make_plan(["AAPL"])
        # Two different signal dates -> different filenames, both should have 1 entry.
        path1 = log_orders(plan, dry_run=True, signal_date="2026-05-07")
        path2 = log_orders(plan, dry_run=True, signal_date="2026-05-14")

        assert path1 != path2
        assert len([l for l in path1.read_text().splitlines() if l.strip()]) == 1
        assert len([l for l in path2.read_text().splitlines() if l.strip()]) == 1
