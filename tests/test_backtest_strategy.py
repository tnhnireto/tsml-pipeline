"""Tests for backtest turnover-control signal rules."""

from __future__ import annotations

import pandas as pd
import pytest

from tsml.portfolio.backtest_strategy import generate_backtest_signals


def _ranking(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestHysteresis:
    def test_existing_holding_kept_between_thresholds(self):
        ranking = _ranking([
            {"symbol": "NVDA", "score": 0.70, "above_sma_200": True},
            {"symbol": "AAPL", "score": 0.55, "above_sma_200": True},
        ])
        signals = generate_backtest_signals(
            ranking,
            {"AAPL"},
            top_n=2,
            buy_threshold=0.58,
            sell_threshold=0.52,
            weeks_held={"AAPL": 3},
            min_hold_weeks=0,
        )
        actions = {s.symbol: s.action for s in signals}
        assert actions["AAPL"] == "hold"
        assert actions.get("NVDA") == "buy"

    def test_new_position_requires_buy_threshold(self):
        ranking = _ranking([
            {"symbol": "META", "score": 0.56, "above_sma_200": True},
            {"symbol": "MSFT", "score": 0.60, "above_sma_200": True},
        ])
        signals = generate_backtest_signals(
            ranking,
            set(),
            top_n=5,
            buy_threshold=0.58,
            sell_threshold=0.52,
        )
        actions = {s.symbol: s.action for s in signals}
        assert "META" not in actions or actions.get("META") != "buy"
        assert actions["MSFT"] == "buy"


class TestMinHoldWeeks:
    def test_min_hold_prevents_early_sell(self):
        ranking = _ranking([
            {"symbol": "AAPL", "score": 0.50, "above_sma_200": True},
        ])
        signals = generate_backtest_signals(
            ranking,
            {"AAPL"},
            top_n=5,
            buy_threshold=0.58,
            sell_threshold=0.52,
            weeks_held={"AAPL": 1},
            min_hold_weeks=2,
        )
        assert any(s.symbol == "AAPL" and s.action == "hold" for s in signals)

    def test_can_sell_after_min_hold_weeks(self):
        ranking = _ranking([
            {"symbol": "AAPL", "score": 0.50, "above_sma_200": True},
        ])
        signals = generate_backtest_signals(
            ranking,
            {"AAPL"},
            top_n=5,
            buy_threshold=0.58,
            sell_threshold=0.52,
            weeks_held={"AAPL": 2},
            min_hold_weeks=2,
        )
        assert any(s.symbol == "AAPL" and s.action == "sell" for s in signals)


class TestHardRiskFilter:
    def test_hard_risk_forces_sell_despite_min_hold(self):
        ranking = _ranking([
            {"symbol": "AAPL", "score": 0.60, "above_sma_200": False},
        ])
        signals = generate_backtest_signals(
            ranking,
            {"AAPL"},
            top_n=5,
            min_score_downtrend=0.62,
            buy_threshold=0.58,
            sell_threshold=0.52,
            weeks_held={"AAPL": 0},
            min_hold_weeks=4,
        )
        sell = [s for s in signals if s.symbol == "AAPL" and s.action == "sell"]
        assert len(sell) == 1
        assert "hard risk" in sell[0].reason or "SMA200" in sell[0].reason


class TestMaxPositions:
    def test_portfolio_does_not_exceed_top_n(self):
        ranking = _ranking([
            {"symbol": f"S{i}", "score": 0.80 - i * 0.02, "above_sma_200": True}
            for i in range(8)
        ])
        held = {f"H{i}" for i in range(3)}
        signals = generate_backtest_signals(
            ranking,
            held,
            top_n=5,
            buy_threshold=0.58,
            sell_threshold=0.52,
            weeks_held={sym: 5 for sym in held},
            min_hold_weeks=0,
        )
        active = {s.symbol for s in signals if s.action in ("buy", "hold")}
        assert len(active) <= 5


class TestTurnoverReduction:
    def test_turnover_control_reduces_trades_in_synthetic_simulation(self):
        from tsml.data_loader.base import DataLoader
        from tsml.models.baselines import CalibratedLogisticRegressionModel
        from tsml.portfolio.simulator import simulate
        from tsml.validation.splitters import WalkForwardSplit
        import numpy as np

        def _make_ohlcv(n: int, seed: int) -> pd.DataFrame:
            rng = np.random.default_rng(seed)
            dates = pd.bdate_range("2020-01-02", periods=n, freq="B", tz="UTC")
            close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, size=n))
            return pd.DataFrame(
                {
                    "open": close * 0.999,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
                },
                index=dates,
            )

        class StubLoader(DataLoader):
            def __init__(self) -> None:
                self._data = {
                    sym: _make_ohlcv(400, seed=i + 1)
                    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"])
                }

            def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
                return self._data[symbol]

        splitter = WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)
        model = CalibratedLogisticRegressionModel()
        loader = StubLoader()
        common = dict(
            symbols=list(loader._data.keys()),
            model=model,
            splitter=splitter,
            start_date="2020-01-01",
            end_date="2021-06-30",
            target="threshold",
            top_n=5,
            min_score=0.55,
            min_score_downtrend=0.62,
            costs_bps=5.0,
            loader=loader,
        )

        legacy = simulate(**common, turnover_control=False)
        controlled = simulate(
            **common,
            turnover_control=True,
            buy_threshold=0.58,
            sell_threshold=0.52,
            min_hold_weeks=2,
        )

        if legacy.trades_log.empty and controlled.trades_log.empty:
            pytest.skip("Synthetic run produced no trades")

        assert controlled.turnover_total <= legacy.turnover_total
        assert len(controlled.trades_log) <= len(legacy.trades_log)
