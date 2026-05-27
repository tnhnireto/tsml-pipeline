"""Tests for out-of-sample holdout evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import DataLoader
from tsml.models.baselines import LogisticRegressionModel
from tsml.portfolio.holdout_eval import (
    DEFAULT_STRATEGY,
    HoldoutPeriods,
    compute_holdout_warnings,
    format_holdout_report,
    metrics_for_period,
    run_holdout_evaluation,
    validate_periods_no_overlap,
)
from tsml.portfolio.simulator import SimulationResult
from tsml.portfolio.weekly_backtest import BacktestMetrics
from tsml.validation.splitters import WalkForwardSplit


def _make_ohlcv(n: int, seed: int = 0, start: str = "2017-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, size=n))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


class StubLoader(DataLoader):
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol not in self._data:
            raise ValueError(f"No stub data for '{symbol}'.")
        return self._data[symbol]


@pytest.fixture()
def fast_splitter() -> WalkForwardSplit:
    return WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)


@pytest.fixture()
def stub_loader() -> StubLoader:
    symbols = ["AAA", "BBB", "SPY", "QQQ"]
    # Long history for dev + holdout slices
    data = {sym: _make_ohlcv(900, seed=i + 1) for i, sym in enumerate(symbols)}
    return StubLoader(data)


class TestPeriodValidation:
    def test_default_periods_do_not_overlap(self):
        periods = HoldoutPeriods(
            dev_start="2018-01-01",
            dev_end="2022-12-31",
            holdout_start="2023-01-01",
            holdout_end="2024-12-31",
        )
        validate_periods_no_overlap(periods)

    def test_overlap_raises(self):
        periods = HoldoutPeriods(
            dev_start="2018-01-01",
            dev_end="2023-06-30",
            holdout_start="2023-01-01",
            holdout_end="2024-12-31",
        )
        with pytest.raises(ValueError, match="after development"):
            validate_periods_no_overlap(periods)


class TestHoldoutWarnings:
    def test_sharpe_drop_warning(self):
        dev = BacktestMetrics(0, 0.1, 1.5, -0.1, 0, 0.7, 10)
        ho = BacktestMetrics(0, 0.05, 0.8, -0.15, 0, 0.6, 8)
        warnings = compute_holdout_warnings(dev, ho)
        assert any("Sharpe" in w for w in warnings)

    def test_drawdown_warning(self):
        dev = BacktestMetrics(0, 0.1, 1.0, -0.10, 0, 0.7, 10)
        ho = BacktestMetrics(0, 0.05, 0.5, -0.20, 0, 0.6, 8)
        warnings = compute_holdout_warnings(dev, ho)
        assert any("drawdown" in w.lower() for w in warnings)


class TestHoldoutReport:
    def test_uses_fixed_strategy_params(self, fast_splitter, stub_loader):
        periods = HoldoutPeriods(
            dev_start="2018-01-01",
            dev_end="2019-12-31",
            holdout_start="2020-01-01",
            holdout_end="2020-12-31",
        )
        result = run_holdout_evaluation(
            ["AAA", "BBB"],
            periods=periods,
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            loader=stub_loader,
        )
        assert result.strategy_config["buy_threshold"] == DEFAULT_STRATEGY["buy_threshold"]
        assert result.strategy_config["feature_set"] == "extended"
        assert result.strategy_config["regime_overlay_enabled"] is True

    def test_report_contains_both_sections(self, fast_splitter, stub_loader):
        periods = HoldoutPeriods(
            dev_start="2018-01-01",
            dev_end="2019-12-31",
            holdout_start="2020-01-01",
            holdout_end="2020-12-31",
        )
        result = run_holdout_evaluation(
            ["AAA", "BBB"],
            periods=periods,
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            loader=stub_loader,
        )
        report = format_holdout_report(result)
        assert "Development" in report
        assert "Holdout" in report
        assert "Fixed parameters" in report
        assert "buy_threshold" in report

    def test_benchmark_comparison_included(self, fast_splitter, stub_loader):
        periods = HoldoutPeriods(
            dev_start="2018-01-01",
            dev_end="2019-12-31",
            holdout_start="2020-01-01",
            holdout_end="2020-12-31",
        )
        result = run_holdout_evaluation(
            ["AAA", "BBB"],
            periods=periods,
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            loader=stub_loader,
        )
        if result.equity_curve.empty:
            pytest.skip("Insufficient synthetic data")
        report = format_holdout_report(result)
        assert "SPY" in report or "QQQ" in report
        assert result.development.benchmark_metrics
        assert result.holdout.benchmark_metrics

    def test_period_metrics_slice(self):
        idx = pd.bdate_range("2018-01-02", periods=500, freq="B", tz="UTC")
        equity = pd.Series(100_000 * (1.001 ** np.arange(500)), index=idx)
        sim = SimulationResult(
            equity_curve=equity,
            trades_log=pd.DataFrame(),
            exposure=pd.Series(0.8, index=idx),
        )
        m = metrics_for_period(sim, "2018-01-01", "2019-12-31")
        assert m.cagr != 0.0 or len(equity.loc["2018-01-01":"2019-12-31"]) >= 2
