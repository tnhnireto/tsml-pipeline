"""Tests for robustness and stress-testing analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.portfolio.robustness import (
    build_universe_variants,
    compute_rolling_metrics,
    evaluate_regime_metrics,
    metrics_for_period,
    summarise_score_calibration,
)
from tsml.portfolio.simulator import SimulationResult
from tsml.portfolio.weekly_backtest import BacktestMetrics


def _equity_curve(n: int = 400, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
    rets = rng.normal(0.0004, 0.01, n)
    return pd.Series(100_000 * np.cumprod(1 + rets), index=idx, name="portfolio_value")


def _simulation_from_equity(eq: pd.Series) -> SimulationResult:
    trades = pd.DataFrame(
        {
            "date": eq.index[::50],
            "symbol": ["AAPL"] * len(eq.index[::50]),
            "action": ["buy"] * len(eq.index[::50]),
            "score": [0.6] * len(eq.index[::50]),
        }
    )
    rebalance = pd.DataFrame(
        {
            "date": eq.index[::5],
            "turnover": [0.05] * len(eq.index[::5]),
            "n_positions": [3] * len(eq.index[::5]),
        }
    )
    return SimulationResult(
        equity_curve=eq,
        trades_log=trades,
        exposure=pd.Series(0.9, index=eq.index),
        turnover_total=float(rebalance["turnover"].sum()),
        rebalance_log=rebalance,
    )


class TestRegimeEvaluation:
    def test_regime_metrics_cover_requested_periods(self):
        eq = _equity_curve(1800, seed=2)
        sim = _simulation_from_equity(eq)
        regimes = evaluate_regime_metrics(sim)
        names = {r.name for r in regimes}
        assert "2018-2019" in names
        assert "2022 bear" in names

    def test_regime_metrics_have_required_fields(self):
        eq = _equity_curve(800)
        sim = _simulation_from_equity(eq)
        regimes = evaluate_regime_metrics(sim)
        for rm in regimes:
            assert isinstance(rm.metrics, BacktestMetrics)
            assert rm.metrics.n_trades >= 0


class TestRollingPerformance:
    def test_rolling_metrics_length(self):
        eq = _equity_curve(400)
        rolling = compute_rolling_metrics(eq, window=252)
        rets = eq.pct_change().dropna()
        assert len(rolling.rolling_sharpe) == len(rets)
        assert len(rolling.rolling_drawdown) == len(eq)

    def test_rolling_drawdown_non_positive(self):
        eq = _equity_curve(400)
        rolling = compute_rolling_metrics(eq)
        assert (rolling.rolling_drawdown <= 1e-9).all()


class TestUniverseVariants:
    def test_build_default_variants(self):
        universe = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
        variants = build_universe_variants(universe, include_defaults=True)
        assert "full" in variants
        assert "without_NVDA" in variants
        assert "without_QQQ" in variants
        assert "NVDA" not in variants["without_NVDA"]

    def test_random_subset_size(self):
        universe = list("ABCDEFGHIJ")
        variants = build_universe_variants(
            universe, random_subset_size=5, seed=7, include_defaults=False
        )
        assert "random_1_n5" in variants
        assert len(variants["random_1_n5"]) == 5


class TestScoreCalibration:
    def test_bucket_summary(self):
        df = pd.DataFrame(
            {
                "score": [0.52, 0.57, 0.62, 0.70, 0.53],
                "forward_return": [0.01, -0.02, 0.03, 0.04, -0.01],
            }
        )
        stats = summarise_score_calibration(df)
        assert len(stats) == 4
        high = next(s for s in stats if s.bucket == "0.65+")
        assert high.n_observations == 1
        assert high.win_rate == pytest.approx(1.0)


class TestMetricsForPeriod:
    def test_period_turnover_from_rebalance_log(self):
        eq = _equity_curve(100)
        sim = _simulation_from_equity(eq)
        m = metrics_for_period(
            eq,
            sim.trades_log,
            sim.rebalance_log,
            sim.exposure,
        )
        assert m.turnover > 0
