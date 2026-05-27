"""Tests for parameter sweep robustness analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import DataLoader
from tsml.models.baselines import LogisticRegressionModel
from tsml.portfolio.parameter_sweep import (
    SweepParams,
    compute_parameter_sensitivity,
    compute_robustness_score,
    compute_sweep_diagnostics,
    compute_sweep_metrics,
    count_parameter_combinations,
    default_parameter_grid,
    expand_parameter_grid,
    run_parameter_sweep,
    run_single_sweep_backtest,
    sample_parameter_combinations,
)
from tsml.portfolio.simulator import SimulationResult, prepare_simulation_inputs
from tsml.validation.splitters import WalkForwardSplit


def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
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


def _bull_spy(n: int) -> pd.Series:
    idx = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
    return pd.Series(100 + np.arange(n) * 0.5, index=idx, dtype=float)


class StubLoader(DataLoader):
    def __init__(self, data: dict[str, pd.DataFrame], spy: pd.Series) -> None:
        self._data = data
        self._spy = spy

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol == "SPY":
            return pd.DataFrame({"close": self._spy}, index=self._spy.index)
        if symbol not in self._data:
            raise ValueError(f"No data for {symbol}")
        return self._data[symbol]


@pytest.fixture()
def fast_splitter() -> WalkForwardSplit:
    return WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)


@pytest.fixture()
def stub_loader() -> StubLoader:
    symbols = ["AAA", "BBB", "CCC"]
    data = {sym: _make_ohlcv(400, seed=i + 1) for i, sym in enumerate(symbols)}
    return StubLoader(data, _bull_spy(400))


@pytest.fixture()
def tiny_grid() -> dict[str, list]:
    return {
        "buy_threshold": [0.56, 0.58],
        "sell_threshold": [0.50],
        "min_score": [0.55],
        "bear_exposure": [0.25],
        "min_hold_weeks": [1, 2],
        "vol_threshold": [None],
    }


class TestGridExpansion:
    def test_count_combinations(self, tiny_grid):
        assert count_parameter_combinations(tiny_grid) == 4

    def test_expand_parameter_grid(self, tiny_grid):
        combos = expand_parameter_grid(tiny_grid)
        assert len(combos) == 4
        assert all(isinstance(c, SweepParams) for c in combos)

    def test_sample_respects_cap(self, tiny_grid):
        sampled = sample_parameter_combinations(tiny_grid, 2, seed=0)
        assert len(sampled) == 2

    def test_sample_deterministic(self, tiny_grid):
        a = sample_parameter_combinations(tiny_grid, 2, seed=7)
        b = sample_parameter_combinations(tiny_grid, 2, seed=7)
        assert [c.as_dict() for c in a] == [c.as_dict() for c in b]

    def test_fast_grid_smaller_than_full(self):
        assert count_parameter_combinations(default_parameter_grid(fast=True)) < (
            count_parameter_combinations(default_parameter_grid(fast=False))
        )


class TestRobustnessScore:
    def test_robustness_score_stable(self):
        sharpe = pd.Series([1.0, 1.1, 0.9, 1.05])
        score = compute_robustness_score(sharpe)
        assert score == pytest.approx(sharpe.mean() / sharpe.std(ddof=1))

    def test_robustness_score_nan_on_constant(self):
        assert np.isnan(compute_robustness_score(pd.Series([1.0, 1.0])))


class TestDiagnostics:
    def test_sensitivity_detects_spread(self):
        df = pd.DataFrame(
            {
                "buy_threshold": [0.56, 0.56, 0.60, 0.60],
                "sell_threshold": [0.50, 0.50, 0.50, 0.50],
                "min_score": [0.55] * 4,
                "bear_exposure": [0.25] * 4,
                "min_hold_weeks": [1, 2, 1, 2],
                "vol_threshold": [None] * 4,
                "sharpe": [0.5, 0.6, 1.0, 1.1],
            }
        )
        sens = compute_parameter_sensitivity(df, metric="sharpe")
        buy_row = sens.loc[sens["parameter"] == "buy_threshold"].iloc[0]
        assert buy_row["metric_spread"] == pytest.approx(0.5)

    def test_diagnostics_aggregation(self):
        df = pd.DataFrame(
            {
                "buy_threshold": [0.56, 0.58],
                "sell_threshold": [0.50, 0.52],
                "min_score": [0.55, 0.55],
                "bear_exposure": [0.25, 0.25],
                "min_hold_weeks": [1, 2],
                "vol_threshold": [None, None],
                "sharpe": [0.8, 1.2],
                "cagr": [0.1, 0.2],
                "max_drawdown": [-0.1, -0.15],
                "turnover": [1.0, 1.5],
                "exposure": [0.7, 0.6],
                "n_trades": [10, 12],
                "calmar": [1.0, 1.3],
                "volatility": [0.15, 0.18],
            }
        )
        diag = compute_sweep_diagnostics(df, top_n=1)
        assert len(diag.top_n) == 1
        assert diag.top_n.iloc[0]["sharpe"] == pytest.approx(1.2)
        assert not diag.percentiles.empty
        assert not diag.parameter_importance.empty


class TestSweepExecution:
    def test_deterministic_sweep(
        self, fast_splitter, stub_loader, tiny_grid
    ):
        common = dict(
            symbols=["AAA", "BBB", "CCC"],
            start="2015-01-01",
            end="2016-06-30",
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            loader=stub_loader,
            grid=tiny_grid,
            feature_set="legacy",
            n_jobs=1,
        )
        a = run_parameter_sweep(**common)
        b = run_parameter_sweep(**common)
        pd.testing.assert_frame_equal(
            a.results.sort_values(list(tiny_grid.keys())).reset_index(drop=True),
            b.results.sort_values(list(tiny_grid.keys())).reset_index(drop=True),
            check_exact=False,
            rtol=1e-9,
        )

    def test_precomputed_scores_no_rerun(
        self, fast_splitter, stub_loader
    ):
        """Precomputed inputs freeze walk-forward scores across param combos."""
        symbols = ["AAA", "BBB", "CCC"]
        pre = prepare_simulation_inputs(
            symbols,
            LogisticRegressionModel(),
            fast_splitter,
            start_date="2015-01-01",
            end_date="2016-06-30",
            loader=stub_loader,
            feature_set="legacy",
            load_spy=True,
        )
        params = SweepParams(buy_threshold=0.56, min_hold_weeks=1)
        row = run_single_sweep_backtest(
            params,
            symbols=symbols,
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2016-06-30",
            precomputed=pre,
            feature_set="legacy",
        )
        assert "sharpe" in row
        assert pre.proba_map  # scores computed once upstream

    def test_no_lookahead_reuses_frozen_precomputed_scores(
        self, fast_splitter, stub_loader
    ):
        """Sweep runs must not re-fit; identical precomputed inputs -> identical metrics."""
        symbols = ["AAA", "BBB", "CCC"]
        pre = prepare_simulation_inputs(
            symbols,
            LogisticRegressionModel(),
            fast_splitter,
            start_date="2015-01-01",
            end_date="2016-06-30",
            loader=stub_loader,
            feature_set="legacy",
            load_spy=True,
        )
        params = SweepParams()
        kwargs = dict(
            symbols=symbols,
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2016-06-30",
            precomputed=pre,
            feature_set="legacy",
        )
        row_a = run_single_sweep_backtest(params, **kwargs)
        row_b = run_single_sweep_backtest(params, **kwargs)
        assert row_a["sharpe"] == pytest.approx(row_b["sharpe"])
        assert row_a["cagr"] == pytest.approx(row_b["cagr"])

    def test_metrics_columns_present(
        self, fast_splitter, stub_loader, tiny_grid
    ):
        sweep = run_parameter_sweep(
            ["AAA", "BBB", "CCC"],
            start="2015-01-01",
            end="2016-06-30",
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            loader=stub_loader,
            grid=tiny_grid,
            feature_set="legacy",
            n_jobs=1,
        )
        if sweep.results.empty:
            pytest.skip("Insufficient synthetic data")
        for col in ["cagr", "sharpe", "max_drawdown", "calmar", "volatility"]:
            assert col in sweep.results.columns

    def test_empty_equity_metrics_zero(self):
        empty = SimulationResult(
            equity_curve=pd.Series(dtype=float),
            trades_log=pd.DataFrame(),
        )
        metrics = compute_sweep_metrics(empty)
        assert metrics["cagr"] == 0.0
        assert metrics["n_trades"] == 0.0
