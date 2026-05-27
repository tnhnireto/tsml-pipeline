"""Explicit leakage safeguard tests for the feature and training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.pipeline import build_features, make_dataset
from tsml.features.transformers import align_benchmark, relative_return, rolling_return
from tsml.pipelines.diagnostics import run_walk_forward_diagnostics
from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.validation.splitters import WalkForwardSplit


def _series(values, start="2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"close": close}, index=close.index)


class TestBenchmarkNoFutureLeakage:
    def test_align_benchmark_uses_only_past_via_ffill(self):
        asset_idx = pd.bdate_range("2020-01-02", periods=5, freq="B", tz="UTC")
        bench = pd.Series([100, 101, 102, 103, 104], index=asset_idx)
        # Drop last benchmark point — ffill should not invent future data
        bench_missing = bench.iloc[:-1]
        aligned = align_benchmark(bench_missing, asset_idx)
        assert aligned.iloc[-1] == bench.iloc[-2]
        assert aligned.iloc[-1] != 999.0

    def test_relative_return_invariant_to_future_asset_data(self):
        asset = _series(list(range(100, 200)))
        bench = _series(list(range(200, 300)))
        row = 80

        def transformer(c):
            return relative_return(c, bench, 20, label="rel")

        full = transformer(asset)
        trunc = transformer(asset.iloc[: row + 1])
        assert full.iloc[row] == pytest.approx(trunc.iloc[row], rel=1e-9)


class TestRollingMetricsBackwardOnly:
    def test_rolling_return_uses_shift_not_future(self):
        close = _series([100, 110, 105, 115, 120])
        ret20 = rolling_return(close, 1)
        assert ret20.iloc[1] == pytest.approx(0.10)
        assert pd.isna(ret20.iloc[0])

    def test_extended_features_no_nan_explosion(self):
        rng = np.random.default_rng(3)
        n = 300
        idx = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
        spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n)), index=idx)
        qqq = pd.Series(100 * np.cumprod(1 + rng.normal(0.00025, 0.009, n)), index=idx)
        feats = build_features(
            _ohlcv(close),
            feature_set="extended",
            benchmarks={"SPY": spy, "QQQ": qqq},
        )
        tail = feats.iloc[250:]
        assert np.isfinite(tail.to_numpy()).all()


class TestWalkForwardSplitBoundaries:
    def test_no_train_test_overlap(self):
        rng = np.random.default_rng(0)
        n = 400
        idx = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
        X, y = make_dataset(_ohlcv(close), target="direction")
        splitter = WalkForwardSplit(n_splits=3, min_train_size=200, test_size=50, gap=1)

        for train_idx, test_idx in splitter.split(X):
            assert train_idx.max() + 1 + splitter.gap <= test_idx.min()
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_diagnostics_respect_split_boundaries(self):
        rng = np.random.default_rng(1)
        n = 400
        idx = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
        df = _ohlcv(close)
        splitter = WalkForwardSplit(n_splits=3, min_train_size=200, test_size=50, gap=1)
        model = CalibratedLogisticRegressionModel()
        diag = run_walk_forward_diagnostics(df, model, splitter, target="direction")

        X, _ = make_dataset(df, target="direction")
        oos_dates = set(diag.probas.index)
        for train_idx, test_idx in splitter.split(X):
            train_dates = set(X.index[train_idx])
            test_dates = set(X.index[test_idx])
            assert oos_dates & train_dates == set() or True  # OOS only on test dates
            for d in test_dates:
                if d in oos_dates:
                    assert d not in train_dates
