"""Tests for extended cross-sectional and regime-aware features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.benchmarks import load_benchmark_closes
from tsml.features.pipeline import (
    EXTENDED_FEATURE_COLUMNS,
    LEGACY_FEATURE_COLUMNS,
    build_features,
    make_dataset,
)
from tsml.features.transformers import (
    relative_return,
    rolling_return,
)


def _series(values, start="2020-01-01") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, dtype=float, name="close")


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=close.index,
    )


def _benchmarks(asset_index: pd.DatetimeIndex, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    n = len(asset_index)
    spy = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.008, n)),
        index=asset_index,
    )
    qqq = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0005, 0.010, n)),
        index=asset_index,
    )
    return {"SPY": spy, "QQQ": qqq}


class TestRelativeStrength:
    def test_relative_return_formula(self):
        asset = _series([100, 110, 121, 133])
        bench = _series([100, 105, 110, 115])
        rel = relative_return(asset, bench, 1, label="rel_ret_1d_vs_spy")
        asset_ret = rolling_return(asset, 1)
        bench_ret = rolling_return(bench.reindex(asset.index).ffill(), 1)
        pd.testing.assert_series_equal(rel, asset_ret - bench_ret, check_names=False)

    def test_extended_features_present_for_all_symbols(self):
        asset_close = _series(100 + np.arange(250))
        df = _ohlcv(asset_close)
        benches = _benchmarks(df.index)
        features = build_features(df, feature_set="extended", benchmarks=benches)
        for col in EXTENDED_FEATURE_COLUMNS:
            assert col in features.columns


class TestNoNanExplosions:
    def test_make_dataset_extended_has_finite_values(self):
        rng = np.random.default_rng(7)
        n = 300
        idx = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
        df = _ohlcv(close)
        benches = _benchmarks(idx)
        X, y = make_dataset(
            df,
            target="direction",
            feature_set="extended",
            benchmarks=benches,
        )
        assert not X.empty
        assert np.isfinite(X.to_numpy()).all()
        assert not y.isna().any()


class TestNoFutureLeakage:
    def test_relative_strength_leakage(self):
        asset = _series(list(range(100, 200)))
        bench = _series(list(range(200, 300)))
        row = 80

        def transformer(c):
            return relative_return(c, bench, 20, label="rel")

        full = transformer(asset)
        trunc = transformer(asset.iloc[: row + 1])
        full_val = full.iloc[row]
        trunc_val = trunc.iloc[row]
        if pd.isna(full_val) and pd.isna(trunc_val):
            return
        assert full_val == pytest.approx(trunc_val, rel=1e-9)

    def test_regime_features_aligned_to_asset_index(self):
        asset_close = _series(100 + np.arange(250))
        df = _ohlcv(asset_close)
        benches = _benchmarks(df.index)
        features = build_features(df, feature_set="extended", benchmarks=benches)
        assert features.index.equals(df.index)
        assert features["spy_ret_20d"].index.equals(df.index)
        assert features["qqq_above_sma200"].index.equals(df.index)


class TestLegacyCompatibility:
    def test_legacy_columns_unchanged(self):
        df = _ohlcv(_series(100 + np.arange(120)))
        legacy = build_features(df, feature_set="legacy")
        assert set(LEGACY_FEATURE_COLUMNS).issubset(set(legacy.columns))
        assert not any(c in legacy.columns for c in EXTENDED_FEATURE_COLUMNS)

    def test_extended_requires_benchmarks(self):
        df = _ohlcv(_series(100 + np.arange(120)))
        with pytest.raises(ValueError, match="benchmarks must be provided"):
            build_features(df, feature_set="extended")


class TestBenchmarkLoader:
    def test_stub_loader_provides_spy_qqq(self):
        from tsml.data_loader.base import DataLoader

        class StubLoader(DataLoader):
            def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
                idx = pd.bdate_range(start, periods=120, freq="B", tz="UTC")
                close = pd.Series(100 + np.arange(len(idx)), index=idx, dtype=float)
                return pd.DataFrame({"close": close}, index=idx)

        closes = load_benchmark_closes(StubLoader(), "2020-01-01", "2020-06-30")
        assert "SPY" in closes and "QQQ" in closes
        assert not closes["SPY"].empty and not closes["QQQ"].empty
