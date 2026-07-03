"""
Tests for the "extended_v2" feature set.

extended_v2 = extended with a stationary base block:
  - "rolling_mean_10" (raw price level) is replaced by "price_vs_mean_10"
    (close / SMA10 - 1).
  - "log_return_1d" is removed (redundant with "return_1d").
  - "return_1d" is kept so the PreviousDirection baseline still works.

The legacy and extended feature sets must remain byte-for-byte unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.pipeline import (
    BENCHMARK_FEATURE_SETS,
    EXTENDED_FEATURE_COLUMNS,
    EXTENDED_V2_FEATURE_COLUMNS,
    LEGACY_FEATURE_COLUMNS,
    build_features,
    make_dataset,
)
from tsml.models.baselines import PreviousDirection


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


@pytest.fixture()
def random_walk_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    close = _series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)))
    return _ohlcv(close)


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


class TestColumns:
    def test_v2_columns_match_constant(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        features = build_features(
            random_walk_df, feature_set="extended_v2", benchmarks=benches
        )
        assert tuple(features.columns) == EXTENDED_V2_FEATURE_COLUMNS

    def test_v2_has_no_raw_price_or_log_return(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        features = build_features(
            random_walk_df, feature_set="extended_v2", benchmarks=benches
        )
        assert "rolling_mean_10" not in features.columns
        assert "log_return_1d" not in features.columns

    def test_v2_keeps_return_1d_and_extended_block(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        features = build_features(
            random_walk_df, feature_set="extended_v2", benchmarks=benches
        )
        assert "return_1d" in features.columns
        for col in EXTENDED_FEATURE_COLUMNS:
            assert col in features.columns

    def test_benchmark_feature_sets_constant(self):
        assert BENCHMARK_FEATURE_SETS == ("extended", "extended_v2")


# ---------------------------------------------------------------------------
# Stationary replacement feature
# ---------------------------------------------------------------------------


class TestPriceVsMean10:
    def test_formula_close_over_sma10_minus_1(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        features = build_features(
            random_walk_df, feature_set="extended_v2", benchmarks=benches
        )
        close = random_walk_df["close"]
        sma10 = close.rolling(window=10, min_periods=10).mean()
        expected = (close - sma10) / sma10
        pd.testing.assert_series_equal(
            features["price_vs_mean_10"], expected, check_names=False
        )

    def test_is_backward_looking(self, random_walk_df):
        """Truncating future rows must not change past feature values."""
        benches = _benchmarks(random_walk_df.index)
        full = build_features(
            random_walk_df, feature_set="extended_v2", benchmarks=benches
        )
        cut = 250
        truncated = build_features(
            random_walk_df.iloc[:cut],
            feature_set="extended_v2",
            benchmarks={k: v.iloc[:cut] for k, v in _benchmarks(random_walk_df.index).items()},
        )
        pd.testing.assert_series_equal(
            full["price_vs_mean_10"].iloc[:cut],
            truncated["price_vs_mean_10"],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Backward compatibility of existing feature sets
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_legacy_columns_unchanged(self, random_walk_df):
        features = build_features(random_walk_df, feature_set="legacy")
        assert tuple(features.columns) == LEGACY_FEATURE_COLUMNS
        assert "rolling_mean_10" in features.columns
        assert "log_return_1d" in features.columns

    def test_extended_columns_unchanged(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        features = build_features(
            random_walk_df, feature_set="extended", benchmarks=benches
        )
        assert tuple(features.columns) == (
            LEGACY_FEATURE_COLUMNS + EXTENDED_FEATURE_COLUMNS
        )

    def test_invalid_feature_set_raises(self, random_walk_df):
        with pytest.raises(ValueError, match="feature_set"):
            build_features(random_walk_df, feature_set="extended_v3")

    def test_v2_requires_benchmarks(self, random_walk_df):
        with pytest.raises(ValueError, match="benchmarks"):
            build_features(random_walk_df, feature_set="extended_v2")


# ---------------------------------------------------------------------------
# make_dataset integration
# ---------------------------------------------------------------------------


class TestMakeDataset:
    def test_make_dataset_v2_clean_and_finite(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        X, y = make_dataset(
            random_walk_df,
            target="direction_5d",
            feature_set="extended_v2",
            benchmarks=benches,
        )
        assert list(X.columns) == list(EXTENDED_V2_FEATURE_COLUMNS)
        assert not X.isna().any().any()
        assert np.isfinite(X.to_numpy()).all()
        assert len(X) == len(y)

    def test_previous_direction_baseline_works_on_v2(self, random_walk_df):
        benches = _benchmarks(random_walk_df.index)
        X, y = make_dataset(
            random_walk_df,
            target="direction",
            feature_set="extended_v2",
            benchmarks=benches,
        )
        model = PreviousDirection().fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)) <= {0, 1}
        assert len(preds) == len(X)
