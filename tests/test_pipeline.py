"""
Tests for build_features and make_dataset.

These are integration-level tests: they verify that the pipeline
produces the right shape, drops NaN correctly, and keeps X and y aligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.pipeline import build_features, make_dataset


def _ohlcv(n: int = 100) -> pd.DataFrame:
    """Minimal synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = 300.0 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.5, 2.0, n)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.Index(dates, name="date"),
    )


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = _ohlcv()
        features = build_features(df)
        assert isinstance(features, pd.DataFrame)

    def test_index_matches_input(self):
        df = _ohlcv()
        features = build_features(df)
        assert features.index.equals(df.index)

    def test_expected_columns_present(self):
        expected = {
            "return_1d",
            "log_return_1d",
            "return_lag1",
            "return_lag2",
            "rolling_mean_10",
            "rolling_vol_10",
            "sma_ratio_5_20",
            "rsi_14",
        }
        features = build_features(_ohlcv())
        assert expected.issubset(set(features.columns))

    def test_early_rows_have_nans(self):
        """The longest rolling window (RSI, window=14) needs warmup."""
        features = build_features(_ohlcv())
        # Row 0 should have NaNs from all rolling features
        assert features.iloc[0].isna().any()

    def test_late_rows_have_no_nans(self):
        """After the warmup period, all features should be non-NaN."""
        features = build_features(_ohlcv(n=100))
        # After row 20 (safe warmup), no NaN expected
        assert not features.iloc[25:].isna().any().any()


class TestMakeDataset:
    def test_direction_target(self):
        X, y = make_dataset(_ohlcv(), target="direction")
        assert y.name == "target_direction"
        assert set(y.unique()).issubset({0.0, 1.0})

    def test_return_target(self):
        X, y = make_dataset(_ohlcv(), target="return")
        assert y.name == "target_return"
        assert y.dtype == float

    def test_no_nans_in_output(self):
        X, y = make_dataset(_ohlcv())
        assert not X.isna().any().any(), "X contains NaNs"
        assert not y.isna().any(), "y contains NaNs"

    def test_x_and_y_are_aligned(self):
        """X and y must share the exact same index."""
        X, y = make_dataset(_ohlcv())
        assert X.index.equals(y.index)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target must be"):
            make_dataset(_ohlcv(), target="price")

    def test_dataset_is_shorter_than_raw(self):
        """Dropping NaN rows must reduce the length."""
        df = _ohlcv(n=100)
        X, y = make_dataset(df)
        assert len(X) < len(df)

    def test_last_row_dropped(self):
        """The last row of the raw data must not appear (target is NaN there)."""
        df = _ohlcv(n=50)
        X, _ = make_dataset(df)
        assert df.index[-1] not in X.index
