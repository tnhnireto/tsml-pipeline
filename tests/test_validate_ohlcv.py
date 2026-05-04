"""
Tests for the OHLCV validation helper.

Each test mutates a valid DataFrame in exactly one way and asserts that
`validate_ohlcv` rejects it.
"""

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import validate_ohlcv


def test_valid_df_passes(sample_ohlcv):
    """A well-formed DataFrame should pass without raising."""
    validate_ohlcv(sample_ohlcv, "TEST")


def test_missing_column_raises(sample_ohlcv):
    df = sample_ohlcv.drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_ohlcv(df, "TEST")


def test_non_datetime_index_raises(sample_ohlcv):
    df = sample_ohlcv.reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv(df, "TEST")


def test_naive_index_raises(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_ohlcv(df, "TEST")


def test_unsorted_index_raises(sample_ohlcv):
    df = sample_ohlcv.iloc[::-1].copy()
    with pytest.raises(ValueError, match="sorted"):
        validate_ohlcv(df, "TEST")


def test_duplicate_dates_raises(sample_ohlcv):
    df = pd.concat([sample_ohlcv.iloc[:5], sample_ohlcv.iloc[:5]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(df, "TEST")


def test_nan_in_close_raises(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[10, df.columns.get_loc("close")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_ohlcv(df, "TEST")


def test_high_less_than_low_raises(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[0, df.columns.get_loc("high")] = df.iloc[0]["low"] - 1
    with pytest.raises(ValueError, match="high < low"):
        validate_ohlcv(df, "TEST")
