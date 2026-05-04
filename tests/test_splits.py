"""
Tests for the time-based train/val/test split.

The critical invariants are:
  1. No row appears in more than one split.
  2. max(train.index) < min(val.index) < min(test.index)
  3. The three splits cover every row of the original DataFrame.
  4. Bad boundary arguments produce clear errors.
"""

import pandas as pd
import pytest

from tsml.data_loader.splits import train_val_test_split


def test_splits_are_non_overlapping(sample_ohlcv):
    train, val, test = train_val_test_split(
        sample_ohlcv, train_end="2021-01-01", val_end="2021-06-01"
    )
    assert train.index.max() < val.index.min(), "Train bleeds into val"
    assert val.index.max() < test.index.min(), "Val bleeds into test"


def test_splits_cover_full_dataset(sample_ohlcv):
    train, val, test = train_val_test_split(
        sample_ohlcv, train_end="2021-01-01", val_end="2021-06-01"
    )
    combined = pd.concat([train, val, test])
    assert len(combined) == len(sample_ohlcv), "Splits don't cover all rows"


def test_no_row_in_two_splits(sample_ohlcv):
    train, val, test = train_val_test_split(
        sample_ohlcv, train_end="2021-01-01", val_end="2021-06-01"
    )
    assert set(train.index).isdisjoint(val.index), "Row appears in train and val"
    assert set(val.index).isdisjoint(test.index), "Row appears in val and test"
    assert set(train.index).isdisjoint(test.index), "Row appears in train and test"


def test_train_end_before_val_end_required(sample_ohlcv):
    with pytest.raises(ValueError, match="before val_end"):
        train_val_test_split(
            sample_ohlcv, train_end="2021-12-31", val_end="2021-01-01"
        )


def test_equal_boundaries_raise(sample_ohlcv):
    with pytest.raises(ValueError, match="before val_end"):
        train_val_test_split(
            sample_ohlcv, train_end="2021-06-01", val_end="2021-06-01"
        )


def test_non_datetime_index_raises():
    df = pd.DataFrame({"close": [1, 2, 3]}, index=[0, 1, 2])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        train_val_test_split(df, train_end="2021-01-01", val_end="2021-06-01")


def test_splits_are_not_empty(sample_ohlcv):
    train, val, test = train_val_test_split(
        sample_ohlcv, train_end="2021-01-01", val_end="2021-06-01"
    )
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_split_boundary_is_inclusive(sample_ohlcv):
    """The train_end date, if it is a trading day, must appear in train."""
    # 2021-01-04 is the first business day of 2021 — it's in sample_ohlcv.
    train_end = "2021-01-04"
    train, val, _ = train_val_test_split(
        sample_ohlcv, train_end=train_end, val_end="2021-06-01"
    )
    ts = pd.Timestamp(train_end, tz="UTC")
    assert ts in train.index, f"{train_end} should be the last row of train"
    assert ts not in val.index, f"{train_end} must not appear in val"
