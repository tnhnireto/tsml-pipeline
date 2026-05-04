"""
Time-based train / validation / test split for time series data.

All splits are deterministic and time-ordered — no randomness, ever.

Terminology
-----------
train_end   Last date (inclusive) of the training window.
val_end     Last date (inclusive) of the validation window.
            Everything after val_end is the held-out test set.

Example
-------
>>> df = yfinance_loader.load("SPY", "2010-01-01", "2023-12-31")
>>> train, val, test = train_val_test_split(
...     df, train_end="2020-12-31", val_end="2021-12-31"
... )
"""

from __future__ import annotations

import pandas as pd


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into three non-overlapping windows.

    Parameters
    ----------
    df:
        DataFrame with a sorted DatetimeIndex.
    train_end:
        Last date (inclusive) of the training set.
    val_end:
        Last date (inclusive) of the validation set.
        Must be strictly after ``train_end``.

    Returns
    -------
    (train, val, test)
        Three DataFrames that together cover every row in ``df``.
        No row appears in more than one split.

    Raises
    ------
    ValueError
        If the boundaries are inconsistent or the resulting splits are empty.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")

    if train_end_ts >= val_end_ts:
        raise ValueError(
            f"train_end ({train_end}) must be before val_end ({val_end})."
        )

    # Use only tz-aware Timestamps as slice bounds.  Mixing a Timestamp
    # with a plain string raises "Both dates must have the same UTC offset"
    # in pandas >= 2.2 when the index is tz-aware.
    train = df.loc[:train_end_ts]
    val = df.loc[train_end_ts + pd.Timedelta(days=1) : val_end_ts]
    test = df.loc[val_end_ts + pd.Timedelta(days=1) :]

    if train.empty:
        raise ValueError("Training split is empty. Check train_end vs df range.")
    if val.empty:
        raise ValueError("Validation split is empty. Check val_end vs df range.")
    if test.empty:
        raise ValueError("Test split is empty. Check val_end vs df range.")

    return train, val, test
