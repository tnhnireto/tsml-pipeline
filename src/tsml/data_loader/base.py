"""
Abstract base class for all data loaders.

Every concrete loader must return a DataFrame that satisfies the schema
checked by `validate_ohlcv`.  This keeps the rest of the pipeline
independent of where the data comes from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# Columns that every loader must provide (lowercase).
REQUIRED_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> None:
    """
    Raise ValueError if `df` does not satisfy the OHLCV contract:

    - Required columns present.
    - DatetimeIndex, UTC-aware, sorted, no duplicate dates.
    - No NaN values inside the date range.
    - High >= Low for every row.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{symbol}] Missing columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{symbol}] Index must be a DatetimeIndex.")

    if df.index.tz is None:
        raise ValueError(f"[{symbol}] Index must be timezone-aware (UTC).")

    # Check duplicates before sort order: a concat of two identical DataFrames
    # triggers duplicates first (the concat result has alternating rows, which
    # also fails the sort check, so we must test in this order).
    if df.index.duplicated().any():
        raise ValueError(f"[{symbol}] Index contains duplicate dates.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{symbol}] Index is not sorted ascending.")

    nan_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"[{symbol}] NaN values found in columns: {nan_cols}")

    if (df["high"] < df["low"]).any():
        raise ValueError(f"[{symbol}] Found rows where high < low.")


class DataLoader(ABC):
    """Base class for OHLCV data loaders."""

    @abstractmethod
    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Return a UTC-indexed OHLCV DataFrame for `symbol` covering
        [start, end] (inclusive, date strings like '2020-01-01').
        """
        ...
