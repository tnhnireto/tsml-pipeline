"""
Shared fixtures for all tests.

The `sample_ohlcv` fixture builds a tiny synthetic OHLCV DataFrame so
tests can run without a network connection.  The fixture spans three full
calendar years to make split tests meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    A synthetic OHLCV DataFrame that looks like real market data.

    - 500 trading days starting 2020-01-02 (UTC-aware DatetimeIndex).
    - All OHLCV columns present, no NaN, high >= low always.
    """
    rng = np.random.default_rng(42)
    n = 500

    dates = pd.bdate_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = 300.0 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.5, 2.0, n)

    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
