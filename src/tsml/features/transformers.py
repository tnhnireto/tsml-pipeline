"""
Feature transformers for financial time series.

All functions are pure: they take a pandas Series (or DataFrame) and
return a new Series.  Every function is strictly backward-looking:
the value at time t depends only on data at time t or earlier.

Naming convention
-----------------
The returned Series keeps the same DatetimeIndex as the input.
Rows that can't be computed (e.g. the first row of a rolling window)
are left as NaN — callers decide when to drop them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def daily_returns(close: pd.Series) -> pd.Series:
    """
    Percentage return for each day.

        r_t = (close_t - close_{t-1}) / close_{t-1}

    The first row is NaN (no previous price).
    """
    return close.pct_change().rename("return_1d")


def log_returns(close: pd.Series) -> pd.Series:
    """
    Natural log return for each day.

        lr_t = ln(close_t / close_{t-1})

    Mathematically equivalent to ln(1 + r_t) for small r.
    The first row is NaN.
    """
    return np.log(close / close.shift(1)).rename("log_return_1d")


def lagged_returns(close: pd.Series, lag: int) -> pd.Series:
    """
    Simple percentage return shifted backward by `lag` trading days.

    lag=1 gives yesterday's return at today's row, which is safe to use
    as a predictor because it contains no future information.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}.")
    return daily_returns(close).shift(lag).rename(f"return_lag{lag}")


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling arithmetic mean over the last `window` observations.

    The value at t is the mean of [t-window+1, ..., t].
    Requires min_periods=window so partial windows are NaN, not misleading.
    """
    return series.rolling(window=window, min_periods=window).mean().rename(
        f"rolling_mean_{window}"
    )


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of daily returns.

    A common proxy for realised volatility.  The value at t uses returns
    from [t-window+1, ..., t], which are all in the past.
    """
    rets = daily_returns(close)
    return rets.rolling(window=window, min_periods=window).std().rename(
        f"rolling_vol_{window}"
    )


def sma_ratio(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    """
    Ratio of the short-term SMA to the long-term SMA.

        ratio_t = SMA(close, short) / SMA(close, long)

    > 1 means the short average is above the long average (upward trend).
    < 1 means the opposite (downward trend).

    The first `long_window - 1` rows are NaN because the long SMA needs
    at least `long_window` observations.
    """
    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be less than long_window ({long_window})."
        )
    short_sma = close.rolling(window=short_window, min_periods=short_window).mean()
    long_sma = close.rolling(window=long_window, min_periods=long_window).mean()
    return (short_sma / long_sma).rename(f"sma_ratio_{short_window}_{long_window}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes on a
    scale of 0–100.  Readings above 70 are typically considered overbought;
    readings below 30 are oversold.

    Implementation uses a simple rolling mean (Wilder's original uses an
    exponential average; this version is easier to reason about and test).

        delta  = close_t - close_{t-1}
        gains  = max(delta, 0)
        losses = max(-delta, 0)
        RS     = mean(gains, window) / mean(losses, window)
        RSI    = 100 - 100 / (1 + RS)

    When avg_loss == 0 (all gains) RS is inf and RSI is 100.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")
