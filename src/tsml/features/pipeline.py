"""
Feature pipeline: turn a raw OHLCV DataFrame into a model-ready dataset.

`build_features` is intentionally explicit: every feature column is named
and constructed in one place so the pipeline is easy to read and modify.

`make_dataset` combines features and a target, drops NaN rows, and returns
X (features) and y (target) as aligned DataFrames.
"""

from __future__ import annotations

import pandas as pd

from tsml.features.targets import next_day_direction, next_day_return
from tsml.features.transformers import (
    daily_returns,
    lagged_returns,
    log_returns,
    rolling_mean,
    rolling_volatility,
    rsi,
    sma_ratio,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from a raw OHLCV DataFrame.

    Every column in the returned DataFrame is strictly backward-looking:
    the value at row t uses only data from rows 0 … t.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a DatetimeIndex and at least a 'close' column.

    Returns
    -------
    pd.DataFrame
        Same index as ``df``.  Many early rows will contain NaN (the warmup
        period for rolling windows).  Call ``make_dataset`` to drop them.
    """
    close = df["close"]

    features = pd.DataFrame(index=df.index)
    features["return_1d"] = daily_returns(close)
    features["log_return_1d"] = log_returns(close)
    features["return_lag1"] = lagged_returns(close, lag=1)
    features["return_lag2"] = lagged_returns(close, lag=2)
    features["rolling_mean_10"] = rolling_mean(close, window=10)
    features["rolling_vol_10"] = rolling_volatility(close, window=10)
    features["sma_ratio_5_20"] = sma_ratio(close, short_window=5, long_window=20)
    features["rsi_14"] = rsi(close, window=14)

    return features


def make_dataset(
    df: pd.DataFrame,
    target: str = "direction",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a clean (X, y) pair ready for model training.

    Steps:
    1. Compute features from the OHLCV DataFrame.
    2. Compute the requested target.
    3. Concatenate features and target into one DataFrame.
    4. Drop any row that has a NaN in any column.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    target:
        ``"direction"`` (binary classification) or ``"return"`` (regression).

    Returns
    -------
    X : pd.DataFrame  — feature matrix, no NaNs
    y : pd.Series     — target vector, aligned with X
    """
    if target not in ("direction", "return"):
        raise ValueError(f"target must be 'direction' or 'return', got '{target}'.")

    close = df["close"]
    X = build_features(df)

    if target == "direction":
        y = next_day_direction(close)
    else:
        y = next_day_return(close)

    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined[X.columns]
    y_clean = combined[y.name]

    return X_clean, y_clean
