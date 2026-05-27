"""
Feature pipeline: turn a raw OHLCV DataFrame into a model-ready dataset.

`build_features` is intentionally explicit: every feature column is named
and constructed in one place so the pipeline is easy to read and modify.

`make_dataset` combines features and a target, drops NaN rows, and returns
X (features) and y (target) as aligned DataFrames.
"""

from __future__ import annotations

import pandas as pd

from tsml.features.benchmarks import DEFAULT_BENCHMARKS
from tsml.features.targets import (
    next_5day_direction,
    next_day_direction,
    next_day_return,
    threshold_direction,
)
from tsml.features.transformers import (
    above_sma,
    daily_returns,
    distance_from_rolling_high,
    fraction_positive_days,
    lagged_returns,
    log_returns,
    price_vs_mean,
    relative_return,
    rolling_mean,
    rolling_return,
    rolling_up_streak,
    rolling_vol_ratio,
    rolling_volatility,
    rsi,
    sma_ratio,
    vol_adjusted_return,
)

LEGACY_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "log_return_1d",
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "rolling_mean_10",
    "rolling_vol_10",
    "sma_ratio_5_20",
    "rsi_14",
    "vol_ratio_5_20",
    "price_vs_mean_20",
)

EXTENDED_FEATURE_COLUMNS: tuple[str, ...] = (
    "rel_ret_20d_vs_spy",
    "rel_ret_60d_vs_spy",
    "rel_ret_20d_vs_qqq",
    "rel_ret_60d_vs_qqq",
    "spy_above_sma200",
    "qqq_above_sma200",
    "spy_ret_20d",
    "spy_vol_20d",
    "qqq_ret_20d",
    "fraction_positive_days_20d",
    "fraction_positive_days_60d",
    "rolling_up_streak",
    "distance_from_20d_high",
    "distance_from_60d_high",
    "ret20d_over_vol20d",
    "ret60d_over_vol20d",
)

_VALID_FEATURE_SETS = ("legacy", "extended")


def build_features(
    df: pd.DataFrame,
    *,
    feature_set: str = "legacy",
    benchmarks: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """
    Compute all features from a raw OHLCV DataFrame.

    Every column in the returned DataFrame is strictly backward-looking:
    the value at row t uses only data from rows 0 … t.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a DatetimeIndex and at least a 'close' column.
    feature_set:
        ``"legacy"`` — original 11 single-asset features.
        ``"extended"`` — legacy features plus cross-sectional / regime
        features (requires SPY and QQQ benchmark closes).
    benchmarks:
        Mapping of benchmark ticker to close price series.  Required when
        ``feature_set="extended"``.

    Returns
    -------
    pd.DataFrame
        Same index as ``df``.  Many early rows will contain NaN (the warmup
        period for rolling windows).  Call ``make_dataset`` to drop them.
    """
    if feature_set not in _VALID_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {_VALID_FEATURE_SETS}, got '{feature_set}'."
        )

    close = df["close"]

    features = pd.DataFrame(index=df.index)
    features["return_1d"]         = daily_returns(close)
    features["log_return_1d"]     = log_returns(close)
    features["return_lag1"]       = lagged_returns(close, lag=1)
    features["return_lag2"]       = lagged_returns(close, lag=2)
    features["return_lag3"]       = lagged_returns(close, lag=3)
    features["rolling_mean_10"]   = rolling_mean(close, window=10)
    features["rolling_vol_10"]    = rolling_volatility(close, window=10)
    features["sma_ratio_5_20"]    = sma_ratio(close, short_window=5, long_window=20)
    features["rsi_14"]            = rsi(close, window=14)
    features["vol_ratio_5_20"]    = rolling_vol_ratio(close, short_window=5, long_window=20)
    features["price_vs_mean_20"]  = price_vs_mean(close, window=20)

    if feature_set == "extended":
        if benchmarks is None:
            raise ValueError(
                "benchmarks must be provided when feature_set='extended'."
            )
        missing = set(DEFAULT_BENCHMARKS) - set(benchmarks)
        if missing:
            raise ValueError(
                f"benchmarks missing required symbols: {sorted(missing)}."
            )

        spy = benchmarks["SPY"]
        qqq = benchmarks["QQQ"]

        features["rel_ret_20d_vs_spy"] = relative_return(
            close, spy, 20, label="rel_ret_20d_vs_spy"
        )
        features["rel_ret_60d_vs_spy"] = relative_return(
            close, spy, 60, label="rel_ret_60d_vs_spy"
        )
        features["rel_ret_20d_vs_qqq"] = relative_return(
            close, qqq, 20, label="rel_ret_20d_vs_qqq"
        )
        features["rel_ret_60d_vs_qqq"] = relative_return(
            close, qqq, 60, label="rel_ret_60d_vs_qqq"
        )

        features["spy_above_sma200"] = above_sma(spy.reindex(df.index).ffill(), 200).astype(float)
        features["qqq_above_sma200"] = above_sma(qqq.reindex(df.index).ffill(), 200).astype(float)
        features["spy_ret_20d"] = rolling_return(spy.reindex(df.index).ffill(), 20)
        features["spy_vol_20d"] = rolling_volatility(spy.reindex(df.index).ffill(), 20)
        features["qqq_ret_20d"] = rolling_return(qqq.reindex(df.index).ffill(), 20)

        features["fraction_positive_days_20d"] = fraction_positive_days(close, 20)
        features["fraction_positive_days_60d"] = fraction_positive_days(close, 60)
        features["rolling_up_streak"] = rolling_up_streak(close)
        features["distance_from_20d_high"] = distance_from_rolling_high(close, 20)
        features["distance_from_60d_high"] = distance_from_rolling_high(close, 60)
        features["ret20d_over_vol20d"] = vol_adjusted_return(close, 20)
        features["ret60d_over_vol20d"] = vol_adjusted_return(close, 60)

    return features


_VALID_TARGETS = ("direction", "return", "direction_5d", "threshold")


def make_dataset(
    df: pd.DataFrame,
    target: str = "direction",
    *,
    feature_set: str = "legacy",
    benchmarks: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a clean (X, y) pair ready for model training.

    Steps:
    1. Compute features from the OHLCV DataFrame.
    2. Compute the requested target.
    3. Concatenate features and target into one DataFrame.
    4. Drop any row that has a NaN in any column (includes neutral rows
       for the ``"threshold"`` target and the warm-up period for all
       targets).

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    target:
        One of:

        ``"direction"``
            Binary next-day direction (0/1).  Default.
        ``"return"``
            Continuous next-day return (regression).
        ``"direction_5d"``
            Binary 5-day-forward direction (0/1).  Lower noise, longer
            horizon.  The last 5 rows of each fold are implicitly excluded.
        ``"threshold"``
            Binary target that keeps only high-conviction days.  Returns
            1 (up > 0.5 %) or 0 (down > 0.5 %); neutral days are NaN and
            dropped.  The model is trained only on strongly-directional
            days, then applied to all test dates.
    feature_set:
        ``"legacy"`` or ``"extended"`` — see :func:`build_features`.
    benchmarks:
        SPY/QQQ close series required when ``feature_set="extended"``.

    Returns
    -------
    X : pd.DataFrame  — feature matrix, no NaNs
    y : pd.Series     — target vector, aligned with X
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"target must be one of {_VALID_TARGETS}, got '{target}'."
        )

    close = df["close"]
    X = build_features(df, feature_set=feature_set, benchmarks=benchmarks)

    if target == "direction":
        y = next_day_direction(close)
    elif target == "return":
        y = next_day_return(close)
    elif target == "direction_5d":
        y = next_5day_direction(close)
    else:  # "threshold"
        y = threshold_direction(close, threshold=0.005)

    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined[X.columns]
    y_clean = combined[y.name]

    return X_clean, y_clean
