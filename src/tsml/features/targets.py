"""
Target builders for supervised learning on financial time series.

All functions use a *forward shift*: the label for day t is derived from
price data after day t.  This means:

    - Features at time t use data up to and including time t.
    - The target at time t uses data from t+1 (or further ahead).
    - There is no information overlap — no leakage.

Rows for which the target cannot be computed (the last N rows) are set to
NaN.  ``make_dataset`` drops them before training.
"""

from __future__ import annotations

import pandas as pd


def next_day_direction(close: pd.Series) -> pd.Series:
    """
    Binary classification target.

        y_t = 1  if close_{t+1} > close_t
              0  otherwise
              NaN for the last row (no next-day price)

    This is the most common starting target for a direction model.
    A model that beats 50 % accuracy on this is doing something real,
    but beating a simple "always-long" strategy on Sharpe is harder.
    """
    future_close = close.shift(-1)
    direction = (future_close > close).astype(float)
    direction.iloc[-1] = float("nan")  # no next-day price for the last row
    return direction.rename("target_direction")


def next_day_return(close: pd.Series) -> pd.Series:
    """
    Regression target: the percentage return of the *next* trading day.

        y_t = (close_{t+1} - close_t) / close_t

    The last row is NaN.

    Use this when you want a regression model instead of a classifier,
    or when you want to size positions proportionally to predicted return.
    """
    future_return = close.pct_change().shift(-1)
    return future_return.rename("target_return")


def next_5day_direction(close: pd.Series) -> pd.Series:
    """
    Binary classification target over a 5-day forward horizon.

        y_t = 1  if close_{t+5} > close_t  (market higher in 5 days)
              0  otherwise
              NaN for the last 5 rows

    Training on a multi-day horizon produces a smoother, lower-noise
    signal than the 1-day direction target.  Note that the backtest still
    executes a *1-day* position: this target identifies medium-term trend
    direction while position management remains daily.
    """
    future_close = close.shift(-5)
    direction = (future_close > close).astype(float)
    direction.iloc[-5:] = float("nan")
    return direction.rename("target_direction_5d")


def threshold_direction(close: pd.Series, threshold: float = 0.005) -> pd.Series:
    """
    Binary classification target that drops low-conviction days.

        y_t = 1    if  (close_{t+1} - close_t) / close_t  >  threshold
              0    if  (close_{t+1} - close_t) / close_t  < -threshold
              NaN  otherwise  (neutral; dropped by ``make_dataset``)

    Neutral rows are returned as NaN rather than a third class so that
    the standard ``dropna()`` step in ``make_dataset`` removes them
    automatically, leaving only clearly directional days in the training
    set.  At inference time the model predicts on all dates and outputs
    0 or 1; neutral training days simply don't contribute to the
    decision boundary.

    Parameters
    ----------
    close:
        Closing price Series.
    threshold:
        Minimum absolute return (as a fraction) to be labelled as
        directional.  Default 0.005 = 0.5 %.
    """
    fwd_return = close.pct_change().shift(-1)
    target = pd.Series(float("nan"), index=close.index, name="target_threshold")
    target[fwd_return > threshold]  = 1.0
    target[fwd_return < -threshold] = 0.0
    return target
