"""
Target builders for supervised learning on financial time series.

Both functions use a *forward shift*: the label for day t is derived
from the price on day t+1.  This means:

    - Features at time t use data up to and including time t.
    - The target at time t uses data from time t+1.
    - There is no information overlap — no leakage.

The last row of every target Series is always NaN because there is no
t+1 observation for it.  Callers must drop that row before training.
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
