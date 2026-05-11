"""
Vectorised backtest engine.

`run_backtest` translates model predictions into a strategy return series.

The no-lookahead rule
----------------------
A prediction made at the close of day t (using only data up to and
including t) cannot influence a position until day t+1.  This is encoded
with a single explicit shift:

    shifted[t] = prediction[t-1]

For holding_period=1 (default):

    position[t] = shifted[t] = prediction[t-1]

For holding_period=N > 1:

    position[t] = max(shifted[t], shifted[t-1], ..., shifted[t-N+1])
                = max(prediction[t-1], prediction[t-2], ..., prediction[t-N])

A single long signal at day k therefore creates a position for days
k+1, k+2, ..., k+N.  Overlapping signals (another signal before the first
expires) extend the exposure but never push position above 1.0 because the
rolling maximum of binary (0/1) inputs is always in {0, 1}.

Transaction costs are applied on every day the *position* changes, regardless
of holding period.

Output columns
--------------
close           : asset close price
asset_return    : daily return of the asset (buy-and-hold benchmark)
prediction      : raw model output (0 or 1)
position        : effective holding derived from predictions (see above)
strategy_return : position * asset_return  (optionally minus costs)
cumulative      : (1 + strategy_return).cumprod()  — strategy equity curve
buy_and_hold    : (1 + asset_return).cumprod()     — passive benchmark
"""

from __future__ import annotations

import pandas as pd


def run_backtest(
    predictions: pd.Series,
    close: pd.Series,
    costs_bps: float = 0.0,
    holding_period: int = 1,
) -> pd.DataFrame:
    """
    Convert model predictions into a strategy performance DataFrame.

    Parameters
    ----------
    predictions:
        Date-indexed Series of model outputs (typically 0 or 1).
        Usually the output of ``run_walk_forward``.
    close:
        Date-indexed Series of asset closing prices.
        Must cover at least all prediction dates plus the day before
        the first prediction (so that asset_return can be computed).
    costs_bps:
        Round-trip transaction cost in basis points, applied each time
        the position changes.  Default 0.0 (no costs).
        Example: costs_bps=10 means 0.10 % per trade.
    holding_period:
        Number of trading days to hold a long position after a signal.
        Default 1 reproduces the original single-day behaviour exactly.

        For holding_period=N, a prediction of 1 on day t creates a
        position of 1 on days t+1 through t+N.  Overlapping signals
        (new signal before the previous one expires) keep the position
        at 1 — they never push it above 1.0.

        The no-lookahead guarantee is preserved: positions are derived
        from a 1-day shift of predictions before the rolling window is
        applied, so signals are never executed on the day they are formed.

    Returns
    -------
    pd.DataFrame
        One row per tradeable day (first row dropped due to position shift).
        Columns: close, asset_return, prediction, position,
                 strategy_return, cumulative, buy_and_hold.

    Raises
    ------
    ValueError
        If predictions and close share no common dates, or if
        holding_period < 1.
    """
    if holding_period < 1:
        raise ValueError(
            f"holding_period must be >= 1, got {holding_period}."
        )
    # ------------------------------------------------------------------ #
    # 1. Align predictions and close on a common date range.              #
    #    We need close prices for one day BEFORE the first prediction     #
    #    in order to compute the asset return on the first signal day.    #
    # ------------------------------------------------------------------ #
    common_idx = predictions.index.intersection(close.index)
    if common_idx.empty:
        raise ValueError(
            "predictions and close share no common dates. "
            "Check that both use the same DatetimeIndex."
        )

    preds = predictions.loc[common_idx].sort_index()
    close_aligned = close.loc[common_idx].sort_index()

    # ------------------------------------------------------------------ #
    # 2. Compute daily asset returns.                                     #
    #    return[t] = (close[t] - close[t-1]) / close[t-1]               #
    #    We look one row back within close_aligned, so the first row      #
    #    of asset_return is NaN (no prior close in this window).          #
    # ------------------------------------------------------------------ #
    asset_return = close_aligned.pct_change().rename("asset_return")

    # ------------------------------------------------------------------ #
    # 3. Build positions.                                                 #
    #                                                                     #
    # Step A — shift by 1 day.                                           #
    #   shifted[t] = prediction[t-1]                                     #
    #   This is the no-lookahead guarantee: a signal formed at close of  #
    #   day t cannot create a position until day t+1.                    #
    #                                                                     #
    # Step B — extend holding period (only when holding_period > 1).    #
    #   position[t] = rolling_max(shifted, window=holding_period)[t]    #
    #               = max(prediction[t-1], ..., prediction[t-N])         #
    #   A single long signal therefore holds for N consecutive days.     #
    #   Overlapping signals keep the position at 1 (rolling max of 0/1  #
    #   binary inputs is always in {0, 1}, so no leverage is created).  #
    # ------------------------------------------------------------------ #
    shifted = preds.shift(1)

    if holding_period == 1:
        position = shifted.rename("position")
    else:
        position = (
            shifted
            .rolling(window=holding_period, min_periods=1)
            .max()
            .clip(upper=1.0)
            .rename("position")
        )

    # ------------------------------------------------------------------ #
    # 4. Compute strategy returns.                                        #
    #    strategy_return[t] = position[t] * asset_return[t]              #
    # ------------------------------------------------------------------ #
    strategy_return = (position * asset_return).rename("strategy_return")

    # ------------------------------------------------------------------ #
    # 5. Apply transaction costs (optional).                              #
    #    Cost is charged whenever the position changes (turnover).        #
    #    turnover[t] = |position[t] - position[t-1]|                     #
    #    cost[t]     = turnover[t] * costs_bps / 10_000                  #
    # ------------------------------------------------------------------ #
    if costs_bps > 0.0:
        # position.diff() produces NaN on the very first entry because there
        # is no prior position in the series.  We fill that NaN with 0.0,
        # meaning: assume we start flat (no entry cost on the first signal).
        turnover = position.diff().abs().fillna(0.0)
        cost = turnover * costs_bps / 10_000
        strategy_return = strategy_return - cost

    # ------------------------------------------------------------------ #
    # 6. Assemble the result DataFrame and drop the first NaN row.       #
    # ------------------------------------------------------------------ #
    result = pd.DataFrame(
        {
            "close": close_aligned,
            "asset_return": asset_return,
            "prediction": preds,
            "position": position,
            "strategy_return": strategy_return,
        }
    )

    # Drop the first row: position is NaN there (no prior prediction).
    result = result.dropna(subset=["position", "asset_return"])

    # ------------------------------------------------------------------ #
    # 7. Compute cumulative equity curves.                                #
    #    Both start at 1.0 on the first tradeable day.                    #
    # ------------------------------------------------------------------ #
    result["cumulative"] = (1 + result["strategy_return"]).cumprod()
    result["buy_and_hold"] = (1 + result["asset_return"]).cumprod()

    return result
