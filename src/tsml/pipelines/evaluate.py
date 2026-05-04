"""
Evaluation helper: combine ML metrics and backtest metrics in one call.

`evaluate` is the single function a user calls after running
`run_walk_forward` + `run_backtest`.  It returns a nested dict so every
metric can be accessed by name without remembering column positions.

Example
-------
>>> predictions = run_walk_forward(df, model, splitter)
>>> bt = run_backtest(predictions, df["close"])
>>> y_true = y.loc[predictions.index]
>>> report = evaluate(predictions, y_true, bt)
>>> print(report["strategy"]["sharpe"])
0.42
>>> print(report["ml"]["accuracy"])
0.53
"""

from __future__ import annotations

import pandas as pd

from tsml.metrics import ml as ml_metrics
from tsml.metrics import returns as ret_metrics


def evaluate(
    predictions: pd.Series,
    y_true: pd.Series,
    backtest_result: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, dict[str, float]]:
    """
    Compute ML and financial metrics from walk-forward predictions and
    a backtest result DataFrame.

    Parameters
    ----------
    predictions:
        Model output Series (0/1), indexed by date.
        Usually the output of ``run_walk_forward``.
    y_true:
        Actual target labels aligned with ``predictions``.
        Typically ``y.loc[predictions.index]`` where y comes from
        ``make_dataset``.
    backtest_result:
        DataFrame returned by ``run_backtest``.
        Must contain columns: strategy_return, asset_return, position.
    risk_free_rate:
        Annual risk-free rate for Sharpe calculation (default 0.0).
    periods_per_year:
        Trading days per year for annualisation (default 252).

    Returns
    -------
    dict with three nested dicts:

        {
          "ml": {
            "accuracy":  float,
            "precision": float,
            "recall":    float,
          },
          "strategy": {
            "total_return": float,
            "cagr":         float,
            "volatility":   float,
            "sharpe":       float,
            "max_drawdown": float,
            "hit_rate":     float,
            "turnover":     float,
          },
          "buy_and_hold": {
            "total_return": float,
            "cagr":         float,
            "volatility":   float,
            "sharpe":       float,
            "max_drawdown": float,
            "hit_rate":     float,
          },
        }

    Raises
    ------
    ValueError
        If predictions and y_true do not share the same index.
    KeyError
        If backtest_result is missing required columns.
    """
    # ------------------------------------------------------------------ #
    # Validate alignment between predictions and y_true.                  #
    # ------------------------------------------------------------------ #
    if not predictions.index.equals(y_true.index):
        raise ValueError(
            "predictions and y_true must share the same index. "
            "Use y.loc[predictions.index] to align them."
        )

    required_cols = {"strategy_return", "asset_return", "position"}
    missing = required_cols - set(backtest_result.columns)
    if missing:
        raise KeyError(f"backtest_result is missing columns: {missing}")

    # ------------------------------------------------------------------ #
    # ML metrics: align predictions with y_true on the common index.     #
    # Only dates that appear in both are evaluated.                       #
    # ------------------------------------------------------------------ #
    common = predictions.index.intersection(y_true.index)
    ml_report = ml_metrics.summary(
        y_true=y_true.loc[common],
        y_pred=predictions.loc[common],
    )

    # ------------------------------------------------------------------ #
    # Strategy metrics (from the backtest result).                        #
    # ------------------------------------------------------------------ #
    strategy_rets = backtest_result["strategy_return"]
    strategy_pos = backtest_result["position"]

    strategy_report = ret_metrics.summary(
        returns=strategy_rets,
        positions=strategy_pos,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    # ------------------------------------------------------------------ #
    # Buy-and-hold metrics (passive benchmark, no position column needed).#
    # ------------------------------------------------------------------ #
    bnh_rets = backtest_result["asset_return"]

    bnh_report = ret_metrics.summary(
        returns=bnh_rets,
        positions=None,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    return {
        "ml": ml_report,
        "strategy": strategy_report,
        "buy_and_hold": bnh_report,
    }
