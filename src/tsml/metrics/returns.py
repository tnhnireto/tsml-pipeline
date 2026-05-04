"""
Financial return metrics.

All functions accept a pandas Series of *daily* returns (e.g. 0.01 = 1 %).
The default annualisation factor is 252 trading days per year.

Each function is a pure computation — no side effects, no plotting.

Formulas used
-------------
total_return       = ∏(1 + r_t) − 1
cagr               = (1 + total_return)^(252/n) − 1
volatility         = std(r) * √252
sharpe_ratio       = (mean(r) − rf/252) / std(r) * √252
max_drawdown       = min((equity − peak) / peak)   over all t
hit_rate           = fraction of days with r > 0
turnover           = mean(|position_t − position_{t−1}|)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# Default annualisation factor for daily returns.
TRADING_DAYS_PER_YEAR: int = 252


def _require_non_empty(series: pd.Series, name: str) -> None:
    if len(series) == 0:
        raise ValueError(f"{name} must not be empty.")


def total_return(returns: pd.Series) -> float:
    """
    Compounded total return over the full period.

        total_return = ∏(1 + r_t) − 1

    A value of 0.25 means the strategy grew by 25 % in total.
    """
    _require_non_empty(returns, "returns")
    return float((1 + returns).prod() - 1)


def cagr(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compound Annual Growth Rate (annualised return).

        cagr = (1 + total_return)^(periods_per_year / n) − 1

    where n is the number of return observations.

    Annualising with n observations converts "grew X % over n days" into
    "equivalent annual rate".
    """
    _require_non_empty(returns, "returns")
    n = len(returns)
    total = total_return(returns)
    return float((1 + total) ** (periods_per_year / n) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised standard deviation of daily returns.

        volatility = std(r) * √periods_per_year

    Uses ddof=1 (sample standard deviation), consistent with most
    financial risk tools.
    """
    _require_non_empty(returns, "returns")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised Sharpe ratio.

        excess_return_daily = returns − risk_free_rate / periods_per_year
        sharpe = mean(excess) / std(excess) * √periods_per_year

    Returns NaN if volatility is zero (all returns identical).

    A Sharpe above 1.0 is considered decent for a daily strategy.
    Note: this metric is notoriously noisy with fewer than 3 years of data.
    """
    _require_non_empty(returns, "returns")
    rf_daily = risk_free_rate / periods_per_year
    excess = returns - rf_daily
    vol = excess.std(ddof=1)
    # Use a small absolute threshold rather than == 0.  A constant return
    # series can produce a non-zero vol of ~1e-18 due to floating-point
    # cancellation in the mean computation, which would give a meaningless
    # Sharpe of ~1e+16 instead of NaN.
    if vol < 1e-14:
        return float("nan")
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in the cumulative return curve.

        equity    = (1 + r).cumprod()
        peak      = equity.cummax()
        drawdown  = (equity − peak) / peak
        MDD       = min(drawdown)

    Returns a negative number (e.g. −0.20 for a 20 % drawdown).
    A value of 0.0 means the strategy never went below a previous high.
    """
    _require_non_empty(returns, "returns")
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def hit_rate(returns: pd.Series) -> float:
    """
    Fraction of days with a positive return.

        hit_rate = count(r > 0) / n

    0.5 means the strategy made money on exactly half of all days.
    Note: a high hit rate does not guarantee profitability if the losing
    days are large — see also Sharpe and max_drawdown.
    """
    _require_non_empty(returns, "returns")
    return float((returns > 0).mean())


def turnover(positions: pd.Series) -> float:
    """
    Average absolute daily position change.

        turnover = mean(|position_t − position_{t−1}|)

    For binary positions (0/1):
        - 0.0  = never trade  (always long or always flat)
        - 1.0  = trade every single day
        - 0.1  = trade on about 10 % of days

    High turnover increases transaction costs and erodes strategy returns.
    """
    _require_non_empty(positions, "positions")
    return float(positions.diff().abs().mean())


def summary(
    returns: pd.Series,
    positions: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    label: str = "strategy",
) -> dict[str, float]:
    """
    Compute all return metrics at once and return them as a dict.

    Parameters
    ----------
    returns:
        Daily return series.
    positions:
        Optional daily position series.  If given, turnover is included.
    risk_free_rate:
        Annual risk-free rate for Sharpe computation.
    periods_per_year:
        Annualisation factor.
    label:
        Prefix used in the returned dict keys (e.g. "strategy" or "buy_and_hold").

    Returns
    -------
    dict with keys: total_return, cagr, volatility, sharpe, max_drawdown,
    hit_rate, and optionally turnover.
    """
    metrics: dict[str, float] = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }
    if positions is not None:
        metrics["turnover"] = turnover(positions)
    return metrics
