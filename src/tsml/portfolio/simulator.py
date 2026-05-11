"""
Portfolio simulator — periodic rebalancing over a date range.

``simulate`` runs a full walk-forward pre-computation for every symbol,
then steps through calendar time, rebalancing on the first trading day of
each week.  At each rebalance date it ranks the universe, generates trading
signals, executes trades (with optional transaction costs), and records the
daily portfolio value.

Leakage guarantee
-----------------
Walk-forward probabilities are pre-computed once per symbol using
``run_walk_forward_proba``, which ensures every OOS probability at date *t*
was computed with data only up to *t*.  The probability matrix is then
**shifted forward by one trading day** so the score used on rebalance date
*d* is the value that was available at the close of the previous trading day.
This replicates the convention used throughout this project: signal at *t*,
execution at *t+1*.

Portfolio mechanics
-------------------
- Positions are **equal-weight** (1/N per held symbol; cash earns 0 %).
- Rebalancing is triggered on the **first trading day of each ISO calendar
  week** (i.e. Monday, or the next available trading day if Monday is a
  holiday).
- Transaction costs are applied as a fraction of **one-way turnover**:
  ``cost = one_way_turnover * portfolio_value * costs_bps * 1e-4``.
  One-way turnover is ``sum(|Δweight|) / 2`` across all symbols, which
  accounts for both entry, exit, and weight changes from position-count
  changes.
- The portfolio value cannot drop below zero.

Output
------
``SimulationResult`` carries:

- ``equity_curve``: daily ``pd.Series`` of portfolio value (starts at
  ``initial_capital``), indexed by the same trading-day DatetimeIndex as
  the price data.
- ``trades_log``: ``pd.DataFrame`` with columns
  ``[date, symbol, action, score]`` — one row per executed trade (buys and
  sells; holds are omitted).

Example
-------
>>> from tsml.models.baselines import CalibratedLogisticRegressionModel
>>> from tsml.portfolio.simulator import simulate
>>> from tsml.validation import WalkForwardSplit
>>>
>>> result = simulate(
...     symbols=["SPY", "QQQ", "MSFT"],
...     model=CalibratedLogisticRegressionModel(),
...     splitter=WalkForwardSplit(n_splits=5, min_train_size=252, test_size=63),
...     start_date="2020-01-01",
...     end_date="2023-12-31",
...     top_n=2,
...     min_score=0.55,
...     costs_bps=5.0,
... )
>>> result.equity_curve.plot(title="Portfolio equity curve")
>>> print(result.trades_log.head())
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.pipelines.train import run_walk_forward_proba
from tsml.portfolio.strategy import generate_signals
from tsml.validation.splitters import WalkForwardSplit


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Return value of :func:`simulate`.

    Attributes
    ----------
    equity_curve:
        Daily portfolio value indexed by trading-day timestamps.
        Starts at ``initial_capital`` on the first day.
    trades_log:
        One row per executed trade (buys and sells only; holds are omitted).
        Columns: ``date``, ``symbol``, ``action`` (``"buy"``/``"sell"``),
        ``score`` (P(up) used to rank that symbol on that day).
    """

    equity_curve: pd.Series
    trades_log: pd.DataFrame


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def simulate(
    symbols: Sequence[str],
    model: Any,
    splitter: WalkForwardSplit,
    *,
    start_date: str,
    end_date: str,
    target: str = "direction",
    top_n: int = 5,
    min_score: float = 0.55,
    costs_bps: float = 5.0,
    initial_capital: float = 1.0,
    rebalance_frequency: str = "weekly",
    loader: DataLoader | None = None,
) -> SimulationResult:
    """
    Simulate periodic portfolio rebalancing driven by walk-forward model scores.

    Parameters
    ----------
    symbols:
        Ticker strings for the universe to trade (e.g. ``["SPY", "QQQ"]``).
    model:
        Any object with ``.fit(X, y)`` and ``.predict_proba(X)`` methods.
        Reused across all symbols; state is overwritten at each fold.
    splitter:
        A configured ``WalkForwardSplit``.  Symbols whose cleaned dataset is
        too small to yield even one valid fold are skipped.
    start_date:
        Inclusive start date for data loading (``"YYYY-MM-DD"``).
    end_date:
        Inclusive end date for data loading (``"YYYY-MM-DD"``).
    target:
        Target type passed to ``make_dataset``.  Must be one of
        ``"direction"``, ``"direction_5d"``, ``"threshold"``, ``"return"``.
        Default ``"direction"``.
    top_n:
        Maximum number of symbols to hold simultaneously.
    min_score:
        Minimum P(up) score for a symbol to be eligible to buy.  Held
        positions that fall below this threshold are sold.
    costs_bps:
        One-way transaction cost in basis points applied to traded notional
        at each rebalance.  Default 5 bps.
    initial_capital:
        Starting portfolio value (any positive float, or 1.0 for a
        normalised equity curve).
    rebalance_frequency:
        Only ``"weekly"`` is currently supported.
    loader:
        Optional ``DataLoader``.  Defaults to
        ``YFinanceLoader(cache_dir="data/raw")``.

    Returns
    -------
    SimulationResult
        Contains ``equity_curve`` (daily ``pd.Series``) and ``trades_log``
        (``pd.DataFrame``).  If every symbol fails pre-computation, both
        fields are empty.

    Raises
    ------
    ValueError
        If ``rebalance_frequency`` is not ``"weekly"``.
    """
    if rebalance_frequency != "weekly":
        raise ValueError(
            f"rebalance_frequency must be 'weekly'; got '{rebalance_frequency}'."
        )
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    # ------------------------------------------------------------------
    # Phase 1: Load prices and compute walk-forward probabilities
    # ------------------------------------------------------------------

    close_map: dict[str, pd.Series] = {}
    proba_map: dict[str, pd.Series] = {}

    for symbol in symbols:
        try:
            df     = loader.load(symbol, start_date, end_date)
            probas = run_walk_forward_proba(df, model, splitter, target=target)
            close_map[symbol] = df["close"]
            proba_map[symbol] = probas
        except Exception as exc:  # noqa: BLE001
            print(
                f"[simulate] skipping {symbol}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if not close_map:
        return SimulationResult(
            equity_curve=pd.Series(dtype=float, name="portfolio_value"),
            trades_log=pd.DataFrame(columns=["date", "symbol", "action", "score"]),
        )

    # ------------------------------------------------------------------
    # Phase 2: Shared time axis, daily returns, score matrix
    # ------------------------------------------------------------------

    close_df    = pd.DataFrame(close_map)
    trading_days = close_df.index
    daily_rets  = close_df.pct_change()

    # Probability matrix: rows = trading days, columns = symbols.
    # Forward-fill so every row has the most recent known probability.
    # Shift by 1 so the score on day t is the value known at t-1 close.
    proba_df = _build_proba_matrix(proba_map, trading_days)
    score_df = proba_df.shift(1)          # strict past — no lookahead

    # ------------------------------------------------------------------
    # Phase 3: Rebalance dates
    # ------------------------------------------------------------------

    rebalance_dates = _weekly_rebalance_dates(trading_days)

    # ------------------------------------------------------------------
    # Phase 4: Day-by-day simulation loop
    # ------------------------------------------------------------------

    portfolio_value: float         = initial_capital
    current_positions: set[str]    = set()
    equity_rows: list[dict]        = []
    trade_rows:  list[dict]        = []

    for date in trading_days:

        # ── Rebalance ──────────────────────────────────────────────────
        if date in rebalance_dates:
            ranking = _build_ranking(score_df, date)

            if not ranking.empty:
                signals = generate_signals(
                    ranking,
                    current_positions,
                    top_n=top_n,
                    min_score=min_score,
                )

                new_positions = {
                    s.symbol for s in signals if s.action in ("buy", "hold")
                }
                cost = _compute_cost(
                    current_positions, new_positions, portfolio_value, costs_bps
                )
                portfolio_value    = max(0.0, portfolio_value - cost)
                current_positions  = new_positions

                for s in signals:
                    if s.action != "hold":
                        trade_rows.append(
                            {
                                "date":   date,
                                "symbol": s.symbol,
                                "action": s.action,
                                "score":  s.score,
                            }
                        )

        # ── Daily return ───────────────────────────────────────────────
        if current_positions:
            portfolio_value = _apply_daily_return(
                portfolio_value, current_positions, daily_rets, date
            )

        equity_rows.append({"date": date, "value": portfolio_value})

    # ------------------------------------------------------------------
    # Phase 5: Assemble output
    # ------------------------------------------------------------------

    equity_curve = (
        pd.DataFrame(equity_rows)
        .set_index("date")["value"]
        .rename("portfolio_value")
    )

    trades_log = pd.DataFrame(
        trade_rows if trade_rows else [],
        columns=["date", "symbol", "action", "score"],
    )

    return SimulationResult(equity_curve=equity_curve, trades_log=trades_log)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_proba_matrix(
    proba_map: dict[str, pd.Series],
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Reindex each symbol's OOS probability series to all trading days,
    then forward-fill gaps (dates before the first OOS prediction are NaN).
    """
    if not proba_map:
        return pd.DataFrame(index=trading_days)

    return pd.DataFrame(
        {sym: series.reindex(trading_days).ffill() for sym, series in proba_map.items()},
        index=trading_days,
    )


def _weekly_rebalance_dates(trading_days: pd.DatetimeIndex) -> frozenset[pd.Timestamp]:
    """
    Return the first trading day of each ISO calendar week.

    This is normally Monday; if Monday is a holiday the next available
    trading day is used instead.
    """
    if trading_days.empty:
        return frozenset()

    iso   = trading_days.isocalendar()
    # Composite key: "YYYY-WW"
    keys  = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    first = pd.Series(trading_days, index=keys).groupby(level=0).first()
    return frozenset(first)


def _build_ranking(
    score_df: pd.DataFrame, date: pd.Timestamp
) -> pd.DataFrame:
    """
    Return a ranking DataFrame for ``date`` from the pre-shifted score matrix.

    Symbols whose score is NaN on this date (no probability available yet)
    are excluded.
    """
    if score_df.empty or date not in score_df.index:
        return pd.DataFrame(columns=["symbol", "score"])

    row = score_df.loc[date].dropna()
    if row.empty:
        return pd.DataFrame(columns=["symbol", "score"])

    return (
        pd.DataFrame({"symbol": row.index.astype(str), "score": row.values.astype(float)})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def _compute_cost(
    old_positions: set[str],
    new_positions: set[str],
    portfolio_value: float,
    costs_bps: float,
) -> float:
    """
    Transaction cost for moving from ``old_positions`` to ``new_positions``.

    Computed as ``one_way_turnover * portfolio_value * costs_bps * 1e-4``.

    One-way turnover is ``sum(|Δweight|) / 2`` computed across *all*
    constituents including the implicit cash position.  Including cash is
    essential: going from 100 % cash to 2 equal-weight positions deploys
    the full portfolio value and must incur the full one-way cost.

    Δweight for each equity is ``1/N_new − 1/N_old``; Δcash is the
    complementary change ``(1 − Σnew_weights) − (1 − Σold_weights)``.
    Dividing the sum of absolute changes by 2 converts the two-sided
    turnover to a one-way traded notional fraction.
    """
    n_old = len(old_positions)
    n_new = len(new_positions)

    if n_old == 0 and n_new == 0:
        return 0.0

    old_w = {sym: 1.0 / n_old for sym in old_positions} if n_old else {}
    new_w = {sym: 1.0 / n_new for sym in new_positions} if n_new else {}

    # Implicit cash weights (remainder not held in equities).
    old_cash = 1.0 - sum(old_w.values())
    new_cash = 1.0 - sum(new_w.values())

    all_syms   = set(old_w) | set(new_w)
    equity_to  = sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in all_syms)
    cash_to    = abs(new_cash - old_cash)
    one_way_to = (equity_to + cash_to) / 2.0

    return one_way_to * portfolio_value * costs_bps * 1e-4


def _apply_daily_return(
    portfolio_value: float,
    positions: set[str],
    daily_rets: pd.DataFrame,
    date: pd.Timestamp,
) -> float:
    """
    Update ``portfolio_value`` for one day's equal-weight portfolio return.

    Symbols missing from the price data or with NaN return are treated as
    contributing zero return for that day (conservative: does not fabricate
    data, does not crash).
    """
    if not positions or date not in daily_rets.index:
        return portfolio_value

    weight   = 1.0 / len(positions)
    port_ret = 0.0

    for sym in positions:
        if sym not in daily_rets.columns:
            continue
        r = daily_rets.loc[date, sym]
        if not np.isnan(r):
            port_ret += weight * r

    return portfolio_value * (1.0 + port_ret)
