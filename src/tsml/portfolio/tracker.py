"""
Portfolio tracker.

Two concerns live here:

1. Order-log replay (``load_orders`` + ``build_equity_curve``)
   Reads JSONL order files written by ``execution.log_orders``, replays
   approved BUY/SELL trades against historical close prices, and produces
   a day-by-day equity curve, cash series, and holdings matrix.

2. Performance statistics (``compute_portfolio_stats``)
   Takes any equity curve (from replay or simulation) plus a benchmark
   close price series and returns a ``PortfolioStats`` summary.

Execution convention
--------------------
Orders are logged on the signal date *t*.  In replay mode we assume
execution at the **close of that same day** — a conservative approximation
since the project default is signal-at-t / fill-at-t+1.  The difference
only matters for live trading; for historical analysis it is negligible.

Order log format (one JSON line per order, ``logs/orders/YYYY-MM-DD.jsonl``)
----------------------------------------------------------------------------
Required fields read by ``load_orders``:
    timestamp       ISO-8601 UTC string, e.g. "2026-05-12T21:00:00+00:00"
    symbol          Ticker string
    side            "BUY" or "SELL"
    amount          USD notional (BUY); 0.0 for SELL (full position close)
    score           Signal score (float)
    risk_approved   bool — only True rows are loaded

Example
-------
>>> from tsml.data_loader import YFinanceLoader
>>> from tsml.portfolio.tracker import load_orders, build_equity_curve
>>>
>>> orders = load_orders("logs/orders")
>>> symbols = orders["symbol"].unique().tolist()
>>>
>>> loader = YFinanceLoader(cache_dir="data/raw")
>>> prices = {s: loader.load(s, "2024-01-01", "2026-05-12")["close"] for s in symbols}
>>> price_df = pd.DataFrame(prices)
>>>
>>> history = build_equity_curve(orders, price_df, initial_capital=10_000.0)
>>> print(history.equity_curve.tail())
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tsml.metrics.returns import (
    annualized_volatility,
    cagr as _cagr,
    max_drawdown as _max_drawdown,
    sharpe_ratio as _sharpe_ratio,
    total_return as _total_return,
)


# ===========================================================================
# Section 1 — Order log types
# ===========================================================================

@dataclass
class TradeRecord:
    """
    A single executed trade as parsed from a JSONL order log.

    Attributes
    ----------
    date:
        UTC-midnight timestamp on which the order was logged (signal date).
        Execution is assumed at the close of this day.
    symbol:
        Ticker string.
    side:
        ``"BUY"`` or ``"SELL"``.
    amount:
        USD notional for BUY orders.  Always 0.0 for SELL (full position
        close); actual proceeds are computed from ``shares * close_price``.
    score:
        P(up) score that generated the signal.
    """

    date: pd.Timestamp
    symbol: str
    side: str
    amount: float
    score: float


# ===========================================================================
# Section 2 — Portfolio history type
# ===========================================================================

@dataclass
class PortfolioHistory:
    """
    Day-by-day portfolio state produced by ``build_equity_curve``.

    Attributes
    ----------
    equity_curve:
        Daily portfolio value (cash + market value of all holdings).
        ``pd.Series`` indexed by UTC trading-day timestamps, named
        ``"portfolio_value"``.
    cash:
        Daily cash balance (``pd.Series``, same index, named ``"cash"``).
    positions:
        ``pd.DataFrame`` with trading-day index and one column per traded
        symbol, containing the number of shares held on each day.
        Symbols with no open position on a day show 0.0.
    """

    equity_curve: pd.Series
    cash: pd.Series
    positions: pd.DataFrame


# ===========================================================================
# Section 3 — Performance statistics type
# ===========================================================================

@dataclass
class PortfolioStats:
    """
    Performance statistics for a strategy compared against a benchmark.

    Attributes
    ----------
    label:
        Display name of the strategy.
    benchmark_label:
        Display name of the benchmark (e.g. ``"SPY"``).
    start_date:
        First date in the aligned evaluation period.
    end_date:
        Last date in the aligned evaluation period.
    n_days:
        Calendar days between start_date and end_date (inclusive).

    Strategy metrics
    ----------------
    total_return:
        Compounded total return (e.g. 0.25 = +25 %).
    cagr:
        Compound annual growth rate.
    volatility:
        Annualised standard deviation of daily returns.
    sharpe:
        Annualised Sharpe ratio (uses the supplied risk-free rate).
    max_drawdown:
        Maximum peak-to-trough decline — always ≤ 0.

    Benchmark metrics
    -----------------
    Same definitions, prefixed ``benchmark_``.

    Relative metric
    ---------------
    excess_return:
        Strategy CAGR minus benchmark CAGR.
        Positive means the strategy outperformed the benchmark.
    """

    label: str
    benchmark_label: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int

    # Strategy
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float

    # Benchmark
    benchmark_total_return: float
    benchmark_cagr: float
    benchmark_volatility: float
    benchmark_sharpe: float
    benchmark_max_drawdown: float

    # Relative
    excess_return: float


# ===========================================================================
# Section 4 — Order log loading
# ===========================================================================

def load_orders(
    logs_dir: str | Path = "logs/orders",
) -> pd.DataFrame:
    """
    Load all approved orders from JSONL files in ``logs_dir``.

    Each ``.jsonl`` file contains one JSON object per line.  Only rows
    where ``risk_approved`` is ``True`` are included — rejected orders are
    skipped silently.

    Parameters
    ----------
    logs_dir:
        Directory containing ``YYYY-MM-DD.jsonl`` files.
        Defaults to ``"logs/orders"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (UTC Timestamp), ``symbol``, ``side``,
        ``amount`` (float), ``score`` (float).
        Sorted by ``date`` ascending.  Returns an empty DataFrame
        (same schema) if no files exist or no approved orders are found.
    """
    logs_dir = Path(logs_dir)
    rows: list[dict] = []

    for path in sorted(logs_dir.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not entry.get("risk_approved", False):
                    continue
                rows.append(
                    {
                        "date":   _parse_order_date(entry["timestamp"]),
                        "symbol": entry["symbol"],
                        "side":   entry["side"],
                        "amount": float(entry.get("amount", 0.0)),
                        "score":  float(entry.get("score", float("nan"))),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[load_orders] skipping {path.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "side", "amount", "score"])

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _parse_order_date(timestamp: str) -> pd.Timestamp:
    """
    Parse an ISO-8601 timestamp string to a UTC-midnight Timestamp.

    Order timestamps record the exact time the plan was generated
    (e.g. ``"2026-05-12T21:00:00+00:00"``).  We normalise to midnight
    UTC so the date aligns with the price index from ``YFinanceLoader``.
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()  # midnight UTC


# ===========================================================================
# Section 5 — Equity curve construction from orders
# ===========================================================================

def build_equity_curve(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> PortfolioHistory:
    """
    Replay order history against close prices to build a portfolio equity curve.

    Portfolio mechanics
    -------------------
    - **BUY**: ``amount`` USD is spent; shares acquired = ``amount / close``.
      Cash decreases by ``amount``.
    - **SELL**: full position closed; proceeds = ``shares * close``.
      Cash increases by proceeds; holdings set to 0.
    - Fractional shares are allowed (no rounding).
    - Multiple BUYs of the same symbol accumulate shares (dollar-cost
      averaging).
    - Orders with no price data on their date are skipped with a warning.

    Daily portfolio value = ``cash + Σ(shares_i × close_i)``

    Parameters
    ----------
    orders:
        DataFrame from ``load_orders()``.  Must have columns
        ``date``, ``symbol``, ``side``, ``amount``.
    prices:
        Wide DataFrame: index = UTC-midnight DatetimeIndex (trading days),
        columns = ticker symbols, values = close prices.
        The date range should span from the earliest order date to the
        desired end date.
    initial_capital:
        Starting cash balance in USD.

    Returns
    -------
    PortfolioHistory
        Contains ``equity_curve``, ``cash``, and ``positions`` (shares per
        symbol per day).

    Notes
    -----
    - Trading days are taken from ``prices.index``; only those days appear
      in the output.
    - Symbols in orders that are absent from ``prices.columns`` are skipped
      with a warning.
    - Cash can never go below zero; if a BUY would drive it negative the
      order is skipped with a warning.
    """
    _EMPTY_COLS = ["date", "symbol", "side", "amount", "score"]

    if prices.empty or orders.empty or orders[orders["side"].isin(["BUY", "SELL"])].empty:
        idx = prices.index if not prices.empty else pd.DatetimeIndex([])
        flat = pd.Series(initial_capital, index=idx, name="portfolio_value")
        cash_s = pd.Series(initial_capital, index=idx, name="cash")
        pos_df = pd.DataFrame(index=idx, columns=prices.columns if not prices.empty else [], dtype=float).fillna(0.0)
        return PortfolioHistory(equity_curve=flat, cash=cash_s, positions=pos_df)

    trading_days = prices.index
    symbols_in_prices = set(prices.columns)

    # Build a lookup: date → list of order rows (already UTC-midnight)
    orders_by_date: dict[pd.Timestamp, list[dict]] = {}
    for row in orders.to_dict("records"):
        d = row["date"]
        orders_by_date.setdefault(d, []).append(row)

    # State
    cash: float = initial_capital
    holdings: dict[str, float] = {}   # symbol → shares

    # Output collectors
    equity_vals: list[float] = []
    cash_vals:   list[float] = []
    # positions: we record holdings snapshot per day for each traded symbol
    traded_symbols: set[str] = set(
        r["symbol"] for r in orders.to_dict("records")
        if r["symbol"] in symbols_in_prices
    )
    pos_rows: dict[str, list[float]] = {sym: [] for sym in traded_symbols}

    for date in trading_days:
        # ── Execute orders logged on this date ────────────────────────────
        for order in orders_by_date.get(date, []):
            sym    = order["symbol"]
            side   = order["side"]
            amount = float(order["amount"])

            if sym not in symbols_in_prices:
                print(
                    f"[build_equity_curve] {sym} not in prices — skipping "
                    f"{side} on {date.date()}",
                    file=sys.stderr,
                )
                continue

            close = prices.loc[date, sym]
            if pd.isna(close) or close <= 0:
                print(
                    f"[build_equity_curve] no valid price for {sym} on "
                    f"{date.date()} — skipping {side}",
                    file=sys.stderr,
                )
                continue

            if side == "BUY":
                if amount > cash:
                    print(
                        f"[build_equity_curve] insufficient cash for BUY "
                        f"{sym} ${amount:.2f} on {date.date()} — skipping",
                        file=sys.stderr,
                    )
                    continue
                shares = amount / close
                holdings[sym] = holdings.get(sym, 0.0) + shares
                cash -= amount

            elif side == "SELL":
                shares_held = holdings.get(sym, 0.0)
                if shares_held <= 0:
                    continue  # nothing to sell (may have been sold already)
                proceeds = shares_held * close
                cash += proceeds
                holdings[sym] = 0.0

        # ── Snapshot daily portfolio value ────────────────────────────────
        equity = cash
        for sym, shares in holdings.items():
            if sym in symbols_in_prices and shares > 0:
                px = prices.loc[date, sym]
                if not pd.isna(px) and px > 0:
                    equity += shares * px

        equity_vals.append(equity)
        cash_vals.append(cash)

        for sym in traded_symbols:
            pos_rows[sym].append(holdings.get(sym, 0.0))

    equity_curve = pd.Series(equity_vals, index=trading_days, name="portfolio_value")
    cash_series  = pd.Series(cash_vals,   index=trading_days, name="cash")
    positions_df = pd.DataFrame(pos_rows, index=trading_days).fillna(0.0)

    return PortfolioHistory(
        equity_curve=equity_curve,
        cash=cash_series,
        positions=positions_df,
    )


# ===========================================================================
# Section 6 — Weekly returns helper
# ===========================================================================

def weekly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Compute weekly portfolio returns from a daily equity curve.

    Resamples to end-of-week (Friday) using the last available trading day
    in each week.  Returns are computed as ``(end / start) - 1`` between
    consecutive week-end values.

    Parameters
    ----------
    equity_curve:
        Daily portfolio value ``pd.Series`` with a ``DatetimeIndex``.

    Returns
    -------
    pd.Series
        Weekly return series (fraction, e.g. 0.02 = 2 %), indexed by
        week-end dates, named ``"weekly_return"``.
        Returns an empty Series if ``equity_curve`` has fewer than 2 values.
    """
    if len(equity_curve) < 2:
        return pd.Series(dtype=float, name="weekly_return")
    weekly = equity_curve.resample("W-FRI").last().dropna()
    return weekly.pct_change().dropna().rename("weekly_return")


# ===========================================================================
# Section 7 — Performance statistics
# ===========================================================================

def compute_portfolio_stats(
    equity_curve: pd.Series,
    benchmark_close: pd.Series,
    *,
    label: str = "portfolio",
    benchmark_label: str = "SPY",
    risk_free_rate: float = 0.0,
) -> PortfolioStats:
    """
    Compute strategy and benchmark performance statistics.

    Parameters
    ----------
    equity_curve:
        Daily portfolio *value* series (e.g. from
        ``PortfolioHistory.equity_curve`` or
        ``SimulationResult.equity_curve``).
        Must have a ``DatetimeIndex`` and at least 2 rows.  Converted to
        daily percentage returns via ``pct_change().dropna()``.
    benchmark_close:
        Daily benchmark *close price* series (e.g. SPY from
        ``YFinanceLoader``).  Must have a ``DatetimeIndex``.  Aligned to
        the same dates as ``equity_curve`` by inner join.
    label:
        Display name for the strategy.
    benchmark_label:
        Display name for the benchmark.
    risk_free_rate:
        Annual risk-free rate for the Sharpe computation.  Default 0.0.

    Returns
    -------
    PortfolioStats

    Raises
    ------
    ValueError
        If ``equity_curve`` has fewer than 2 rows, or the two series share
        fewer than 2 common dates.
    """
    if len(equity_curve) < 2:
        raise ValueError(
            f"equity_curve must have at least 2 rows, got {len(equity_curve)}."
        )

    common = equity_curve.index.intersection(benchmark_close.index)
    if len(common) < 2:
        raise ValueError(
            "equity_curve and benchmark_close share fewer than 2 common dates. "
            "Check that both cover the same period."
        )

    strat_eq = equity_curve.loc[common].sort_index()
    bench_px = benchmark_close.loc[common].sort_index()

    strat_ret = strat_eq.pct_change().dropna()
    bench_ret = bench_px.pct_change().dropna()

    strat_cagr = _cagr(strat_ret)
    bench_cagr = _cagr(bench_ret)

    return PortfolioStats(
        label=label,
        benchmark_label=benchmark_label,
        start_date=strat_eq.index[0],
        end_date=strat_eq.index[-1],
        n_days=int((strat_eq.index[-1] - strat_eq.index[0]).days),
        # Strategy
        total_return=_total_return(strat_ret),
        cagr=strat_cagr,
        volatility=annualized_volatility(strat_ret),
        sharpe=_sharpe_ratio(strat_ret, risk_free_rate),
        max_drawdown=_max_drawdown(strat_ret),
        # Benchmark
        benchmark_total_return=_total_return(bench_ret),
        benchmark_cagr=bench_cagr,
        benchmark_volatility=annualized_volatility(bench_ret),
        benchmark_sharpe=_sharpe_ratio(bench_ret, risk_free_rate),
        benchmark_max_drawdown=_max_drawdown(bench_ret),
        # Relative
        excess_return=strat_cagr - bench_cagr,
    )
