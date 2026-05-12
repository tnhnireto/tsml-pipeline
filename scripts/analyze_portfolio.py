"""
Portfolio performance analyser.

Reads executed order logs from ``logs/orders/``, fetches close prices for
all traded symbols plus a SPY benchmark, reconstructs the portfolio equity
curve, and prints a performance summary.

Usage
-----
    python scripts/analyze_portfolio.py
    python scripts/analyze_portfolio.py --logs logs/orders --capital 10000
    python scripts/analyze_portfolio.py --start 2025-01-01 --benchmark QQQ
    python scripts/analyze_portfolio.py --weeks 12

All flags are optional.  The date range defaults to the first/last order
dates found in the log files; use ``--start`` / ``--end`` to override.

Output
------
  - Performance table: total return, CAGR, volatility, Sharpe, max drawdown
  - Benchmark comparison (SPY by default)
  - Weekly return table (most recent N weeks)
  - Equity curve plot saved to reports/portfolio_from_orders.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsml.data_loader import YFinanceLoader
from tsml.portfolio.tracker import (
    PortfolioHistory,
    PortfolioStats,
    build_equity_curve,
    compute_portfolio_stats,
    load_orders,
    weekly_returns,
)
from tsml.reporting import plot_portfolio_vs_benchmark


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_stats(stats: PortfolioStats) -> None:
    W = 24

    def pct(v: float) -> str:
        return f"{v:+.2%}"

    def pos_pct(v: float) -> str:
        return f"{v:.2%}"

    print()
    print("=" * 54)
    print("  Portfolio Performance")
    print(f"  {stats.start_date.date()} \u2192 {stats.end_date.date()}  ({stats.n_days}d)")
    print("=" * 54)
    header = f"  {'Metric':<{W}}{'Strategy':>12}{'Benchmark':>12}"
    print(header)
    print("  " + "\u2500" * (len(header) - 2))
    rows = [
        ("Total Return",  pct(stats.total_return),          pct(stats.benchmark_total_return)),
        ("CAGR",          pct(stats.cagr),                  pct(stats.benchmark_cagr)),
        ("Volatility",    pos_pct(stats.volatility),        pos_pct(stats.benchmark_volatility)),
        ("Sharpe",        f"{stats.sharpe:.2f}",            f"{stats.benchmark_sharpe:.2f}"),
        ("Max Drawdown",  pct(stats.max_drawdown),          pct(stats.benchmark_max_drawdown)),
        ("Excess Return", pct(stats.excess_return),         ""),
    ]
    for label, strat_val, bench_val in rows:
        print(f"  {label:<{W}}{strat_val:>12}{bench_val:>12}")
    print("=" * 54)


def _print_weekly_returns(wr: pd.Series, n_weeks: int = 8) -> None:
    if wr.empty:
        print("\n  (no weekly return data)\n")
        return

    tail = wr.tail(n_weeks)
    print()
    print(f"  Weekly Returns (last {len(tail)} weeks)")
    print("  " + "\u2500" * 30)
    for date, ret in tail.items():
        bar = _sparkbar(ret)
        sign = "+" if ret >= 0 else ""
        print(f"  {date.date()}   {sign}{ret:.2%}  {bar}")
    print()


def _sparkbar(ret: float, scale: float = 0.05) -> str:
    """Mini ASCII bar chart for a weekly return value."""
    n = int(abs(ret) / scale * 10)
    n = min(n, 20)
    char = "\u2588" if ret >= 0 else "\u2591"
    return char * n


def _print_positions_snapshot(history: PortfolioHistory) -> None:
    """Print the most recent open positions."""
    last_pos = history.positions.iloc[-1]
    open_pos = last_pos[last_pos > 0]
    last_cash = history.cash.iloc[-1]
    last_equity = history.equity_curve.iloc[-1]

    print()
    print("  Current Positions")
    print("  " + "\u2500" * 36)
    if open_pos.empty:
        print("  (no open positions — fully in cash)")
    else:
        for sym, shares in open_pos.items():
            pct_of_port = shares  # shares, not pct — we'd need price to show %
            print(f"  {sym:<8}  {shares:.4f} shares")
    print(f"  {'Cash':<8}  ${last_cash:>12,.2f}")
    print(f"  {'Total':<8}  ${last_equity:>12,.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse portfolio performance from order logs."
    )
    parser.add_argument(
        "--logs", default="logs/orders",
        help="Directory containing JSONL order log files (default: logs/orders)",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Starting portfolio value in USD (default: 10000)",
    )
    parser.add_argument(
        "--start",
        help="Override start date YYYY-MM-DD (default: first order date)",
    )
    parser.add_argument(
        "--end",
        help="Override end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--benchmark", default="SPY",
        help="Benchmark ticker (default: SPY)",
    )
    parser.add_argument(
        "--weeks", type=int, default=8,
        help="Number of recent weeks to show in the weekly return table (default: 8)",
    )
    parser.add_argument(
        "--output", default="reports/portfolio_from_orders.png",
        help="Output plot path (default: reports/portfolio_from_orders.png)",
    )
    args = parser.parse_args()

    # ── 1. Load orders ──────────────────────────────────────────────────────
    print(f"\nLoading orders from {args.logs} ...")
    orders = load_orders(args.logs)

    if orders.empty:
        print(
            "No approved orders found in the log directory.\n"
            "Run run_etoro_demo.py first to generate order logs.",
            file=sys.stderr,
        )
        sys.exit(1)

    symbols = sorted(orders["symbol"].unique().tolist())
    print(f"Found {len(orders)} approved orders across {len(symbols)} symbols: "
          f"{', '.join(symbols)}")

    # ── 2. Date range ───────────────────────────────────────────────────────
    first_order_date = orders["date"].min().strftime("%Y-%m-%d")
    last_order_date  = orders["date"].max().strftime("%Y-%m-%d")
    start_date = args.start or first_order_date
    end_date   = args.end   or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    print(f"Date range: {start_date} \u2192 {end_date}")

    # ── 3. Fetch prices ─────────────────────────────────────────────────────
    loader = YFinanceLoader(cache_dir="data/raw")
    all_symbols = symbols + ([args.benchmark] if args.benchmark not in symbols else [])

    price_map: dict[str, pd.Series] = {}
    for sym in all_symbols:
        try:
            df = loader.load(sym, start_date, end_date)
            price_map[sym] = df["close"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not load {sym}: {exc}", file=sys.stderr)

    if not price_map:
        print("No price data available.", file=sys.stderr)
        sys.exit(1)

    # Build wide price DataFrame (trading days × symbols)
    price_df = pd.DataFrame(price_map)
    bench_close = price_map.get(args.benchmark)

    # ── 4. Build equity curve ───────────────────────────────────────────────
    print("Building equity curve ...")
    history = build_equity_curve(
        orders=orders,
        prices=price_df.drop(columns=[args.benchmark], errors="ignore"),
        initial_capital=args.capital,
    )

    if history.equity_curve.empty:
        print("Equity curve is empty — no tradeable orders in the date range.",
              file=sys.stderr)
        sys.exit(1)

    # ── 5. Performance stats ────────────────────────────────────────────────
    if bench_close is not None and len(history.equity_curve) >= 2:
        try:
            stats = compute_portfolio_stats(
                history.equity_curve,
                bench_close,
                label="Portfolio",
                benchmark_label=args.benchmark,
            )
            _print_stats(stats)
        except ValueError as exc:
            print(f"  [warn] could not compute benchmark stats: {exc}", file=sys.stderr)
    else:
        print("  (benchmark data unavailable — skipping benchmark comparison)")

    # ── 6. Weekly returns ───────────────────────────────────────────────────
    wr = weekly_returns(history.equity_curve)
    _print_weekly_returns(wr, n_weeks=args.weeks)

    # ── 7. Positions snapshot ───────────────────────────────────────────────
    if not history.positions.empty:
        _print_positions_snapshot(history)

    # ── 8. Plot ─────────────────────────────────────────────────────────────
    if bench_close is not None:
        try:
            out = plot_portfolio_vs_benchmark(
                history.equity_curve,
                bench_close,
                title=f"Portfolio vs {args.benchmark}  ({start_date} \u2013 {end_date})",
                output_path=args.output,
                strategy_label="Portfolio",
                benchmark_label=args.benchmark,
            )
            print(f"Plot saved \u2192 {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] plot failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
