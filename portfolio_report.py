"""
Portfolio performance report.

Runs the multi-asset simulation over a configurable date range, fetches SPY
as the benchmark, computes PortfolioStats, prints a summary table to stdout,
and saves an equity-curve plot to reports/.

All flags have sensible defaults so the script is safe to run without arguments.

Usage
-----
    python portfolio_report.py
    python portfolio_report.py --start 2021-01-01 --end 2023-12-31
    python portfolio_report.py --top-n 3 --costs-bps 10 --output reports/my_run.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsml.data_loader import YFinanceLoader
from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.portfolio import simulate
from tsml.portfolio.tracker import PortfolioStats, compute_portfolio_stats
from tsml.reporting import plot_portfolio_vs_benchmark
from tsml.validation import WalkForwardSplit

UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "JPM", "V", "UNH",
    "XOM", "JNJ", "WMT", "MA", "HD",
]
BENCHMARK = "SPY"


def _print_stats(stats: PortfolioStats) -> None:
    """Print a formatted two-column summary table to stdout."""
    col = 26

    def pct(v: float) -> str:
        return f"{v:+.2%}"

    def pos_pct(v: float) -> str:
        return f"{v:.2%}"

    print()
    print("=" * 54)
    print("  Portfolio Performance Report")
    print(
        f"  {stats.start_date.date()} \u2192 {stats.end_date.date()}"
        f"  ({stats.n_days} days)"
    )
    print("=" * 54)
    print(f"  {'Metric':<{col}}{'Strategy':>12}{'Benchmark':>12}")
    print("  " + "-" * 50)
    rows = [
        ("Total Return",  pct(stats.total_return),          pct(stats.benchmark_total_return)),
        ("CAGR",          pct(stats.cagr),                  pct(stats.benchmark_cagr)),
        ("Volatility",    pos_pct(stats.volatility),        pos_pct(stats.benchmark_volatility)),
        ("Sharpe",        f"{stats.sharpe:.2f}",            f"{stats.benchmark_sharpe:.2f}"),
        ("Max Drawdown",  pct(stats.max_drawdown),          pct(stats.benchmark_max_drawdown)),
        ("Excess Return", pct(stats.excess_return),         ""),
    ]
    for label, strat_val, bench_val in rows:
        print(f"  {label:<{col}}{strat_val:>12}{bench_val:>12}")
    print("=" * 54)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio performance report")
    parser.add_argument("--start",     default="2020-01-01",            help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default="2023-12-31",            help="Simulation end date (YYYY-MM-DD)")
    parser.add_argument("--top-n",     type=int,   default=5,           help="Max symbols held simultaneously")
    parser.add_argument("--min-score", type=float, default=0.55,        help="Minimum P(up) score to enter")
    parser.add_argument("--costs-bps", type=float, default=5.0,         help="One-way transaction costs in bps")
    parser.add_argument("--output",    default="reports/portfolio_report.png", help="Output plot path")
    args = parser.parse_args()

    loader   = YFinanceLoader(cache_dir="data/raw")
    splitter = WalkForwardSplit(n_splits=5, min_train_size=252, test_size=63)

    print(f"Running simulation {args.start} \u2192 {args.end} ...")
    result = simulate(
        symbols=UNIVERSE,
        model=CalibratedLogisticRegressionModel(),
        splitter=splitter,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        min_score=args.min_score,
        costs_bps=args.costs_bps,
        loader=loader,
    )

    if result.equity_curve.empty:
        print("Simulation produced no data. Check your date range and universe.")
        return

    print(f"Loading {BENCHMARK} benchmark ...")
    spy_df = loader.load(BENCHMARK, args.start, args.end)

    stats = compute_portfolio_stats(
        result.equity_curve,
        spy_df["close"],
        label="Strategy",
        benchmark_label=BENCHMARK,
    )

    _print_stats(stats)

    out = plot_portfolio_vs_benchmark(
        result.equity_curve,
        spy_df["close"],
        title=f"Portfolio vs {BENCHMARK}  ({args.start} \u2013 {args.end})",
        output_path=args.output,
        strategy_label="Strategy",
        benchmark_label=BENCHMARK,
    )
    print(f"Plot saved \u2192 {out}")


if __name__ == "__main__":
    main()
