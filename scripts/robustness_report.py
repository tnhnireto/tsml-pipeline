"""
Robustness and stress-test report for the weekly ML portfolio strategy.

Usage::

    python scripts/robustness_report.py --start 2018-01-01 --end 2024-06-30
    python scripts/robustness_report.py --exclude-symbols NVDA,QQQ --random-subset-size 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.portfolio.robustness import (
    build_universe_variants,
    format_robustness_report,
    run_robustness_analysis,
)
from tsml.reporting.robustness_plots import plot_rolling_performance
from tsml.validation.splitters import WalkForwardSplit, coverage_n_splits

UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "JNJ", "XOM", "V", "GS", "NFLX",
]

TARGET = "direction_5d"
FEATURE_SET = "extended_v2"


# gap must cover the 5-day label horizon of direction_5d.
def _make_splitter(start: str, end: str) -> WalkForwardSplit:
    """Size folds to the date range so OOS scores cover the whole backtest
    instead of forward-filling stale probabilities past fold 5."""
    return WalkForwardSplit(
        n_splits=coverage_n_splits(
            start, end, min_train_size=252, test_size=63, gap=5
        ),
        min_train_size=252,
        test_size=63,
        gap=5,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness and stress-test report for weekly ML portfolio.",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--exclude-symbols",
        default="",
        help="Comma-separated symbols to exclude as separate variants.",
    )
    parser.add_argument(
        "--random-subset-size",
        type=int,
        default=None,
        help="If set, run two random universe subsets of this size.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for ASCII report and plots.",
    )
    parser.add_argument(
        "--skip-universe-variants",
        action="store_true",
        help="Skip universe robustness runs (faster).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    end = args.end or date.today().isoformat()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = [s.strip() for s in args.exclude_symbols.split(",") if s.strip()]

    variants = None
    if not args.skip_universe_variants:
        variants = build_universe_variants(
            UNIVERSE,
            exclude_symbols=exclude,
            random_subset_size=args.random_subset_size,
            seed=args.random_seed,
            include_defaults=True,
        )

    print(f"Running robustness analysis  {args.start} -> {end}")
    print(f"Universe: {len(UNIVERSE)} symbols, feature_set={FEATURE_SET}, target={TARGET}")
    if variants:
        print(f"Universe variants: {', '.join(variants.keys())}")
    print()

    model = CalibratedLogisticRegressionModel()
    report = run_robustness_analysis(
        UNIVERSE,
        start=args.start,
        end=end,
        model=model,
        splitter=_make_splitter(args.start, end),
        universe_variants=variants,
        target=TARGET,
        feature_set=FEATURE_SET,
        top_n=5,
        min_score=0.55,
        min_score_downtrend=0.62,
        cash_buffer_pct=0.05,
        costs_bps=5.0,
        initial_capital=100_000.0,
        turnover_control=True,
        buy_threshold=0.58,
        sell_threshold=0.52,
        min_hold_weeks=2,
    )

    text = format_robustness_report(report)
    print(text)

    report_path = out_dir / "robustness_report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(f"\nReport saved to {report_path.resolve()}")

    if not report.rolling.rolling_sharpe.empty:
        plot_path = plot_rolling_performance(
            report.rolling,
            title=f"Rolling Performance  ({args.start} - {end})",
            output_path=out_dir / "robustness_rolling_performance.png",
        )
        print(f"Rolling performance plot saved to {plot_path}")

    if not report.feature_importance.empty:
        imp_path = out_dir / "robustness_feature_importance.csv"
        report.feature_importance.to_csv(imp_path, index=False)
        print(f"Feature importance saved to {imp_path.resolve()}")


if __name__ == "__main__":
    main()
