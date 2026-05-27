"""
Out-of-sample holdout evaluation report for the weekly ML portfolio strategy.

Evaluates fixed strategy parameters on a development window and reports
generalization to a later unseen holdout period.

Usage::

    python scripts/holdout_report.py
    python scripts/holdout_report.py --dev-start 2018-01-01 --holdout-end 2024-12-31
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
from tsml.portfolio.holdout_eval import (
    HoldoutPeriods,
    format_holdout_report,
    run_holdout_evaluation,
)
from tsml.reporting.holdout_plots import plot_holdout_equity
from tsml.validation.splitters import WalkForwardSplit

UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "JNJ", "XOM", "V", "GS", "NFLX",
]

WALK_FORWARD = WalkForwardSplit(n_splits=5, min_train_size=252, test_size=63, gap=1)

REPORT_PATH = Path("reports/holdout_report.txt")
EQUITY_PLOT_PATH = Path("reports/holdout_equity.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-sample holdout evaluation for weekly ML portfolio.",
    )
    parser.add_argument("--dev-start", default="2018-01-01")
    parser.add_argument("--dev-end", default="2022-12-31")
    parser.add_argument("--holdout-start", default="2023-01-01")
    parser.add_argument("--holdout-end", default=None)
    parser.add_argument(
        "--report-path",
        default=str(REPORT_PATH),
        help="Path for ASCII report.",
    )
    parser.add_argument(
        "--equity-plot",
        default=str(EQUITY_PLOT_PATH),
        help="Path for equity curve PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    holdout_end = args.holdout_end or date.today().isoformat()

    periods = HoldoutPeriods(
        dev_start=args.dev_start,
        dev_end=args.dev_end,
        holdout_start=args.holdout_start,
        holdout_end=holdout_end,
    )

    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Development: {periods.dev_start} -> {periods.dev_end}")
    print(f"Holdout:     {periods.holdout_start} -> {periods.holdout_end}")
    print()

    result = run_holdout_evaluation(
        UNIVERSE,
        periods=periods,
        model=CalibratedLogisticRegressionModel(),
        splitter=WALK_FORWARD,
    )

    report = format_holdout_report(result)
    print(report)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path.resolve()}")

    if result.equity_curve.empty:
        print("No equity data — skipping plot.")
        return

    plot_path = plot_holdout_equity(
        result.equity_curve,
        result.benchmark_closes,
        periods,
        args.equity_plot,
    )
    print(f"Equity plot saved to {plot_path}")


if __name__ == "__main__":
    main()
