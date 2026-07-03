"""
Parameter robustness / sensitivity sweep for the weekly ML portfolio strategy.

Usage::

    python scripts/parameter_sweep.py --start 2018-01-01
    python scripts/parameter_sweep.py --fast
    python scripts/parameter_sweep.py --max-combinations 50 --jobs 4
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
from tsml.portfolio.parameter_sweep import (
    count_parameter_combinations,
    default_parameter_grid,
    format_parameter_sweep_report,
    run_parameter_sweep,
)
from tsml.reporting.parameter_sweep_plots import generate_parameter_sweep_plots
from tsml.validation.splitters import (
    AdaptiveWalkForwardParams,
    make_adaptive_walk_forward_splitter,
)

UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "JNJ", "XOM", "V", "GS", "NFLX",
]

TARGET = "direction_5d"
FEATURE_SET = "extended_v2"


# gap must cover the 5-day label horizon of direction_5d.
_WALK_FORWARD_PARAMS = AdaptiveWalkForwardParams(gap=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter robustness sweep for weekly ML portfolio backtest.",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller parameter grid for quick runs.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Cap the number of grid combinations (random sample if exceeded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when sampling combinations.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes (default 1).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/parameter_sweep",
        help="Directory for CSV report and plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top parameter sets to highlight.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    end = args.end or date.today().isoformat()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = default_parameter_grid(fast=args.fast)
    full_size = count_parameter_combinations(grid)

    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Period:   {args.start} -> {end}")
    print(f"Grid:     {full_size} combinations"
          + (" (fast mode)" if args.fast else ""))
    if args.max_combinations is not None:
        print(f"Cap:      {args.max_combinations} runs (seed={args.seed})")
    print(f"Workers:  {args.jobs}")
    print()

    sweep = run_parameter_sweep(
        UNIVERSE,
        start=args.start,
        end=end,
        model=CalibratedLogisticRegressionModel(),
        splitter=make_adaptive_walk_forward_splitter(
            args.start, end, _WALK_FORWARD_PARAMS
        ),
        grid=grid,
        target=TARGET,
        feature_set=FEATURE_SET,
        fast=args.fast,
        max_combinations=args.max_combinations,
        seed=args.seed,
        n_jobs=args.jobs,
        top_n_results=args.top_n,
    )

    print(format_parameter_sweep_report(sweep))
    print()

    csv_path = out_dir / "parameter_sweep_results.csv"
    sweep.results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    if sweep.results.empty:
        print("No results produced — try a longer date range.")
        return

    plot_paths = generate_parameter_sweep_plots(
        sweep.results,
        sweep.diagnostics.parameter_importance,
        out_dir,
    )
    for name, path in plot_paths.items():
        print(f"Plot {name}: {path}")


if __name__ == "__main__":
    main()
