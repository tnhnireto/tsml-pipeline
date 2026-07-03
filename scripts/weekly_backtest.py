"""
Historical weekly portfolio backtest mirroring the live signal workflow.

Uses the same universe, model config, and portfolio rules as
``run_weekly_signal.py`` / ``run_etoro_demo.py``.

Usage::

    python scripts/weekly_backtest.py
    python scripts/weekly_backtest.py --start 2018-01-01 --end 2024-12-31
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
from tsml.portfolio.regime_overlay import RegimeOverlayConfig
from tsml.portfolio.weekly_backtest import (
    format_backtest_report,
    format_feature_set_comparison,
    format_regime_overlay_comparison,
    run_feature_set_comparison,
    run_regime_overlay_comparison,
    run_weekly_backtest,
)
from tsml.reporting.exposure_plots import plot_exposure_timeline
from tsml.reporting.plots import plot_strategy_vs_benchmarks
from tsml.validation.splitters import (
    AdaptiveWalkForwardParams,
    make_adaptive_walk_forward_splitter,
)

# Same universe and config as run_weekly_signal.py / run_etoro_demo.py
UNIVERSE = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "XOM",
    "V",
    "GS",
    "NFLX",
]

TOP_N = 5
MIN_SCORE = 0.55
MIN_SCORE_DOWNTREND = 0.62
TARGET = "direction_5d"
FEATURE_SET = "extended_v2"
CASH_BUFFER_PCT = 0.05
COSTS_BPS = 5.0
INITIAL_CAPITAL = 100_000.0

# gap must cover the 5-day label horizon of direction_5d.
_WALK_FORWARD_PARAMS = AdaptiveWalkForwardParams(gap=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weekly portfolio backtest (mirrors live signal workflow).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/weekly_backtest_equity.png",
        help="Path for equity curve plot.",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.58,
        help="Minimum score to open a new position (turnover control).",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=0.52,
        help="Minimum score to keep an existing position (turnover control).",
    )
    parser.add_argument(
        "--min-hold-weeks",
        type=int,
        default=2,
        help="Minimum weekly rebalance periods before discretionary sell.",
    )
    parser.add_argument(
        "--no-turnover-control",
        action="store_true",
        help="Disable hysteresis / min-hold rules (legacy backtest behaviour).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip legacy baseline comparison run.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("legacy", "extended", "extended_v2"),
        default=FEATURE_SET,
        help="Model feature set (default: extended_v2).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help="Minimum score for buy eligibility.",
    )
    parser.add_argument(
        "--min-score-downtrend",
        type=float,
        default=MIN_SCORE_DOWNTREND,
        help="Stricter min score when below SMA200.",
    )
    parser.add_argument(
        "--compare-features",
        action="store_true",
        help="Run side-by-side legacy vs extended feature set backtest.",
    )
    parser.add_argument(
        "--regime-overlay",
        action="store_true",
        help="Enable SPY regime exposure scaling (backtest only).",
    )
    parser.add_argument(
        "--compare-regime-overlay",
        action="store_true",
        help="Compare backtest with and without regime overlay.",
    )
    parser.add_argument(
        "--bear-exposure",
        type=float,
        default=0.25,
        help="Target exposure when SPY is below SMA200 (default 0.25).",
    )
    parser.add_argument(
        "--high-vol-exposure",
        type=float,
        default=0.0,
        help="Target exposure when SPY below SMA200 and vol exceeds threshold.",
    )
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=None,
        help="Daily vol threshold for high-vol bear regime (optional).",
    )
    return parser.parse_args()


def _regime_overlay_config(args: argparse.Namespace) -> RegimeOverlayConfig:
    high_vol = args.high_vol_exposure
    vol_thresh = args.vol_threshold
    return RegimeOverlayConfig(
        enabled=True,
        bull_exposure=1.0,
        bear_exposure=args.bear_exposure,
        high_vol_exposure=high_vol if vol_thresh is not None else None,
        vol_threshold=vol_thresh,
    )


def main() -> None:
    args = _parse_args()
    end = args.end or date.today().isoformat()
    walk_forward = make_adaptive_walk_forward_splitter(
        args.start, end, _WALK_FORWARD_PARAMS
    )

    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Period:   {args.start} -> {end}")
    print(f"Config:   target={TARGET}, top_n={TOP_N}, "
          f"min_score={args.min_score}, min_score_downtrend={args.min_score_downtrend}")
    print(f"          feature_set={args.feature_set}")
    print(f"          cash_buffer={CASH_BUFFER_PCT:.0%}, costs={COSTS_BPS} bps")
    if not args.no_turnover_control:
        print(f"          turnover control: buy>={args.buy_threshold}, "
              f"sell>={args.sell_threshold}, min_hold={args.min_hold_weeks}w")
    else:
        print("          turnover control: disabled (legacy rules)")
    if args.regime_overlay or args.compare_regime_overlay:
        vol_note = f", vol>{args.vol_threshold}" if args.vol_threshold else ""
        print(f"          regime overlay: bear={args.bear_exposure:.0%}, "
              f"high_vol={args.high_vol_exposure:.0%}{vol_note}")
    print()

    simulate_kw = dict(
        target=TARGET,
        top_n=TOP_N,
        min_score=args.min_score,
        min_score_downtrend=args.min_score_downtrend,
        cash_buffer_pct=CASH_BUFFER_PCT,
        costs_bps=COSTS_BPS,
        initial_capital=INITIAL_CAPITAL,
        turnover_control=not args.no_turnover_control,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        min_hold_weeks=args.min_hold_weeks,
        feature_set=args.feature_set,
    )

    if args.compare_regime_overlay:
        comparison = run_regime_overlay_comparison(
            UNIVERSE,
            start=args.start,
            end=end,
            model=CalibratedLogisticRegressionModel(),
            splitter=walk_forward,
            regime_overlay=_regime_overlay_config(args),
            **simulate_kw,
        )
        print(format_regime_overlay_comparison(comparison))
        print()
        return

    if args.compare_features:
        comparison = run_feature_set_comparison(
            UNIVERSE,
            start=args.start,
            end=end,
            model=CalibratedLogisticRegressionModel(),
            splitter=walk_forward,
            **simulate_kw,
        )
        print(format_feature_set_comparison(comparison))
        print()
        return

    regime_overlay = (
        _regime_overlay_config(args)
        if args.regime_overlay
        else None
    )

    result = run_weekly_backtest(
        UNIVERSE,
        start=args.start,
        end=end,
        model=CalibratedLogisticRegressionModel(),
        splitter=walk_forward,
        benchmark_symbols=("SPY", "QQQ"),
        compare_baseline=not args.no_baseline,
        regime_overlay=regime_overlay,
        **simulate_kw,
    )

    print(format_backtest_report(result))
    print()

    if result.equity_curve.empty:
        print("No simulation data produced. Use a longer date range "
              "(walk-forward needs ~568 trading days of history per symbol).")
        return

    plot_path = plot_strategy_vs_benchmarks(
        result.equity_curve,
        result.benchmark_closes,
        title="Weekly Portfolio Backtest vs SPY / QQQ",
        output_path=args.output,
        strategy_label="Weekly signal",
    )
    print(f"Equity curve saved to {plot_path}")

    if regime_overlay is not None and not result.simulation.exposure.empty:
        exp_plot = plot_exposure_timeline(
            result.simulation.exposure,
            result.simulation.regime_target,
            title="Portfolio Exposure vs Regime Target",
            output_path=Path(args.output).parent / "regime_exposure_timeline.png",
        )
        print(f"Exposure timeline saved to {exp_plot}")


if __name__ == "__main__":
    main()
