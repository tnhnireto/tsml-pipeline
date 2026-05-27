"""
Out-of-sample holdout evaluation for the weekly ML portfolio strategy.

Runs a single continuous backtest with fixed parameters, then reports
development and holdout metrics from non-overlapping date slices.
Parameters must be chosen on the development period only; holdout is
for validation, not tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.portfolio.regime_overlay import RegimeOverlayConfig
from tsml.portfolio.simulator import SimulationResult
from tsml.portfolio.weekly_backtest import (
    BacktestMetrics,
    _buy_and_hold_equity,
    _metrics_from_equity,
    compute_backtest_metrics,
    run_weekly_backtest,
)
from tsml.validation.splitters import WalkForwardSplit

DEFAULT_DEV_START = "2018-01-01"
DEFAULT_DEV_END = "2022-12-31"
DEFAULT_HOLDOUT_START = "2023-01-01"
DEFAULT_HOLDOUT_END = "2024-12-31"

# Fixed strategy parameters (chosen on development data — not tuned on holdout).
DEFAULT_STRATEGY = dict(
    feature_set="extended",
    regime_overlay_enabled=True,
    buy_threshold=0.58,
    sell_threshold=0.52,
    min_hold_weeks=2,
    bear_exposure=0.25,
    min_score=0.55,
    min_score_downtrend=0.62,
    top_n=5,
    target="threshold",
    cash_buffer_pct=0.05,
    costs_bps=5.0,
    turnover_control=True,
)

SHARPE_DROP_WARN = 0.50
DRAWDOWN_WORSEN_RATIO = 1.50


@dataclass(frozen=True)
class HoldoutPeriods:
    """Development and holdout date ranges (inclusive)."""

    dev_start: str
    dev_end: str
    holdout_start: str
    holdout_end: str

    def full_start(self) -> str:
        return self.dev_start

    def full_end(self) -> str:
        return self.holdout_end


@dataclass
class PeriodEvaluation:
    """Backtest metrics for one evaluation window."""

    label: str
    start: str
    end: str
    metrics: BacktestMetrics
    benchmark_metrics: dict[str, BacktestMetrics]


@dataclass
class HoldoutEvaluationResult:
    """Full development + holdout evaluation output."""

    periods: HoldoutPeriods
    strategy_config: dict[str, Any]
    development: PeriodEvaluation
    holdout: PeriodEvaluation
    simulation: SimulationResult
    equity_curve: pd.Series
    benchmark_closes: dict[str, pd.Series]
    warnings: list[str]


def default_holdout_periods() -> HoldoutPeriods:
    return HoldoutPeriods(
        dev_start=DEFAULT_DEV_START,
        dev_end=DEFAULT_DEV_END,
        holdout_start=DEFAULT_HOLDOUT_START,
        holdout_end=DEFAULT_HOLDOUT_END,
    )


def validate_periods_no_overlap(periods: HoldoutPeriods) -> None:
    """Raise if development and holdout windows overlap or are mis-ordered."""
    dev_end = pd.Timestamp(periods.dev_end)
    holdout_start = pd.Timestamp(periods.holdout_start)
    dev_start = pd.Timestamp(periods.dev_start)
    holdout_end = pd.Timestamp(periods.holdout_end)

    if dev_start > dev_end:
        raise ValueError(
            f"dev_start ({periods.dev_start}) must be on or before "
            f"dev_end ({periods.dev_end})."
        )
    if holdout_start > holdout_end:
        raise ValueError(
            f"holdout_start ({periods.holdout_start}) must be on or before "
            f"holdout_end ({periods.holdout_end})."
        )
    if holdout_start <= dev_end:
        raise ValueError(
            f"Holdout must start after development ends: "
            f"dev_end={periods.dev_end}, holdout_start={periods.holdout_start}."
        )


def fixed_regime_overlay(config: dict[str, Any]) -> RegimeOverlayConfig | None:
    if not config.get("regime_overlay_enabled", False):
        return None
    return RegimeOverlayConfig(
        enabled=True,
        bull_exposure=1.0,
        bear_exposure=float(config["bear_exposure"]),
        high_vol_exposure=None,
    )


def _to_timestamp(date_str: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(date_str)
    if index.tz is not None and ts.tz is None:
        ts = ts.tz_localize(index.tz)
    elif index.tz is not None and ts.tz != index.tz:
        ts = ts.tz_convert(index.tz)
    return ts


def metrics_for_period(
    simulation: SimulationResult,
    start: str,
    end: str,
    *,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Compute strategy metrics for an inclusive date slice."""
    eq = simulation.equity_curve
    if eq.empty:
        return compute_backtest_metrics(simulation, risk_free_rate=risk_free_rate)

    start_ts = _to_timestamp(start, eq.index)
    end_ts = _to_timestamp(end, eq.index)
    window = eq.loc[start_ts:end_ts]
    if len(window) < 2:
        return _metrics_from_equity(window, risk_free_rate=risk_free_rate)

    exposure = (
        simulation.exposure.loc[start_ts:end_ts]
        if not simulation.exposure.empty
        else None
    )

    turnover = 0.0
    if not simulation.rebalance_log.empty:
        log = simulation.rebalance_log.copy()
        log["date"] = pd.to_datetime(log["date"])
        if log["date"].dt.tz is None and start_ts.tz is not None:
            log["date"] = log["date"].dt.tz_localize(start_ts.tz)
        mask = (log["date"] >= start_ts) & (log["date"] <= end_ts)
        turnover = float(log.loc[mask, "turnover"].sum())

    n_trades = 0
    if not simulation.trades_log.empty:
        trades = simulation.trades_log.copy()
        trades["date"] = pd.to_datetime(trades["date"])
        if trades["date"].dt.tz is None and start_ts.tz is not None:
            trades["date"] = trades["date"].dt.tz_localize(start_ts.tz)
        mask = (trades["date"] >= start_ts) & (trades["date"] <= end_ts)
        n_trades = int(mask.sum())

    return _metrics_from_equity(
        window,
        turnover=turnover,
        exposure=exposure,
        n_trades=n_trades,
        avg_holding_weeks=simulation.avg_holding_weeks,
        risk_free_rate=risk_free_rate,
    )


def benchmark_metrics_for_period(
    benchmark_closes: dict[str, pd.Series],
    strategy_index: pd.DatetimeIndex,
    start: str,
    end: str,
    *,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.0,
) -> dict[str, BacktestMetrics]:
    """Buy-and-hold benchmark metrics for a date slice."""
    start_ts = _to_timestamp(start, strategy_index)
    end_ts = _to_timestamp(end, strategy_index)
    window_index = strategy_index[(strategy_index >= start_ts) & (strategy_index <= end_ts)]

    out: dict[str, BacktestMetrics] = {}
    for sym, close in benchmark_closes.items():
        bench_eq = _buy_and_hold_equity(close, window_index, initial_capital)
        out[sym] = _metrics_from_equity(bench_eq, risk_free_rate=risk_free_rate)
    return out


def compute_holdout_warnings(
    development: BacktestMetrics,
    holdout: BacktestMetrics,
    *,
    sharpe_drop_threshold: float = SHARPE_DROP_WARN,
    drawdown_worsen_ratio: float = DRAWDOWN_WORSEN_RATIO,
) -> list[str]:
    """Return warning messages when holdout degrades vs development."""
    warnings: list[str] = []

    dev_sharpe = development.sharpe
    ho_sharpe = holdout.sharpe
    if not pd.isna(dev_sharpe) and not pd.isna(ho_sharpe):
        drop = dev_sharpe - ho_sharpe
        if drop >= sharpe_drop_threshold:
            warnings.append(
                f"Holdout Sharpe ({ho_sharpe:.2f}) is {drop:.2f} below "
                f"development Sharpe ({dev_sharpe:.2f})."
            )
        elif dev_sharpe > 0 and ho_sharpe < 0.5 * dev_sharpe:
            warnings.append(
                f"Holdout Sharpe ({ho_sharpe:.2f}) is less than half of "
                f"development Sharpe ({dev_sharpe:.2f})."
            )

    dev_mdd = development.max_drawdown
    ho_mdd = holdout.max_drawdown
    if dev_mdd < 0 and ho_mdd < 0:
        if abs(ho_mdd) >= abs(dev_mdd) * drawdown_worsen_ratio:
            warnings.append(
                f"Holdout max drawdown ({ho_mdd:+.1%}) is materially worse than "
                f"development ({dev_mdd:+.1%})."
            )

    return warnings


def run_holdout_evaluation(
    symbols: Sequence[str],
    *,
    periods: HoldoutPeriods | None = None,
    model: Any,
    splitter: WalkForwardSplit,
    strategy_config: dict[str, Any] | None = None,
    benchmark_symbols: Sequence[str] = ("SPY", "QQQ"),
    loader: DataLoader | None = None,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.0,
) -> HoldoutEvaluationResult:
    """
    Evaluate fixed strategy parameters on development and holdout slices.

    A single continuous simulation from ``dev_start`` through ``holdout_end``
    preserves walk-forward history across the split without using holdout
    labels for parameter selection.
    """
    if periods is None:
        periods = default_holdout_periods()
    validate_periods_no_overlap(periods)

    config = dict(DEFAULT_STRATEGY)
    if strategy_config is not None:
        config.update(strategy_config)

    regime = fixed_regime_overlay(config)

    full = run_weekly_backtest(
        symbols,
        start=periods.full_start(),
        end=periods.full_end(),
        model=model,
        splitter=splitter,
        target=config["target"],
        top_n=config["top_n"],
        min_score=config["min_score"],
        min_score_downtrend=config["min_score_downtrend"],
        cash_buffer_pct=config["cash_buffer_pct"],
        costs_bps=config["costs_bps"],
        initial_capital=initial_capital,
        benchmark_symbols=benchmark_symbols,
        loader=loader,
        risk_free_rate=risk_free_rate,
        turnover_control=config["turnover_control"],
        buy_threshold=config["buy_threshold"],
        sell_threshold=config["sell_threshold"],
        min_hold_weeks=config["min_hold_weeks"],
        compare_baseline=False,
        feature_set=config["feature_set"],
        regime_overlay=regime,
    )

    sim = full.simulation
    dev_metrics = metrics_for_period(
        sim, periods.dev_start, periods.dev_end, risk_free_rate=risk_free_rate
    )
    ho_metrics = metrics_for_period(
        sim, periods.holdout_start, periods.holdout_end, risk_free_rate=risk_free_rate
    )

    eq_index = full.equity_curve.index
    dev_bench = benchmark_metrics_for_period(
        full.benchmark_closes,
        eq_index,
        periods.dev_start,
        periods.dev_end,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
    )
    ho_bench = benchmark_metrics_for_period(
        full.benchmark_closes,
        eq_index,
        periods.holdout_start,
        periods.holdout_end,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
    )

    development = PeriodEvaluation(
        label="Development",
        start=periods.dev_start,
        end=periods.dev_end,
        metrics=dev_metrics,
        benchmark_metrics=dev_bench,
    )
    holdout = PeriodEvaluation(
        label="Holdout",
        start=periods.holdout_start,
        end=periods.holdout_end,
        metrics=ho_metrics,
        benchmark_metrics=ho_bench,
    )

    warnings = compute_holdout_warnings(dev_metrics, ho_metrics)

    return HoldoutEvaluationResult(
        periods=periods,
        strategy_config=config,
        development=development,
        holdout=holdout,
        simulation=sim,
        equity_curve=full.equity_curve,
        benchmark_closes=full.benchmark_closes,
        warnings=warnings,
    )


def _format_metrics_block(
    period: PeriodEvaluation,
    bench_syms: Sequence[str],
) -> list[str]:
    def pct(v: float) -> str:
        return f"{v:+.2%}"

    lines = [
        f"  {period.label} ({period.start} -> {period.end})",
        "  " + "-" * 50,
        f"  {'Total Return':<22}{pct(period.metrics.total_return):>12}",
        f"  {'CAGR':<22}{pct(period.metrics.cagr):>12}",
        f"  {'Sharpe':<22}{period.metrics.sharpe:>12.2f}",
        f"  {'Max Drawdown':<22}{pct(period.metrics.max_drawdown):>12}",
        f"  {'Turnover (cum.)':<22}{period.metrics.turnover:>12.2%}",
        f"  {'Avg Exposure':<22}{period.metrics.exposure:>12.2%}",
        f"  {'Trades':<22}{period.metrics.n_trades:>12d}",
    ]

    if period.benchmark_metrics and bench_syms:
        col_w = 12
        lines.append("")
        header = f"  {'Benchmark':<22}" + "".join(f"{s:>{col_w}}" for s in bench_syms)
        lines.append(header)
        rows = [
            ("CAGR", "cagr", pct),
            ("Sharpe", "sharpe", lambda v: f"{v:.2f}"),
            ("Max Drawdown", "max_drawdown", pct),
        ]
        for label, attr, fmt in rows:
            vals = "".join(
                f"{fmt(period.benchmark_metrics[s].__dict__[attr]):>{col_w}}"
                for s in bench_syms
                if s in period.benchmark_metrics
            )
            lines.append(f"  {label:<22}{vals}")

    return lines


def format_holdout_report(result: HoldoutEvaluationResult) -> str:
    """Format development vs holdout ASCII report."""
    cfg = result.strategy_config
    bench_syms = list(result.development.benchmark_metrics.keys()) or ["SPY", "QQQ"]

    lines = [
        "=" * 68,
        "  Weekly Strategy - Out-of-Sample Holdout Evaluation",
        "=" * 68,
        "",
        "  Fixed parameters (not tuned on holdout)",
        "  " + "-" * 50,
        f"  {'feature_set':<22}{cfg.get('feature_set', 'extended'):>12}",
        f"  {'regime_overlay':<22}{str(cfg.get('regime_overlay_enabled', True)):>12}",
        f"  {'buy_threshold':<22}{cfg.get('buy_threshold', 0.58):>12.2f}",
        f"  {'sell_threshold':<22}{cfg.get('sell_threshold', 0.52):>12.2f}",
        f"  {'min_hold_weeks':<22}{cfg.get('min_hold_weeks', 2):>12d}",
        f"  {'bear_exposure':<22}{cfg.get('bear_exposure', 0.25):>12.0%}",
        f"  {'cash_buffer':<22}{cfg.get('cash_buffer_pct', 0.05):>12.0%}",
        f"  {'costs_bps':<22}{cfg.get('costs_bps', 5.0):>12.1f}",
        "",
    ]

    lines.extend(_format_metrics_block(result.development, bench_syms))
    lines.append("")
    lines.extend(_format_metrics_block(result.holdout, bench_syms))

    lines.append("")
    lines.append("  Holdout vs development")
    lines.append("  " + "-" * 50)
    d = result.development.metrics
    h = result.holdout.metrics
    lines.append(
        f"  {'Sharpe delta':<22}{h.sharpe - d.sharpe:>+12.2f}"
    )
    lines.append(
        f"  {'CAGR delta':<22}{h.cagr - d.cagr:>+12.2%}"
    )
    lines.append(
        f"  {'MDD delta':<22}{h.max_drawdown - d.max_drawdown:>+12.2%}"
    )

    if result.warnings:
        lines.append("")
        lines.append("  WARNINGS")
        lines.append("  " + "-" * 50)
        for w in result.warnings:
            lines.append(f"  ! {w}")
    else:
        lines.append("")
        lines.append("  No holdout degradation warnings.")

    lines.append("=" * 68)
    return "\n".join(lines)
