"""
Weekly backtest orchestration and reporting.

Mirrors the live signal workflow in ``run_weekly_signal.py`` by delegating
simulation to :func:`~tsml.portfolio.simulator.simulate` and attaching
benchmark comparisons plus summary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.metrics.returns import (
    cagr as _cagr,
    max_drawdown as _max_drawdown,
    sharpe_ratio as _sharpe_ratio,
    total_return as _total_return,
)
from tsml.portfolio.regime_overlay import RegimeOverlayConfig
from tsml.portfolio.regime_periods import REGIME_PERIODS
from tsml.portfolio.simulator import SimulationResult, simulate
from tsml.validation.splitters import WalkForwardSplit


@dataclass
class BacktestMetrics:
    """Performance summary for a strategy or buy-and-hold benchmark."""

    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    turnover: float
    exposure: float
    n_trades: int
    avg_holding_weeks: float = 0.0


def exposure_adjusted_turnover(turnover: float, avg_exposure: float) -> float:
    """Normalize cumulative turnover by average equity exposure."""
    if avg_exposure <= 0.0:
        return 0.0
    return turnover / avg_exposure


@dataclass
class FeatureSetComparison:
    """Side-by-side backtest metrics for legacy vs extended features."""

    legacy: BacktestMetrics
    extended: BacktestMetrics


@dataclass
class RegimeOverlayStats:
    """Summary statistics for regime exposure overlay."""

    avg_exposure: float
    avg_regime_target: float
    pct_days_full: float
    pct_days_bear: float
    pct_days_zero: float
    exposure_by_regime: dict[str, float]


@dataclass
class RegimeOverlayComparison:
    """Side-by-side backtest with and without regime overlay."""

    without_overlay: BacktestMetrics
    with_overlay: BacktestMetrics
    overlay_stats: RegimeOverlayStats


@dataclass
class WeeklyBacktestResult:
    """Full output of :func:`run_weekly_backtest`."""

    simulation: SimulationResult
    equity_curve: pd.Series
    metrics: BacktestMetrics
    benchmark_metrics: dict[str, BacktestMetrics]
    benchmark_closes: dict[str, pd.Series]
    baseline_metrics: BacktestMetrics | None = None


def _metrics_from_equity(
    equity: pd.Series,
    *,
    turnover: float = 0.0,
    exposure: pd.Series | None = None,
    n_trades: int = 0,
    avg_holding_weeks: float = 0.0,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    if len(equity) < 2:
        return BacktestMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            turnover=turnover,
            exposure=0.0,
            n_trades=n_trades,
            avg_holding_weeks=avg_holding_weeks,
        )

    rets = equity.pct_change().dropna()
    avg_exposure = float(exposure.mean()) if exposure is not None and not exposure.empty else 0.0

    return BacktestMetrics(
        total_return=_total_return(rets),
        cagr=_cagr(rets),
        sharpe=_sharpe_ratio(rets, risk_free_rate),
        max_drawdown=_max_drawdown(rets),
        turnover=turnover,
        exposure=avg_exposure,
        n_trades=n_trades,
        avg_holding_weeks=avg_holding_weeks,
    )


def compute_backtest_metrics(
    result: SimulationResult,
    *,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Derive strategy metrics from a :class:`SimulationResult`."""
    return _metrics_from_equity(
        result.equity_curve,
        turnover=result.turnover_total,
        exposure=result.exposure,
        n_trades=len(result.trades_log),
        avg_holding_weeks=result.avg_holding_weeks,
        risk_free_rate=risk_free_rate,
    )


def _buy_and_hold_equity(
    close: pd.Series,
    strategy_index: pd.DatetimeIndex,
    initial_capital: float,
) -> pd.Series:
    aligned = close.reindex(strategy_index).ffill()
    if aligned.isna().all():
        return pd.Series(dtype=float)
    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(dtype=float)
    aligned = aligned.loc[first_valid:]
    rets = aligned.pct_change().fillna(0.0)
    equity = initial_capital * (1.0 + rets).cumprod()
    return equity.reindex(strategy_index).ffill()


def _simulate_common_kwargs(
    symbols: Sequence[str],
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    loader: DataLoader,
    *,
    target: str,
    top_n: int,
    min_score: float,
    min_score_downtrend: float,
    cash_buffer_pct: float,
    costs_bps: float,
    initial_capital: float,
) -> dict:
    return dict(
        symbols=list(symbols),
        model=model,
        splitter=splitter,
        start_date=start,
        end_date=end,
        target=target,
        top_n=top_n,
        min_score=min_score,
        min_score_downtrend=min_score_downtrend,
        cash_buffer_pct=cash_buffer_pct,
        costs_bps=costs_bps,
        initial_capital=initial_capital,
        loader=loader,
    )


def run_weekly_backtest(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    target: str = "threshold",
    top_n: int = 5,
    min_score: float = 0.55,
    min_score_downtrend: float = 0.62,
    cash_buffer_pct: float = 0.05,
    costs_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    benchmark_symbols: Sequence[str] = ("SPY", "QQQ"),
    loader: DataLoader | None = None,
    risk_free_rate: float = 0.0,
    turnover_control: bool = True,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.52,
    min_hold_weeks: int = 2,
    compare_baseline: bool = True,
    feature_set: str = "extended",
    regime_overlay: RegimeOverlayConfig | None = None,
) -> WeeklyBacktestResult:
    """
    Run a weekly-rebalance backtest and compare against buy-and-hold benchmarks.

    When ``turnover_control`` is enabled (default), the simulator applies
    hysteresis thresholds and a minimum holding period to reduce churn.
    Optionally runs a legacy baseline (no turnover control) for comparison.
    """
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    common = _simulate_common_kwargs(
        symbols,
        start,
        end,
        model,
        splitter,
        loader,
        target=target,
        top_n=top_n,
        min_score=min_score,
        min_score_downtrend=min_score_downtrend,
        cash_buffer_pct=cash_buffer_pct,
        costs_bps=costs_bps,
        initial_capital=initial_capital,
    )

    simulation = simulate(
        **common,
        turnover_control=turnover_control,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        min_hold_weeks=min_hold_weeks,
        feature_set=feature_set,
        regime_overlay=regime_overlay,
    )

    metrics = compute_backtest_metrics(simulation, risk_free_rate=risk_free_rate)

    baseline_metrics: BacktestMetrics | None = None
    if turnover_control and compare_baseline:
        baseline_sim = simulate(**common, turnover_control=False)
        baseline_metrics = compute_backtest_metrics(
            baseline_sim, risk_free_rate=risk_free_rate
        )

    benchmark_closes: dict[str, pd.Series] = {}
    benchmark_metrics: dict[str, BacktestMetrics] = {}

    eq_index = simulation.equity_curve.index
    for sym in benchmark_symbols:
        try:
            df = loader.load(sym, start, end)
            close = df["close"]
            benchmark_closes[sym] = close
            bench_eq = _buy_and_hold_equity(
                close, eq_index, initial_capital
            )
            benchmark_metrics[sym] = _metrics_from_equity(
                bench_eq,
                risk_free_rate=risk_free_rate,
            )
        except Exception:
            continue

    return WeeklyBacktestResult(
        simulation=simulation,
        equity_curve=simulation.equity_curve,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        benchmark_closes=benchmark_closes,
        baseline_metrics=baseline_metrics,
    )


def format_backtest_report(result: WeeklyBacktestResult) -> str:
    """Return a human-readable ASCII report for stdout."""
    m = result.metrics
    lines: list[str] = []

    def pct(v: float) -> str:
        return f"{v:+.2%}"

    def pos_pct(v: float) -> str:
        return f"{v:.2%}"

    lines.append("=" * 60)
    lines.append("  Weekly Portfolio Backtest")
    if not result.equity_curve.empty:
        start = result.equity_curve.index[0].date()
        end = result.equity_curve.index[-1].date()
        lines.append(f"  {start} -> {end}  ({len(result.equity_curve)} days)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  Strategy")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Total Return':<22}{pct(m.total_return):>12}")
    lines.append(f"  {'CAGR':<22}{pct(m.cagr):>12}")
    lines.append(f"  {'Sharpe':<22}{m.sharpe:>12.2f}")
    lines.append(f"  {'Max Drawdown':<22}{pct(m.max_drawdown):>12}")
    lines.append(f"  {'Turnover (cum.)':<22}{pos_pct(m.turnover):>12}")
    adj_to = exposure_adjusted_turnover(m.turnover, m.exposure)
    if m.exposure > 0:
        lines.append(f"  {'Turnover (exp-adj.)':<22}{adj_to:>12.2f}")
    lines.append(f"  {'Avg Exposure':<22}{pos_pct(m.exposure):>12}")
    lines.append(f"  {'Trades':<22}{m.n_trades:>12d}")
    lines.append(f"  {'Avg Hold (weeks)':<22}{m.avg_holding_weeks:>12.1f}")

    if result.baseline_metrics is not None:
        b = result.baseline_metrics
        lines.append("")
        lines.append("  Turnover control vs baseline (legacy rules)")
        lines.append("  " + "-" * 40)
        if b.turnover > 0:
            reduction = (b.turnover - m.turnover) / b.turnover
            lines.append(
                f"  {'Turnover reduction':<22}{pos_pct(reduction):>12}"
            )
        else:
            lines.append(f"  {'Turnover reduction':<22}{'n/a':>12}")
        lines.append(
            f"  {'Baseline turnover':<22}{pos_pct(b.turnover):>12}"
        )
        lines.append(
            f"  {'Baseline trades':<22}{b.n_trades:>12d}"
        )

    lines.append("")

    if result.benchmark_metrics:
        bench_syms = list(result.benchmark_metrics.keys())
        col_w = 12
        header = f"  {'Metric':<22}" + "".join(f"{s:>{col_w}}" for s in bench_syms)
        lines.append("  Benchmarks (buy & hold)")
        lines.append("  " + "-" * (22 + col_w * len(bench_syms)))
        lines.append(header)

        rows = [
            ("Total Return", "total_return", pct),
            ("CAGR", "cagr", pct),
            ("Sharpe", "sharpe", lambda v: f"{v:.2f}"),
            ("Max Drawdown", "max_drawdown", pct),
        ]
        for label, attr, fmt in rows:
            vals = "".join(
                f"{fmt(getattr(result.benchmark_metrics[s], attr)):>{col_w}}"
                for s in bench_syms
            )
            lines.append(f"  {label:<22}{vals}")

    overlay_active = (
        not result.simulation.regime_target.empty
        and (result.simulation.regime_target < 0.99).any()
    )
    if overlay_active:
        stats = compute_regime_overlay_stats(result.simulation)
        lines.append("")
        lines.append("  Regime overlay statistics")
        lines.append("  " + "-" * 40)
        lines.append(
            f"  {'Avg regime target':<22}{pos_pct(stats.avg_regime_target):>12}"
        )
        lines.append(
            f"  {'Days at full exposure':<22}{pos_pct(stats.pct_days_full):>12}"
        )
        lines.append(
            f"  {'Days in bear scaling':<22}{pos_pct(stats.pct_days_bear):>12}"
        )
        lines.append(
            f"  {'Days at zero exposure':<22}{pos_pct(stats.pct_days_zero):>12}"
        )
        if stats.exposure_by_regime:
            lines.append("")
            lines.append("  Average exposure by historical regime")
            lines.append("  " + "-" * 40)
            for name, exp in stats.exposure_by_regime.items():
                lines.append(f"  {name:<28}{pos_pct(exp):>12}")

    lines.append("=" * 60)
    return "\n".join(lines)


def compute_regime_overlay_stats(simulation: SimulationResult) -> RegimeOverlayStats:
    """Derive overlay statistics from a simulation with regime targeting."""
    exposure = simulation.exposure
    target = simulation.regime_target
    if exposure.empty or target.empty:
        return RegimeOverlayStats(0, 0, 0, 0, 0, {})

    aligned = pd.concat([exposure, target], axis=1).dropna()
    aligned.columns = ["actual", "target"]

    by_regime: dict[str, float] = {}
    for name, start, end in REGIME_PERIODS:
        start_ts = pd.Timestamp(start, tz=aligned.index.tz)
        end_ts = pd.Timestamp(end, tz=aligned.index.tz)
        mask = (aligned.index >= start_ts) & (aligned.index <= end_ts)
        if mask.any():
            by_regime[name] = float(aligned.loc[mask, "actual"].mean())

    return RegimeOverlayStats(
        avg_exposure=float(aligned["actual"].mean()),
        avg_regime_target=float(aligned["target"].mean()),
        pct_days_full=float((aligned["target"] >= 0.99).mean()),
        pct_days_bear=float((aligned["target"] < 0.99).mean()),
        pct_days_zero=float((aligned["target"] <= 0.0).mean()),
        exposure_by_regime=by_regime,
    )


def run_regime_overlay_comparison(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    loader: DataLoader | None = None,
    risk_free_rate: float = 0.0,
    regime_overlay: RegimeOverlayConfig,
    **simulate_kwargs: Any,
) -> RegimeOverlayComparison:
    """Run backtest without and with regime exposure overlay."""
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    common = _simulate_common_kwargs(
        symbols,
        start,
        end,
        model,
        splitter,
        loader,
        target=simulate_kwargs.get("target", "threshold"),
        top_n=simulate_kwargs.get("top_n", 5),
        min_score=simulate_kwargs.get("min_score", 0.55),
        min_score_downtrend=simulate_kwargs.get("min_score_downtrend", 0.62),
        cash_buffer_pct=simulate_kwargs.get("cash_buffer_pct", 0.05),
        costs_bps=simulate_kwargs.get("costs_bps", 5.0),
        initial_capital=simulate_kwargs.get("initial_capital", 100_000.0),
    )

    sim_kw = {
        "turnover_control": simulate_kwargs.get("turnover_control", True),
        "buy_threshold": simulate_kwargs.get("buy_threshold", 0.58),
        "sell_threshold": simulate_kwargs.get("sell_threshold", 0.52),
        "min_hold_weeks": simulate_kwargs.get("min_hold_weeks", 2),
        "feature_set": simulate_kwargs.get("feature_set", "extended"),
    }

    without = simulate(
        **common,
        regime_overlay=RegimeOverlayConfig(enabled=False),
        **sim_kw,
    )
    with_overlay = simulate(
        **common,
        regime_overlay=regime_overlay,
        **sim_kw,
    )

    return RegimeOverlayComparison(
        without_overlay=compute_backtest_metrics(without, risk_free_rate=risk_free_rate),
        with_overlay=compute_backtest_metrics(with_overlay, risk_free_rate=risk_free_rate),
        overlay_stats=compute_regime_overlay_stats(with_overlay),
    )


def format_regime_overlay_comparison(comp: RegimeOverlayComparison) -> str:
    """Format regime overlay on/off comparison report."""

    def pct(v: float) -> str:
        return f"{v:+.2%}"

    def pos_pct(v: float) -> str:
        return f"{v:.2%}"

    col_w = 14
    lines = [
        "=" * 68,
        "  Regime Overlay Comparison (off vs on)",
        "=" * 68,
        f"  {'Metric':<22}{'No overlay':>{col_w}}{'With overlay':>{col_w}}",
        "  " + "-" * (22 + col_w * 2),
    ]

    rows = [
        ("CAGR", "cagr", pct),
        ("Sharpe", "sharpe", lambda v: f"{v:.2f}"),
        ("Max Drawdown", "max_drawdown", pct),
        ("Turnover (cum.)", "turnover", pos_pct),
        ("Avg Exposure", "exposure", pos_pct),
        ("Trades", "n_trades", lambda v: f"{v:d}"),
    ]
    for label, attr, fmt in rows:
        lines.append(
            f"  {label:<22}"
            f"{fmt(getattr(comp.without_overlay, attr)):>{col_w}}"
            f"{fmt(getattr(comp.with_overlay, attr)):>{col_w}}"
        )

    wo_adj = exposure_adjusted_turnover(
        comp.without_overlay.turnover, comp.without_overlay.exposure
    )
    w_adj = exposure_adjusted_turnover(
        comp.with_overlay.turnover, comp.with_overlay.exposure
    )
    lines.append(
        f"  {'Turnover (exp-adj.)':<22}{wo_adj:>14.2f}{w_adj:>14.2f}"
    )

    stats = comp.overlay_stats
    lines.extend([
        "",
        "  Regime overlay statistics (with overlay run)",
        "  " + "-" * 40,
        f"  {'Avg regime target':<22}{pos_pct(stats.avg_regime_target):>12}",
        f"  {'Avg actual exposure':<22}{pos_pct(stats.avg_exposure):>12}",
        f"  {'Days at full exposure':<22}{pos_pct(stats.pct_days_full):>12}",
        f"  {'Days in bear scaling':<22}{pos_pct(stats.pct_days_bear):>12}",
        f"  {'Days at zero exposure':<22}{pos_pct(stats.pct_days_zero):>12}",
    ])

    if stats.exposure_by_regime:
        lines.append("")
        lines.append("  Average exposure by historical regime")
        lines.append("  " + "-" * 40)
        for name, exp in stats.exposure_by_regime.items():
            lines.append(f"  {name:<28}{pos_pct(exp):>12}")

    lines.append("=" * 68)
    return "\n".join(lines)


def run_feature_set_comparison(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    loader: DataLoader | None = None,
    risk_free_rate: float = 0.0,
    **simulate_kwargs: Any,
) -> FeatureSetComparison:
    """
    Run the same backtest with legacy and extended feature sets.

    Portfolio rules (turnover control, thresholds, etc.) are identical;
    only the model input features differ.
    """
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    common = _simulate_common_kwargs(
        symbols,
        start,
        end,
        model,
        splitter,
        loader,
        target=simulate_kwargs.get("target", "threshold"),
        top_n=simulate_kwargs.get("top_n", 5),
        min_score=simulate_kwargs.get("min_score", 0.55),
        min_score_downtrend=simulate_kwargs.get("min_score_downtrend", 0.62),
        cash_buffer_pct=simulate_kwargs.get("cash_buffer_pct", 0.05),
        costs_bps=simulate_kwargs.get("costs_bps", 5.0),
        initial_capital=simulate_kwargs.get("initial_capital", 100_000.0),
    )

    sim_kw = {
        "turnover_control": simulate_kwargs.get("turnover_control", True),
        "buy_threshold": simulate_kwargs.get("buy_threshold", 0.58),
        "sell_threshold": simulate_kwargs.get("sell_threshold", 0.52),
        "min_hold_weeks": simulate_kwargs.get("min_hold_weeks", 2),
    }

    legacy_sim = simulate(**common, feature_set="legacy", **sim_kw)
    extended_sim = simulate(**common, feature_set="extended", **sim_kw)

    return FeatureSetComparison(
        legacy=compute_backtest_metrics(legacy_sim, risk_free_rate=risk_free_rate),
        extended=compute_backtest_metrics(extended_sim, risk_free_rate=risk_free_rate),
    )


def format_feature_set_comparison(comp: FeatureSetComparison) -> str:
    """Format legacy vs extended feature set backtest comparison."""

    def pct(v: float) -> str:
        return f"{v:+.2%}"

    def pos_pct(v: float) -> str:
        return f"{v:.2%}"

    col_w = 14
    lines = [
        "=" * 60,
        "  Feature Set Comparison (legacy vs extended)",
        "=" * 60,
        f"  {'Metric':<22}{'Legacy':>{col_w}}{'Extended':>{col_w}}",
        "  " + "-" * (22 + col_w * 2),
    ]

    rows = [
        ("CAGR", "cagr", pct),
        ("Sharpe", "sharpe", lambda v: f"{v:.2f}"),
        ("Max Drawdown", "max_drawdown", pct),
        ("Turnover (cum.)", "turnover", pos_pct),
        ("Trades", "n_trades", lambda v: f"{v:d}"),
    ]
    for label, attr, fmt in rows:
        lines.append(
            f"  {label:<22}"
            f"{fmt(getattr(comp.legacy, attr)):>{col_w}}"
            f"{fmt(getattr(comp.extended, attr)):>{col_w}}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
