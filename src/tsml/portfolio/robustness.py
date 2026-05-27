"""
Robustness and stress-testing analysis for the weekly ML portfolio strategy.

Evaluates regime dependence, rolling performance, universe sensitivity,
score calibration, and feature-importance stability without changing
live execution or broker code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.features.benchmarks import load_benchmark_closes
from tsml.metrics.returns import cagr, max_drawdown, sharpe_ratio, total_return
from tsml.pipelines.diagnostics import (
    FoldImportance,
    aggregate_fold_importance,
    run_walk_forward_diagnostics,
)
from tsml.portfolio.regime_periods import REGIME_PERIODS
from tsml.portfolio.simulator import SimulationResult, simulate
from tsml.portfolio.weekly_backtest import BacktestMetrics, compute_backtest_metrics
from tsml.validation.splitters import WalkForwardSplit


SCORE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.50-0.55", 0.50, 0.55),
    ("0.55-0.60", 0.55, 0.60),
    ("0.60-0.65", 0.60, 0.65),
    ("0.65+", 0.65, 1.01),
)


@dataclass
class RegimeMetrics:
    """Performance metrics for one historical regime."""

    name: str
    start: str
    end: str
    metrics: BacktestMetrics


@dataclass
class RollingMetrics:
    """Rolling performance series derived from an equity curve."""

    rolling_sharpe: pd.Series
    rolling_cagr: pd.Series
    rolling_drawdown: pd.Series


@dataclass
class ScoreBucketStats:
    """Calibration stats for one predicted-score bucket."""

    bucket: str
    win_rate: float
    avg_forward_return: float
    n_observations: int


@dataclass
class UniverseVariantResult:
    """Backtest summary for one universe variant."""

    label: str
    symbols: list[str]
    metrics: BacktestMetrics


@dataclass
class RobustnessReport:
    """Consolidated robustness analysis output."""

    simulation: SimulationResult
    full_metrics: BacktestMetrics
    regime_metrics: list[RegimeMetrics]
    rolling: RollingMetrics
    score_calibration: list[ScoreBucketStats]
    feature_importance: pd.DataFrame
    fold_importances: list[FoldImportance]
    universe_variants: list[UniverseVariantResult] = field(default_factory=list)


def _slice_period(
    equity: pd.Series,
    trades: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    start_ts = pd.Timestamp(start, tz=equity.index.tz)
    end_ts = pd.Timestamp(end, tz=equity.index.tz)
    mask = (equity.index >= start_ts) & (equity.index <= end_ts)
    eq = equity.loc[mask]
    tr = trades[(trades["date"] >= start_ts) & (trades["date"] <= end_ts)]
    rb = rebalance_log[
        (rebalance_log["date"] >= start_ts) & (rebalance_log["date"] <= end_ts)
    ]
    return eq, tr, rb


def metrics_for_period(
    equity: pd.Series,
    trades: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    exposure: pd.Series,
    *,
    avg_holding_weeks: float = 0.0,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Compute backtest metrics for a sliced equity curve."""
    if len(equity) < 2:
        return BacktestMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            turnover=float(rebalance_log["turnover"].sum()) if not rebalance_log.empty else 0.0,
            exposure=0.0,
            n_trades=len(trades),
            avg_holding_weeks=avg_holding_weeks,
        )

    rets = equity.pct_change().dropna()
    exp_slice = exposure.reindex(equity.index).fillna(0.0)
    return BacktestMetrics(
        total_return=total_return(rets),
        cagr=cagr(rets),
        sharpe=sharpe_ratio(rets, risk_free_rate),
        max_drawdown=max_drawdown(rets),
        turnover=float(rebalance_log["turnover"].sum()) if not rebalance_log.empty else 0.0,
        exposure=float(exp_slice.mean()),
        n_trades=len(trades),
        avg_holding_weeks=avg_holding_weeks,
    )


def evaluate_regime_metrics(
    simulation: SimulationResult,
    *,
    regimes: Sequence[tuple[str, str, str]] = REGIME_PERIODS,
    risk_free_rate: float = 0.0,
) -> list[RegimeMetrics]:
    """Split simulation results into historical regime windows."""
    results: list[RegimeMetrics] = []
    eq = simulation.equity_curve
    if eq.empty:
        return results

    for name, start, end in regimes:
        eq_slice, tr, rb = _slice_period(
            eq, simulation.trades_log, simulation.rebalance_log, start, end
        )
        if len(eq_slice) < 2:
            continue
        m = metrics_for_period(
            eq_slice,
            tr,
            rb,
            simulation.exposure,
            risk_free_rate=risk_free_rate,
        )
        results.append(RegimeMetrics(name=name, start=start, end=end, metrics=m))
    return results


def compute_rolling_metrics(
    equity: pd.Series,
    *,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> RollingMetrics:
    """Compute rolling Sharpe, CAGR, and drawdown from an equity curve."""
    if len(equity) < window + 1:
        empty = pd.Series(dtype=float)
        return RollingMetrics(empty, empty, empty)

    rets = equity.pct_change().dropna()
    rolling_sharpe = rets.rolling(window).apply(
        lambda x: sharpe_ratio(pd.Series(x), risk_free_rate),
        raw=False,
    )
    rolling_cagr = rets.rolling(window).apply(
        lambda x: cagr(pd.Series(x)),
        raw=False,
    )

    peak = equity.cummax()
    rolling_dd = (equity - peak) / peak

    return RollingMetrics(
        rolling_sharpe=rolling_sharpe.rename("rolling_sharpe_252d"),
        rolling_cagr=rolling_cagr.rename("rolling_cagr_252d"),
        rolling_drawdown=rolling_dd.rename("rolling_drawdown"),
    )


def build_universe_variants(
    universe: Sequence[str],
    *,
    exclude_symbols: Sequence[str] | None = None,
    random_subset_size: int | None = None,
    seed: int = 42,
    include_defaults: bool = True,
) -> dict[str, list[str]]:
    """
    Build universe variants for robustness testing.

    Always includes ``full``.  When ``include_defaults`` is True, also adds
    ``without_NVDA`` and ``without_QQQ`` when those symbols are present.
    """
    base = list(universe)
    variants: dict[str, list[str]] = {"full": base}

    if include_defaults:
        for sym in ("NVDA", "QQQ"):
            if sym in base:
                variants[f"without_{sym}"] = [s for s in base if s != sym]

    for sym in exclude_symbols or []:
        sym = sym.strip().upper()
        if sym:
            variants[f"without_{sym}"] = [s for s in base if s != sym]

    if random_subset_size is not None and random_subset_size > 0:
        rng = np.random.default_rng(seed)
        size = min(random_subset_size, len(base))
        for i in range(2):
            chosen = sorted(rng.choice(base, size=size, replace=False).tolist())
            variants[f"random_{i + 1}_n{size}"] = chosen

    return variants


def collect_score_calibration(
    symbols: Sequence[str],
    loader: DataLoader,
    model: Any,
    splitter: WalkForwardSplit,
    *,
    start: str,
    end: str,
    target: str = "threshold",
    feature_set: str = "extended",
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Collect OOS predicted scores with realised forward returns.

    Forward returns are used for evaluation only — never as model inputs.
    """
    benchmarks = None
    if feature_set == "extended":
        benchmarks = load_benchmark_closes(loader, start, end)

    records: list[dict] = []
    for symbol in symbols:
        try:
            df = loader.load(symbol, start, end)
            diag = run_walk_forward_diagnostics(
                df,
                model,
                splitter,
                target=target,
                feature_set=feature_set,
                benchmarks=benchmarks,
            )
            close = df["close"]
            fwd = close.shift(-forward_days) / close - 1.0
            for date, score in diag.probas.items():
                if date not in fwd.index:
                    continue
                fr = fwd.loc[date]
                if pd.isna(fr):
                    continue
                records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "score": float(score),
                        "forward_return": float(fr),
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(records)


def summarise_score_calibration(calibration_df: pd.DataFrame) -> list[ScoreBucketStats]:
    """Group calibration observations into score buckets."""
    if calibration_df.empty:
        return []

    stats: list[ScoreBucketStats] = []
    for label, lo, hi in SCORE_BUCKETS:
        if label.endswith("+"):
            mask = calibration_df["score"] >= lo
        else:
            mask = (calibration_df["score"] >= lo) & (calibration_df["score"] < hi)
        bucket = calibration_df.loc[mask]
        if bucket.empty:
            stats.append(ScoreBucketStats(label, 0.0, 0.0, 0))
            continue
        stats.append(
            ScoreBucketStats(
                bucket=label,
                win_rate=float((bucket["forward_return"] > 0).mean()),
                avg_forward_return=float(bucket["forward_return"].mean()),
                n_observations=len(bucket),
            )
        )
    return stats


def collect_feature_importance_diagnostics(
    symbols: Sequence[str],
    loader: DataLoader,
    model: Any,
    splitter: WalkForwardSplit,
    *,
    start: str,
    end: str,
    target: str = "threshold",
    feature_set: str = "extended",
    max_symbols: int = 5,
) -> tuple[pd.DataFrame, list[FoldImportance]]:
    """
    Aggregate walk-forward feature importance across symbols and folds.

    Uses up to ``max_symbols`` successfully scored symbols to keep runtime
    reasonable while capturing cross-sectional stability.
    """
    benchmarks = None
    if feature_set == "extended":
        benchmarks = load_benchmark_closes(loader, start, end)

    all_folds: list[FoldImportance] = []
    scored = 0
    for symbol in symbols:
        if scored >= max_symbols:
            break
        try:
            df = loader.load(symbol, start, end)
            diag = run_walk_forward_diagnostics(
                df,
                model,
                splitter,
                target=target,
                feature_set=feature_set,
                benchmarks=benchmarks,
            )
            for fi in diag.fold_importances:
                tagged = FoldImportance(
                    fold=fi.fold,
                    train_end=fi.train_end,
                    test_start=fi.test_start,
                    test_end=fi.test_end,
                    importance=fi.importance.assign(symbol=symbol),
                )
                all_folds.append(tagged)
            scored += 1
        except Exception:
            continue

    if not all_folds:
        return (
            pd.DataFrame(
                columns=["feature", "mean_importance", "std_importance", "stability"]
            ),
            [],
        )

    # Pool importances across symbols — each fold×symbol is one snapshot.
    plain_folds = [
        FoldImportance(
            fold=i,
            train_end=fi.train_end,
            test_start=fi.test_start,
            test_end=fi.test_end,
            importance=fi.importance[["feature", "importance"]],
        )
        for i, fi in enumerate(all_folds)
    ]
    summary = aggregate_fold_importance(plain_folds)
    return summary, all_folds


def run_universe_robustness(
    variants: dict[str, list[str]],
    *,
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    loader: DataLoader,
    risk_free_rate: float = 0.0,
    **simulate_kwargs: Any,
) -> list[UniverseVariantResult]:
    """Run backtests for each universe variant."""
    results: list[UniverseVariantResult] = []
    for label, syms in variants.items():
        if len(syms) < 2:
            continue
        sim = simulate(
            syms,
            model=model,
            splitter=splitter,
            start_date=start,
            end_date=end,
            loader=loader,
            **simulate_kwargs,
        )
        results.append(
            UniverseVariantResult(
                label=label,
                symbols=syms,
                metrics=compute_backtest_metrics(sim, risk_free_rate=risk_free_rate),
            )
        )
    return results


def run_robustness_analysis(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    loader: DataLoader | None = None,
    risk_free_rate: float = 0.0,
    universe_variants: dict[str, list[str]] | None = None,
    **simulate_kwargs: Any,
) -> RobustnessReport:
    """
    Run the full robustness analysis suite on one backtest configuration.
    """
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    simulation = simulate(
        list(symbols),
        model=model,
        splitter=splitter,
        start_date=start,
        end_date=end,
        loader=loader,
        **simulate_kwargs,
    )

    full_metrics = compute_backtest_metrics(simulation, risk_free_rate=risk_free_rate)
    regime_metrics = evaluate_regime_metrics(simulation, risk_free_rate=risk_free_rate)
    rolling = compute_rolling_metrics(simulation.equity_curve, risk_free_rate=risk_free_rate)

    cal_df = collect_score_calibration(
        symbols,
        loader,
        model,
        splitter,
        start=start,
        end=end,
        target=simulate_kwargs.get("target", "threshold"),
        feature_set=simulate_kwargs.get("feature_set", "extended"),
    )
    score_calibration = summarise_score_calibration(cal_df)

    feat_imp, fold_snaps = collect_feature_importance_diagnostics(
        symbols,
        loader,
        model,
        splitter,
        start=start,
        end=end,
        target=simulate_kwargs.get("target", "threshold"),
        feature_set=simulate_kwargs.get("feature_set", "extended"),
    )

    variant_results: list[UniverseVariantResult] = []
    if universe_variants:
        variant_results = run_universe_robustness(
            universe_variants,
            start=start,
            end=end,
            model=model,
            splitter=splitter,
            loader=loader,
            risk_free_rate=risk_free_rate,
            **simulate_kwargs,
        )

    return RobustnessReport(
        simulation=simulation,
        full_metrics=full_metrics,
        regime_metrics=regime_metrics,
        rolling=rolling,
        score_calibration=score_calibration,
        feature_importance=feat_imp,
        fold_importances=fold_snaps,
        universe_variants=variant_results,
    )


def format_robustness_report(report: RobustnessReport) -> str:
    """Render consolidated ASCII robustness report."""

    def pct(v: float) -> str:
        return f"{v:+.2%}"

    def pos_pct(v: float) -> str:
        return f"{v:.2%}"

    lines: list[str] = [
        "=" * 72,
        "  Weekly ML Portfolio — Robustness & Stress Test Report",
        "=" * 72,
        "",
        "  Full-period summary",
        "  " + "-" * 40,
    ]
    m = report.full_metrics
    lines.extend([
        f"  {'CAGR':<22}{pct(m.cagr):>12}",
        f"  {'Sharpe':<22}{m.sharpe:>12.2f}",
        f"  {'Max Drawdown':<22}{pct(m.max_drawdown):>12}",
        f"  {'Turnover (cum.)':<22}{pos_pct(m.turnover):>12}",
        f"  {'Trades':<22}{m.n_trades:>12d}",
        "",
        "  Regime breakdown",
        "  " + "-" * 68,
        f"  {'Regime':<22}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}{'Turnover':>10}{'Trades':>8}",
    ])

    for rm in report.regime_metrics:
        rm_m = rm.metrics
        lines.append(
            f"  {rm.name:<22}"
            f"{pct(rm_m.cagr):>10}"
            f"{rm_m.sharpe:>10.2f}"
            f"{pct(rm_m.max_drawdown):>10}"
            f"{pos_pct(rm_m.turnover):>10}"
            f"{rm_m.n_trades:>8d}"
        )

    lines.extend([
        "",
        "  Score calibration (5d forward return by predicted score)",
        "  " + "-" * 60,
        f"  {'Bucket':<12}{'Win rate':>12}{'Avg fwd ret':>14}{'Count':>10}",
    ])
    for b in report.score_calibration:
        lines.append(
            f"  {b.bucket:<12}"
            f"{pos_pct(b.win_rate):>12}"
            f"{pct(b.avg_forward_return):>14}"
            f"{b.n_observations:>10d}"
        )

    if not report.feature_importance.empty:
        lines.extend([
            "",
            "  Feature importance (top 10 by mean |coef| across folds)",
            "  " + "-" * 60,
            f"  {'Feature':<28}{'Mean':>10}{'Std':>10}{'Stability':>10}",
        ])
        for _, row in report.feature_importance.head(10).iterrows():
            lines.append(
                f"  {row['feature']:<28}"
                f"{row['mean_importance']:>10.4f}"
                f"{row['std_importance']:>10.4f}"
                f"{row['stability']:>10.2f}"
            )

    if report.universe_variants:
        lines.extend([
            "",
            "  Universe robustness",
            "  " + "-" * 60,
            f"  {'Variant':<22}{'CAGR':>12}{'Sharpe':>10}{'MaxDD':>12}{'Trades':>8}",
        ])
        for uv in report.universe_variants:
            uv_m = uv.metrics
            lines.append(
                f"  {uv.label:<22}"
                f"{pct(uv_m.cagr):>12}"
                f"{uv_m.sharpe:>10.2f}"
                f"{pct(uv_m.max_drawdown):>12}"
                f"{uv_m.n_trades:>8d}"
            )

    lines.append("=" * 72)
    return "\n".join(lines)
