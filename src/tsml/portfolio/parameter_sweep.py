"""
Parameter robustness / sensitivity analysis for the weekly ML portfolio strategy.

Runs many backtests over a Cartesian grid of portfolio-rule parameters while
reusing a single walk-forward score pre-computation (no re-fitting per combo).
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.metrics.returns import (
    annualized_volatility,
    cagr as _cagr,
    max_drawdown as _max_drawdown,
    sharpe_ratio as _sharpe_ratio,
    total_return as _total_return,
)
from tsml.portfolio.regime_overlay import RegimeOverlayConfig
from tsml.portfolio.simulator import (
    SimulationInputs,
    SimulationResult,
    prepare_simulation_inputs,
    simulate,
)
from tsml.validation.splitters import WalkForwardSplit

SWEEP_PARAM_NAMES: tuple[str, ...] = (
    "buy_threshold",
    "sell_threshold",
    "min_score",
    "bear_exposure",
    "min_hold_weeks",
    "vol_threshold",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "cagr",
    "sharpe",
    "max_drawdown",
    "turnover",
    "exposure",
    "n_trades",
    "calmar",
    "volatility",
)

_VOL_THRESHOLD_SENTINEL = "__disabled__"


@dataclass(frozen=True)
class SweepParams:
    """One parameter combination in a grid search."""

    buy_threshold: float = 0.58
    sell_threshold: float = 0.52
    min_score: float = 0.55
    bear_exposure: float = 0.25
    min_hold_weeks: int = 2
    vol_threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "min_score": self.min_score,
            "bear_exposure": self.bear_exposure,
            "min_hold_weeks": self.min_hold_weeks,
            "vol_threshold": self.vol_threshold,
        }


@dataclass
class SweepDiagnostics:
    """Stability and sensitivity summaries for a parameter sweep."""

    robustness_score: float
    top_n: pd.DataFrame
    percentiles: pd.DataFrame
    sensitivity: pd.DataFrame
    parameter_importance: pd.Series


@dataclass
class ParameterSweepResult:
    """Full output of a parameter grid search."""

    results: pd.DataFrame
    diagnostics: SweepDiagnostics
    grid_size: int
    sampled: bool
    precomputed: SimulationInputs | None = None


def default_parameter_grid(*, fast: bool = False) -> dict[str, list[Any]]:
    """
    Return default sweep ranges.

    ``fast=True`` uses a smaller grid suitable for smoke tests and CI.
    """
    if fast:
        return {
            "buy_threshold": [0.56, 0.58],
            "sell_threshold": [0.50, 0.52],
            "min_score": [0.55],
            "bear_exposure": [0.0, 0.25],
            "min_hold_weeks": [1, 2],
            "vol_threshold": [None],
        }
    return {
        "buy_threshold": [0.56, 0.58, 0.60],
        "sell_threshold": [0.50, 0.52, 0.54],
        "min_score": [0.53, 0.55, 0.57],
        "bear_exposure": [0.0, 0.25, 0.5],
        "min_hold_weeks": [1, 2, 4],
        "vol_threshold": [None, 0.015, 0.02],
    }


def count_parameter_combinations(grid: Mapping[str, Sequence[Any]]) -> int:
    """Return the Cartesian product size for ``grid``."""
    if not grid:
        return 0
    total = 1
    for values in grid.values():
        if len(values) == 0:
            return 0
        total *= len(values)
    return total


def expand_parameter_grid(grid: Mapping[str, Sequence[Any]]) -> list[SweepParams]:
    """Expand a parameter grid into a list of :class:`SweepParams`."""
    keys = [k for k in SWEEP_PARAM_NAMES if k in grid]
    if not keys:
        return [SweepParams()]

    value_lists = [list(grid[k]) for k in keys]
    combos: list[SweepParams] = []
    for values in itertools.product(*value_lists):
        raw = dict(zip(keys, values, strict=True))
        combos.append(
            SweepParams(
                buy_threshold=float(raw.get("buy_threshold", 0.58)),
                sell_threshold=float(raw.get("sell_threshold", 0.52)),
                min_score=float(raw.get("min_score", 0.55)),
                bear_exposure=float(raw.get("bear_exposure", 0.25)),
                min_hold_weeks=int(raw.get("min_hold_weeks", 2)),
                vol_threshold=raw.get("vol_threshold"),
            )
        )
    return combos


def sample_parameter_combinations(
    grid: Mapping[str, Sequence[Any]],
    max_combinations: int,
    *,
    seed: int = 42,
) -> list[SweepParams]:
    """
    Return up to ``max_combinations`` combinations from the full Cartesian grid.

    When the grid is larger than the cap, combinations are sampled uniformly
    without replacement using ``seed`` for reproducibility.
    """
    all_combos = expand_parameter_grid(grid)
    if len(all_combos) <= max_combinations:
        return all_combos
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_combos), size=max_combinations, replace=False)
    return [all_combos[int(i)] for i in sorted(indices)]


def _serialize_vol_threshold(value: float | None) -> str | float:
    if value is None:
        return _VOL_THRESHOLD_SENTINEL
    return value


def _deserialize_vol_threshold(value: str | float | None) -> float | None:
    if value is None or value == _VOL_THRESHOLD_SENTINEL:
        return None
    return float(value)


def _regime_overlay_from_params(params: SweepParams) -> RegimeOverlayConfig:
    high_vol = 0.0 if params.vol_threshold is not None else None
    return RegimeOverlayConfig(
        enabled=True,
        bull_exposure=1.0,
        bear_exposure=params.bear_exposure,
        high_vol_exposure=high_vol,
        vol_threshold=params.vol_threshold,
    )


def compute_sweep_metrics(
    result: SimulationResult,
    *,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Derive sweep metrics from a simulation result."""
    if len(result.equity_curve) < 2:
        return {col: 0.0 for col in METRIC_COLUMNS}

    rets = result.equity_curve.pct_change().dropna()
    cagr = _cagr(rets)
    mdd = _max_drawdown(rets)
    vol = annualized_volatility(rets)
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")

    avg_exposure = (
        float(result.exposure.mean())
        if not result.exposure.empty
        else 0.0
    )

    return {
        "cagr": cagr,
        "sharpe": _sharpe_ratio(rets, risk_free_rate),
        "max_drawdown": mdd,
        "turnover": result.turnover_total,
        "exposure": avg_exposure,
        "n_trades": float(len(result.trades_log)),
        "calmar": calmar,
        "volatility": vol,
        "total_return": _total_return(rets),
    }


def run_single_sweep_backtest(
    params: SweepParams,
    *,
    symbols: Sequence[str],
    model: Any,
    splitter: WalkForwardSplit,
    start_date: str,
    end_date: str,
    loader: DataLoader | None = None,
    precomputed: SimulationInputs | None = None,
    target: str = "threshold",
    top_n: int = 5,
    min_score_downtrend: float = 0.62,
    cash_buffer_pct: float = 0.05,
    costs_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    feature_set: str = "extended",
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Run one backtest for ``params`` and return parameters + metrics."""
    if precomputed is not None:
        loader = None
    regime = _regime_overlay_from_params(params)
    result = simulate(
        symbols=list(symbols),
        model=model,
        splitter=splitter,
        start_date=start_date,
        end_date=end_date,
        target=target,
        top_n=top_n,
        min_score=params.min_score,
        min_score_downtrend=min_score_downtrend,
        cash_buffer_pct=cash_buffer_pct,
        costs_bps=costs_bps,
        initial_capital=initial_capital,
        loader=loader,
        turnover_control=True,
        buy_threshold=params.buy_threshold,
        sell_threshold=params.sell_threshold,
        min_hold_weeks=params.min_hold_weeks,
        feature_set=feature_set,
        regime_overlay=regime,
        precomputed=precomputed,
    )
    row = params.as_dict()
    row["vol_threshold"] = _serialize_vol_threshold(params.vol_threshold)
    row.update(compute_sweep_metrics(result, risk_free_rate=risk_free_rate))
    return row


# ---------------------------------------------------------------------------
# Parallel worker support (module-level for pickling)
# ---------------------------------------------------------------------------

_WORKER_PRECOMPUTED: SimulationInputs | None = None
_WORKER_CONTEXT: dict[str, Any] = {}


def _init_sweep_worker(precomputed: SimulationInputs, context: dict[str, Any]) -> None:
    global _WORKER_PRECOMPUTED, _WORKER_CONTEXT
    _WORKER_PRECOMPUTED = precomputed
    _WORKER_CONTEXT = context


def _sweep_worker(params_dict: dict[str, Any]) -> dict[str, Any]:
    params = SweepParams(
        buy_threshold=float(params_dict["buy_threshold"]),
        sell_threshold=float(params_dict["sell_threshold"]),
        min_score=float(params_dict["min_score"]),
        bear_exposure=float(params_dict["bear_exposure"]),
        min_hold_weeks=int(params_dict["min_hold_weeks"]),
        vol_threshold=_deserialize_vol_threshold(params_dict.get("vol_threshold")),
    )
    return run_single_sweep_backtest(
        params,
        precomputed=_WORKER_PRECOMPUTED,
        **_WORKER_CONTEXT,
    )


def run_parameter_sweep(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    model: Any,
    splitter: WalkForwardSplit,
    grid: Mapping[str, Sequence[Any]] | None = None,
    fast: bool = False,
    max_combinations: int | None = None,
    seed: int = 42,
    n_jobs: int = 1,
    loader: DataLoader | None = None,
    top_n: int = 5,
    min_score_downtrend: float = 0.62,
    cash_buffer_pct: float = 0.05,
    costs_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    target: str = "threshold",
    feature_set: str = "extended",
    risk_free_rate: float = 0.0,
    top_n_results: int = 10,
) -> ParameterSweepResult:
    """
    Run a parameter grid search with optional parallel execution.

    Walk-forward scores are precomputed once and shared across all runs.
    """
    if grid is None:
        grid = default_parameter_grid(fast=fast)

    full_size = count_parameter_combinations(grid)
    sampled = False
    if max_combinations is not None and full_size > max_combinations:
        combos = sample_parameter_combinations(grid, max_combinations, seed=seed)
        sampled = True
    else:
        combos = expand_parameter_grid(grid)

    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    precomputed = prepare_simulation_inputs(
        symbols,
        model,
        splitter,
        start_date=start,
        end_date=end,
        target=target,
        loader=loader,
        feature_set=feature_set,
        load_spy=True,
    )

    context = dict(
        symbols=list(symbols),
        model=model,
        splitter=splitter,
        start_date=start,
        end_date=end,
        target=target,
        top_n=top_n,
        min_score_downtrend=min_score_downtrend,
        cash_buffer_pct=cash_buffer_pct,
        costs_bps=costs_bps,
        initial_capital=initial_capital,
        feature_set=feature_set,
        risk_free_rate=risk_free_rate,
    )

    rows: list[dict[str, Any]] = []
    param_dicts = [
        {**c.as_dict(), "vol_threshold": _serialize_vol_threshold(c.vol_threshold)}
        for c in combos
    ]

    if n_jobs <= 1 or len(param_dicts) <= 1:
        for params in combos:
            rows.append(
                run_single_sweep_backtest(
                    params,
                    precomputed=precomputed,
                    **context,
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_sweep_worker,
            initargs=(precomputed, context),
        ) as pool:
            futures = [pool.submit(_sweep_worker, p) for p in param_dicts]
            for fut in as_completed(futures):
                rows.append(fut.result())

    results = pd.DataFrame(rows)
    if "vol_threshold" in results.columns:
        results["vol_threshold"] = results["vol_threshold"].map(
            _deserialize_vol_threshold
        )

    diagnostics = compute_sweep_diagnostics(results, top_n=top_n_results)
    return ParameterSweepResult(
        results=results,
        diagnostics=diagnostics,
        grid_size=full_size,
        sampled=sampled,
        precomputed=precomputed,
    )


def compute_robustness_score(sharpe_values: pd.Series) -> float:
    """
    Robustness score = mean(Sharpe) / std(Sharpe).

    Higher values indicate stable performance across parameter settings.
    Returns NaN when std is zero or fewer than two observations.
    """
    if len(sharpe_values) < 2:
        return float("nan")
    std = float(sharpe_values.std(ddof=1))
    if std < 1e-14:
        return float("nan")
    return float(sharpe_values.mean() / std)


def compute_sweep_diagnostics(
    results: pd.DataFrame,
    *,
    top_n: int = 10,
    metric: str = "sharpe",
) -> SweepDiagnostics:
    """Compute stability diagnostics from sweep results."""
    if results.empty:
        empty = pd.DataFrame()
        return SweepDiagnostics(
            robustness_score=float("nan"),
            top_n=empty,
            percentiles=empty,
            sensitivity=empty,
            parameter_importance=pd.Series(dtype=float),
        )

    sharpe = results["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    robustness = compute_robustness_score(sharpe)

    top = (
        results.sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    pct_cols = [c for c in METRIC_COLUMNS if c in results.columns]
    percentiles = results[pct_cols].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).T

    sensitivity = compute_parameter_sensitivity(results, metric=metric)
    importance = sensitivity.set_index("parameter")["metric_spread"].sort_values(
        ascending=False
    )

    return SweepDiagnostics(
        robustness_score=robustness,
        top_n=top,
        percentiles=percentiles,
        sensitivity=sensitivity,
        parameter_importance=importance,
    )


def compute_parameter_sensitivity(
    results: pd.DataFrame,
    *,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """
    Measure how much ``metric`` changes when each parameter varies.

    ``metric_spread`` is max(group mean) − min(group mean) for that parameter.
    """
    rows: list[dict[str, Any]] = []
    for param in SWEEP_PARAM_NAMES:
        if param not in results.columns:
            continue
        grouped = results.groupby(param, dropna=False)[metric].agg(
            ["mean", "std", "min", "max", "count"]
        )
        if grouped.empty:
            continue
        spread = float(grouped["mean"].max() - grouped["mean"].min())
        rows.append(
            {
                "parameter": param,
                "metric_spread": spread,
                "mean_of_means": float(grouped["mean"].mean()),
                "worst_level_mean": float(grouped["mean"].min()),
                "best_level_mean": float(grouped["mean"].max()),
                "levels": int(len(grouped)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("metric_spread", ascending=False).reset_index(drop=True)


def format_parameter_sweep_report(result: ParameterSweepResult) -> str:
    """Format a human-readable ASCII summary."""
    d = result.diagnostics
    lines = [
        "=" * 68,
        "  Parameter Sweep - Robustness Summary",
        "=" * 68,
        f"  Grid size (full):     {result.grid_size}",
        f"  Runs completed:       {len(result.results)}"
        + (" (sampled)" if result.sampled else ""),
        f"  Robustness score:     {d.robustness_score:.3f}"
        if not math.isnan(d.robustness_score)
        else "  Robustness score:     n/a",
        "",
        "  Metric percentiles",
        "  " + "-" * 40,
    ]

    if not d.percentiles.empty:
        for metric_name in ["sharpe", "cagr", "max_drawdown", "calmar"]:
            if metric_name not in d.percentiles.index:
                continue
            row = d.percentiles.loc[metric_name]
            lines.append(
                f"  {metric_name:<14} p10={row.get('10%', float('nan')):+.3f}  "
                f"p50={row.get('50%', float('nan')):+.3f}  "
                f"p90={row.get('90%', float('nan')):+.3f}"
            )

    if not d.parameter_importance.empty:
        lines.extend(["", "  Parameter importance (Sharpe spread)", "  " + "-" * 40])
        for param, spread in d.parameter_importance.items():
            lines.append(f"  {param:<22}{spread:>10.4f}")

    if not d.sensitivity.empty:
        lines.extend(["", "  Sensitivity detail", "  " + "-" * 40])
        for _, row in d.sensitivity.iterrows():
            lines.append(
                f"  {row['parameter']:<18}"
                f" spread={row['metric_spread']:.4f}  "
                f"best={row['best_level_mean']:.3f}  "
                f"worst={row['worst_level_mean']:.3f}"
            )

    if not d.top_n.empty:
        lines.extend(["", f"  Top {len(d.top_n)} parameter sets (by Sharpe)", "  " + "-" * 40])
        for i, row in d.top_n.iterrows():
            vol = row.get("vol_threshold")
            vol_s = "off" if vol is None or (isinstance(vol, float) and math.isnan(vol)) else f"{vol:.3f}"
            lines.append(
                f"  #{i + 1} Sharpe={row['sharpe']:.2f}  CAGR={row['cagr']:+.1%}  "
                f"MDD={row['max_drawdown']:+.1%}  "
                f"buy={row['buy_threshold']:.2f} sell={row['sell_threshold']:.2f}  "
                f"min_score={row['min_score']:.2f} bear={row['bear_exposure']:.0%}  "
                f"hold={int(row['min_hold_weeks'])}w vol={vol_s}"
            )

    lines.append("=" * 68)
    return "\n".join(lines)
