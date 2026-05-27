"""Holdout evaluation equity plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tsml.portfolio.holdout_eval import HoldoutPeriods
from tsml.reporting.plots import _BNH_COLOUR, _STRATEGY_COLOURS


def plot_holdout_equity(
    equity_curve: pd.Series,
    benchmark_closes: dict[str, pd.Series],
    periods: HoldoutPeriods,
    output_path: str | Path,
    *,
    title: str = "Weekly Strategy: Development vs Holdout",
) -> Path:
    """
    Plot normalized equity with a vertical line marking the holdout start.
    """
    if not benchmark_closes:
        raise ValueError("benchmark_closes must contain at least one series.")

    common = equity_curve.index
    for close in benchmark_closes.values():
        common = common.intersection(close.index)
    if common.empty:
        raise ValueError("No overlapping dates between strategy and benchmarks.")

    strat_norm = equity_curve.loc[common].sort_index()
    strat_norm = strat_norm / strat_norm.iloc[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    bench_colours = [_BNH_COLOUR, "#607D8B", "#8D6E63", "#78909C"]
    for i, (label, close) in enumerate(benchmark_closes.items()):
        bench = close.loc[common].sort_index()
        bench_norm = bench / bench.iloc[0]
        colour = bench_colours[i % len(bench_colours)]
        ax.plot(
            bench_norm.index,
            bench_norm.values,
            color=colour,
            linewidth=1.5,
            linestyle="--",
            label=f"{label}  ({bench_norm.iloc[-1] - 1:+.1%})",
            zorder=2,
        )

    ax.plot(
        strat_norm.index,
        strat_norm.values,
        color=_STRATEGY_COLOURS[0],
        linewidth=1.8,
        label=f"Strategy  ({strat_norm.iloc[-1] - 1:+.1%})",
        zorder=3,
    )

    holdout_ts = pd.Timestamp(periods.holdout_start)
    if common.tz is not None and holdout_ts.tz is None:
        holdout_ts = holdout_ts.tz_localize(common.tz)
    ax.axvline(
        holdout_ts,
        color="#FF5722",
        linewidth=1.5,
        linestyle=":",
        label="Holdout start",
        zorder=4,
    )

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Normalised Value (1 = start)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
