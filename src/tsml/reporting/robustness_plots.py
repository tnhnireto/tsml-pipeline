"""Rolling performance plots for robustness analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tsml.portfolio.robustness import RollingMetrics


def plot_rolling_performance(
    rolling: RollingMetrics,
    title: str,
    output_path: str | Path,
) -> Path:
    """
    Save a three-panel plot of rolling Sharpe, CAGR, and drawdown.

    Parameters
    ----------
    rolling:
        Output of :func:`~tsml.portfolio.robustness.compute_rolling_metrics`.
    title:
        Figure suptitle.
    output_path:
        Destination PNG path.  Parent directories are created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    if not rolling.rolling_sharpe.empty:
        axes[0].plot(
            rolling.rolling_sharpe.index,
            rolling.rolling_sharpe.values,
            color="#2196F3",
            linewidth=1.5,
        )
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("Rolling Sharpe (252d)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    if not rolling.rolling_cagr.empty:
        axes[1].plot(
            rolling.rolling_cagr.index,
            rolling.rolling_cagr.values,
            color="#4CAF50",
            linewidth=1.5,
        )
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("Rolling CAGR (252d)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    if not rolling.rolling_drawdown.empty:
        axes[2].fill_between(
            rolling.rolling_drawdown.index,
            rolling.rolling_drawdown.values,
            0,
            color="#F44336",
            alpha=0.4,
        )
        axes[2].plot(
            rolling.rolling_drawdown.index,
            rolling.rolling_drawdown.values,
            color="#C62828",
            linewidth=1.2,
        )
    axes[2].set_ylabel("Drawdown")
    axes[2].set_xlabel("Date")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
