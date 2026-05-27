"""Exposure timeline plots for regime overlay analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_exposure_timeline(
    actual_exposure: pd.Series,
    regime_target: pd.Series,
    title: str,
    output_path: str | Path,
) -> Path:
    """
    Plot actual portfolio exposure vs regime target exposure over time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common = actual_exposure.index.intersection(regime_target.index)
    actual = actual_exposure.loc[common].sort_index()
    target = regime_target.loc[common].sort_index()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(
        target.index,
        target.values,
        alpha=0.2,
        color="#FF9800",
        label="Regime target",
    )
    ax.plot(
        target.index,
        target.values,
        color="#FF9800",
        linewidth=1.2,
        linestyle="--",
    )
    ax.plot(
        actual.index,
        actual.values,
        color="#2196F3",
        linewidth=1.5,
        label="Actual exposure",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Portfolio exposure")
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
