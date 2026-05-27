"""Visualizations for parameter sweep robustness analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsml.portfolio.parameter_sweep import SWEEP_PARAM_NAMES


def _ensure_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_sharpe_distribution(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Sharpe Ratio Distribution Across Parameter Grid",
) -> Path:
    """Histogram of Sharpe ratios from all parameter combinations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sharpe = results["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sharpe, bins=min(30, max(5, len(sharpe) // 3)), color="#2196F3", alpha=0.85)
    ax.axvline(sharpe.mean(), color="#FF5722", linestyle="--", label=f"Mean={sharpe.mean():.2f}")
    ax.axvline(sharpe.median(), color="#4CAF50", linestyle=":", label=f"Median={sharpe.median():.2f}")
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def plot_cagr_vs_drawdown(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "CAGR vs Max Drawdown",
) -> Path:
    """Scatter of CAGR against max drawdown, coloured by Sharpe."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = results.dropna(subset=["cagr", "max_drawdown", "sharpe"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["max_drawdown"] * 100,
        df["cagr"] * 100,
        c=df["sharpe"],
        cmap="RdYlGn",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
        s=40,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe")
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def plot_parameter_importance(
    importance: pd.Series,
    output_path: str | Path,
    *,
    title: str = "Parameter Importance (Sharpe Sensitivity)",
) -> Path:
    """Horizontal bar chart of parameter sensitivity ranking."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if importance.empty:
        return output_path

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(importance))))
    y_pos = np.arange(len(importance))
    ax.barh(y_pos, importance.values, color="#673AB7", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance.index)
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe spread (max − min group mean)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def plot_parameter_heatmap(
    results: pd.DataFrame,
    param_x: str,
    param_y: str,
    output_path: str | Path,
    *,
    metric: str = "sharpe",
    title: str | None = None,
) -> Path:
    """
    Heatmap of mean ``metric`` for pairs of parameter values.

    Other parameters are averaged (mean of means per cell).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = results.pivot_table(
        index=param_y,
        columns=param_x,
        values=metric,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.8)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    plt.colorbar(im, ax=ax, label=f"Mean {metric}")
    ax.set_title(
        title or f"Mean {metric}: {param_y} vs {param_x}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def generate_parameter_sweep_plots(
    results: pd.DataFrame,
    importance: pd.Series,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate all standard parameter sweep plots."""
    out = _ensure_dir(output_dir)
    paths: dict[str, Path] = {}

    paths["sharpe_distribution"] = plot_sharpe_distribution(
        results, out / "sharpe_distribution.png"
    )
    paths["cagr_vs_drawdown"] = plot_cagr_vs_drawdown(
        results, out / "cagr_vs_drawdown.png"
    )
    paths["parameter_importance"] = plot_parameter_importance(
        importance, out / "parameter_importance.png"
    )

    heatmap_pairs = [
        ("buy_threshold", "sell_threshold"),
        ("bear_exposure", "min_hold_weeks"),
        ("buy_threshold", "bear_exposure"),
    ]
    for px, py in heatmap_pairs:
        if px in results.columns and py in results.columns:
            name = f"heatmap_{px}_vs_{py}"
            paths[name] = plot_parameter_heatmap(
                results, px, py, out / f"{name}.png"
            )

    return paths
