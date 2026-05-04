"""
Reporting — equity curve plots.

`plot_equity_curves` is the main entry point.  It accepts the dict of
backtest DataFrames produced by `run_backtest`, draws one cumulative
equity line per strategy, overlays the shared buy-and-hold benchmark,
and saves the figure to disk.

Expected DataFrame columns (from `run_backtest`):
    cumulative   : (1 + strategy_return).cumprod()
    buy_and_hold : (1 + asset_return).cumprod()

Example
-------
>>> results = {
...     "AlwaysLong":   bt_always_long,
...     "CalibratedLR": bt_calibrated,
...     "RandomForest": bt_rf,
... }
>>> plot_equity_curves(results, title="NVDA 2020-2023", output_path="reports/nvda.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Colour palette — distinct, print-safe colours for up to 8 strategies.
_STRATEGY_COLOURS = [
    "#2196F3",  # blue
    "#E91E63",  # pink
    "#FF9800",  # orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#F44336",  # red
    "#795548",  # brown
]
_BNH_COLOUR = "#9E9E9E"   # grey — passive benchmark always in the background


def plot_equity_curves(
    backtest_results: dict[str, pd.DataFrame],
    title: str,
    output_path: str | Path,
) -> Path:
    """
    Plot cumulative equity curves for one or more strategies.

    Each entry in ``backtest_results`` contributes one strategy line drawn
    from the ``cumulative`` column.  The buy-and-hold benchmark is taken
    from the ``buy_and_hold`` column of the *first* DataFrame (all
    strategies share the same underlying asset, so they are identical).

    Parameters
    ----------
    backtest_results:
        ``{strategy_name: backtest_df}`` mapping.  Each DataFrame must be
        the output of ``run_backtest`` and contain at least the columns
        ``cumulative`` and ``buy_and_hold``.
    title:
        Figure title shown at the top of the plot.
    output_path:
        File path for the saved PNG.  Parent directories are created
        automatically.

    Returns
    -------
    Path
        Resolved absolute path of the saved file.

    Raises
    ------
    ValueError
        If ``backtest_results`` is empty, or a required column is missing.
    """
    if not backtest_results:
        raise ValueError("backtest_results must contain at least one entry.")

    required = {"cumulative", "buy_and_hold"}
    for name, df in backtest_results.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"backtest_results['{name}'] is missing columns: {missing}"
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Buy-and-hold benchmark — plotted first so it sits behind strategy lines.
    first_df = next(iter(backtest_results.values()))
    ax.plot(
        first_df.index,
        first_df["buy_and_hold"],
        color=_BNH_COLOUR,
        linewidth=1.5,
        linestyle="--",
        label="Buy & Hold",
        zorder=2,
    )

    # One line per strategy.
    for (name, df), colour in zip(
        backtest_results.items(), _STRATEGY_COLOURS
    ):
        final = df["cumulative"].iloc[-1]
        label = f"{name}  ({final - 1:+.1%})"
        ax.plot(
            df.index,
            df["cumulative"],
            color=colour,
            linewidth=1.8,
            label=label,
            zorder=3,
        )

    # Reference line at 1.0 (no gain / no loss).
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Cumulative Return (1 = starting value)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path.resolve()
