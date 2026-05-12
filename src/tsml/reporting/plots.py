"""
Reporting — equity curve plots.

Functions
---------
plot_equity_curves
    Plot cumulative equity curves from ``run_backtest`` results (single-asset,
    multi-strategy comparison).

plot_portfolio_vs_benchmark
    Plot a portfolio equity curve against a benchmark price series, both
    normalised to 1.0 at the shared start date.  Intended for use with the
    multi-asset ``simulate()`` output.

Example
-------
>>> results = {
...     "AlwaysLong":   bt_always_long,
...     "CalibratedLR": bt_calibrated,
...     "RandomForest": bt_rf,
... }
>>> plot_equity_curves(results, title="NVDA 2020-2023", output_path="reports/nvda.png")
>>>
>>> plot_portfolio_vs_benchmark(
...     result.equity_curve, spy_df["close"],
...     title="Portfolio vs SPY", output_path="reports/portfolio.png",
... )
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


def plot_portfolio_vs_benchmark(
    equity_curve: pd.Series,
    benchmark_close: pd.Series,
    title: str,
    output_path: str | Path,
    *,
    strategy_label: str = "Portfolio",
    benchmark_label: str = "SPY",
) -> Path:
    """
    Plot a portfolio equity curve vs a benchmark, normalised to 1.0 at start.

    Both series are aligned by inner join on their shared dates and then
    normalised to start at 1.0, so they can be compared on the same scale
    regardless of their absolute values.

    Parameters
    ----------
    equity_curve:
        Daily portfolio value series (e.g. ``SimulationResult.equity_curve``).
        Must have a ``DatetimeIndex``.
    benchmark_close:
        Benchmark close price series (e.g. SPY from ``YFinanceLoader``).
        Must have a ``DatetimeIndex``.
    title:
        Figure title.
    output_path:
        Destination PNG file path.  Parent directories are created automatically.
    strategy_label:
        Legend label for the portfolio line.
    benchmark_label:
        Legend label for the benchmark line.

    Returns
    -------
    Path
        Resolved absolute path of the saved PNG.

    Raises
    ------
    ValueError
        If ``equity_curve`` and ``benchmark_close`` share no common dates.
    """
    common = equity_curve.index.intersection(benchmark_close.index)
    if common.empty:
        raise ValueError(
            "equity_curve and benchmark_close share no common dates. "
            "Check that both cover the same period."
        )

    strat = equity_curve.loc[common].sort_index()
    bench = benchmark_close.loc[common].sort_index()

    # Normalise both to 1.0 at the shared start date.
    strat_norm = strat / strat.iloc[0]
    bench_norm = bench / bench.iloc[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        bench_norm.index,
        bench_norm.values,
        color=_BNH_COLOUR,
        linewidth=1.5,
        linestyle="--",
        label=f"{benchmark_label}  ({bench_norm.iloc[-1] - 1:+.1%})",
        zorder=2,
    )
    ax.plot(
        strat_norm.index,
        strat_norm.values,
        color=_STRATEGY_COLOURS[0],
        linewidth=1.8,
        label=f"{strategy_label}  ({strat_norm.iloc[-1] - 1:+.1%})",
        zorder=3,
    )

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Normalised Value (1 = start)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path.resolve()
