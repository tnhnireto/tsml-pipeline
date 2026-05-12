"""
analyze_signals.py — backward analysis of saved weekly signals.

For every (date, symbol) row saved by run_weekly_signal.py, this script
loads close prices and appends three forward-return columns:

    return_5d_forward   — 5 trading days after signal date
    return_20d_forward  — 20 trading days after signal date
    return_60d_forward  — 60 trading days after signal date

Forward returns are from the signal-date close to the Nth-trading-day
close.  The entry convention is: signal generated at close of day t,
trade executed at close of t+1.  For simplicity this script computes
close-to-close from day t (slight underestimate of slippage — acceptable
for exploratory analysis).

Summary statistics are then printed, grouped by:

    1. action
    2. action × above_sma_200
    3. action × score bucket  (<0.55 | 0.55-0.60 | 0.60-0.65 | >0.65)

Each group shows:  count, rows with valid forward data, average return,
and win rate (fraction with return > 0) for each horizon.

Usage
-----
    python analyze_signals.py

The enriched DataFrame is saved to signals/analysis.csv so it can be
loaded into a notebook or spreadsheet for further exploration.

Note on NaN forward returns
---------------------------
If a signal file is very recent, forward data will not yet be available:

    5d  returns require 5 trading days  (~1 calendar week)
    20d returns require 20 trading days (~1 calendar month)
    60d returns require 60 trading days (~3 calendar months)

All NaN results are expected for a freshly generated signal file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tsml.data_loader import YFinanceLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGNALS_DIR  = Path("signals")
LOAD_START   = "2018-01-01"                              # covers any past signal date
LOAD_END     = pd.Timestamp.today().strftime("%Y-%m-%d") # fetch up to today for fwd data
HORIZONS     = [5, 20, 60]                               # trading-day forward horizons

SCORE_BINS   = [0.00, 0.55, 0.60, 0.65, 1.01]
SCORE_LABELS = ["<0.55", "0.55-0.60", "0.60-0.65", ">0.65"]

_W = 88   # table width (wide enough for the longest label in table 2/3)


# ---------------------------------------------------------------------------
# Step 1: Load all signal CSVs
# ---------------------------------------------------------------------------

def load_all_signals() -> pd.DataFrame:
    """Read every *.csv in SIGNALS_DIR and return a combined DataFrame."""
    files = sorted(SIGNALS_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {SIGNALS_DIR}/.", file=sys.stderr)
        sys.exit(1)

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Parse date → UTC Timestamp (matches the UTC index in YFinanceLoader)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")

    # Parse score
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Parse above_sma_200: CSV stores "True"/"False" strings or empty
    df["above_sma_200"] = (
        df["above_sma_200"]
        .map(lambda v: True if v is True or str(v) == "True"
                       else (False if v is False or str(v) == "False"
                             else None))
    )

    # Normalise action: empty string → NaN → "(no action)"
    df["action"] = df["action"].fillna("(no action)").replace("", "(no action)")

    n_files   = len(files)
    n_rows    = len(df)
    n_symbols = df["symbol"].nunique()
    n_dates   = df["date"].nunique()
    print(
        f"Loaded {n_rows} rows  |  {n_files} signal file(s)  |  "
        f"{n_symbols} unique symbols  |  {n_dates} signal date(s)"
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: Append forward returns
# ---------------------------------------------------------------------------

def _entry_pos(prices: pd.Series, signal_date: pd.Timestamp) -> int | None:
    """
    Return the integer index of the first available trading day on or after
    signal_date.  Returns None when the date is beyond the loaded range.
    """
    pos = int(prices.index.searchsorted(signal_date))
    return pos if pos < len(prices) else None


def add_forward_returns(
    df: pd.DataFrame,
    loader: YFinanceLoader,
) -> pd.DataFrame:
    """
    For each row, look up the close price on the signal date and N trading
    days later, and compute the percentage return.

    Price data is loaded once per symbol and reused for all signal dates.
    """
    # Load close prices for every unique symbol (served from cache).
    price_map: dict[str, pd.Series] = {}
    print(f"\nLoading price data for {df['symbol'].nunique()} symbols ...")
    for sym in sorted(df["symbol"].unique()):
        try:
            raw              = loader.load(sym, LOAD_START, LOAD_END)
            price_map[sym]   = raw["close"].sort_index()
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not load {sym}: {exc}", file=sys.stderr)

    print(f"  Loaded {len(price_map)} symbol(s).\n")

    # Build forward-return columns
    fwd: dict[str, list[float]] = {f"return_{h}d_forward": [] for h in HORIZONS}

    for _, row in df.iterrows():
        sym  = str(row["symbol"])
        date = row["date"]

        if sym not in price_map:
            for h in HORIZONS:
                fwd[f"return_{h}d_forward"].append(float("nan"))
            continue

        prices   = price_map[sym]
        pos      = _entry_pos(prices, date)

        if pos is None:
            for h in HORIZONS:
                fwd[f"return_{h}d_forward"].append(float("nan"))
            continue

        entry_px = float(prices.iloc[pos])

        for h in HORIZONS:
            target = pos + h
            if target >= len(prices):
                fwd[f"return_{h}d_forward"].append(float("nan"))
            else:
                fwd[f"return_{h}d_forward"].append(
                    float(prices.iloc[target]) / entry_px - 1
                )

    result = df.copy()
    for col, vals in fwd.items():
        result[col] = vals

    n_valid = result["return_5d_forward"].notna().sum()
    print(
        f"Forward returns appended.  "
        f"Rows with valid 5d data: {n_valid} / {len(result)}"
    )
    if n_valid == 0:
        print(
            "\n  All forward returns are NaN -- this is expected when all "
            "signal files are\n"
            "  very recent.  Re-run after 5+ trading days to see 5d results,\n"
            "  after ~20 trading days for 20d, after ~60 for 60d."
        )
    return result


# ---------------------------------------------------------------------------
# Step 3: Summary printing helpers
# ---------------------------------------------------------------------------

def _summarize_group(sub: pd.DataFrame) -> dict:
    stats: dict[str, float | int] = {"n": len(sub)}
    for h in HORIZONS:
        col    = f"return_{h}d_forward"
        series = sub[col].dropna()
        n      = len(series)
        stats[f"n_valid_{h}d"]  = int(n)
        stats[f"avg_{h}d"]      = float(series.mean()) if n else float("nan")
        stats[f"win_{h}d"]      = float((series > 0).mean()) if n else float("nan")
    return stats


def _pct(v: float, signed: bool = True, width: int = 7) -> str:
    if pd.isna(v):
        return "n/a".rjust(width)
    fmt = f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"
    return fmt.rjust(width)


def _win(v: float, width: int = 5) -> str:
    if pd.isna(v):
        return "n/a".rjust(width)
    return f"{v * 100:.0f}%".rjust(width)


def _print_summary_table(
    title: str,
    groups: dict[str, dict],
    label_width: int = 22,
) -> None:
    lw = label_width
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)
    hdr = (
        f"  {'Group':<{lw}}  {'N':>5}  "
        f"{'Avg5d':>7}  {'Win5d':>5}  "
        f"{'Avg20d':>7}  {'Win20d':>5}  "
        f"{'Avg60d':>7}  {'Win60d':>5}"
    )
    print(hdr)
    print("-" * _W)

    for key, s in groups.items():
        print(
            f"  {str(key):<{lw}}  {s['n']:>5}  "
            f"{_pct(s['avg_5d'])}  {_win(s['win_5d'])}  "
            f"{_pct(s['avg_20d'])}  {_win(s['win_20d'])}  "
            f"{_pct(s['avg_60d'])}  {_win(s['win_60d'])}"
        )
    print("=" * _W)


def _sma_label(v: object) -> str:
    if v is True or v is np.bool_(True):
        return "above_SMA"
    if v is False or v is np.bool_(False):
        return "below_SMA"
    return "SMA_unknown"


# ---------------------------------------------------------------------------
# Step 4: Three summary tables
# ---------------------------------------------------------------------------

def print_summaries(df: pd.DataFrame) -> None:
    df = df.copy()
    df["score_bucket"] = pd.cut(
        df["score"], bins=SCORE_BINS, labels=SCORE_LABELS, right=False
    )

    # ── Table 1: by action ──────────────────────────────────────────────
    by_action = {
        str(action): _summarize_group(grp)
        for action, grp in df.groupby("action", dropna=False)
    }
    _print_summary_table("1. Summary by action", by_action)

    # ── Table 2: by action × above_sma_200 ─────────────────────────────
    by_action_sma: dict[str, dict] = {}
    for (action, sma), grp in df.groupby(
        ["action", "above_sma_200"], dropna=False
    ):
        label = f"{action} / {_sma_label(sma)}"
        by_action_sma[label] = _summarize_group(grp)
    _print_summary_table(
        "2. Summary by action x above_sma_200",
        by_action_sma,
        label_width=30,
    )

    # ── Table 3: by action × score bucket ──────────────────────────────
    by_action_score: dict[str, dict] = {}
    for (action, bucket), grp in df.groupby(
        ["action", "score_bucket"], dropna=False
    ):
        label = f"{action} / {bucket}"
        by_action_score[label] = _summarize_group(grp)
    _print_summary_table(
        "3. Summary by action x score bucket",
        by_action_score,
        label_width=28,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = YFinanceLoader(cache_dir="data/raw")

    # ── Load ────────────────────────────────────────────────────────────
    df = load_all_signals()

    # ── Append forward returns ──────────────────────────────────────────
    df = add_forward_returns(df, loader)

    # ── Save enriched CSV ───────────────────────────────────────────────
    out_path = SIGNALS_DIR / "analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nEnriched data saved: {out_path}  ({len(df)} rows, {len(df.columns)} columns)")

    # ── Print summaries ─────────────────────────────────────────────────
    print_summaries(df)
