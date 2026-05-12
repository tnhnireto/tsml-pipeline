"""
run_weekly_signal.py -- weekly portfolio signal generation.

Ranks a defined symbol universe by model conviction, enriches the ranking
with plain-language market context, and suggests buy / hold / sell actions
for the next trading week.

Target: "threshold"
  Only high-conviction training days are used (|next-day return| > 0.5 %).
  This focuses the model on clear directional moves and drops ambiguous days,
  at the cost of fewer training samples.  Symbols with too little post-filter
  data are skipped automatically.

Usage
-----
    python run_weekly_signal.py

Data is downloaded from Yahoo Finance on the first run and cached under
data/raw/.  Subsequent runs use the cache.

Expected runtime: 1-3 minutes on first run, ~15 seconds with cache.

Configuration
-------------
Edit the constants in the CONFIGURATION section below.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

from tsml.data_loader import YFinanceLoader
from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.portfolio import enrich_with_context, generate_signals, rank_universe
from tsml.validation import WalkForwardSplit

# ===========================================================================
# CONFIGURATION -- edit this section before running
# ===========================================================================

UNIVERSE: list[str] = [
    # Broad market ETFs
    "SPY", "QQQ",
    # US mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    # Other large-caps
    "TSLA", "JPM", "JNJ", "XOM", "V", "GS", "NFLX",
]

START_DATE: str = "2019-01-01"
END_DATE:   str = date.today().strftime("%Y-%m-%d")

N_SPLITS:       int = 5
MIN_TRAIN_SIZE: int = 252
TEST_SIZE:      int = 63
GAP:            int = 1

TOP_N:                int   = 5
MIN_SCORE:            float = 0.55   # base eligibility threshold
MIN_SCORE_DOWNTREND:  float = 0.62   # stricter threshold when above_sma_200 is False
TARGET:               str   = "threshold"   # high-conviction days only

# Edit this to reflect actual open positions before each run.
# Example:  CURRENT_POSITIONS = {"MSFT", "AAPL", "QQQ"}
CURRENT_POSITIONS: set[str] = set()

# ===========================================================================
# FORMATTING HELPERS
# ===========================================================================

_W = 76
_SEP  = "-" * _W
_SEP2 = "=" * _W


def _pct(v: float, *, signed: bool = True, width: int = 7) -> str:
    """Format a fraction as a percentage string, right-aligned to `width`."""
    if math.isnan(v):
        return "n/a".rjust(width)
    fmt = f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"
    return fmt.rjust(width)


def _bool_str(v: bool | None, width: int = 5) -> str:
    if v is None:
        return "n/a".ljust(width)
    return ("Yes" if v else "No").ljust(width)


# ===========================================================================
# SETUP
# ===========================================================================

print(_SEP2)
print("  Weekly portfolio signal")
print(_SEP2)
print(f"  Universe  : {len(UNIVERSE)} symbols")
print(f"  Period    : {START_DATE}  to  {END_DATE}")
print(f"  Target    : {TARGET}  (high-conviction days, |ret| > 0.5 %)")
print(f"  Top N     : {TOP_N}   |   Min score : {MIN_SCORE}  |  Min score (downtrend) : {MIN_SCORE_DOWNTREND}")
if CURRENT_POSITIONS:
    print(f"  Holdings  : {', '.join(sorted(CURRENT_POSITIONS))}")
else:
    print("  Holdings  : none (all buys will be new entries)")
print(_SEP2)
print()

loader   = YFinanceLoader(cache_dir="data/raw")
splitter = WalkForwardSplit(
    n_splits=N_SPLITS,
    min_train_size=MIN_TRAIN_SIZE,
    test_size=TEST_SIZE,
    gap=GAP,
)
model = CalibratedLogisticRegressionModel()

# ===========================================================================
# RANK UNIVERSE
# ===========================================================================

print("Ranking universe  (first run downloads data, subsequent runs use cache)")
print()

ranking = rank_universe(
    UNIVERSE,
    model=model,
    splitter=splitter,
    target=TARGET,
    start=START_DATE,
    end=END_DATE,
    loader=loader,
)

if ranking.empty:
    print(
        "ERROR: No symbols could be ranked.\n"
        "  - Check that START_DATE gives enough history for the splitter.\n"
        "  - With target='threshold' roughly half the rows are dropped;\n"
        "    try extending START_DATE further back if many symbols fail."
    )
    sys.exit(1)

n_scored  = len(ranking)
n_skipped = len(UNIVERSE) - n_scored
if n_skipped:
    print(f"  Note: {n_skipped} symbol(s) skipped (see stderr for details).")
    print()

# ===========================================================================
# ENRICH WITH MARKET CONTEXT
# ===========================================================================

ranking = enrich_with_context(
    ranking,
    start=START_DATE,
    end=END_DATE,
    loader=loader,
)

# ===========================================================================
# TOP-10 CANDIDATES TABLE
# ===========================================================================

n_display = min(10, n_scored)

print(
    f"Top {n_display} candidates"
    f"  (score = P(up) on last OOS date, target = {TARGET}):"
)
print(_SEP)
print(
    f"  {'#':>3}  {'Symbol':<7} {'Score':>5}  "
    f"{'Ret20d':>7}  {'Ret60d':>7}  {'Vol20d':>7}  {'vsSMA200':>9}  {'>SMA':<5}  Status"
)
print(_SEP)

for rank, (_, row) in enumerate(ranking.head(n_display).iterrows(), start=1):
    sym   = str(row["symbol"])
    score = float(row["score"])

    held_mark = "*" if sym in CURRENT_POSITIONS else " "

    if rank <= TOP_N and score >= MIN_SCORE:
        status = "eligible"
    elif rank <= TOP_N:
        status = "(below min_score)"
    else:
        status = ""

    print(
        f"  {rank:>3}.{held_mark} {sym:<7} {score:>5.3f}  "
        f"{_pct(row['return_20d'])}  "
        f"{_pct(row['return_60d'])}  "
        f"{_pct(row['volatility_20d'], signed=False)}  "
        f"{_pct(row['price_vs_sma_200'], width=9)}  "
        f"{_bool_str(row['above_sma_200'])}  "
        f"{status}"
    )

print(_SEP)
if CURRENT_POSITIONS:
    print("  (* = currently held)")
print()
print(
    "  Columns: Ret20d = 20-day return | Ret60d = 60-day return | "
    "Vol20d = annualised 20-day vol"
)
print(
    "           vsSMA200 = price vs 200-day SMA | >SMA = above SMA 200"
)
print()

# ===========================================================================
# SUGGESTED ACTIONS
# ===========================================================================

signals = generate_signals(
    ranking,
    current_positions=CURRENT_POSITIONS,
    top_n=TOP_N,
    min_score=MIN_SCORE,
    min_score_downtrend=MIN_SCORE_DOWNTREND,
)

# Split into active actions and blocked candidates.
active   = [s for s in signals if s.action != "blocked"]
blocked_ = [s for s in signals if s.action == "blocked"]

# ===========================================================================
# SAVE SIGNAL LOG
# ===========================================================================

_action_map = {s.symbol: s.action for s in signals}
_reason_map = {s.symbol: s.reason  for s in signals}

_log = ranking.copy().reset_index(drop=True)
_log.insert(0, "date", END_DATE)
_log.insert(1, "rank", range(1, len(_log) + 1))
_log["action"] = _log["symbol"].map(_action_map).fillna("")
_log["reason"] = _log["symbol"].map(_reason_map).fillna("")

_LOG_COLS = [
    "date", "rank", "symbol", "score", "action", "reason",
    "return_20d", "return_60d", "volatility_20d",
    "price_vs_sma_200", "above_sma_200",
]
_log = _log[[c for c in _LOG_COLS if c in _log.columns]]

_signals_dir = Path("signals")
_signals_dir.mkdir(exist_ok=True)
_log_path = _signals_dir / f"{END_DATE}.csv"
_log.to_csv(_log_path, index=False)
print(f"Signal log saved: {_log_path}  ({len(_log)} symbols)")
print()

# ===========================================================================
# SUGGESTED ACTIONS
# ===========================================================================

print("Suggested actions for next week:")
print(_SEP)

if not active:
    print("  No action required.")
else:
    _order = {"buy": 0, "hold": 1, "sell": 2}
    for s in sorted(active, key=lambda x: (_order[x.action], -x.score)):
        score_str = f"{s.score:.3f}" if not math.isnan(s.score) else "n/a"
        reason_str = f"  -- {s.reason}" if s.reason else ""
        print(f"  {s.action.upper():<4}  {s.symbol:<8}  score: {score_str}{reason_str}")

print(_SEP)
print()

# ===========================================================================
# BLOCKED CANDIDATES
# ===========================================================================

if blocked_:
    print("Blocked candidates (would rank in top-N but failed risk filter):")
    print(_SEP)
    for s in blocked_:
        score_str = f"{s.score:.3f}" if not math.isnan(s.score) else "n/a"
        print(f"  BLOCKED  {s.symbol:<8}  score: {score_str}  -- {s.reason}")
    print(_SEP)
    print()

print(
    "Next step: execute the suggested trades, then update CURRENT_POSITIONS\n"
    "           at the top of this file before the next weekly run."
)
