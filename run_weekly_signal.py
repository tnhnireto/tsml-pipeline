"""
run_weekly_signal.py -- weekly portfolio signal generation.

Ranks a defined symbol universe by model conviction, prints the top-10
candidates, and suggests buy / hold / sell actions for the next trading
week based on current holdings.

Usage
-----
    python run_weekly_signal.py

On the first run data is downloaded from Yahoo Finance and cached under
data/raw/.  Subsequent runs use the cache and are faster.

Expected runtime: 1-3 minutes depending on universe size and CPU.

Configuration
-------------
Edit the constants in the CONFIGURATION section below to change the
symbol universe, date range, strategy parameters, or current holdings.
"""

from __future__ import annotations

import sys
from datetime import date

from tsml.data_loader import YFinanceLoader
from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.portfolio import generate_signals, rank_universe
from tsml.validation import WalkForwardSplit

# ===========================================================================
# CONFIGURATION -- edit this section before running
# ===========================================================================

# Symbol universe to rank.  Add or remove tickers freely.
UNIVERSE: list[str] = [
    # Broad market ETFs
    "SPY", "QQQ",
    # US mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    # Other large-caps
    "TSLA", "JPM", "JNJ", "XOM", "V", "GS", "NFLX",
]

# Date range used to load and train on historical data.
# START_DATE should be far enough back to give the walk-forward splitter
# sufficient training rows (>= MIN_TRAIN_SIZE trading days before any test fold).
START_DATE: str = "2019-01-01"
END_DATE:   str = date.today().strftime("%Y-%m-%d")  # today

# Walk-forward splitter settings.
N_SPLITS:       int = 5
MIN_TRAIN_SIZE: int = 252   # approximately 1 trading year
TEST_SIZE:      int = 63    # approximately 1 trading quarter
GAP:            int = 1     # 1-day embargo between train and test

# Signal generation settings.
TOP_N:     int   = 5     # maximum number of positions to hold simultaneously
MIN_SCORE: float = 0.55  # minimum P(up) score to be eligible for purchase
TARGET:    str   = "direction"  # model target: "direction" | "direction_5d" | "threshold"

# Current holdings.
# Edit this set to reflect actual open positions before running.
# Example:  CURRENT_POSITIONS = {"MSFT", "AAPL", "JPM"}
CURRENT_POSITIONS: set[str] = set()

# ===========================================================================
# SETUP
# ===========================================================================

_SEP  = "-" * 56
_SEP2 = "=" * 56

print(_SEP2)
print("  Weekly portfolio signal")
print(_SEP2)
print(f"  Universe  : {len(UNIVERSE)} symbols")
print(f"  Period    : {START_DATE}  to  {END_DATE}")
print(f"  Target    : {TARGET}")
print(f"  Top N     : {TOP_N}   |   Min score : {MIN_SCORE}")
if CURRENT_POSITIONS:
    held = ", ".join(sorted(CURRENT_POSITIONS))
    print(f"  Holdings  : {held}")
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
        "Check that START_DATE is far enough back and that data is available."
    )
    sys.exit(1)

n_scored  = len(ranking)
n_skipped = len(UNIVERSE) - n_scored
if n_skipped:
    print(f"  Note: {n_skipped} symbol(s) were skipped (see stderr for details).")
    print()

# ===========================================================================
# TOP-10 CANDIDATES
# ===========================================================================

n_display = min(10, n_scored)
print(f"Top {n_display} candidates  (score = P(up) from last OOS walk-forward fold):")
print(_SEP)
print(f"  {'Rank':>4}   {'Symbol':<8}  {'Score':>6}")
print(_SEP)

for rank, (_, row) in enumerate(ranking.head(n_display).iterrows(), start=1):
    symbol = str(row["symbol"])
    score  = float(row["score"])

    # Mark symbols that are currently held
    held_flag = " *" if symbol in CURRENT_POSITIONS else ""

    # Mark symbols above the min_score threshold
    eligible  = "  eligible" if score >= MIN_SCORE else "  (below threshold)"

    print(f"  {rank:>4}.  {symbol:<8}  {score:.3f}{held_flag}{eligible}")

if CURRENT_POSITIONS:
    print()
    print("  (* = currently held)")

print(_SEP)
print()

# ===========================================================================
# SUGGESTED ACTIONS
# ===========================================================================

signals = generate_signals(
    ranking,
    current_positions=CURRENT_POSITIONS,
    top_n=TOP_N,
    min_score=MIN_SCORE,
)

print("Suggested actions for next week:")
print(_SEP)

if not signals:
    print("  No action required.")
else:
    # Group by action for readability
    _order = {"buy": 0, "hold": 1, "sell": 2}
    for s in sorted(signals, key=lambda x: (_order[x.action], -x.score)):
        action_str = s.action.upper()
        score_str  = f"{s.score:.3f}" if s.score == s.score else "  n/a"  # NaN check
        print(f"  {action_str:<4}  {s.symbol:<8}  score: {score_str}")

print(_SEP)
print()
print(
    "Next step: pass these signals to your broker or update CURRENT_POSITIONS\n"
    "           before the next weekly run."
)
