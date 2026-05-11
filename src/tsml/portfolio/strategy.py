"""
Portfolio strategy — translate a ranked universe into trading actions.

``generate_signals`` compares a fresh ranking from ``rank_universe`` against
a set of currently held positions and decides what to buy, sell, or hold.

Signal generation rules
-----------------------
1. Any symbol whose score is below ``min_score`` is ineligible regardless of
   its rank.  Symbols already held that fall below ``min_score`` receive a
   "sell" action.
2. The eligible symbols are sorted by score descending.  The top ``top_n``
   become the target portfolio.
3. Current positions that remain in the target portfolio → "hold".
4. Current positions that are no longer in the target portfolio → "sell".
5. Target portfolio members that are not currently held → "buy".

Symbols that are neither currently held nor in the target portfolio are not
included in the output (no action required).

Example
-------
>>> from tsml.portfolio.strategy import generate_signals
>>> import pandas as pd
>>>
>>> ranking = pd.DataFrame({
...     "symbol": ["NVDA", "MSFT", "AAPL", "GOOGL", "META", "AMZN"],
...     "score":  [0.72,   0.68,   0.61,   0.58,    0.52,   0.51],
... })
>>> current_positions = {"AAPL", "GOOGL", "AMZN"}
>>>
>>> signals = generate_signals(ranking, current_positions, top_n=3, min_score=0.55)
>>> for s in signals:
...     print(s)
SignalAction(symbol='NVDA', action='buy',  score=0.72)
SignalAction(symbol='MSFT', action='buy',  score=0.68)
SignalAction(symbol='AAPL', action='hold', score=0.61)
SignalAction(symbol='GOOGL', action='sell', score=0.58)  # rank 4 — outside top_n
SignalAction(symbol='AMZN',  action='sell', score=0.51)  # below min_score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import pandas as pd


# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalAction:
    """
    A single trading action for one symbol.

    Attributes
    ----------
    symbol:
        Ticker string (e.g. ``"NVDA"``).
    action:
        One of ``"buy"``, ``"sell"``, or ``"hold"``.
    score:
        The model's P(up) score that drove the decision.
    """

    symbol: str
    action: str   # "buy" | "sell" | "hold"
    score: float

    def __post_init__(self) -> None:
        if self.action not in {"buy", "sell", "hold"}:
            raise ValueError(
                f"action must be 'buy', 'sell', or 'hold'; got '{self.action}'."
            )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_signals(
    ranking_df: pd.DataFrame,
    current_positions: Collection[str],
    *,
    top_n: int = 5,
    min_score: float = 0.55,
) -> list[SignalAction]:
    """
    Compare a ranked universe to current holdings and produce trading actions.

    Parameters
    ----------
    ranking_df:
        DataFrame with at least columns ``"symbol"`` (str) and ``"score"``
        (float), as returned by :func:`tsml.portfolio.ranker.rank_universe`.
        Rows may be in any order; the function re-sorts internally.
    current_positions:
        Any collection (set, list, dict keys, …) of ticker strings that are
        currently held.
    top_n:
        Number of top-scoring eligible symbols to keep in the target
        portfolio.  Must be >= 1.
    min_score:
        Minimum score for a symbol to be eligible for purchase or continued
        holding.  Symbols with ``score < min_score`` are treated as
        sell/ignore candidates.  Must be in [0, 1].

    Returns
    -------
    list[SignalAction]
        One ``SignalAction`` per symbol that requires an action or is
        currently held.  Symbols with no current position and outside the
        target portfolio are omitted.

        The list is ordered as:
        - buys first (sorted by score descending),
        - holds next (sorted by score descending),
        - sells last (sorted by score descending).

    Raises
    ------
    ValueError
        If ``ranking_df`` is missing the required columns, ``top_n < 1``, or
        ``min_score`` is outside [0, 1].

    Notes
    -----
    The function does not track position sizes or weights.  All "buy" and
    "hold" actions imply equal-weight allocation; callers are responsible for
    sizing.
    """
    _validate_inputs(ranking_df, top_n, min_score)

    held: set[str] = set(current_positions)

    # Sort descending so rank order is deterministic regardless of input order.
    ranked = (
        ranking_df[["symbol", "score"]]
        .copy()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # Eligible = score meets threshold; take top_n of those.
    eligible = ranked[ranked["score"] >= min_score]
    target: set[str] = set(eligible.head(top_n)["symbol"])

    # Score lookup for all symbols that will appear in output.
    score_map: dict[str, float] = dict(
        zip(ranked["symbol"], ranked["score"])
    )

    buys:  list[SignalAction] = []
    holds: list[SignalAction] = []
    sells: list[SignalAction] = []

    # Target portfolio → buy or hold.
    for _, row in eligible.head(top_n).iterrows():
        sym   = str(row["symbol"])
        score = float(row["score"])
        if sym in held:
            holds.append(SignalAction(sym, "hold", score))
        else:
            buys.append(SignalAction(sym, "buy", score))

    # Current positions no longer in the target portfolio → sell.
    for sym in held:
        if sym not in target:
            score = score_map.get(sym, float("nan"))
            sells.append(SignalAction(sym, "sell", score))

    # Stable output order: buys → holds → sells, each group by score desc.
    sells.sort(key=lambda s: s.score if s.score == s.score else float("-inf"), reverse=True)

    return buys + holds + sells


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    ranking_df: pd.DataFrame,
    top_n: int,
    min_score: float,
) -> None:
    required = {"symbol", "score"}
    missing  = required - set(ranking_df.columns)
    if missing:
        raise ValueError(
            f"ranking_df is missing required columns: {missing}."
        )
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1; got {top_n}.")
    if not (0.0 <= min_score <= 1.0):
        raise ValueError(f"min_score must be in [0, 1]; got {min_score}.")
