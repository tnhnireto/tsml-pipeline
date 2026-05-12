"""
Portfolio strategy — translate a ranked universe into trading actions.

``generate_signals`` compares a fresh ranking from ``rank_universe`` against
a set of currently held positions and decides what to buy, sell, or hold.

Signal generation rules
-----------------------
1. Any symbol whose score is below ``min_score`` is ineligible regardless of
   its rank.  Symbols already held that fall below ``min_score`` receive a
   "sell" action.
2. **Risk filter (downtrend guard):** if the optional ``above_sma_200`` column
   is present and is ``False`` for a symbol, a stricter threshold
   ``min_score_downtrend`` (default 0.62) is applied.  The symbol is blocked
   from the target portfolio unless its score meets the stricter threshold.
   - Blocked symbols not currently held → ``action="blocked"`` (shown
     separately so the caller can explain the decision).
   - Blocked symbols that *are* currently held → ``action="sell"`` with a
     reason string (exit the position — don't sit in a known downtrend).
   - If ``above_sma_200`` is ``None`` or missing, the stricter threshold is
     not applied (unknown trend = treat as neutral).
3. The remaining eligible symbols are sorted by score descending.  The top
   ``top_n`` become the target portfolio.
4. Current positions that remain in the target portfolio → "hold".
5. Current positions that are no longer in the target portfolio → "sell".
6. Target portfolio members that are not currently held → "buy".

Output order: buys → holds → sells → blocked (each group by score desc).

Example
-------
>>> from tsml.portfolio.strategy import generate_signals
>>> import pandas as pd
>>>
>>> ranking = pd.DataFrame({
...     "symbol":        ["NVDA", "MSFT", "AAPL", "META"],
...     "score":         [0.72,   0.68,   0.61,   0.58],
...     "above_sma_200": [True,   False,  True,   False],
... })
>>> signals = generate_signals(ranking, {"AAPL"}, top_n=3,
...                            min_score=0.55, min_score_downtrend=0.62)
>>> for s in signals:
...     print(s.action, s.symbol, s.reason)
buy  NVDA
buy  AAPL   (wait — held, so it becomes hold)
...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
        One of ``"buy"``, ``"sell"``, ``"hold"``, or ``"blocked"``.

        ``"blocked"`` means the symbol ranked in the eligible score range but
        was rejected by the risk filter (e.g. below SMA 200 with insufficient
        conviction).  A blocked symbol that is *currently held* produces a
        ``"sell"`` instead.
    score:
        The model's P(up) score that drove the decision.
    reason:
        Human-readable explanation for why an action was taken, populated for
        ``"blocked"`` and risk-filter-driven ``"sell"`` actions.  Empty string
        for normal buys, holds, and sells.
    """

    symbol: str
    action: str    # "buy" | "sell" | "hold" | "blocked"
    score:  float
    reason: str = field(default="")

    def __post_init__(self) -> None:
        valid = {"buy", "sell", "hold", "blocked"}
        if self.action not in valid:
            raise ValueError(
                f"action must be one of {sorted(valid)}; got '{self.action}'."
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
    min_score_downtrend: float = 0.62,
) -> list[SignalAction]:
    """
    Compare a ranked universe to current holdings and produce trading actions.

    Parameters
    ----------
    ranking_df:
        DataFrame with at least columns ``"symbol"`` (str) and ``"score"``
        (float), as returned by :func:`tsml.portfolio.ranker.rank_universe`.
        May optionally contain ``"above_sma_200"`` (bool / None) from
        :func:`tsml.portfolio.ranker.enrich_with_context`; if present, the
        downtrend risk filter is applied.
        Rows may be in any order; the function re-sorts internally.
    current_positions:
        Any collection (set, list, dict keys, …) of ticker strings that are
        currently held.
    top_n:
        Number of top-scoring *eligible* symbols to keep in the target
        portfolio.  Must be >= 1.
    min_score:
        Minimum score for a symbol to be considered at all.
        Symbols below this threshold are silently ignored (or sold if held).
        Must be in [0, 1].
    min_score_downtrend:
        Stricter score threshold applied when ``above_sma_200`` is ``False``.
        Symbols that pass ``min_score`` but fail ``min_score_downtrend`` while
        in a confirmed downtrend are returned with ``action="blocked"`` (or
        ``action="sell"`` if currently held).
        Must be in [0, 1] and >= ``min_score``.

    Returns
    -------
    list[SignalAction]
        One ``SignalAction`` per relevant symbol, ordered:
        buys → holds → sells → blocked.  Each group is sorted by score
        descending.  Symbols below ``min_score`` with no open position are
        omitted entirely.

    Raises
    ------
    ValueError
        If ``ranking_df`` is missing required columns, ``top_n < 1``,
        ``min_score`` or ``min_score_downtrend`` is outside [0, 1], or
        ``min_score_downtrend < min_score``.

    Notes
    -----
    The function does not track position sizes or weights.  All "buy" and
    "hold" actions imply equal-weight allocation; callers are responsible for
    sizing.
    """
    _validate_inputs(ranking_df, top_n, min_score, min_score_downtrend)

    held: set[str] = set(current_positions)
    has_sma = "above_sma_200" in ranking_df.columns

    # Re-sort descending for deterministic rank order.
    keep_cols = ["symbol", "score"] + (["above_sma_200"] if has_sma else [])
    ranked = (
        ranking_df[keep_cols]
        .copy()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # Full score lookup (used for sell / blocked actions on held symbols).
    score_map: dict[str, float] = dict(zip(ranked["symbol"], ranked["score"]))

    # Partition symbols that pass min_score into eligible vs. blocked.
    eligible: list[tuple[str, float]] = []   # (symbol, score) — can enter top-N
    blocked:  list[tuple[str, float]] = []   # (symbol, score) — risk-filtered out

    for _, row in ranked.iterrows():
        score = float(row["score"])
        if score < min_score:
            break   # sorted desc; nothing below qualifies
        sym       = str(row["symbol"])
        above_sma = row["above_sma_200"] if has_sma else None

        # Apply stricter threshold only when trend is *confirmed* down.
        if above_sma is False and score < min_score_downtrend:
            blocked.append((sym, score))
        else:
            eligible.append((sym, score))

    target:      set[str] = {sym for sym, _ in eligible[:top_n]}
    blocked_syms: set[str] = {sym for sym, _ in blocked}

    buys:    list[SignalAction] = []
    holds:   list[SignalAction] = []
    sells:   list[SignalAction] = []
    blocked_out: list[SignalAction] = []

    # ── Top-N eligible → buy or hold ──────────────────────────────────
    for sym, score in eligible[:top_n]:
        if sym in held:
            holds.append(SignalAction(sym, "hold", score))
        else:
            buys.append(SignalAction(sym, "buy", score))

    # ── Current positions no longer in target → sell ──────────────────
    for sym in held:
        if sym not in target:
            score  = score_map.get(sym, float("nan"))
            reason = (
                f"blocked: below SMA200 and score {score:.3f}"
                f" < min_score_downtrend {min_score_downtrend}"
                if sym in blocked_syms else ""
            )
            sells.append(SignalAction(sym, "sell", score, reason))

    # ── Blocked, not held → "blocked" for visibility ──────────────────
    for sym, score in blocked:
        if sym not in held:
            reason = (
                f"blocked: below SMA200 and score {score:.3f}"
                f" < min_score_downtrend {min_score_downtrend}"
            )
            blocked_out.append(SignalAction(sym, "blocked", score, reason))

    _by_score = lambda s: s.score if not math.isnan(s.score) else float("-inf")  # noqa: E731
    sells.sort(       key=_by_score, reverse=True)
    blocked_out.sort( key=_by_score, reverse=True)

    return buys + holds + sells + blocked_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    ranking_df: pd.DataFrame,
    top_n: int,
    min_score: float,
    min_score_downtrend: float,
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
    if not (0.0 <= min_score_downtrend <= 1.0):
        raise ValueError(
            f"min_score_downtrend must be in [0, 1]; got {min_score_downtrend}."
        )
    if min_score_downtrend < min_score:
        raise ValueError(
            f"min_score_downtrend ({min_score_downtrend}) must be >= "
            f"min_score ({min_score}); the downtrend threshold is stricter."
        )
