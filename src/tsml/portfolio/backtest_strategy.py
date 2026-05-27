"""
Backtest-only signal generation with turnover-control rules.

Used by :func:`~tsml.portfolio.simulator.simulate` when ``turnover_control``
is enabled.  Live execution continues to use
:func:`~tsml.portfolio.strategy.generate_signals` unchanged.
"""

from __future__ import annotations

import math
from typing import Collection, Mapping

import pandas as pd

from tsml.portfolio.strategy import SignalAction


def generate_backtest_signals(
    ranking_df: pd.DataFrame,
    current_positions: Collection[str],
    *,
    top_n: int = 5,
    min_score_downtrend: float = 0.62,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.52,
    weeks_held: Mapping[str, int] | None = None,
    min_hold_weeks: int = 2,
) -> list[SignalAction]:
    """
    Produce buy / hold / sell actions with hysteresis and minimum hold time.

    Rules
    -----
    * New buys require ``score >= buy_threshold`` and pass the SMA200
      downtrend filter (``score >= min_score_downtrend`` when below SMA200).
    * Held positions are kept while ``score >= sell_threshold``, even if
      they fall out of the top-N ranking.
    * Held positions are not sold before ``min_hold_weeks`` rebalance
      periods have elapsed, unless the hard risk filter triggers.
    * Hard risk exit: ``above_sma_200 is False`` and
      ``score < min_score_downtrend`` — always sells, ignoring min hold.
    """
    _validate_inputs(
        ranking_df,
        top_n,
        buy_threshold,
        sell_threshold,
        min_score_downtrend,
        min_hold_weeks,
    )

    held: set[str] = set(current_positions)
    held_weeks: dict[str, int] = dict(weeks_held or {})
    has_sma = "above_sma_200" in ranking_df.columns

    keep_cols = ["symbol", "score"] + (["above_sma_200"] if has_sma else [])
    ranked = (
        ranking_df[keep_cols]
        .copy()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    score_map: dict[str, float] = dict(zip(ranked["symbol"], ranked["score"]))
    sma_map: dict[str, bool | None] = {}
    if has_sma:
        sma_map = {
            str(row["symbol"]): row["above_sma_200"]
            for _, row in ranked.iterrows()
        }

    def hard_risk_exit(sym: str) -> bool:
        above = sma_map.get(sym)
        score = score_map.get(sym, float("nan"))
        return (
            above is False
            and not math.isnan(score)
            and score < min_score_downtrend
        )

    def should_sell(sym: str) -> bool:
        if sym not in held:
            return False
        if hard_risk_exit(sym):
            return True
        weeks = held_weeks.get(sym, 0)
        if weeks < min_hold_weeks:
            return False
        score = score_map.get(sym, float("nan"))
        return math.isnan(score) or score < sell_threshold

    keepers: set[str] = {sym for sym in held if not should_sell(sym)}

    buy_candidates: list[tuple[str, float]] = []
    blocked: list[tuple[str, float]] = []

    for _, row in ranked.iterrows():
        sym = str(row["symbol"])
        score = float(row["score"])
        if sym in keepers:
            continue
        if score < buy_threshold:
            continue
        above = row["above_sma_200"] if has_sma else None
        if above is False and score < min_score_downtrend:
            blocked.append((sym, score))
            continue
        buy_candidates.append((sym, score))

    slots = max(0, top_n - len(keepers))
    new_buys = buy_candidates[:slots]
    new_buy_syms = {sym for sym, _ in new_buys}
    target = keepers | new_buy_syms

    buys: list[SignalAction] = [
        SignalAction(sym, "buy", score) for sym, score in new_buys
    ]
    holds: list[SignalAction] = [
        SignalAction(sym, "hold", score_map.get(sym, float("nan")))
        for sym in sorted(keepers)
    ]
    sells: list[SignalAction] = []
    for sym in held:
        if sym not in target:
            score = score_map.get(sym, float("nan"))
            reason = ""
            if hard_risk_exit(sym):
                reason = (
                    f"hard risk: below SMA200 and score {score:.3f}"
                    f" < min_score_downtrend {min_score_downtrend}"
                )
            sells.append(SignalAction(sym, "sell", score, reason))

    blocked_out: list[SignalAction] = [
        SignalAction(
            sym,
            "blocked",
            score,
            f"blocked: below SMA200 and score {score:.3f}"
            f" < min_score_downtrend {min_score_downtrend}",
        )
        for sym, score in blocked
        if sym not in held
    ]

    _by_score = lambda s: s.score if not math.isnan(s.score) else float("-inf")  # noqa: E731
    buys.sort(key=_by_score, reverse=True)
    holds.sort(key=_by_score, reverse=True)
    sells.sort(key=_by_score, reverse=True)
    blocked_out.sort(key=_by_score, reverse=True)

    return buys + holds + sells + blocked_out


def _validate_inputs(
    ranking_df: pd.DataFrame,
    top_n: int,
    buy_threshold: float,
    sell_threshold: float,
    min_score_downtrend: float,
    min_hold_weeks: int,
) -> None:
    required = {"symbol", "score"}
    missing = required - set(ranking_df.columns)
    if missing:
        raise ValueError(f"ranking_df is missing required columns: {missing}.")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1; got {top_n}.")
    for name, val in [
        ("buy_threshold", buy_threshold),
        ("sell_threshold", sell_threshold),
        ("min_score_downtrend", min_score_downtrend),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]; got {val}.")
    if sell_threshold > buy_threshold:
        raise ValueError(
            f"sell_threshold ({sell_threshold}) must be <= "
            f"buy_threshold ({buy_threshold})."
        )
    if min_score_downtrend < buy_threshold:
        raise ValueError(
            f"min_score_downtrend ({min_score_downtrend}) must be >= "
            f"buy_threshold ({buy_threshold})."
        )
    if min_hold_weeks < 0:
        raise ValueError(f"min_hold_weeks must be >= 0; got {min_hold_weeks}.")
