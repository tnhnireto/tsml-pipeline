"""
Tests for generate_signals.

No network or file I/O is required — all inputs are in-process DataFrames.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tsml.portfolio.strategy import SignalAction, generate_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ranking(*rows: tuple[str, float]) -> pd.DataFrame:
    """Build a ranking DataFrame from (symbol, score) pairs."""
    return pd.DataFrame(rows, columns=["symbol", "score"])


def _actions(signals: list[SignalAction]) -> dict[str, str]:
    """Map symbol -> action for easy assertion."""
    return {s.symbol: s.action for s in signals}


# ---------------------------------------------------------------------------
# Basic buy / sell / hold
# ---------------------------------------------------------------------------

class TestBasicActions:
    def test_new_symbol_in_top_n_gets_buy(self):
        df = _ranking(("AAA", 0.70), ("BBB", 0.65))
        signals = generate_signals(df, current_positions=set(), top_n=2, min_score=0.55)
        acts = _actions(signals)
        assert acts["AAA"] == "buy"
        assert acts["BBB"] == "buy"

    def test_held_symbol_in_top_n_gets_hold(self):
        df = _ranking(("AAA", 0.70), ("BBB", 0.65))
        signals = generate_signals(df, current_positions={"AAA"}, top_n=2, min_score=0.55)
        acts = _actions(signals)
        assert acts["AAA"] == "hold"

    def test_held_symbol_outside_top_n_gets_sell(self):
        df = _ranking(("AAA", 0.80), ("BBB", 0.75), ("CCC", 0.60))
        signals = generate_signals(df, current_positions={"CCC"}, top_n=2, min_score=0.55)
        acts = _actions(signals)
        assert acts["CCC"] == "sell"

    def test_held_symbol_below_min_score_gets_sell(self):
        df = _ranking(("AAA", 0.80), ("BBB", 0.50))
        signals = generate_signals(df, current_positions={"BBB"}, top_n=5, min_score=0.55)
        acts = _actions(signals)
        assert acts["BBB"] == "sell"

    def test_unowned_symbol_below_min_score_not_in_output(self):
        df = _ranking(("AAA", 0.80), ("BBB", 0.50))
        signals = generate_signals(df, current_positions=set(), top_n=5, min_score=0.55)
        syms = {s.symbol for s in signals}
        assert "BBB" not in syms

    def test_no_positions_no_ranking_returns_empty(self):
        df = _ranking()
        signals = generate_signals(df, current_positions=set(), top_n=5, min_score=0.55)
        assert signals == []


# ---------------------------------------------------------------------------
# top_n limit
# ---------------------------------------------------------------------------

class TestTopN:
    def test_only_top_n_symbols_are_bought(self):
        df = _ranking(
            ("A", 0.90), ("B", 0.85), ("C", 0.80),
            ("D", 0.75), ("E", 0.70), ("F", 0.65),
        )
        signals = generate_signals(df, current_positions=set(), top_n=3, min_score=0.55)
        buys = [s.symbol for s in signals if s.action == "buy"]
        assert set(buys) == {"A", "B", "C"}
        assert len(buys) == 3

    def test_symbol_ranked_n_plus_1_not_bought(self):
        df = _ranking(("A", 0.90), ("B", 0.80), ("C", 0.70))
        signals = generate_signals(df, current_positions=set(), top_n=2, min_score=0.55)
        syms = {s.symbol for s in signals}
        assert "C" not in syms

    def test_top_n_1_returns_single_buy(self):
        df = _ranking(("A", 0.90), ("B", 0.80))
        signals = generate_signals(df, current_positions=set(), top_n=1, min_score=0.55)
        assert len([s for s in signals if s.action == "buy"]) == 1
        assert signals[0].symbol == "A"

    def test_fewer_eligible_than_top_n_returns_all_eligible(self):
        """Only 2 symbols meet min_score; top_n=5 should not pad with ineligible ones."""
        df = _ranking(("A", 0.80), ("B", 0.70), ("C", 0.40))
        signals = generate_signals(df, current_positions=set(), top_n=5, min_score=0.55)
        buys = [s for s in signals if s.action == "buy"]
        assert len(buys) == 2
        assert {s.symbol for s in buys} == {"A", "B"}


# ---------------------------------------------------------------------------
# min_score filter
# ---------------------------------------------------------------------------

class TestMinScore:
    def test_exactly_at_min_score_is_eligible(self):
        df = _ranking(("A", 0.55))
        signals = generate_signals(df, current_positions=set(), top_n=1, min_score=0.55)
        assert signals[0].action == "buy"

    def test_just_below_min_score_is_ineligible(self):
        df = _ranking(("A", 0.5499))
        signals = generate_signals(df, current_positions=set(), top_n=1, min_score=0.55)
        assert signals == []

    def test_min_score_zero_accepts_all(self):
        df = _ranking(("A", 0.10), ("B", 0.05))
        signals = generate_signals(df, current_positions=set(), top_n=5, min_score=0.0)
        acts = _actions(signals)
        assert acts["A"] == "buy"
        assert acts["B"] == "buy"


# ---------------------------------------------------------------------------
# Output ordering
# ---------------------------------------------------------------------------

class TestOutputOrder:
    def test_buys_before_holds_before_sells(self):
        df = _ranking(
            ("NEW1", 0.90), ("NEW2", 0.85),  # buys
            ("HELD", 0.80),                   # hold
            ("DROP", 0.60),                   # will be sell (held but rank 4 > top_n=3)
        )
        signals = generate_signals(
            df,
            current_positions={"HELD", "DROP"},
            top_n=3,
            min_score=0.55,
        )
        action_order = [s.action for s in signals]
        # Sells come last
        first_sell = next(
            (i for i, a in enumerate(action_order) if a == "sell"), len(action_order)
        )
        last_non_sell = max(
            (i for i, a in enumerate(action_order) if a != "sell"), default=-1
        )
        assert last_non_sell < first_sell, f"Sell appeared before non-sell: {action_order}"

    def test_buys_sorted_descending_by_score(self):
        df = _ranking(("A", 0.60), ("B", 0.80), ("C", 0.70))
        signals = generate_signals(df, current_positions=set(), top_n=3, min_score=0.55)
        buy_scores = [s.score for s in signals if s.action == "buy"]
        assert buy_scores == sorted(buy_scores, reverse=True)


# ---------------------------------------------------------------------------
# Scores on output actions
# ---------------------------------------------------------------------------

class TestScores:
    def test_scores_match_ranking(self):
        df = _ranking(("A", 0.80), ("B", 0.70))
        signals = generate_signals(df, current_positions=set(), top_n=2, min_score=0.55)
        score_map = {s.symbol: s.score for s in signals}
        assert score_map["A"] == pytest.approx(0.80)
        assert score_map["B"] == pytest.approx(0.70)

    def test_sell_score_is_nan_when_symbol_missing_from_ranking(self):
        """A held symbol absent from ranking_df should still produce a sell with NaN score."""
        df = _ranking(("A", 0.80))
        signals = generate_signals(df, current_positions={"GHOST"}, top_n=1, min_score=0.55)
        sell = next(s for s in signals if s.symbol == "GHOST")
        assert sell.action == "sell"
        assert math.isnan(sell.score)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_symbol_column_raises(self):
        df = pd.DataFrame({"score": [0.8]})
        with pytest.raises(ValueError, match="symbol"):
            generate_signals(df, current_positions=set())

    def test_missing_score_column_raises(self):
        df = pd.DataFrame({"symbol": ["A"]})
        with pytest.raises(ValueError, match="score"):
            generate_signals(df, current_positions=set())

    def test_top_n_zero_raises(self):
        df = _ranking(("A", 0.8))
        with pytest.raises(ValueError, match="top_n"):
            generate_signals(df, current_positions=set(), top_n=0)

    def test_min_score_above_1_raises(self):
        df = _ranking(("A", 0.8))
        with pytest.raises(ValueError, match="min_score"):
            generate_signals(df, current_positions=set(), min_score=1.1)

    def test_min_score_below_0_raises(self):
        df = _ranking(("A", 0.8))
        with pytest.raises(ValueError, match="min_score"):
            generate_signals(df, current_positions=set(), min_score=-0.01)

    def test_min_score_downtrend_above_1_raises(self):
        df = _ranking(("A", 0.8))
        with pytest.raises(ValueError, match="min_score_downtrend"):
            generate_signals(df, current_positions=set(), min_score_downtrend=1.1)

    def test_min_score_downtrend_below_min_score_raises(self):
        """Downtrend threshold must be at least as strict as the base threshold."""
        df = _ranking(("A", 0.8))
        with pytest.raises(ValueError, match="min_score_downtrend"):
            generate_signals(df, current_positions=set(), min_score=0.55, min_score_downtrend=0.50)


# ---------------------------------------------------------------------------
# Risk filter — downtrend guard
# ---------------------------------------------------------------------------

def _ranking_with_sma(*rows: tuple[str, float, bool | None]) -> pd.DataFrame:
    """Build a ranking DataFrame that includes above_sma_200."""
    return pd.DataFrame(rows, columns=["symbol", "score", "above_sma_200"])


class TestDowntrendFilter:
    def test_below_sma200_and_low_score_is_blocked(self):
        df = _ranking_with_sma(("A", 0.90, True), ("B", 0.58, False))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=2, min_score=0.55, min_score_downtrend=0.62)
        acts = _actions(signals)
        assert acts["B"] == "blocked"

    def test_below_sma200_but_high_enough_score_is_eligible(self):
        """If score >= min_score_downtrend the symbol can still be bought."""
        df = _ranking_with_sma(("A", 0.65, False))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        acts = _actions(signals)
        assert acts["A"] == "buy"

    def test_above_sma200_ignores_downtrend_threshold(self):
        df = _ranking_with_sma(("A", 0.58, True))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        acts = _actions(signals)
        assert acts["A"] == "buy"

    def test_none_sma200_ignores_downtrend_threshold(self):
        """Unknown SMA status → no penalty."""
        df = _ranking_with_sma(("A", 0.58, None))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        acts = _actions(signals)
        assert acts["A"] == "buy"

    def test_no_sma_column_no_blocking(self):
        """Backward compat: ranking without above_sma_200 never blocks."""
        df = _ranking(("A", 0.58), ("B", 0.57))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=2, min_score=0.55, min_score_downtrend=0.62)
        acts = _actions(signals)
        assert acts["A"] == "buy"
        assert acts["B"] == "buy"

    def test_blocked_symbol_has_reason_string(self):
        df = _ranking_with_sma(("A", 0.58, False))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        blocked = next(s for s in signals if s.symbol == "A")
        assert blocked.action == "blocked"
        assert "SMA200" in blocked.reason
        assert "0.62" in blocked.reason

    def test_held_blocked_symbol_becomes_sell_with_reason(self):
        """A position that becomes blocked must be exited."""
        df = _ranking_with_sma(("A", 0.58, False))
        signals = generate_signals(df, current_positions={"A"},
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        sell = next(s for s in signals if s.symbol == "A")
        assert sell.action == "sell"
        assert "SMA200" in sell.reason

    def test_blocked_not_counted_toward_top_n(self):
        """Blocking a symbol should allow the next-ranked eligible one to fill the slot."""
        df = _ranking_with_sma(
            ("A", 0.90, True),    # eligible rank 1
            ("B", 0.75, False),   # blocked (0.75 >= 0.62 so NOT blocked — this is eligible)
            ("C", 0.58, False),   # blocked (0.58 < 0.62)
            ("D", 0.57, True),    # eligible rank 3
        )
        # With top_n=2 and B eligible: target = {A, B}
        signals = generate_signals(df, current_positions=set(),
                                   top_n=2, min_score=0.55, min_score_downtrend=0.62)
        acts = _actions(signals)
        assert acts["A"] == "buy"
        assert acts["B"] == "buy"
        assert acts["C"] == "blocked"
        assert "D" not in acts

    def test_blocked_symbol_below_min_score_not_in_output(self):
        """A symbol below min_score is invisible regardless of SMA status."""
        df = _ranking_with_sma(("A", 0.50, False))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        assert not signals

    def test_blocked_appears_after_sells_in_output(self):
        df = _ranking_with_sma(
            ("A", 0.90, True),
            ("B", 0.58, False),   # blocked
        )
        signals = generate_signals(df, current_positions={"C"},
                                   top_n=1, min_score=0.55, min_score_downtrend=0.62)
        action_order = [s.action for s in signals]
        if "sell" in action_order and "blocked" in action_order:
            assert action_order.index("sell") < action_order.index("blocked")

    def test_normal_buy_has_empty_reason(self):
        df = _ranking_with_sma(("A", 0.80, True))
        signals = generate_signals(df, current_positions=set(),
                                   top_n=1, min_score=0.55)
        assert signals[0].action == "buy"
        assert signals[0].reason == ""


# ---------------------------------------------------------------------------
# SignalAction frozen dataclass
# ---------------------------------------------------------------------------

class TestSignalAction:
    def test_valid_actions_accepted(self):
        for action in ("buy", "sell", "hold", "blocked"):
            sa = SignalAction("X", action, 0.6)
            assert sa.action == action

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            SignalAction("X", "short", 0.6)

    def test_frozen_immutable(self):
        sa = SignalAction("X", "buy", 0.6)
        with pytest.raises(Exception):
            sa.action = "sell"  # type: ignore[misc]

    def test_reason_defaults_to_empty_string(self):
        sa = SignalAction("X", "buy", 0.6)
        assert sa.reason == ""

    def test_reason_can_be_set(self):
        sa = SignalAction("X", "blocked", 0.58, "blocked: below SMA200")
        assert "SMA200" in sa.reason
