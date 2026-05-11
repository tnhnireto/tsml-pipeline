"""
Tests for run_backtest.

The most important invariant is the shift:

    position[t] = prediction[t-1]

Every test that touches position checks this, because a missing or wrong
shift is the most common source of lookahead bias in backtests.

We also verify that the cumulative curve is computed correctly and that
transaction costs are applied only on position changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.backtest.engine import run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, freq="B", tz="UTC")


def _close(values: list[float]) -> pd.Series:
    return pd.Series(values, index=_dates(len(values)), name="close", dtype=float)


def _preds(values: list[int | float]) -> pd.Series:
    return pd.Series(values, index=_dates(len(values)), name="prediction", dtype=float)


# ---------------------------------------------------------------------------
# The shift invariant — the most critical test in this file
# ---------------------------------------------------------------------------

class TestPositionShift:
    def test_position_equals_prediction_shifted_by_one(self):
        """
        position[t] must equal prediction[t-1].

        This is the no-lookahead guarantee: a signal formed at the close
        of day t can only be executed from day t+1 onward.
        """
        preds = _preds([1, 0, 1, 1, 0, 0])
        close = _close([100, 101, 99, 102, 101, 103])
        result = run_backtest(preds, close)

        # First row is dropped (position NaN), so result starts at day 1.
        # position[day_1] = prediction[day_0] = 1
        # position[day_2] = prediction[day_1] = 0
        # ... etc.
        expected_positions = [1.0, 0.0, 1.0, 1.0, 0.0]
        actual_positions = result["position"].tolist()
        assert actual_positions == expected_positions, (
            f"Position shift wrong.\n"
            f"  expected: {expected_positions}\n"
            f"  got:      {actual_positions}"
        )

    def test_first_row_dropped_due_to_nan_position(self):
        """
        The first prediction date has no prior prediction, so position
        is NaN there and that row must be excluded from the result.
        """
        preds = _preds([1, 0, 1])
        close = _close([100, 101, 99])
        result = run_backtest(preds, close)
        assert len(result) == 2  # one row dropped

    def test_result_index_starts_on_second_prediction_date(self):
        preds = _preds([1, 0, 1, 0])
        close = _close([100, 101, 99, 102])
        result = run_backtest(preds, close)
        assert result.index[0] == preds.index[1]

    def test_always_long_position_is_one_everywhere(self):
        preds = _preds([1, 1, 1, 1, 1])
        close = _close([100, 101, 102, 103, 104])
        result = run_backtest(preds, close)
        assert (result["position"] == 1.0).all()

    def test_always_flat_position_is_zero_everywhere(self):
        preds = _preds([0, 0, 0, 0, 0])
        close = _close([100, 101, 100, 99, 102])
        result = run_backtest(preds, close)
        assert (result["position"] == 0.0).all()


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------

class TestReturnCalculations:
    def test_strategy_return_equals_position_times_asset_return(self):
        """Core formula: strategy_return[t] = position[t] * asset_return[t]."""
        preds = _preds([1, 0, 1, 0])
        close = _close([100.0, 110.0, 99.0, 105.0])
        result = run_backtest(preds, close)

        expected_asset = [(110 - 100) / 100, (99 - 110) / 110, (105 - 99) / 99]
        expected_pos = [1.0, 0.0, 1.0]
        expected_strategy = [p * r for p, r in zip(expected_pos, expected_asset)]

        np.testing.assert_allclose(result["asset_return"].tolist(), expected_asset)
        np.testing.assert_allclose(result["strategy_return"].tolist(), expected_strategy)

    def test_flat_position_gives_zero_strategy_return(self):
        """If position=0, we earn nothing regardless of market moves."""
        preds = _preds([0, 0, 0, 0])
        close = _close([100.0, 110.0, 90.0, 120.0])
        result = run_backtest(preds, close)
        assert (result["strategy_return"] == 0.0).all()

    def test_long_position_earns_asset_return(self):
        """If position=1 always, strategy return = asset return."""
        preds = _preds([1, 1, 1, 1])
        close = _close([100.0, 110.0, 90.0, 120.0])
        result = run_backtest(preds, close)
        np.testing.assert_allclose(
            result["strategy_return"].values,
            result["asset_return"].values,
        )

    def test_cumulative_starts_at_correct_value(self):
        """cumulative[0] = 1 + strategy_return[0]."""
        preds = _preds([1, 1, 1])
        close = _close([100.0, 110.0, 121.0])
        result = run_backtest(preds, close)
        first_sr = result["strategy_return"].iloc[0]
        assert result["cumulative"].iloc[0] == pytest.approx(1 + first_sr)

    def test_cumulative_compounds_correctly(self):
        """
        cumulative[n] = product of (1 + strategy_return[0..n]).
        We use known values to verify this precisely.
        """
        # Prices: 100 → 110 → 121, always long (prediction=1)
        # Returns: +10%, +10%
        # Cumulative: 1.10, 1.21
        preds = _preds([1, 1, 1])
        close = _close([100.0, 110.0, 121.0])
        result = run_backtest(preds, close)
        np.testing.assert_allclose(
            result["cumulative"].tolist(),
            [1.10, 1.21],
            rtol=1e-9,
        )

    def test_buy_and_hold_matches_asset_cumulative(self):
        """buy_and_hold must equal (1 + asset_return).cumprod()."""
        preds = _preds([1, 0, 1, 0])
        close = _close([100.0, 110.0, 99.0, 105.0])
        result = run_backtest(preds, close)
        expected = (1 + result["asset_return"]).cumprod()
        np.testing.assert_allclose(result["buy_and_hold"].values, expected.values)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_expected_columns(self):
        preds = _preds([1, 0, 1])
        close = _close([100.0, 101.0, 99.0])
        result = run_backtest(preds, close)
        expected = {"close", "asset_return", "prediction", "position",
                    "strategy_return", "cumulative", "buy_and_hold"}
        assert expected.issubset(set(result.columns))

    def test_no_nans_in_result(self):
        preds = _preds([1, 0, 1, 0, 1])
        close = _close([100.0, 101.0, 99.0, 102.0, 100.0])
        result = run_backtest(preds, close)
        assert not result.isna().any().any()

    def test_result_is_dataframe(self):
        preds = _preds([1, 0, 1])
        close = _close([100.0, 101.0, 99.0])
        result = run_backtest(preds, close)
        assert isinstance(result, pd.DataFrame)

    def test_no_common_dates_raises(self):
        preds = pd.Series([1, 0], index=_dates(2, "2021-01-04"))
        close = pd.Series([100.0, 101.0], index=_dates(2, "2022-01-04"))
        with pytest.raises(ValueError, match="no common dates"):
            run_backtest(preds, close)


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

class TestCosts:
    def test_no_cost_when_position_unchanged(self):
        """If position never changes, costs_bps > 0 has no effect."""
        preds = _preds([1, 1, 1, 1])
        close = _close([100.0, 101.0, 102.0, 103.0])
        result_free = run_backtest(preds, close, costs_bps=0)
        result_cost = run_backtest(preds, close, costs_bps=10)
        # No turnover after the first position, so returns differ only on
        # the first tradeable row (where position changes from NaN → 1).
        # After that row both should be identical.
        np.testing.assert_allclose(
            result_free["strategy_return"].iloc[1:].values,
            result_cost["strategy_return"].iloc[1:].values,
        )

    def test_cost_reduces_return_on_position_change(self):
        """A trade (position flip) incurs a cost that reduces strategy return."""
        # preds = [1, 0, 0, 0]
        # position after shift = [NaN, 1, 0, 0]
        # After dropping NaN row: position = [1, 0, 0] at result indices [0,1,2]
        # The flip from 1 → 0 happens at result index 1.
        preds = _preds([1, 0, 0, 0])
        close = _close([100.0, 110.0, 120.0, 130.0])
        result_free = run_backtest(preds, close, costs_bps=0)
        result_cost = run_backtest(preds, close, costs_bps=100)  # 1 % per trade
        # At the flip day (result index 1), position=0 so free return=0.
        # With costs, we pay 1 % for the position change → return < 0.
        flip_day = 1
        assert result_cost["strategy_return"].iloc[flip_day] < \
               result_free["strategy_return"].iloc[flip_day]

    def test_costs_bps_zero_is_default(self):
        """Explicit costs_bps=0 must produce identical results to no argument."""
        preds = _preds([1, 0, 1, 0])
        close = _close([100.0, 101.0, 99.0, 102.0])
        result_default = run_backtest(preds, close)
        result_zero = run_backtest(preds, close, costs_bps=0.0)
        pd.testing.assert_frame_equal(result_default, result_zero)


# ---------------------------------------------------------------------------
# Holding period
# ---------------------------------------------------------------------------

class TestHoldingPeriod:
    """
    Tests for the holding_period parameter.

    The no-lookahead guarantee must hold for every holding period:
    a signal on day t must not influence the position until day t+1.
    """

    def test_holding_period_1_matches_default_output(self):
        """
        holding_period=1 must produce bit-for-bit identical output to the
        default call (holding_period not specified).  This is the backward-
        compatibility guarantee.
        """
        preds = _preds([1, 0, 1, 1, 0, 0, 1])
        close = _close([100.0, 101.0, 99.0, 102.0, 101.0, 103.0, 105.0])
        result_default = run_backtest(preds, close)
        result_hp1     = run_backtest(preds, close, holding_period=1)
        pd.testing.assert_frame_equal(result_default, result_hp1)

    def test_holding_period_5_creates_exposure_for_five_days(self):
        """
        A single signal on day k must produce position=1 on days
        k+1, k+2, k+3, k+4, k+5 and position=0 everywhere else.

        Prediction series (10 days): [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                                              ^--- signal on day 2

        Expected positions in result (days 1-9, day 0 is dropped):
            day 1: 0  (signal not yet seen)
            day 2: 0  (signal on THIS day — not yet executable)
            day 3: 1  ← day 2 signal, +1
            day 4: 1  ← day 2 signal, +2
            day 5: 1  ← day 2 signal, +3
            day 6: 1  ← day 2 signal, +4
            day 7: 1  ← day 2 signal, +5
            day 8: 0  (signal expired)
            day 9: 0
        """
        preds = _preds([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        close = _close([100.0] * 10)
        result = run_backtest(preds, close, holding_period=5)

        expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        actual   = result["position"].tolist()
        assert actual == expected, (
            f"Expected 5-day exposure after signal.\n"
            f"  expected: {expected}\n"
            f"  got:      {actual}"
        )

    def test_no_position_taken_on_signal_day(self):
        """
        The position on the day a signal is formed must still be based on
        the *previous* day's prediction — the 1-day shift is always applied
        first, even for long holding periods.

        With predictions = [0, 1, 0, ...], the signal is on day 1.
        Position on day 1 = prediction[day 0] = 0  (not prediction[day 1]).
        Position on day 2 = 1  (first day of exposure).
        """
        preds = _preds([0, 1, 0, 0, 0, 0, 0, 0])
        close = _close([100.0] * 8)
        result = run_backtest(preds, close, holding_period=5)

        # Result starts at day 1.  position[day 1] must be 0 (signal day).
        assert result["position"].iloc[0] == 0.0, (
            "Position on signal day must be 0 — lookahead detected."
        )
        # position[day 2] must be 1 (first execution day).
        assert result["position"].iloc[1] == 1.0, (
            "Position on day after signal must be 1."
        )

    def test_overlapping_signals_do_not_exceed_position_of_one(self):
        """
        Consecutive signals must not stack into a position > 1.0.

        With predictions = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] every day is a
        new signal, yet the maximum position must stay at exactly 1.0.
        This verifies that the rolling-max approach never creates leverage.
        """
        preds = _preds([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        close = _close([100.0] * 10)
        result = run_backtest(preds, close, holding_period=5)

        assert result["position"].max() <= 1.0, (
            "Overlapping signals must not push position above 1.0."
        )
        assert result["position"].min() >= 0.0, (
            "Position must never be negative."
        )

    def test_invalid_holding_period_raises(self):
        """holding_period < 1 must raise ValueError."""
        preds = _preds([1, 0, 1])
        close = _close([100.0, 101.0, 99.0])
        with pytest.raises(ValueError, match="holding_period"):
            run_backtest(preds, close, holding_period=0)
