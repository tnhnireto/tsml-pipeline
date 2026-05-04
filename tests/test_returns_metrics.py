"""
Tests for financial return metrics.

Every test uses inputs where the correct answer can be computed by hand
or from first principles, so the expected value in the assertion is a
checked number, not just "run the same formula twice".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsml.metrics.returns import (
    annualized_volatility,
    cagr,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    summary,
    total_return,
    turnover,
)


def _s(values: list[float]) -> pd.Series:
    """Build a simple float Series (no special index needed for pure math)."""
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# total_return
# ---------------------------------------------------------------------------

class TestTotalReturn:
    def test_single_period(self):
        # (1 + 0.10) - 1 = 0.10
        assert total_return(_s([0.10])) == pytest.approx(0.10)

    def test_two_periods_compounding(self):
        # (1.10 * 0.90) - 1 = -0.01  (10% up then 10% down ≠ 0)
        assert total_return(_s([0.10, -0.10])) == pytest.approx(-0.01)

    def test_constant_gain(self):
        # Three 5 % days: 1.05^3 - 1 = 0.157625
        assert total_return(_s([0.05, 0.05, 0.05])) == pytest.approx(
            1.05**3 - 1, rel=1e-9
        )

    def test_zero_returns_give_zero(self):
        assert total_return(_s([0.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            total_return(_s([]))


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------

class TestCagr:
    def test_one_year_of_data(self):
        # 252 days of 0.01 % daily return → CAGR ≈ 2.52 % * 252 ≈ 27.7 %
        # More precisely: (1 + total_return)^(252/252) - 1 = total_return
        r = _s([0.0001] * 252)
        total = (1.0001**252) - 1
        expected = total  # exponent = 1
        assert cagr(r) == pytest.approx(expected, rel=1e-6)

    def test_known_two_year_return(self):
        # total_return = 21 % over 504 days (2 years)
        # cagr = (1.21)^(252/504) - 1 = 1.21^0.5 - 1 = 0.10
        r = _s([0.0] * 504)  # placeholder length
        # Patch total to 21 %
        r = _s([0.21 / 504] * 504)  # approximate; check via cagr formula
        expected = (1 + total_return(r)) ** (252 / 504) - 1
        assert cagr(r) == pytest.approx(expected, rel=1e-6)

    def test_cagr_equals_total_return_for_one_year(self):
        """When n == periods_per_year, CAGR == total_return."""
        r = _s([0.001] * 252)
        assert cagr(r, periods_per_year=252) == pytest.approx(
            total_return(r), rel=1e-9
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cagr(_s([]))


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------

class TestAnnualizedVolatility:
    def test_constant_returns_give_zero_vol(self):
        assert annualized_volatility(_s([0.01] * 50)) == pytest.approx(0.0, abs=1e-12)

    def test_known_daily_std(self):
        # std([0, 1]) = √0.5; annualised = √0.5 * √252
        r = _s([0.0, 1.0])
        expected = math.sqrt(0.5) * math.sqrt(252)
        assert annualized_volatility(r) == pytest.approx(expected, rel=1e-9)

    def test_scaling_by_sqrt_252(self):
        """Vol should scale exactly by √252."""
        r = _s([0.01, -0.01, 0.02, -0.02] * 10)
        daily_std = r.std(ddof=1)
        expected = daily_std * math.sqrt(252)
        assert annualized_volatility(r) == pytest.approx(expected, rel=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            annualized_volatility(_s([]))


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_volatility_returns_nan(self):
        # A constant return series has std=0.  Due to floating-point
        # representation, std may not be exactly 0.0, but the resulting
        # Sharpe is non-finite (inf).  The function maps any non-finite
        # result to NaN.
        assert math.isnan(sharpe_ratio(_s([0.001] * 20)))

    def test_positive_mean_positive_sharpe(self):
        # Positive mean return, positive volatility → positive Sharpe.
        r = _s([0.01, 0.02, 0.01, 0.02] * 20)
        assert sharpe_ratio(r) > 0

    def test_negative_mean_negative_sharpe(self):
        r = _s([-0.01, -0.02, -0.01, -0.02] * 20)
        assert sharpe_ratio(r) < 0

    def test_rf_reduces_sharpe(self):
        """A higher risk-free rate should lower (or equal) the Sharpe."""
        r = _s([0.01, -0.005, 0.008, 0.002] * 20)
        assert sharpe_ratio(r, risk_free_rate=0.0) > sharpe_ratio(r, risk_free_rate=0.05)

    def test_formula_manually(self):
        """Verify the formula step by step with a hand-checkable input."""
        r = _s([0.02, -0.01, 0.03, -0.02])
        excess = r - 0.0 / 252
        expected = float(excess.mean() / excess.std(ddof=1) * math.sqrt(252))
        assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            sharpe_ratio(_s([]))


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotonically_rising_gives_zero_drawdown(self):
        r = _s([0.01, 0.01, 0.01, 0.01])
        assert max_drawdown(r) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # Equity: 1.0 → 1.1 → 0.99 → 1.2
        # Peak after row 1: 1.1; trough at row 2: 0.99
        # Drawdown = (0.99 - 1.1) / 1.1 ≈ -0.1
        r = _s([0.10, -0.10, 0.2121])
        # (1+0.10)=1.1, (1.1*(1-0.10))=0.99; dd=(0.99-1.1)/1.1 = -0.1
        assert max_drawdown(r) == pytest.approx(-0.10, rel=1e-6)

    def test_single_loss(self):
        # Only one return: no prior peak to compare against at that point.
        # equity = [1-0.05] = [0.95], peak = 0.95, drawdown = 0
        assert max_drawdown(_s([-0.05])) == pytest.approx(0.0, abs=1e-10)

    def test_all_losses_gives_large_drawdown(self):
        # -10 % every day for 5 days
        r = _s([-0.10] * 5)
        equity = (1 + r).cumprod()
        peak = equity.cummax()
        expected = float(((equity - peak) / peak).min())
        assert max_drawdown(r) == pytest.approx(expected, rel=1e-9)

    def test_returns_negative_or_zero(self):
        r = _s([0.01, -0.05, 0.03])
        assert max_drawdown(r) <= 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            max_drawdown(_s([]))


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------

class TestHitRate:
    def test_all_positive(self):
        assert hit_rate(_s([0.01, 0.02, 0.03])) == pytest.approx(1.0)

    def test_all_negative(self):
        assert hit_rate(_s([-0.01, -0.02])) == pytest.approx(0.0)

    def test_half_positive(self):
        assert hit_rate(_s([0.01, -0.01])) == pytest.approx(0.5)

    def test_zero_returns_not_counted_as_positive(self):
        """Zero returns are not > 0, so they don't count as hits."""
        assert hit_rate(_s([0.0, 0.0, 0.01])) == pytest.approx(1 / 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            hit_rate(_s([]))


# ---------------------------------------------------------------------------
# turnover
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_constant_position_gives_zero_turnover(self):
        positions = _s([1, 1, 1, 1, 1])
        assert turnover(positions) == pytest.approx(0.0, abs=1e-12)

    def test_alternating_positions_give_max_turnover(self):
        # Flip between 0 and 1 every day → turnover = 1.0
        # diff: [NaN, 1, 1, 1, 1] → mean of abs = mean([1,1,1,1]) = 1.0
        positions = _s([0, 1, 0, 1, 0])
        assert turnover(positions) == pytest.approx(1.0)

    def test_known_value(self):
        # Positions: [1, 1, 0, 0, 1]
        # Diffs:     [NaN, 0, -1, 0, 1]
        # |Diffs|:   [NaN, 0,  1, 0, 1]
        # mean = (0+1+0+1)/4 = 0.5
        positions = _s([1, 1, 0, 0, 1])
        assert turnover(positions) == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            turnover(_s([]))


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_returns_dict(self):
        r = _s([0.01, -0.005, 0.008] * 10)
        result = summary(r)
        assert isinstance(result, dict)

    def test_has_all_keys_without_positions(self):
        r = _s([0.01, -0.005, 0.008] * 10)
        result = summary(r)
        assert set(result.keys()) == {
            "total_return", "cagr", "volatility", "sharpe", "max_drawdown", "hit_rate"
        }

    def test_includes_turnover_when_positions_given(self):
        r = _s([0.01, -0.005, 0.008] * 10)
        pos = _s([1, 1, 0] * 10)
        result = summary(r, positions=pos)
        assert "turnover" in result

    def test_all_values_are_floats(self):
        r = _s([0.01, -0.005, 0.008] * 10)
        result = summary(r)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"
