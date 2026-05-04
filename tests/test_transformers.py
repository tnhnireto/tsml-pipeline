"""
Tests for individual feature transformers.

Tests are grouped by transformer.  Each group checks:
  - basic correctness (known input → expected output)
  - NaN placement (warmup rows are NaN, not silently zero)
  - the no-future-data invariant (leakage test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.transformers import (
    daily_returns,
    lagged_returns,
    log_returns,
    rolling_mean,
    rolling_volatility,
    rsi,
    sma_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(values, start="2020-01-01") -> pd.Series:
    """Build a UTC-indexed Series from a list of floats."""
    index = pd.bdate_range(start, periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, dtype=float, name="close")


def _leakage_check(transformer, close: pd.Series, row: int) -> None:
    """
    Assert that transformer(close)[row] is identical whether computed on
    the full series or on close[:row+1].

    This is the fundamental leakage invariant: the value at time t must
    not change when future data (t+1, t+2, …) is added or removed.
    """
    full_result = transformer(close)
    truncated_result = transformer(close.iloc[: row + 1])

    full_val = full_result.iloc[row]
    trunc_val = truncated_result.iloc[row]

    if pd.isna(full_val) and pd.isna(trunc_val):
        return  # both NaN: the warmup period, invariant still holds

    assert full_val == pytest.approx(trunc_val, rel=1e-9), (
        f"Leakage detected at row {row}: "
        f"full={full_val}, truncated={trunc_val}"
    )


# ---------------------------------------------------------------------------
# daily_returns
# ---------------------------------------------------------------------------

class TestDailyReturns:
    def test_known_values(self):
        close = _series([100, 110, 99])
        r = daily_returns(close)
        assert r.iloc[0] is np.nan or pd.isna(r.iloc[0])
        assert r.iloc[1] == pytest.approx(0.10)   # +10 %
        assert r.iloc[2] == pytest.approx(-0.10)  # -10 %

    def test_first_row_is_nan(self):
        assert pd.isna(daily_returns(_series([100, 101])).iloc[0])

    def test_leakage(self):
        close = _series(list(range(100, 130)))
        for row in [5, 15, 25]:
            _leakage_check(daily_returns, close, row)

    def test_flat_prices_give_zero_returns(self):
        close = _series([100.0] * 10)
        assert (daily_returns(close).dropna() == 0).all()


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

class TestLogReturns:
    def test_known_value(self):
        close = _series([100, 100 * np.e])
        lr = log_returns(close)
        assert lr.iloc[1] == pytest.approx(1.0)

    def test_first_row_is_nan(self):
        assert pd.isna(log_returns(_series([100, 101])).iloc[0])

    def test_leakage(self):
        close = _series(list(range(100, 130)))
        for row in [5, 15, 25]:
            _leakage_check(log_returns, close, row)

    def test_approximately_equals_pct_return_for_small_moves(self):
        """ln(1 + r) ≈ r for small r."""
        close = _series([100.0, 100.5, 101.0])
        lr = log_returns(close).dropna()
        pr = daily_returns(close).dropna()
        assert (abs(lr - pr) < 0.001).all()


# ---------------------------------------------------------------------------
# lagged_returns
# ---------------------------------------------------------------------------

class TestLaggedReturns:
    def test_lag1_shifts_returns_by_one(self):
        close = _series([100, 110, 99, 105])
        r = daily_returns(close)
        lag1 = lagged_returns(close, lag=1)
        # lag1[2] should equal r[1]
        assert lag1.iloc[2] == pytest.approx(r.iloc[1])

    def test_lag2_shifts_by_two(self):
        close = _series([100, 110, 99, 105, 108])
        r = daily_returns(close)
        lag2 = lagged_returns(close, lag=2)
        assert lag2.iloc[3] == pytest.approx(r.iloc[1])

    def test_invalid_lag_raises(self):
        close = _series([100, 110])
        with pytest.raises(ValueError, match="lag must be"):
            lagged_returns(close, lag=0)

    def test_leakage(self):
        close = _series(list(range(100, 130)))
        for row in [5, 15, 25]:
            _leakage_check(lambda c: lagged_returns(c, lag=1), close, row)


# ---------------------------------------------------------------------------
# rolling_mean
# ---------------------------------------------------------------------------

class TestRollingMean:
    def test_known_value(self):
        close = _series([10, 20, 30, 40, 50])
        rm = rolling_mean(close, window=3)
        # First two rows are NaN (warmup)
        assert pd.isna(rm.iloc[0])
        assert pd.isna(rm.iloc[1])
        assert rm.iloc[2] == pytest.approx(20.0)  # mean(10,20,30)
        assert rm.iloc[3] == pytest.approx(30.0)  # mean(20,30,40)

    def test_partial_windows_are_nan(self):
        close = _series(list(range(1, 11)))
        rm = rolling_mean(close, window=5)
        assert rm.iloc[:4].isna().all()

    def test_leakage(self):
        close = _series(list(range(100, 130)))
        for row in [10, 15, 25]:
            _leakage_check(lambda c: rolling_mean(c, window=5), close, row)


# ---------------------------------------------------------------------------
# rolling_volatility
# ---------------------------------------------------------------------------

class TestRollingVolatility:
    def test_constant_prices_give_zero_vol(self):
        close = _series([100.0] * 20)
        vol = rolling_volatility(close, window=5)
        assert (vol.dropna() == 0).all()

    def test_partial_windows_are_nan(self):
        close = _series(list(range(100, 120)))
        vol = rolling_volatility(close, window=5)
        # needs window returns → window+1 prices → window rows of NaN
        assert vol.iloc[:5].isna().all()

    def test_leakage(self):
        close = _series(list(range(100, 130)))
        for row in [10, 15, 25]:
            _leakage_check(lambda c: rolling_volatility(c, window=5), close, row)

    def test_higher_variance_gives_higher_vol(self):
        stable = _series([100, 101, 100, 101, 100, 101, 100])
        volatile = _series([100, 120, 80, 130, 70, 140, 60])
        assert rolling_volatility(volatile, window=5).dropna().mean() > \
               rolling_volatility(stable, window=5).dropna().mean()


# ---------------------------------------------------------------------------
# sma_ratio
# ---------------------------------------------------------------------------

class TestSmaRatio:
    def test_ratio_is_one_for_flat_prices(self):
        close = _series([100.0] * 30)
        ratio = sma_ratio(close, short_window=5, long_window=20).dropna()
        # pytest.approx doesn't broadcast over Series; compare via max deviation.
        assert abs(ratio - 1.0).max() == pytest.approx(0.0, abs=1e-9)

    def test_ratio_above_one_in_uptrend(self):
        # Steadily rising prices: short SMA > long SMA
        close = _series(list(range(100, 140)))
        ratio = sma_ratio(close, short_window=5, long_window=20)
        assert (ratio.dropna() > 1).all()

    def test_ratio_below_one_in_downtrend(self):
        close = _series(list(range(140, 100, -1)))
        ratio = sma_ratio(close, short_window=5, long_window=20)
        assert (ratio.dropna() < 1).all()

    def test_invalid_windows_raise(self):
        close = _series(list(range(100, 130)))
        with pytest.raises(ValueError, match="short_window"):
            sma_ratio(close, short_window=20, long_window=5)

    def test_leakage(self):
        close = _series(list(range(100, 130)))
        for row in [20, 25]:
            _leakage_check(
                lambda c: sma_ratio(c, short_window=5, long_window=20), close, row
            )


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRsi:
    def test_range_is_0_to_100(self):
        import numpy as np
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 100))
        close = _series(prices.tolist())
        result = rsi(close, window=14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_all_gains_gives_rsi_100(self):
        # Prices only go up → RSI should be 100.
        close = _series(list(range(100, 120)))
        result = rsi(close, window=5).dropna()
        assert (result == pytest.approx(100.0)).all()

    def test_all_losses_gives_rsi_0(self):
        # Prices only go down → avg_gain = 0 → RSI = 0.
        close = _series(list(range(120, 100, -1)))
        result = rsi(close, window=5).dropna()
        assert result.abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_warmup_rows_are_nan(self):
        close = _series(list(range(100, 130)))
        result = rsi(close, window=14)
        assert result.iloc[:14].isna().all()

    def test_leakage(self):
        import numpy as np
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, 60))
        close = _series(prices.tolist())
        for row in [20, 30, 50]:
            _leakage_check(lambda c: rsi(c, window=14), close, row)
