"""
Tests for target builders.

Key invariants:
  - target[t] is derived from close[t+1], never close[t] or earlier.
  - The last row of every target is NaN (no t+1 observation).
  - No target value should appear in the feature set (they use a forward shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.targets import next_day_direction, next_day_return


def _series(values, start="2020-01-01") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, dtype=float, name="close")


# ---------------------------------------------------------------------------
# next_day_direction
# ---------------------------------------------------------------------------

class TestNextDayDirection:
    def test_known_values(self):
        #        0    1    2    3
        close = _series([100, 110, 105, 108])
        y = next_day_direction(close)

        assert y.iloc[0] == 1.0   # 110 > 100
        assert y.iloc[1] == 0.0   # 105 < 110
        assert y.iloc[2] == 1.0   # 108 > 105
        assert pd.isna(y.iloc[3]) # no next day

    def test_last_row_is_always_nan(self):
        for n in [3, 10, 50]:
            close = _series(list(range(100, 100 + n)))
            assert pd.isna(next_day_direction(close).iloc[-1])

    def test_output_is_binary(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 100))
        close = _series(prices.tolist())
        y = next_day_direction(close).dropna()
        assert set(y.unique()).issubset({0.0, 1.0})

    def test_equal_close_gives_zero(self):
        # close_{t+1} == close_t → not strictly greater → direction = 0
        close = _series([100, 100, 100])
        y = next_day_direction(close)
        assert y.iloc[0] == 0.0
        assert y.iloc[1] == 0.0

    def test_index_preserved(self):
        close = _series([100, 110, 105])
        y = next_day_direction(close)
        assert y.index.equals(close.index)

    def test_forward_shift_not_current(self):
        """
        The direction label for row 0 must encode what happens on row 1,
        NOT what happened on row 0.  We verify by checking sign consistency.
        """
        close = _series([100, 80, 120])
        y = next_day_direction(close)
        # Row 0→1: price drops, so direction[0] = 0
        assert y.iloc[0] == 0.0
        # Row 1→2: price rises, so direction[1] = 1
        assert y.iloc[1] == 1.0


# ---------------------------------------------------------------------------
# next_day_return
# ---------------------------------------------------------------------------

class TestNextDayReturn:
    def test_known_values(self):
        close = _series([100, 110, 99])
        y = next_day_return(close)
        assert y.iloc[0] == pytest.approx(0.10)   # (110-100)/100
        assert y.iloc[1] == pytest.approx(-0.10)  # (99-110)/110
        assert pd.isna(y.iloc[2])

    def test_last_row_is_always_nan(self):
        for n in [3, 10, 50]:
            close = _series(list(range(100, 100 + n)))
            assert pd.isna(next_day_return(close).iloc[-1])

    def test_index_preserved(self):
        close = _series([100, 110, 105])
        y = next_day_return(close)
        assert y.index.equals(close.index)

    def test_sum_of_returns_consistent_with_total_move(self):
        """
        Compounding all next_day_returns should reproduce the total price move.
        (Within floating-point tolerance.)
        """
        close = _series([100.0, 110.0, 121.0, 133.1])
        y = next_day_return(close).dropna()
        # Product of (1 + r) factors: 1.1 * 1.1 * 1.1 ≈ 1.331
        product = (1 + y).prod()
        assert product == pytest.approx(133.1 / 100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Shared: no target leaks into features
# ---------------------------------------------------------------------------

class TestTargetDoesNotLeakIntoFeatures:
    """
    The target at row t uses close[t+1].  A feature at row t must not
    use close[t+1].  We verify this indirectly: features at row t must
    be identical whether or not close[t+1] is included in the series.
    """

    def test_direction_target_is_only_forward_looking(self):
        from tsml.features.transformers import daily_returns

        close = _series([100, 110, 105, 108, 120])
        feature_full = daily_returns(close)
        feature_trunc = daily_returns(close.iloc[:3])

        # Feature at row 2 should not change when rows 3 and 4 are removed
        assert feature_full.iloc[2] == pytest.approx(feature_trunc.iloc[2])

    def test_return_target_is_only_forward_looking(self):
        from tsml.features.transformers import rolling_mean

        close = _series(list(range(100, 115)))
        feat_full = rolling_mean(close, window=5)
        feat_trunc = rolling_mean(close.iloc[:10], window=5)

        assert feat_full.iloc[9] == pytest.approx(feat_trunc.iloc[9])
