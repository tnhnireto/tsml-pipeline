"""
Tests for the evaluate() helper.

These are integration tests: they verify that evaluate() correctly
connects the ML metrics, return metrics, and backtest DataFrame into
one consistent report structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.backtest.engine import run_backtest
from tsml.pipelines.evaluate import evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=n, freq="B", tz="UTC")


def _make_inputs(n: int = 20):
    """
    Build a consistent set of (predictions, y_true, backtest_result).

    Prices: steadily rising so the backtest result has no edge cases.
    """
    rng = np.random.default_rng(0)
    dates = _dates(n + 1)  # one extra for the close shift

    close = pd.Series(
        300.0 + np.cumsum(rng.normal(0, 1, n + 1)),
        index=dates,
        name="close",
    )

    pred_dates = dates[1:]                   # predictions start on day 1
    predictions = pd.Series(
        rng.integers(0, 2, n).astype(float),
        index=pred_dates,
        name="prediction",
    )
    y_true = pd.Series(
        rng.integers(0, 2, n).astype(float),
        index=pred_dates,
        name="target_direction",
    )

    bt = run_backtest(predictions, close)

    # Trim predictions and y_true to match the backtest result index
    # (run_backtest drops the first row).
    common = predictions.index.intersection(bt.index)
    predictions = predictions.loc[common]
    y_true = y_true.loc[common]

    return predictions, y_true, bt


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestEvaluateStructure:
    def test_returns_dict(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert isinstance(result, dict)

    def test_has_three_top_level_keys(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert set(result.keys()) == {"ml", "strategy", "buy_and_hold"}

    def test_ml_keys(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert set(result["ml"].keys()) == {"accuracy", "precision", "recall"}

    def test_strategy_keys(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        expected = {
            "total_return", "cagr", "volatility", "sharpe",
            "max_drawdown", "hit_rate", "turnover",
        }
        assert expected.issubset(set(result["strategy"].keys()))

    def test_buy_and_hold_keys(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        expected = {
            "total_return", "cagr", "volatility", "sharpe",
            "max_drawdown", "hit_rate",
        }
        assert expected.issubset(set(result["buy_and_hold"].keys()))

    def test_buy_and_hold_has_no_turnover(self):
        """Buy-and-hold is passive: no turnover key expected."""
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert "turnover" not in result["buy_and_hold"]

    def test_all_ml_values_are_floats(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        for k, v in result["ml"].items():
            assert isinstance(v, float), f"ml.{k} is not float"

    def test_all_strategy_values_are_floats(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        for k, v in result["strategy"].items():
            assert isinstance(v, float), f"strategy.{k} is not float"


# ---------------------------------------------------------------------------
# Correctness: spot-check specific values
# ---------------------------------------------------------------------------

class TestEvaluateCorrectness:
    def test_ml_accuracy_in_range(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert 0.0 <= result["ml"]["accuracy"] <= 1.0

    def test_max_drawdown_is_non_positive(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert result["strategy"]["max_drawdown"] <= 0.0
        assert result["buy_and_hold"]["max_drawdown"] <= 0.0

    def test_hit_rate_in_range(self):
        preds, y_true, bt = _make_inputs()
        result = evaluate(preds, y_true, bt)
        assert 0.0 <= result["strategy"]["hit_rate"] <= 1.0
        assert 0.0 <= result["buy_and_hold"]["hit_rate"] <= 1.0

    def test_always_long_total_return_equals_buy_and_hold(self):
        """If we always hold long, strategy return == buy-and-hold return."""
        n = 30
        dates = _dates(n + 1)
        rng = np.random.default_rng(7)
        close = pd.Series(
            300.0 + np.cumsum(rng.normal(0, 1, n + 1)),
            index=dates,
            name="close",
        )
        # Always predict 1 → position always 1 (after shift) → = buy and hold
        predictions = pd.Series(np.ones(n), index=dates[1:], name="prediction")
        y_true = pd.Series(np.ones(n), index=dates[1:], name="target")
        bt = run_backtest(predictions, close)

        common = predictions.index.intersection(bt.index)
        result = evaluate(predictions.loc[common], y_true.loc[common], bt)

        # strategy total return should equal buy-and-hold total return
        assert result["strategy"]["total_return"] == pytest.approx(
            result["buy_and_hold"]["total_return"], rel=1e-6
        )

    def test_always_flat_strategy_has_zero_total_return(self):
        """If we always predict 0, position is always 0 → no return."""
        n = 30
        dates = _dates(n + 1)
        rng = np.random.default_rng(7)
        close = pd.Series(
            300.0 + np.cumsum(rng.normal(0, 1, n + 1)),
            index=dates,
            name="close",
        )
        predictions = pd.Series(np.zeros(n), index=dates[1:], name="prediction")
        y_true = pd.Series(np.zeros(n), index=dates[1:], name="target")
        bt = run_backtest(predictions, close)

        common = predictions.index.intersection(bt.index)
        result = evaluate(predictions.loc[common], y_true.loc[common], bt)

        assert result["strategy"]["total_return"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEvaluateErrors:
    def test_misaligned_indices_raise(self):
        preds, y_true, bt = _make_inputs()
        shifted_y = y_true.copy()
        shifted_y.index = _dates(len(y_true) + 5)[5:]  # shifted by 5 days
        with pytest.raises(ValueError, match="same index"):
            evaluate(preds, shifted_y, bt)

    def test_missing_column_raises(self):
        preds, y_true, bt = _make_inputs()
        bt_missing = bt.drop(columns=["strategy_return"])
        with pytest.raises(KeyError, match="strategy_return"):
            evaluate(preds, y_true, bt_missing)
