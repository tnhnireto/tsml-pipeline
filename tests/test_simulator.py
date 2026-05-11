"""
Tests for simulate().

All tests use a stub DataLoader backed by deterministic synthetic price data
so no network calls are made.  A fast splitter (2 splits, 200-row min-train,
50-row test) keeps runtime low while still producing valid OOS probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import DataLoader
from tsml.models.baselines import LogisticRegressionModel
from tsml.portfolio.simulator import (
    SimulationResult,
    _build_proba_matrix,
    _compute_cost,
    _weekly_rebalance_dates,
    simulate,
)
from tsml.validation import WalkForwardSplit


# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n))
    df = pd.DataFrame(
        {
            "open":   close * 0.999,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class StubLoader(DataLoader):
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol not in self._data:
            raise ValueError(f"No stub data for '{symbol}'.")
        return self._data[symbol]


@pytest.fixture()
def fast_splitter() -> WalkForwardSplit:
    # Minimum rows needed: min_train + gap + test + test = 200+1+50+50 = 301
    return WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)


@pytest.fixture()
def two_symbol_loader() -> StubLoader:
    return StubLoader(
        {
            "AAA": _make_ohlcv(400, seed=1),
            "BBB": _make_ohlcv(400, seed=2),
        }
    )


@pytest.fixture()
def three_symbol_loader() -> StubLoader:
    return StubLoader(
        {
            "AAA": _make_ohlcv(400, seed=1),
            "BBB": _make_ohlcv(400, seed=2),
            "CCC": _make_ohlcv(400, seed=3),
        }
    )


# ---------------------------------------------------------------------------
# Return shape and types
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_simulation_result(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert isinstance(result, SimulationResult)

    def test_equity_curve_is_series(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert isinstance(result.equity_curve, pd.Series)

    def test_trades_log_is_dataframe(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert isinstance(result.trades_log, pd.DataFrame)

    def test_trades_log_columns(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert list(result.trades_log.columns) == ["date", "symbol", "action", "score"]


# ---------------------------------------------------------------------------
# Equity curve properties
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_starts_at_initial_capital(self, fast_splitter, two_symbol_loader):
        cap = 10_000.0
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            initial_capital=cap,
            loader=two_symbol_loader,
        )
        assert result.equity_curve.iloc[0] == pytest.approx(cap)

    def test_length_equals_trading_days(self, fast_splitter, two_symbol_loader):
        data = two_symbol_loader._data
        expected_len = len(data["AAA"])   # both symbols share the same trading days
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert len(result.equity_curve) == expected_len

    def test_equity_never_negative(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            costs_bps=0.0,
            loader=two_symbol_loader,
        )
        assert (result.equity_curve >= 0.0).all()

    def test_name_is_portfolio_value(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert result.equity_curve.name == "portfolio_value"


# ---------------------------------------------------------------------------
# Trade log properties
# ---------------------------------------------------------------------------

class TestTradesLog:
    def test_no_hold_actions_in_log(self, fast_splitter, two_symbol_loader):
        """Holds are not trades — they must not appear in the log."""
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        if not result.trades_log.empty:
            assert "hold" not in result.trades_log["action"].values

    def test_actions_only_buy_or_sell(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        if not result.trades_log.empty:
            assert set(result.trades_log["action"]).issubset({"buy", "sell"})

    def test_scores_are_floats(self, fast_splitter, two_symbol_loader):
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        if not result.trades_log.empty:
            assert result.trades_log["score"].dtype == float


# ---------------------------------------------------------------------------
# Rebalancing behaviour
# ---------------------------------------------------------------------------

class TestRebalancing:
    def test_rebalance_count_is_roughly_weekly(self, fast_splitter, two_symbol_loader):
        """400 trading days ≈ 80 weeks → roughly 80 rebalance events."""
        result = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        if result.trades_log.empty:
            return  # model may not trade often under strict min_score
        n_weeks = 400 / 5  # approx
        n_rebalance_days = result.trades_log["date"].nunique()
        # Very loose bound — just checks rebalance is not happening every day
        assert n_rebalance_days <= n_weeks + 5


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_unknown_symbol_is_skipped(self, fast_splitter, two_symbol_loader):
        """An unknown symbol should be skipped without crashing."""
        result = simulate(
            ["AAA", "MISSING"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert isinstance(result, SimulationResult)
        assert len(result.equity_curve) > 0

    def test_symbol_too_few_rows_is_skipped(self, fast_splitter):
        loader = StubLoader(
            {
                "TINY": _make_ohlcv(50, seed=9),   # not enough for any fold
                "BIG":  _make_ohlcv(400, seed=5),
            }
        )
        result = simulate(
            ["TINY", "BIG"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=loader,
        )
        assert len(result.equity_curve) > 0

    def test_all_fail_returns_empty_result(self, fast_splitter):
        loader = StubLoader({"X": _make_ohlcv(10, seed=0)})
        result = simulate(
            ["X"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=loader,
        )
        assert result.equity_curve.empty
        assert result.trades_log.empty

    def test_failure_logged_to_stderr(self, fast_splitter, two_symbol_loader, capsys):
        simulate(
            ["AAA", "NO_SUCH_SYMBOL"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            loader=two_symbol_loader,
        )
        assert "NO_SUCH_SYMBOL" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    def test_higher_costs_reduce_final_value(self, fast_splitter, two_symbol_loader):
        """Simulation with 20 bps costs should produce lower final value than 0 bps."""
        low = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            costs_bps=0.0,
            min_score=0.0,   # force buying so costs are definitely paid
            loader=two_symbol_loader,
        )
        high = simulate(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            costs_bps=20.0,
            min_score=0.0,
            loader=two_symbol_loader,
        )
        assert high.equity_curve.iloc[-1] <= low.equity_curve.iloc[-1]

    def test_no_cost_result_unchanged_from_zero(self, fast_splitter, two_symbol_loader):
        """Two runs with costs_bps=0 and identical params should yield the same curve."""
        kw = dict(
            model=LogisticRegressionModel(),
            splitter=fast_splitter,
            start_date="2015-01-01",
            end_date="2023-12-31",
            costs_bps=0.0,
            loader=two_symbol_loader,
        )
        r1 = simulate(["AAA", "BBB"], **kw)
        r2 = simulate(["AAA", "BBB"], **kw)
        pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)


# ---------------------------------------------------------------------------
# Unsupported rebalance frequency
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unsupported_frequency_raises(self, fast_splitter, two_symbol_loader):
        with pytest.raises(ValueError, match="weekly"):
            simulate(
                ["AAA"],
                model=LogisticRegressionModel(),
                splitter=fast_splitter,
                start_date="2015-01-01",
                end_date="2023-12-31",
                rebalance_frequency="daily",
                loader=two_symbol_loader,
            )


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestWeeklyRebalanceDates:
    def test_empty_input_returns_empty(self):
        assert _weekly_rebalance_dates(pd.DatetimeIndex([])) == frozenset()

    def test_returns_first_day_of_each_week(self):
        # 10 consecutive business days = 2 ISO weeks
        days = pd.bdate_range("2024-01-01", periods=10, freq="B", tz="UTC")
        dates = _weekly_rebalance_dates(days)
        # Mon 2024-01-01 and Mon 2024-01-08 should be the rebalance dates
        assert pd.Timestamp("2024-01-01", tz="UTC") in dates
        assert pd.Timestamp("2024-01-08", tz="UTC") in dates

    def test_count_equals_number_of_weeks(self):
        # 5 weeks of trading days (25 days) → 5 rebalance dates
        days = pd.bdate_range("2024-01-01", periods=25, freq="B", tz="UTC")
        dates = _weekly_rebalance_dates(days)
        assert len(dates) == 5


class TestComputeCost:
    def test_no_change_zero_cost(self):
        pos = {"A", "B"}
        assert _compute_cost(pos, pos, 10_000.0, 10.0) == 0.0

    def test_from_empty_to_n_positions_costs_full_notional(self):
        """Entering 2 new positions from cash: cost = 1.0 * pv * bps * 1e-4."""
        cost = _compute_cost(set(), {"A", "B"}, 10_000.0, 10.0)
        # one-way turnover = 1.0 (buy 100 % of portfolio)
        assert cost == pytest.approx(10_000.0 * 10.0 * 1e-4)

    def test_zero_costs_bps_gives_zero_cost(self):
        assert _compute_cost({"A"}, {"A", "B"}, 10_000.0, 0.0) == 0.0

    def test_fully_exit_all_positions(self):
        """Selling all positions should cost the same as buying all positions."""
        buy_cost  = _compute_cost(set(),       {"A", "B"}, 10_000.0, 10.0)
        sell_cost = _compute_cost({"A", "B"}, set(),       10_000.0, 10.0)
        assert buy_cost == pytest.approx(sell_cost)


class TestBuildProbaMatrix:
    def test_forward_fill_propagates_last_known_value(self):
        days = pd.bdate_range("2024-01-01", periods=5, freq="B", tz="UTC")
        # Probability only on day 0 and day 2
        s = pd.Series({days[0]: 0.6, days[2]: 0.7}, name="proba_up")
        matrix = _build_proba_matrix({"A": s}, days)
        # day 1 should be forward-filled to 0.6
        assert matrix.loc[days[1], "A"] == pytest.approx(0.6)
        # day 3 should be forward-filled to 0.7
        assert matrix.loc[days[3], "A"] == pytest.approx(0.7)

    def test_before_first_probability_is_nan(self):
        days = pd.bdate_range("2024-01-01", periods=5, freq="B", tz="UTC")
        s = pd.Series({days[3]: 0.65}, name="proba_up")
        matrix = _build_proba_matrix({"A": s}, days)
        assert np.isnan(matrix.loc[days[0], "A"])

    def test_empty_input_returns_empty_dataframe(self):
        days = pd.bdate_range("2024-01-01", periods=5, freq="B", tz="UTC")
        matrix = _build_proba_matrix({}, days)
        assert matrix.empty
        assert len(matrix) == 5
