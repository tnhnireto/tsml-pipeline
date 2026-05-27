"""Tests for the weekly portfolio backtest (mirrors live signal workflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import DataLoader
from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.portfolio.simulator import (
    SimulationResult,
    _enrich_ranking_as_of,
    _prior_trading_day,
    simulate,
)
from tsml.portfolio.weekly_backtest import (
    compute_backtest_metrics,
    run_weekly_backtest,
)
from tsml.validation.splitters import WalkForwardSplit


# ---------------------------------------------------------------------------
# Helpers (same pattern as tests/test_simulator.py)
# ---------------------------------------------------------------------------

UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n))
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
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


def _universe_loader(symbols: list[str], n: int = 400) -> StubLoader:
    return StubLoader({sym: _make_ohlcv(n, seed=i + 1) for i, sym in enumerate(symbols)})


def _fast_splitter() -> WalkForwardSplit:
    return WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)


def _mock_simulate_result(
    symbols: list[str] | None = None,
    *,
    cash_buffer_pct: float = 0.05,
    top_n: int = 5,
    n: int = 400,
    turnover_control: bool = True,
) -> SimulationResult:
    syms = symbols or UNIVERSE
    loader = _universe_loader(syms, n=n)
    return simulate(
        syms,
        model=CalibratedLogisticRegressionModel(),
        splitter=_fast_splitter(),
        start_date="2020-01-01",
        end_date="2021-06-30",
        target="threshold",
        top_n=top_n,
        min_score=0.55,
        min_score_downtrend=0.62,
        cash_buffer_pct=cash_buffer_pct,
        costs_bps=5.0,
        initial_capital=100_000.0,
        loader=loader,
        turnover_control=turnover_control,
        buy_threshold=0.58,
        sell_threshold=0.52,
        min_hold_weeks=2,
    )


# ---------------------------------------------------------------------------
# No lookahead
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_enrich_ranking_uses_prior_trading_day(self):
        close = _make_ohlcv(250, seed=1)["close"]
        close_map = {"AAPL": close}
        idx = close.index
        ranking = pd.DataFrame({"symbol": ["AAPL"], "score": [0.70]})
        rebalance_date = idx[200]

        enriched = _enrich_ranking_as_of(ranking, close_map, rebalance_date, idx)
        prior = _prior_trading_day(idx, rebalance_date)
        assert prior is not None

        from tsml.portfolio.ranker import compute_context_as_of

        expected = compute_context_as_of(close, prior)
        assert enriched.iloc[0]["above_sma_200"] == expected["above_sma_200"]

    def test_context_does_not_use_same_day_close(self):
        close = _make_ohlcv(250, seed=2)["close"]
        close_map = {"AAPL": close}
        idx = close.index
        ranking = pd.DataFrame({"symbol": ["AAPL"], "score": [0.70]})
        rebalance_date = idx[200]
        prior = _prior_trading_day(idx, rebalance_date)
        assert prior is not None

        from tsml.portfolio.ranker import compute_context_as_of

        same_day = compute_context_as_of(close, rebalance_date)
        prior_day = compute_context_as_of(close, prior)
        enriched = _enrich_ranking_as_of(ranking, close_map, rebalance_date, idx)

        assert enriched.iloc[0]["above_sma_200"] == prior_day["above_sma_200"]
        if same_day["above_sma_200"] != prior_day["above_sma_200"]:
            assert enriched.iloc[0]["above_sma_200"] != same_day["above_sma_200"]


# ---------------------------------------------------------------------------
# Weekly rebalance dates
# ---------------------------------------------------------------------------


class TestWeeklyRebalanceDates:
    def test_trades_only_on_week_starts(self):
        result = _mock_simulate_result()

        if result.trades_log.empty:
            pytest.skip("No trades generated in synthetic run")

        rebalance_weeks: set[str] = set()
        for dt in result.trades_log["date"]:
            iso = dt.isocalendar()
            rebalance_weeks.add(f"{iso.year}-{iso.week:02d}")
            assert dt.dayofweek == 0 or dt == result.equity_curve.index[0]


# ---------------------------------------------------------------------------
# Max positions
# ---------------------------------------------------------------------------


class TestMaxPositions:
    def test_never_more_than_top_n_symbols_per_rebalance(self):
        result = _mock_simulate_result(top_n=5)

        holdings: set[str] = set()
        max_seen = 0
        for _, row in result.trades_log.sort_values("date").iterrows():
            if row["action"] == "buy":
                holdings.add(row["symbol"])
            elif row["action"] == "sell":
                holdings.discard(row["symbol"])
            max_seen = max(max_seen, len(holdings))

        assert max_seen <= 5


# ---------------------------------------------------------------------------
# Cash buffer
# ---------------------------------------------------------------------------


class TestCashBuffer:
    def test_exposure_respects_cash_buffer(self):
        buffer = 0.05
        result = _mock_simulate_result(cash_buffer_pct=buffer)

        invested = result.exposure[result.exposure > 0]
        if invested.empty:
            pytest.skip("Strategy stayed in cash for entire synthetic run")

        assert invested.max() <= 1.0 - buffer + 1e-9
        assert invested.min() >= (1.0 - buffer) / 5 - 1e-6


# ---------------------------------------------------------------------------
# No duplicate holdings
# ---------------------------------------------------------------------------


class TestNoDuplicateHoldings:
    def test_no_duplicate_symbols_in_trades_at_same_date(self):
        result = _mock_simulate_result()

        buys = result.trades_log[result.trades_log["action"] == "buy"]
        for dt, group in buys.groupby("date"):
            assert group["symbol"].nunique() == len(group), f"Duplicates on {dt}"


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


class TestBenchmarkComparison:
    def test_run_weekly_backtest_includes_spy_qqq_metrics(self):
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL"]
        loader = _universe_loader(symbols)

        result = run_weekly_backtest(
            symbols,
            start="2020-01-01",
            end="2021-06-30",
            model=CalibratedLogisticRegressionModel(),
            splitter=_fast_splitter(),
            benchmark_symbols=("SPY", "QQQ"),
            loader=loader,
            compare_baseline=False,
        )

        assert "SPY" in result.benchmark_metrics
        assert "QQQ" in result.benchmark_metrics
        assert "SPY" in result.benchmark_closes
        assert "QQQ" in result.benchmark_closes
        assert result.benchmark_metrics["SPY"].cagr is not None
        assert result.benchmark_metrics["QQQ"].total_return is not None

    def test_compute_backtest_metrics_fields(self):
        idx = pd.bdate_range("2020-01-01", periods=252)
        equity = pd.Series(
            100_000 * np.cumprod(1 + np.full(252, 0.0004)),
            index=idx,
            name="portfolio_value",
        )
        exposure = pd.Series(0.95, index=idx)
        trades = pd.DataFrame(
            {
                "date": [idx[0], idx[50]],
                "symbol": ["AAPL", "MSFT"],
                "action": ["buy", "sell"],
                "score": [0.6, 0.5],
            }
        )
        sim = SimulationResult(
            equity_curve=equity,
            trades_log=trades,
            exposure=exposure,
            turnover_total=0.15,
        )
        metrics = compute_backtest_metrics(sim)
        assert metrics.cagr is not None
        assert metrics.sharpe is not None
        assert metrics.max_drawdown <= 0
        assert metrics.total_return > 0
        assert metrics.turnover == pytest.approx(0.15)
        assert metrics.exposure == pytest.approx(0.95)
        assert metrics.n_trades == 2
