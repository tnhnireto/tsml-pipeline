"""
Tests for src/tsml/portfolio/tracker.py.

Covers three areas:
  1. load_orders        — JSONL parsing, filtering, date normalisation
  2. build_equity_curve — portfolio replay: BUY/SELL mechanics, cash,
                          positions, edge cases
  3. compute_portfolio_stats — benchmark comparison, metrics, alignment

All tests use synthetic data or temporary directories.  No network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsml.portfolio.tracker import (
    PortfolioHistory,
    PortfolioStats,
    TradeRecord,
    build_equity_curve,
    compute_portfolio_stats,
    load_orders,
    weekly_returns,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _order_entry(
    symbol: str,
    side: str,
    amount: float,
    date: str = "2024-01-08",
    score: float = 0.62,
    risk_approved: bool = True,
) -> dict:
    """Build a minimal JSONL order entry."""
    return {
        "timestamp":     f"{date}T20:00:00+00:00",
        "dry_run":       True,
        "type":          "approved" if risk_approved else "rejected",
        "symbol":        symbol,
        "side":          side,
        "amount":        amount,
        "score":         score,
        "signal_reason": "test",
        "risk_approved": risk_approved,
        "risk_rule":     "ok",
        "risk_reason":   "",
        "broker_status": "dry_run",
        "broker_order_id": None,
    }


def _prices(
    symbols: list[str],
    n: int = 60,
    start: str = "2024-01-02",
    base: float = 100.0,
    drift: float = 0.001,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic close price DataFrame (UTC-midnight index)."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, freq="B", tz="UTC")
    cols  = {}
    for i, sym in enumerate(symbols):
        prices = base * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
        cols[sym] = prices
    return pd.DataFrame(cols, index=dates)


def _equity_curve(
    n: int = 250,
    drift: float = 0.0005,
    seed: int = 0,
    start: str = "2020-01-02",
) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, freq="B", tz="UTC")
    values = np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    return pd.Series(values, index=dates, name="portfolio_value")


def _benchmark(
    n: int = 250,
    drift: float = 0.0003,
    seed: int = 42,
    start: str = "2020-01-02",
) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, freq="B", tz="UTC")
    prices = 300.0 * np.cumprod(1 + rng.normal(drift, 0.012, size=n))
    return pd.Series(prices, index=dates, name="close")


# ===========================================================================
# 1. load_orders
# ===========================================================================

class TestLoadOrdersEmptyDir:
    def test_missing_dir_returns_empty(self, tmp_path):
        df = load_orders(tmp_path / "nonexistent")
        assert df.empty
        assert list(df.columns) == ["date", "symbol", "side", "amount", "score"]

    def test_empty_dir_returns_empty(self, tmp_path):
        df = load_orders(tmp_path)
        assert df.empty

    def test_dir_with_no_jsonl_returns_empty(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        df = load_orders(tmp_path)
        assert df.empty


class TestLoadOrdersBasic:
    def test_approved_order_is_loaded(self, tmp_path):
        _write_jsonl(
            tmp_path / "2024-01-08.jsonl",
            [_order_entry("AAPL", "BUY", 1_000.0)],
        )
        df = load_orders(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["side"] == "BUY"
        assert df.iloc[0]["amount"] == pytest.approx(1_000.0)

    def test_rejected_order_is_excluded(self, tmp_path):
        _write_jsonl(
            tmp_path / "2024-01-08.jsonl",
            [_order_entry("AAPL", "BUY", 1_000.0, risk_approved=False)],
        )
        df = load_orders(tmp_path)
        assert df.empty

    def test_mixed_file_keeps_only_approved(self, tmp_path):
        entries = [
            _order_entry("AAPL", "BUY",  1_000.0, risk_approved=True),
            _order_entry("TSLA", "BUY",  500.0,   risk_approved=False),
            _order_entry("MSFT", "SELL", 0.0,     risk_approved=True),
        ]
        _write_jsonl(tmp_path / "2024-01-08.jsonl", entries)
        df = load_orders(tmp_path)
        assert len(df) == 2
        assert set(df["symbol"]) == {"AAPL", "MSFT"}


class TestLoadOrdersMultipleFiles:
    def test_multiple_files_combined(self, tmp_path):
        _write_jsonl(tmp_path / "2024-01-08.jsonl",
                     [_order_entry("AAPL", "BUY", 1_000.0, date="2024-01-08")])
        _write_jsonl(tmp_path / "2024-01-15.jsonl",
                     [_order_entry("MSFT", "BUY", 800.0,   date="2024-01-15")])
        df = load_orders(tmp_path)
        assert len(df) == 2
        assert set(df["symbol"]) == {"AAPL", "MSFT"}

    def test_sorted_by_date(self, tmp_path):
        _write_jsonl(tmp_path / "2024-01-15.jsonl",
                     [_order_entry("MSFT", "BUY", 800.0, date="2024-01-15")])
        _write_jsonl(tmp_path / "2024-01-08.jsonl",
                     [_order_entry("AAPL", "BUY", 1_000.0, date="2024-01-08")])
        df = load_orders(tmp_path)
        assert df.iloc[0]["symbol"] == "AAPL"   # earlier date first


class TestLoadOrdersDateParsing:
    def test_date_is_utc_midnight(self, tmp_path):
        _write_jsonl(
            tmp_path / "2024-01-08.jsonl",
            [_order_entry("AAPL", "BUY", 1_000.0, date="2024-01-08")],
        )
        df = load_orders(tmp_path)
        ts = df.iloc[0]["date"]
        assert ts.hour == 0
        assert ts.minute == 0
        assert str(ts.tzinfo) in ("UTC", "utc") or ts.tz is not None

    def test_score_parsed_as_float(self, tmp_path):
        _write_jsonl(
            tmp_path / "2024-01-08.jsonl",
            [_order_entry("AAPL", "BUY", 500.0, score=0.73)],
        )
        df = load_orders(tmp_path)
        assert df.iloc[0]["score"] == pytest.approx(0.73)


# ===========================================================================
# 2. build_equity_curve
# ===========================================================================

class TestBuildEquityCurveReturnType:
    def test_returns_portfolio_history(self):
        orders = pd.DataFrame(columns=["date", "symbol", "side", "amount", "score"])
        prices = _prices(["AAPL"], n=20)
        history = build_equity_curve(orders, prices)
        assert isinstance(history, PortfolioHistory)

    def test_empty_orders_gives_flat_curve(self):
        orders = pd.DataFrame(columns=["date", "symbol", "side", "amount", "score"])
        prices = _prices(["AAPL"], n=20)
        history = build_equity_curve(orders, prices, initial_capital=5_000.0)
        assert (history.equity_curve == 5_000.0).all()

    def test_equity_curve_name(self):
        orders = pd.DataFrame(columns=["date", "symbol", "side", "amount", "score"])
        prices = _prices(["AAPL"])
        h = build_equity_curve(orders, prices)
        assert h.equity_curve.name == "portfolio_value"

    def test_cash_name(self):
        orders = pd.DataFrame(columns=["date", "symbol", "side", "amount", "score"])
        prices = _prices(["AAPL"])
        h = build_equity_curve(orders, prices)
        assert h.cash.name == "cash"


class TestBuildEquityCurveBuyMechanics:
    def _make_single_buy(self, tmp_path: Path, date: str, amount: float) -> pd.DataFrame:
        """Return a one-row orders DataFrame for a BUY on the given date."""
        _write_jsonl(tmp_path / "orders.jsonl", [_order_entry("AAPL", "BUY", amount, date=date)])
        return load_orders(tmp_path)

    def test_buy_reduces_cash(self, tmp_path):
        prices = _prices(["AAPL"], n=20, start="2024-01-02")
        buy_date = prices.index[5].strftime("%Y-%m-%d")
        orders = self._make_single_buy(tmp_path, buy_date, 1_000.0)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        # Cash after the buy date must be less than initial capital
        post_buy_cash = h.cash.loc[prices.index[5]:]
        assert (post_buy_cash < 10_000.0).all()

    def test_buy_creates_position(self, tmp_path):
        prices = _prices(["AAPL"], n=20, start="2024-01-02")
        buy_date = prices.index[5].strftime("%Y-%m-%d")
        orders = self._make_single_buy(tmp_path, buy_date, 1_000.0)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        after_buy = h.positions.loc[prices.index[5]:]
        assert (after_buy["AAPL"] > 0).all()

    def test_buy_shares_equal_amount_over_price(self, tmp_path):
        prices = _prices(["AAPL"], n=20, start="2024-01-02")
        buy_date_ts = prices.index[5]
        buy_date    = buy_date_ts.strftime("%Y-%m-%d")
        amount      = 1_000.0
        orders = self._make_single_buy(tmp_path, buy_date, amount)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        expected_shares = amount / prices.loc[buy_date_ts, "AAPL"]
        actual_shares   = h.positions.loc[buy_date_ts, "AAPL"]
        assert actual_shares == pytest.approx(expected_shares)

    def test_equity_tracks_price_after_buy(self, tmp_path):
        """After a single BUY, equity should grow if price rises."""
        # Construct a strictly rising price series
        dates  = pd.bdate_range("2024-01-02", periods=10, freq="B", tz="UTC")
        prices_rising = pd.DataFrame(
            {"AAPL": [100.0 + i for i in range(10)]}, index=dates
        )
        buy_date = dates[2].strftime("%Y-%m-%d")
        _write_jsonl(tmp_path / "orders.jsonl",
                     [_order_entry("AAPL", "BUY", 1_000.0, date=buy_date)])
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices_rising, initial_capital=10_000.0)
        # Equity after buy should be strictly increasing (price is rising)
        post_buy = h.equity_curve.iloc[3:]
        assert (post_buy.diff().dropna() > 0).all()


class TestBuildEquityCurveSellMechanics:
    def test_sell_clears_position(self, tmp_path):
        dates  = pd.bdate_range("2024-01-02", periods=20, freq="B", tz="UTC")
        prices = pd.DataFrame({"AAPL": [100.0 + i * 0.5 for i in range(20)]}, index=dates)

        entries = [
            _order_entry("AAPL", "BUY",  1_000.0, date=dates[2].strftime("%Y-%m-%d")),
            _order_entry("AAPL", "SELL", 0.0,     date=dates[8].strftime("%Y-%m-%d")),
        ]
        _write_jsonl(tmp_path / "orders.jsonl", entries)
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        # After sell date, shares should be 0
        after_sell = h.positions.loc[dates[8]:, "AAPL"]
        assert (after_sell == 0.0).all()

    def test_sell_increases_cash(self, tmp_path):
        dates  = pd.bdate_range("2024-01-02", periods=20, freq="B", tz="UTC")
        prices = pd.DataFrame({"AAPL": [100.0] * 20}, index=dates)  # flat price

        buy_date  = dates[2].strftime("%Y-%m-%d")
        sell_date = dates[8].strftime("%Y-%m-%d")
        entries = [
            _order_entry("AAPL", "BUY",  1_000.0, date=buy_date),
            _order_entry("AAPL", "SELL", 0.0,     date=sell_date),
        ]
        _write_jsonl(tmp_path / "orders.jsonl", entries)
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        cash_before_sell = h.cash.loc[dates[7]]
        cash_after_sell  = h.cash.loc[dates[8]]
        assert cash_after_sell > cash_before_sell

    def test_buy_sell_at_same_price_restores_capital(self, tmp_path):
        """Buy then sell at constant price: portfolio value ≈ initial_capital."""
        dates  = pd.bdate_range("2024-01-02", periods=20, freq="B", tz="UTC")
        prices = pd.DataFrame({"AAPL": [100.0] * 20}, index=dates)

        buy_date  = dates[2].strftime("%Y-%m-%d")
        sell_date = dates[10].strftime("%Y-%m-%d")
        entries = [
            _order_entry("AAPL", "BUY",  2_000.0, date=buy_date),
            _order_entry("AAPL", "SELL", 0.0,     date=sell_date),
        ]
        _write_jsonl(tmp_path / "orders.jsonl", entries)
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        final_value = h.equity_curve.iloc[-1]
        assert final_value == pytest.approx(10_000.0, rel=1e-6)


class TestBuildEquityCurveEdgeCases:
    def test_insufficient_cash_order_skipped(self, tmp_path, capsys):
        dates  = pd.bdate_range("2024-01-02", periods=10, freq="B", tz="UTC")
        prices = pd.DataFrame({"AAPL": [100.0] * 10}, index=dates)
        # Buy amount exceeds initial capital
        _write_jsonl(tmp_path / "orders.jsonl",
                     [_order_entry("AAPL", "BUY", 20_000.0, date=dates[2].strftime("%Y-%m-%d"))])
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices, initial_capital=5_000.0)
        # Capital should be unchanged (order skipped)
        assert h.equity_curve.iloc[-1] == pytest.approx(5_000.0, rel=0.01)
        assert "insufficient cash" in capsys.readouterr().err

    def test_symbol_not_in_prices_skipped(self, tmp_path, capsys):
        prices = _prices(["MSFT"], n=10)
        _write_jsonl(tmp_path / "orders.jsonl",
                     [_order_entry("AAPL", "BUY", 1_000.0, date=prices.index[2].strftime("%Y-%m-%d"))])
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        # Portfolio should stay flat (order skipped)
        assert h.equity_curve.iloc[-1] == pytest.approx(10_000.0)
        assert "not in prices" in capsys.readouterr().err

    def test_sell_with_no_position_is_silent_noop(self, tmp_path):
        prices = _prices(["AAPL"], n=10)
        _write_jsonl(tmp_path / "orders.jsonl",
                     [_order_entry("AAPL", "SELL", 0.0, date=prices.index[5].strftime("%Y-%m-%d"))])
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices, initial_capital=10_000.0)
        assert h.equity_curve.iloc[-1] == pytest.approx(10_000.0)

    def test_equity_curve_length_equals_price_index(self, tmp_path):
        prices = _prices(["AAPL"], n=30)
        orders = pd.DataFrame(columns=["date", "symbol", "side", "amount", "score"])
        h = build_equity_curve(orders, prices)
        assert len(h.equity_curve) == len(prices)

    def test_positions_index_matches_price_index(self, tmp_path):
        prices = _prices(["AAPL"], n=30)
        _write_jsonl(tmp_path / "orders.jsonl",
                     [_order_entry("AAPL", "BUY", 500.0, date=prices.index[5].strftime("%Y-%m-%d"))])
        orders = load_orders(tmp_path)
        h = build_equity_curve(orders, prices)
        pd.testing.assert_index_equal(h.positions.index, prices.index)


# ===========================================================================
# 3. weekly_returns
# ===========================================================================

class TestWeeklyReturns:
    def test_returns_series_named_weekly_return(self):
        eq = _equity_curve(100)
        wr = weekly_returns(eq)
        assert wr.name == "weekly_return"

    def test_length_is_roughly_number_of_weeks(self):
        eq = _equity_curve(250)   # ~50 weeks
        wr = weekly_returns(eq)
        # Loose: 40 < weeks < 60
        assert 40 < len(wr) < 60

    def test_too_short_returns_empty(self):
        eq = _equity_curve(1)
        wr = weekly_returns(eq)
        assert wr.empty

    def test_flat_curve_returns_zero(self):
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B", tz="UTC")
        eq = pd.Series(1.0, index=dates, name="portfolio_value")
        wr = weekly_returns(eq)
        assert (wr.abs() < 1e-12).all()


# ===========================================================================
# 4. compute_portfolio_stats (retained from first version of tracker)
# ===========================================================================

class TestComputePortfolioStats:
    def test_returns_portfolio_stats(self):
        stats = compute_portfolio_stats(_equity_curve(300), _benchmark(300))
        assert isinstance(stats, PortfolioStats)

    def test_default_labels(self):
        stats = compute_portfolio_stats(_equity_curve(300), _benchmark(300))
        assert stats.label == "portfolio"
        assert stats.benchmark_label == "SPY"

    def test_custom_labels_stored(self):
        stats = compute_portfolio_stats(
            _equity_curve(300), _benchmark(300),
            label="myport", benchmark_label="QQQ",
        )
        assert stats.label == "myport"
        assert stats.benchmark_label == "QQQ"

    def test_volatility_positive(self):
        stats = compute_portfolio_stats(_equity_curve(300), _benchmark(300))
        assert stats.volatility > 0
        assert stats.benchmark_volatility > 0

    def test_max_drawdown_non_positive(self):
        stats = compute_portfolio_stats(_equity_curve(300), _benchmark(300))
        assert stats.max_drawdown <= 0
        assert stats.benchmark_max_drawdown <= 0

    def test_excess_return_equals_cagr_difference(self):
        stats = compute_portfolio_stats(_equity_curve(300), _benchmark(300))
        assert stats.excess_return == pytest.approx(
            stats.cagr - stats.benchmark_cagr
        )

    def test_sharpe_changes_with_risk_free_rate(self):
        eq = _equity_curve(300)
        bm = _benchmark(300)
        s0  = compute_portfolio_stats(eq, bm, risk_free_rate=0.0)
        s05 = compute_portfolio_stats(eq, bm, risk_free_rate=0.05)
        assert s0.sharpe != s05.sharpe

    def test_too_short_equity_raises(self):
        with pytest.raises(ValueError, match="2 rows"):
            compute_portfolio_stats(_equity_curve(1), _benchmark(10))

    def test_no_common_dates_raises(self):
        eq = _equity_curve(100, start="2020-01-02")
        bm = _benchmark(100, start="2025-01-02")
        with pytest.raises(ValueError, match="common dates"):
            compute_portfolio_stats(eq, bm)

    def test_always_rising_has_zero_drawdown(self):
        dates = pd.bdate_range("2020-01-02", periods=100, freq="B", tz="UTC")
        eq = pd.Series(np.linspace(1.0, 2.0, 100), index=dates)
        bm = _benchmark(100)
        stats = compute_portfolio_stats(eq, bm)
        assert stats.max_drawdown == pytest.approx(0.0, abs=1e-10)

    def test_date_range_fields(self):
        stats = compute_portfolio_stats(_equity_curve(300), _benchmark(300))
        assert stats.start_date < stats.end_date
        assert stats.n_days > 0
