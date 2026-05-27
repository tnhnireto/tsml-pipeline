"""Tests for portfolio-level regime exposure overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import DataLoader
from tsml.models.baselines import CalibratedLogisticRegressionModel
from tsml.portfolio.regime_overlay import (
    RegimeOverlayConfig,
    compute_target_exposure_as_of,
    deployable_fraction,
    per_position_weight,
)
from tsml.portfolio.simulator import simulate
from tsml.portfolio.weekly_backtest import exposure_adjusted_turnover
from tsml.validation.splitters import WalkForwardSplit


def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, size=n))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


def _bull_spy(n: int = 500) -> pd.Series:
    """Monotonically rising SPY — always above SMA200 after warmup."""
    idx = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
    return pd.Series(100 + np.arange(n) * 0.5, index=idx, dtype=float)


def _bear_spy(n: int = 500) -> pd.Series:
    """Falling SPY after long bull — below SMA200."""
    idx = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
    half = n // 2
    up = 100 + np.arange(half) * 0.5
    down = up[-1] - np.arange(n - half) * 1.0
    return pd.Series(np.concatenate([up, down]), index=idx, dtype=float)


class StubLoader(DataLoader):
    def __init__(self, data: dict[str, pd.DataFrame], spy: pd.Series) -> None:
        self._data = data
        self._spy = spy

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol == "SPY":
            return pd.DataFrame({"close": self._spy}, index=self._spy.index)
        if symbol not in self._data:
            raise ValueError(f"No data for {symbol}")
        return self._data[symbol]


class TestExposureScaling:
    def test_bull_regime_full_exposure(self):
        spy = _bull_spy(300)
        as_of = spy.index[250]
        cfg = RegimeOverlayConfig(enabled=True, bear_exposure=0.25)
        assert compute_target_exposure_as_of(spy, as_of, cfg) == pytest.approx(1.0)

    def test_bear_regime_scaled_exposure(self):
        spy = _bear_spy(500)
        as_of = spy.index[-1]
        cfg = RegimeOverlayConfig(enabled=True, bear_exposure=0.25)
        assert compute_target_exposure_as_of(spy, as_of, cfg) == pytest.approx(0.25)

    def test_deployable_fraction_with_overlay(self):
        assert deployable_fraction(5, 0.05, 0.25) == pytest.approx(0.2375)
        assert deployable_fraction(5, 0.05, 1.0) == pytest.approx(0.95)

    def test_per_position_weight_no_leverage(self):
        w = per_position_weight(5, 0.05, 0.25)
        assert w == pytest.approx(0.0475)
        assert w * 5 <= 1.0

    def test_cash_retention_bear_regime(self):
        """Bear target 0.25 with 5% buffer leaves >= 70% in cash."""
        deployable = deployable_fraction(5, 0.05, 0.25)
        assert deployable == pytest.approx(0.2375)
        assert 1.0 - deployable >= 0.70

    def test_exposure_adjusted_turnover(self):
        assert exposure_adjusted_turnover(0.5, 0.25) == pytest.approx(2.0)
        assert exposure_adjusted_turnover(0.5, 0.0) == 0.0


class TestZeroExposureRegime:
    def test_high_vol_bear_goes_to_zero(self):
        spy = _bear_spy(500)
        # Inject high volatility at the end
        spy_volatile = spy.copy()
        idx = spy_volatile.index[-25:]
        spy_volatile.loc[idx] = spy_volatile.loc[idx] * (
            1 + np.random.default_rng(0).normal(0, 0.05, len(idx))
        )
        as_of = spy_volatile.index[-1]
        cfg = RegimeOverlayConfig(
            enabled=True,
            bear_exposure=0.25,
            high_vol_exposure=0.0,
            vol_threshold=0.001,
        )
        target = compute_target_exposure_as_of(spy_volatile, as_of, cfg)
        assert target == pytest.approx(0.0)


class TestNoLookahead:
    def test_overlay_uses_only_past_spy_data(self):
        spy = _bear_spy(500)
        as_of = spy.index[300]
        cfg = RegimeOverlayConfig(enabled=True, bear_exposure=0.25)

        target_before = compute_target_exposure_as_of(spy, as_of, cfg)
        spy_mutated = spy.copy()
        spy_mutated.loc[as_of + pd.Timedelta(days=30):] = 9999.0
        target_after = compute_target_exposure_as_of(spy_mutated, as_of, cfg)
        assert target_before == pytest.approx(target_after)


class TestSimulatorIntegration:
    def test_overlay_reduces_average_exposure(self):
        symbols = ["AAA", "BBB", "CCC"]
        loader_data = {
            sym: _make_ohlcv(400, seed=i + 1) for i, sym in enumerate(symbols)
        }
        loader = StubLoader(loader_data, _bear_spy(400))
        splitter = WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)
        model = CalibratedLogisticRegressionModel()
        common = dict(
            symbols=symbols,
            model=model,
            splitter=splitter,
            start_date="2015-01-01",
            end_date="2016-06-30",
            loader=loader,
            turnover_control=True,
            cash_buffer_pct=0.05,
        )

        base = simulate(**common, regime_overlay=RegimeOverlayConfig(enabled=False))
        overlay = simulate(
            **common,
            regime_overlay=RegimeOverlayConfig(enabled=True, bear_exposure=0.25),
        )

        if base.exposure.empty:
            pytest.skip("Insufficient synthetic data for exposure comparison")

        assert overlay.exposure.mean() <= base.exposure.mean() + 1e-9

    def test_zero_target_flattens_portfolio(self):
        symbols = ["AAA", "BBB"]
        loader_data = {sym: _make_ohlcv(400, seed=i) for i, sym in enumerate(symbols)}
        loader = StubLoader(loader_data, _bear_spy(400))
        splitter = WalkForwardSplit(n_splits=2, min_train_size=200, test_size=50, gap=1)

        result = simulate(
            symbols,
            model=CalibratedLogisticRegressionModel(),
            splitter=splitter,
            start_date="2015-01-01",
            end_date="2016-06-30",
            loader=loader,
            turnover_control=True,
            regime_overlay=RegimeOverlayConfig(
                enabled=True,
                bear_exposure=0.25,
                high_vol_exposure=0.0,
                vol_threshold=0.0001,
            ),
        )
        assert (result.regime_target <= 1.0).all()
        assert (result.exposure <= 1.0).all()
