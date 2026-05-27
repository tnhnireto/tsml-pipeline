"""
Portfolio-level regime exposure overlay for backtests.

Scales target equity exposure based on SPY trend and optional volatility
regime.  Uses only information available through the prior trading day's
close — the same cutoff as walk-forward scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsml.features.transformers import rolling_volatility


@dataclass
class RegimeOverlayConfig:
    """
    Configuration for regime-based portfolio exposure scaling.

    Attributes
    ----------
    enabled:
        When ``False``, target exposure is always 1.0 (no overlay).
    bull_exposure:
        Target equity fraction when SPY is above its 200-day SMA.
    bear_exposure:
        Target equity fraction when SPY is below its 200-day SMA.
    high_vol_exposure:
        Target when SPY is below SMA200 *and* 20-day vol exceeds
        ``vol_threshold``.  Set to ``None`` to disable this rule.
    vol_threshold:
        Daily realised-vol threshold (same units as ``spy_vol_20d``).
        Required when ``high_vol_exposure`` is set.
    benchmark_symbol:
        Benchmark ticker used for regime signals (default SPY).
    """

    enabled: bool = False
    bull_exposure: float = 1.0
    bear_exposure: float = 0.25
    high_vol_exposure: float | None = 0.0
    vol_threshold: float | None = None
    benchmark_symbol: str = "SPY"

    def __post_init__(self) -> None:
        for name, val in [
            ("bull_exposure", self.bull_exposure),
            ("bear_exposure", self.bear_exposure),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]; got {val}.")
        if self.high_vol_exposure is not None and not (0.0 <= self.high_vol_exposure <= 1.0):
            raise ValueError(
                f"high_vol_exposure must be in [0, 1]; got {self.high_vol_exposure}."
            )


def compute_target_exposure_as_of(
    benchmark_close: pd.Series,
    as_of: pd.Timestamp,
    config: RegimeOverlayConfig,
) -> float:
    """
    Return target portfolio equity exposure using data through ``as_of``.

    Parameters
    ----------
    benchmark_close:
        Benchmark daily close prices (typically SPY).
    as_of:
        Last date whose close may be used (inclusive).
    config:
        Overlay thresholds and exposure levels.
    """
    if not config.enabled:
        return 1.0

    hist = benchmark_close.loc[:as_of]
    if hist.empty:
        return config.bull_exposure

    sma200 = hist.rolling(window=200, min_periods=200).mean()
    if pd.isna(sma200.iloc[-1]):
        return config.bull_exposure

    above_sma = hist.iloc[-1] > sma200.iloc[-1]
    if above_sma:
        return config.bull_exposure

    if (
        config.high_vol_exposure is not None
        and config.vol_threshold is not None
    ):
        vol20 = rolling_volatility(hist, 20)
        current_vol = vol20.iloc[-1]
        if not pd.isna(current_vol) and current_vol > config.vol_threshold:
            return config.high_vol_exposure

    return config.bear_exposure


def build_regime_target_series(
    benchmark_close: pd.Series,
    trading_days: pd.DatetimeIndex,
    config: RegimeOverlayConfig,
) -> pd.Series:
    """
    Build a daily target-exposure series aligned to ``trading_days``.

    Each value uses the benchmark close through the **prior** trading day,
    matching the simulator's signal lag convention.
    """
    if not config.enabled:
        return pd.Series(1.0, index=trading_days, name="regime_target")

    targets: list[float] = []
    for i, date in enumerate(trading_days):
        if i == 0:
            targets.append(config.bull_exposure)
            continue
        prior = trading_days[i - 1]
        targets.append(
            compute_target_exposure_as_of(benchmark_close, prior, config)
        )

    return pd.Series(targets, index=trading_days, name="regime_target")


def deployable_fraction(
    n_positions: int,
    cash_buffer_pct: float,
    regime_target: float,
) -> float:
    """
    Equity fraction deployed across ``n_positions`` after buffer and overlay.

    Returns 0 when there are no positions or regime target is zero.
    """
    if n_positions <= 0 or regime_target <= 0.0:
        return 0.0
    return (1.0 - cash_buffer_pct) * regime_target


def per_position_weight(
    n_positions: int,
    cash_buffer_pct: float,
    regime_target: float,
) -> float:
    """Equal weight per held symbol within the deployable sleeve."""
    deployable = deployable_fraction(n_positions, cash_buffer_pct, regime_target)
    if n_positions <= 0:
        return 0.0
    return deployable / n_positions
