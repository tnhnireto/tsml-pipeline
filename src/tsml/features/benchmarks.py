"""Benchmark price loading for cross-sectional and regime features."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from tsml.data_loader.base import DataLoader

DEFAULT_BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ")


def load_benchmark_closes(
    loader: DataLoader,
    start: str,
    end: str,
    symbols: Sequence[str] = DEFAULT_BENCHMARKS,
) -> dict[str, pd.Series]:
    """
    Load daily close prices for benchmark symbols.

    Returns
    -------
    dict
        Mapping ticker → close ``Series`` indexed by trading dates.
    """
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        df = loader.load(sym, start, end)
        closes[sym] = df["close"]
    return closes
