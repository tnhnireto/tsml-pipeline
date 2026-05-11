"""
Tests for rank_universe.

We avoid network calls by injecting a stub DataLoader that returns synthetic
OHLCV DataFrames built from deterministic price sequences.  This keeps the
tests fast, offline, and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.data_loader.base import DataLoader
from tsml.models.baselines import AlwaysLong, LogisticRegressionModel
from tsml.portfolio.ranker import rank_universe
from tsml.validation import WalkForwardSplit


# ---------------------------------------------------------------------------
# Stub loader — returns synthetic data without hitting the network
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with n trading-day rows."""
    rng    = np.random.default_rng(seed)
    dates  = pd.bdate_range("2015-01-02", periods=n, freq="B", tz="UTC")
    close  = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n))
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
    """Returns pre-built DataFrames keyed by symbol; raises for unknown ones."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol not in self._data:
            raise ValueError(f"No stub data for symbol '{symbol}'.")
        return self._data[symbol]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def splitter() -> WalkForwardSplit:
    return WalkForwardSplit(n_splits=3, min_train_size=252, test_size=63, gap=1)


@pytest.fixture()
def loader_3symbols() -> StubLoader:
    return StubLoader(
        {
            "AAA": _make_ohlcv(800, seed=1),
            "BBB": _make_ohlcv(800, seed=2),
            "CCC": _make_ohlcv(800, seed=3),
        }
    )


# ---------------------------------------------------------------------------
# Return shape and content
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_dataframe(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB", "CCC"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_symbol_and_score(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert list(result.columns) == ["symbol", "score"]

    def test_one_row_per_successful_symbol(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB", "CCC"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert len(result) == 3
        assert set(result["symbol"]) == {"AAA", "BBB", "CCC"}

    def test_index_is_reset(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB", "CCC"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert list(result.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSorting:
    def test_sorted_descending_by_score(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB", "CCC"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        scores = result["score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending: {scores}"
        )

    def test_scores_are_probabilities_between_0_and_1(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB", "CCC"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert (result["score"] >= 0.0).all(), "Score below 0"
        assert (result["score"] <= 1.0).all(), "Score above 1"


# ---------------------------------------------------------------------------
# Graceful failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_unknown_symbol_is_skipped(self, splitter, loader_3symbols):
        """A symbol not in the loader raises; rank_universe must skip it."""
        result = rank_universe(
            ["AAA", "UNKNOWN", "CCC"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert "UNKNOWN" not in result["symbol"].values
        assert len(result) == 2

    def test_symbol_with_too_few_rows_is_skipped(self, splitter):
        """A symbol with insufficient data (splitter raises ValueError) is skipped."""
        tiny_loader = StubLoader(
            {
                "TINY": _make_ohlcv(50, seed=9),   # way too few rows
                "BIG":  _make_ohlcv(800, seed=5),
            }
        )
        result = rank_universe(
            ["TINY", "BIG"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=tiny_loader,
        )
        assert "TINY" not in result["symbol"].values
        assert "BIG" in result["symbol"].values

    def test_all_symbols_fail_returns_empty_dataframe(self, splitter):
        """When every symbol fails the result is an empty DataFrame."""
        empty_loader = StubLoader(
            {
                "X": _make_ohlcv(10, seed=0),
                "Y": _make_ohlcv(10, seed=1),
            }
        )
        result = rank_universe(
            ["X", "Y"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=empty_loader,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ["symbol", "score"]

    def test_failure_warning_written_to_stderr(
        self, splitter, loader_3symbols, capsys
    ):
        """Skipped symbols must emit a warning line to stderr."""
        rank_universe(
            ["AAA", "MISSING"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        captured = capsys.readouterr()
        assert "MISSING" in captured.err
        assert "rank_universe" in captured.err


# ---------------------------------------------------------------------------
# Target parameter forwarding
# ---------------------------------------------------------------------------

class TestTargetForwarding:
    def test_direction_5d_target_runs_without_error(self, splitter, loader_3symbols):
        result = rank_universe(
            ["AAA", "BBB"],
            model=LogisticRegressionModel(),
            splitter=splitter,
            target="direction_5d",
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert len(result) == 2

    def test_always_long_model_scores_all_above_0(self, splitter, loader_3symbols):
        """AlwaysLong always predicts 1, so P(up) should be 1.0 for every symbol."""
        result = rank_universe(
            ["AAA", "BBB", "CCC"],
            model=AlwaysLong(),
            splitter=splitter,
            start="2015-01-01",
            end="2023-12-31",
            loader=loader_3symbols,
        )
        assert (result["score"] == 1.0).all()
