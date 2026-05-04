"""
Tests for YFinanceLoader.

Most tests use a patched yfinance.download so they don't hit the network.
One optional integration test is skipped unless explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tsml.data_loader.yfinance_loader import YFinanceLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_yf_df(start: str = "2020-01-02", periods: int = 50) -> pd.DataFrame:
    """
    Return a DataFrame that mimics what yfinance.download returns:
    UTC-aware DatetimeIndex, simple column names (no MultiIndex).
    """
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start, periods=periods, freq="B", tz="UTC")
    close = 300.0 + np.cumsum(rng.normal(0, 1, periods))

    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, periods).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

class TestYFinanceLoaderUnit:
    def test_download_is_called_when_no_cache(self, tmp_path):
        raw = _make_raw_yf_df()

        with patch("yfinance.download", return_value=raw) as mock_dl:
            loader = YFinanceLoader(cache_dir=tmp_path)
            df = loader.load("SPY", "2020-01-02", "2020-03-20")

        mock_dl.assert_called_once()
        assert not df.empty

    def test_cache_is_written_after_download(self, tmp_path):
        raw = _make_raw_yf_df()

        with patch("yfinance.download", return_value=raw):
            loader = YFinanceLoader(cache_dir=tmp_path)
            loader.load("SPY", "2020-01-02", "2020-03-20")

        cache_file = tmp_path / "SPY.parquet"
        assert cache_file.exists(), "Parquet file should be written after download"

    def test_cache_is_used_on_second_call(self, tmp_path):
        # The mock returns 50 business days starting 2020-01-02 (~2020-03-11).
        # Both requests must fall within that range so the cache is sufficient.
        raw = _make_raw_yf_df()

        with patch("yfinance.download", return_value=raw) as mock_dl:
            loader = YFinanceLoader(cache_dir=tmp_path)
            loader.load("SPY", "2020-01-02", "2020-03-05")
            loader.load("SPY", "2020-01-02", "2020-03-05")

        assert mock_dl.call_count == 1, "Second call should serve from cache"

    def test_re_download_when_cache_too_short(self, tmp_path):
        """If the cached range doesn't cover the requested range, re-download."""
        short_raw = _make_raw_yf_df(periods=10)    # only 10 days
        full_raw = _make_raw_yf_df(periods=50)

        with patch("yfinance.download", return_value=short_raw) as mock_dl:
            loader = YFinanceLoader(cache_dir=tmp_path)
            loader.load("SPY", "2020-01-02", "2020-01-15")

        # Now request a wider range — should trigger a second download.
        with patch("yfinance.download", return_value=full_raw) as mock_dl2:
            loader.load("SPY", "2020-01-02", "2020-03-20")

        mock_dl2.assert_called_once()

    def test_returned_df_has_lowercase_columns(self, tmp_path):
        raw = _make_raw_yf_df()

        with patch("yfinance.download", return_value=raw):
            loader = YFinanceLoader(cache_dir=tmp_path)
            df = loader.load("SPY", "2020-01-02", "2020-03-20")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_returned_df_index_is_utc(self, tmp_path):
        raw = _make_raw_yf_df()

        with patch("yfinance.download", return_value=raw):
            loader = YFinanceLoader(cache_dir=tmp_path)
            df = loader.load("SPY", "2020-01-02", "2020-03-20")

        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_empty_download_raises(self, tmp_path):
        empty = pd.DataFrame()

        with patch("yfinance.download", return_value=empty):
            loader = YFinanceLoader(cache_dir=tmp_path)
            with pytest.raises(ValueError, match="no data"):
                loader.load("INVALID", "2020-01-02", "2020-03-20")

    def test_cache_dir_is_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "cache"
        assert not new_dir.exists()
        YFinanceLoader(cache_dir=new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# Optional integration test (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Integration test — hits Yahoo Finance network")
def test_integration_real_download(tmp_path):
    loader = YFinanceLoader(cache_dir=tmp_path)
    df = loader.load("SPY", "2022-01-01", "2022-03-31")

    assert not df.empty
    assert "close" in df.columns
    assert df.index.tz is not None
