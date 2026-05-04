"""
YFinanceLoader — downloads daily OHLCV data from Yahoo Finance.

Data is cached as a Parquet file on disk so repeated calls do not
hit the network.  The cache is keyed by symbol; if the requested date
range falls outside the cached range the file is re-downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from tsml.data_loader.base import DataLoader, validate_ohlcv


class YFinanceLoader(DataLoader):
    """
    Downloads and caches daily OHLCV bars from Yahoo Finance.

    Parameters
    ----------
    cache_dir:
        Directory where Parquet files are stored.
        One file per symbol: ``<cache_dir>/<SYMBOL>.parquet``.
    """

    def __init__(self, cache_dir: str | Path = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Return a UTC-indexed OHLCV DataFrame.

        The result covers the trading days in [start, end].  Data is
        served from disk if a cached file already covers that range;
        otherwise Yahoo Finance is queried and the result is cached.
        """
        cache_path = self.cache_dir / f"{symbol.upper()}.parquet"

        df = self._load_from_cache(cache_path, start, end)
        if df is None:
            df = self._download(symbol, start, end)
            df.to_parquet(cache_path)

        df = self._slice(df, start, end)
        validate_ohlcv(df, symbol)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_cache(
        self, path: Path, start: str, end: str
    ) -> pd.DataFrame | None:
        """Return cached data if it fully covers [start, end], else None."""
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        cache_start = df.index.min().strftime("%Y-%m-%d")
        cache_end = df.index.max().strftime("%Y-%m-%d")

        if cache_start <= start and cache_end >= end:
            return df

        return None

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download from Yahoo Finance and normalise the DataFrame."""
        raw = yf.download(
            symbol,
            start=start,
            # yfinance end is exclusive, so add one day.
            end=pd.Timestamp(end) + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for '{symbol}' "
                f"between {start} and {end}."
            )

        df = self._normalise(raw, symbol)
        return df

    def _normalise(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Turn the raw yfinance DataFrame into the project's standard shape:
        lowercase column names, UTC-aware DatetimeIndex.
        """
        # yfinance may return a MultiIndex when a single ticker is requested
        # with certain versions — flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]

        # Ensure the index is UTC-aware.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[start:end]
