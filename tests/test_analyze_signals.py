"""
Tests for the date-parsing logic in analyze_signals.load_all_signals.

The function must accept:
  - plain YYYY-MM-DD strings   ("2026-05-12")
  - full UTC timestamps        ("2026-05-14 00:00:00+00:00")
  - mixed formats in one load  (older files use plain dates, newer use timestamps)

In all cases the resulting ``date`` column must be UTC-aware
(dtype = datetime64[ns, UTC]).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = (
    "date,rank,symbol,score,action,reason,"
    "return_20d,return_60d,volatility_20d,price_vs_sma_200,above_sma_200"
)

def _make_csv(date_str: str, symbol: str = "AAPL") -> str:
    """Return a minimal valid signal CSV string with the given date value."""
    return (
        f"{_CSV_COLUMNS}\n"
        f"{date_str},1,{symbol},0.62,buy,,0.05,0.12,0.18,0.03,True\n"
    )


def _load_from_csv_strings(tmp_path: Path, *csv_contents: str) -> pd.DataFrame:
    """
    Write each string as a separate .csv file in tmp_path, then call
    load_all_signals with SIGNALS_DIR patched to tmp_path.
    """
    import analyze_signals as mod

    for i, content in enumerate(csv_contents):
        (tmp_path / f"2026-05-{12 + i:02d}.csv").write_text(content, encoding="utf-8")

    original = mod.SIGNALS_DIR
    mod.SIGNALS_DIR = tmp_path
    try:
        return mod.load_all_signals()
    finally:
        mod.SIGNALS_DIR = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDateParsing:
    def test_plain_date_string_is_parsed(self, tmp_path):
        """'YYYY-MM-DD' dates should parse without error."""
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-12"))
        assert len(df) == 1

    def test_plain_date_string_is_utc_aware(self, tmp_path):
        """A plain date string must yield a UTC-aware Timestamp."""
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-12"))
        ts = df["date"].iloc[0]
        assert ts.tzinfo is not None, "Timestamp must be timezone-aware"
        assert str(ts.tzinfo) == "UTC"

    def test_plain_date_string_correct_value(self, tmp_path):
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-12"))
        ts = df["date"].iloc[0]
        expected = pd.Timestamp("2026-05-12", tz="UTC")
        assert ts == expected

    def test_full_utc_timestamp_is_parsed(self, tmp_path):
        """'YYYY-MM-DD HH:MM:SS+00:00' timestamps should parse without error."""
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-14 00:00:00+00:00"))
        assert len(df) == 1

    def test_full_utc_timestamp_is_utc_aware(self, tmp_path):
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-14 00:00:00+00:00"))
        ts = df["date"].iloc[0]
        assert ts.tzinfo is not None
        assert str(ts.tzinfo) == "UTC"

    def test_full_utc_timestamp_correct_value(self, tmp_path):
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-14 00:00:00+00:00"))
        ts = df["date"].iloc[0]
        expected = pd.Timestamp("2026-05-14", tz="UTC")
        assert ts == expected

    def test_mixed_formats_in_same_dataset(self, tmp_path):
        """
        Older CSV uses plain date; newer CSV uses full UTC timestamp.
        Both must load and all dates must be UTC-aware.
        """
        df = _load_from_csv_strings(
            tmp_path,
            _make_csv("2026-05-12"),                    # plain date (old style)
            _make_csv("2026-05-14 00:00:00+00:00"),     # full timestamp (new style)
        )
        assert len(df) == 2
        for ts in df["date"]:
            assert ts.tzinfo is not None, f"Expected UTC-aware, got naive: {ts}"
            assert str(ts.tzinfo) == "UTC"

    def test_mixed_formats_correct_values(self, tmp_path):
        df = _load_from_csv_strings(
            tmp_path,
            _make_csv("2026-05-12"),
            _make_csv("2026-05-14 00:00:00+00:00"),
        )
        dates = sorted(df["date"].tolist())
        assert dates[0] == pd.Timestamp("2026-05-12", tz="UTC")
        assert dates[1] == pd.Timestamp("2026-05-14", tz="UTC")

    def test_column_dtype_is_datetime_utc(self, tmp_path):
        """The date column dtype must be datetime64[ns, UTC] or equivalent."""
        df = _load_from_csv_strings(
            tmp_path,
            _make_csv("2026-05-12"),
            _make_csv("2026-05-14 00:00:00+00:00"),
        )
        dtype = df["date"].dtype
        # Accept both the legacy and new pandas datetime-with-tz dtypes
        assert hasattr(dtype, "tz"), f"Expected a tz-aware dtype, got {dtype}"
        assert str(dtype.tz) == "UTC"

    def test_non_date_columns_unaffected(self, tmp_path):
        """Parsing fix must not alter other columns."""
        df = _load_from_csv_strings(tmp_path, _make_csv("2026-05-12", symbol="NVDA"))
        assert df["symbol"].iloc[0] == "NVDA"
        assert pytest.approx(df["score"].iloc[0], abs=1e-9) == 0.62
        assert df["action"].iloc[0] == "buy"
