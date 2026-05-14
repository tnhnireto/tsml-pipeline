"""
Tests for scripts/analyze_portfolio.py.

Focus: console output must contain only ASCII characters so the script
does not raise UnicodeEncodeError on Windows terminals using cp1252 or
any other narrow encoding.

The module is imported via importlib so it can live outside the installed
package tree.  No network calls or file I/O are performed; all heavy
dependencies are replaced by lightweight mock objects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load scripts/analyze_portfolio.py as a module
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def ap():
    """Return the analyze_portfolio module object."""
    spec = importlib.util.spec_from_file_location(
        "analyze_portfolio",
        _SCRIPTS_DIR / "analyze_portfolio.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_ascii(text: str, context: str = "") -> None:
    """Raise AssertionError if *text* contains any non-ASCII character."""
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        offending = text[max(0, exc.start - 10) : exc.end + 10]
        raise AssertionError(
            f"Non-ASCII character in {context or 'output'}: "
            f"...{offending!r}...  (full error: {exc})"
        ) from exc


def _make_stats(ap) -> object:
    """Build a minimal PortfolioStats-like mock accepted by _print_stats."""
    stats = MagicMock()
    stats.start_date.date.return_value = pd.Timestamp("2025-01-01").date()
    stats.end_date.date.return_value   = pd.Timestamp("2025-12-31").date()
    stats.n_days                       = 365
    stats.total_return                 = 0.12
    stats.benchmark_total_return       = 0.08
    stats.cagr                         = 0.11
    stats.benchmark_cagr               = 0.07
    stats.volatility                   = 0.18
    stats.benchmark_volatility         = 0.15
    stats.sharpe                       = 0.72
    stats.benchmark_sharpe             = 0.55
    stats.max_drawdown                 = -0.14
    stats.benchmark_max_drawdown       = -0.10
    stats.excess_return                = 0.04
    return stats


# ===========================================================================
# _sparkbar
# ===========================================================================

class TestSparkbar:
    def test_positive_return_is_ascii(self, ap):
        bar = ap._sparkbar(0.05)
        _assert_ascii(bar, "_sparkbar(positive)")

    def test_negative_return_is_ascii(self, ap):
        bar = ap._sparkbar(-0.05)
        _assert_ascii(bar, "_sparkbar(negative)")

    def test_zero_return_is_ascii(self, ap):
        bar = ap._sparkbar(0.0)
        _assert_ascii(bar, "_sparkbar(zero)")

    def test_large_return_is_ascii(self, ap):
        bar = ap._sparkbar(0.99)
        _assert_ascii(bar, "_sparkbar(large)")

    def test_positive_uses_hash(self, ap):
        bar = ap._sparkbar(0.05)
        assert "#" in bar

    def test_negative_uses_dot(self, ap):
        bar = ap._sparkbar(-0.05)
        assert "." in bar

    def test_empty_for_zero_return(self, ap):
        bar = ap._sparkbar(0.0)
        assert bar == ""


# ===========================================================================
# _print_stats
# ===========================================================================

class TestPrintStats:
    def test_output_is_ascii_only(self, ap, capsys):
        stats = _make_stats(ap)
        ap._print_stats(stats)
        out = capsys.readouterr().out
        _assert_ascii(out, "_print_stats output")

    def test_contains_total_return_label(self, ap, capsys):
        stats = _make_stats(ap)
        ap._print_stats(stats)
        out = capsys.readouterr().out
        assert "Total Return" in out

    def test_contains_sharpe_label(self, ap, capsys):
        stats = _make_stats(ap)
        ap._print_stats(stats)
        out = capsys.readouterr().out
        assert "Sharpe" in out

    def test_date_range_uses_ascii_arrow(self, ap, capsys):
        stats = _make_stats(ap)
        ap._print_stats(stats)
        out = capsys.readouterr().out
        assert "->" in out
        assert "\u2192" not in out   # Unicode → must not appear

    def test_separator_uses_hyphens(self, ap, capsys):
        stats = _make_stats(ap)
        ap._print_stats(stats)
        out = capsys.readouterr().out
        assert "---" in out          # at least three consecutive hyphens
        assert "\u2500" not in out   # BOX DRAWINGS LIGHT HORIZONTAL must not appear


# ===========================================================================
# _print_weekly_returns
# ===========================================================================

class TestPrintWeeklyReturns:
    def _make_weekly_returns(self) -> pd.Series:
        dates = pd.date_range("2025-01-06", periods=6, freq="W-MON", tz="UTC")
        return pd.Series(
            [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
            index=dates,
        )

    def test_output_is_ascii_only(self, ap, capsys):
        wr = self._make_weekly_returns()
        ap._print_weekly_returns(wr, n_weeks=6)
        out = capsys.readouterr().out
        _assert_ascii(out, "_print_weekly_returns output")

    def test_sparkbar_chars_are_ascii(self, ap, capsys):
        wr = self._make_weekly_returns()
        ap._print_weekly_returns(wr, n_weeks=6)
        out = capsys.readouterr().out
        assert "\u2588" not in out   # FULL BLOCK must not appear
        assert "\u2591" not in out   # LIGHT SHADE must not appear

    def test_empty_series_no_error(self, ap, capsys):
        ap._print_weekly_returns(pd.Series(dtype=float), n_weeks=8)
        out = capsys.readouterr().out
        _assert_ascii(out, "_print_weekly_returns(empty)")


# ===========================================================================
# _print_positions_snapshot
# ===========================================================================

class TestPrintPositionsSnapshot:
    def _make_history(self, ap):
        from tsml.portfolio.tracker import PortfolioHistory
        idx   = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
        hist  = PortfolioHistory(
            equity_curve=pd.Series([10_000.0, 10_200.0, 10_150.0], index=idx),
            positions=pd.DataFrame(
                {"AAPL": [0.0, 1.0, 1.0], "MSFT": [0.0, 0.0, 2.0]},
                index=idx,
            ),
            cash=pd.Series([10_000.0, 8_000.0, 7_500.0], index=idx),
        )
        return hist

    def test_output_is_ascii_only(self, ap, capsys):
        hist = self._make_history(ap)
        ap._print_positions_snapshot(hist)
        out = capsys.readouterr().out
        _assert_ascii(out, "_print_positions_snapshot output")

    def test_separator_uses_hyphens(self, ap, capsys):
        hist = self._make_history(ap)
        ap._print_positions_snapshot(hist)
        out = capsys.readouterr().out
        assert "\u2500" not in out


# ===========================================================================
# Source-level regression: no \uXXXX escape sequences in print calls
# ===========================================================================

class TestNoUnicodeEscapesInPrintCalls:
    """
    Guard against re-introducing Unicode escape sequences inside print() calls.

    The previous bug was that ``print(f"Date range: {start_date} \\u2192 {end_date}")``
    appeared to pass the non-ASCII byte grep (because \\u2192 is stored as 6
    ASCII bytes in the source file) but still raises UnicodeEncodeError at
    runtime on Windows cp1252 terminals, because Python evaluates the escape
    at runtime to produce the actual U+2192 character before passing it to
    stdout.

    This test reads the source files directly and asserts that no ``print(``
    call contains a ``\\uXXXX`` sequence.
    """

    _SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

    def _print_lines(self, filename: str) -> list[str]:
        """Return lines from *filename* that call print(...)."""
        path = self._SCRIPTS_DIR / filename
        return [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if "print(" in line
        ]

    def _has_unicode_escape(self, line: str) -> bool:
        import re
        return bool(re.search(r"\\u[0-9a-fA-F]{4}", line))

    def test_analyze_portfolio_no_unicode_escapes_in_print(self):
        offending = [
            line for line in self._print_lines("analyze_portfolio.py")
            if self._has_unicode_escape(line)
        ]
        assert offending == [], (
            "Unicode escapes found in print() calls:\n" +
            "\n".join(f"  {line.strip()}" for line in offending)
        )

    def test_weekly_job_no_unicode_escapes_in_print(self):
        offending = [
            line for line in self._print_lines("weekly_job.py")
            if self._has_unicode_escape(line)
        ]
        assert offending == [], (
            "Unicode escapes found in print() calls:\n" +
            "\n".join(f"  {line.strip()}" for line in offending)
        )
