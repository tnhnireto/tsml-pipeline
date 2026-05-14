"""
Tests for signal-file selection and safety validation in run_etoro_demo.py.

Coverage
--------
- _load_latest_signals  : file selection (dated only, latest wins)
- _validate_signal_file : rejects multi-date files and missing date column
- _df_to_signal_actions : blocked rows are excluded
- Integration           : blocked rows never appear in proposed orders
                          (JPM-style regression)

No network calls, no broker API calls, no subprocess spawning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load run_etoro_demo.py as a module
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo():
    """Return the run_etoro_demo module object."""
    spec = importlib.util.spec_from_file_location(
        "run_etoro_demo",
        _ROOT / "run_etoro_demo.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "date,rank,symbol,score,action,reason,"
    "return_20d,return_60d,volatility_20d,price_vs_sma_200,above_sma_200"
)


def _write_signal_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal signal CSV with the given rows."""
    lines = [_CSV_HEADER]
    for r in rows:
        lines.append(
            f"{r.get('date','2026-05-14')},"
            f"{r.get('rank',1)},"
            f"{r.get('symbol','AAPL')},"
            f"{r.get('score',0.62)},"
            f"{r.get('action','buy')},"
            f"{r.get('reason','')},"
            f"{r.get('return_20d',0.05)},"
            f"{r.get('return_60d',0.12)},"
            f"{r.get('volatility_20d',0.18)},"
            f"{r.get('price_vs_sma_200',0.03)},"
            f"{r.get('above_sma_200','True')}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# _load_latest_signals — file selection
# ===========================================================================

class TestLoadLatestSignals:
    def test_picks_dated_file_over_analysis_csv(self, demo, tmp_path):
        """
        analysis.csv sorts after YYYY-MM-DD.csv alphabetically ('a' > '2').
        It must be excluded even though it would be selected by a naive
        sorted(...)[−1] approach.
        """
        _write_signal_csv(tmp_path / "2026-05-14.csv", [{"symbol": "AAPL", "action": "buy"}])
        _write_signal_csv(tmp_path / "analysis.csv",   [{"symbol": "SPY",  "action": "sell"}])

        df, path = demo._load_latest_signals(tmp_path)

        assert path.name == "2026-05-14.csv"
        assert "AAPL" in df["symbol"].values
        assert "SPY"  not in df["symbol"].values

    def test_analysis_csv_alone_causes_exit(self, demo, tmp_path):
        """If only analysis.csv exists (no dated file), the function exits."""
        _write_signal_csv(tmp_path / "analysis.csv", [{"symbol": "SPY", "action": "buy"}])

        with pytest.raises(SystemExit):
            demo._load_latest_signals(tmp_path)

    def test_latest_of_multiple_dated_files_selected(self, demo, tmp_path):
        """Given several dated files the one with the most recent date wins."""
        _write_signal_csv(tmp_path / "2026-05-10.csv", [{"symbol": "MSFT", "action": "buy"}])
        _write_signal_csv(tmp_path / "2026-05-12.csv", [{"symbol": "NVDA", "action": "buy"}])
        _write_signal_csv(tmp_path / "2026-05-14.csv", [{"symbol": "AAPL", "action": "buy"}])

        df, path = demo._load_latest_signals(tmp_path)

        assert path.name == "2026-05-14.csv"
        assert "AAPL" in df["symbol"].values

    def test_older_dated_file_not_selected(self, demo, tmp_path):
        _write_signal_csv(tmp_path / "2026-05-12.csv", [{"symbol": "OLD",  "action": "buy"}])
        _write_signal_csv(tmp_path / "2026-05-14.csv", [{"symbol": "AAPL", "action": "buy"}])

        df, _ = demo._load_latest_signals(tmp_path)

        assert "OLD" not in df["symbol"].values

    def test_non_date_csv_excluded(self, demo, tmp_path):
        """Files like summary.csv, report.csv are excluded."""
        _write_signal_csv(tmp_path / "2026-05-14.csv", [{"symbol": "AAPL", "action": "buy"}])
        _write_signal_csv(tmp_path / "summary.csv",    [{"symbol": "BAD",  "action": "buy"}])
        _write_signal_csv(tmp_path / "report.csv",     [{"symbol": "ALSO_BAD", "action": "sell"}])

        df, path = demo._load_latest_signals(tmp_path)

        assert path.name == "2026-05-14.csv"
        assert "BAD"      not in df["symbol"].values
        assert "ALSO_BAD" not in df["symbol"].values

    def test_empty_signals_dir_causes_exit(self, demo, tmp_path):
        with pytest.raises(SystemExit):
            demo._load_latest_signals(tmp_path)

    def test_returns_dataframe_and_path(self, demo, tmp_path):
        _write_signal_csv(tmp_path / "2026-05-14.csv", [{"symbol": "AAPL", "action": "buy"}])
        df, path = demo._load_latest_signals(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(path, Path)


# ===========================================================================
# _validate_signal_file — multi-date rejection
# ===========================================================================

class TestValidateSignalFile:
    def test_single_date_passes(self, demo, tmp_path):
        path = tmp_path / "2026-05-14.csv"
        _write_signal_csv(path, [
            {"date": "2026-05-14", "symbol": "AAPL"},
            {"date": "2026-05-14", "symbol": "MSFT"},
        ])
        df = pd.read_csv(path)
        demo._validate_signal_file(df, path)   # must not raise

    def test_multi_date_raises_value_error(self, demo, tmp_path):
        """A file with two unique dates must be rejected."""
        path = tmp_path / "2026-05-14.csv"
        _write_signal_csv(path, [
            {"date": "2026-05-12", "symbol": "AAPL"},
            {"date": "2026-05-14", "symbol": "MSFT"},
        ])
        df = pd.read_csv(path)
        with pytest.raises(ValueError, match="unique dates"):
            demo._validate_signal_file(df, path)

    def test_multi_date_error_message_lists_dates(self, demo, tmp_path):
        path = tmp_path / "2026-05-14.csv"
        _write_signal_csv(path, [
            {"date": "2026-05-12", "symbol": "AAPL"},
            {"date": "2026-05-14", "symbol": "MSFT"},
        ])
        df = pd.read_csv(path)
        with pytest.raises(ValueError) as exc_info:
            demo._validate_signal_file(df, path)
        msg = str(exc_info.value)
        assert "2026-05-12" in msg or "2026-05-14" in msg

    def test_missing_date_column_raises(self, demo, tmp_path):
        path = tmp_path / "2026-05-14.csv"
        path.write_text("symbol,score,action\nAAPL,0.62,buy\n", encoding="utf-8")
        df = pd.read_csv(path)
        with pytest.raises(ValueError, match="date"):
            demo._validate_signal_file(df, path)

    def test_load_rejects_multi_date_file(self, demo, tmp_path):
        """
        _load_latest_signals must propagate the ValueError from
        _validate_signal_file so callers see a clear error.
        """
        path = tmp_path / "2026-05-14.csv"
        _write_signal_csv(path, [
            {"date": "2026-05-12", "symbol": "AAPL"},
            {"date": "2026-05-14", "symbol": "MSFT"},
        ])
        with pytest.raises(ValueError, match="unique dates"):
            demo._load_latest_signals(tmp_path)


# ===========================================================================
# _df_to_signal_actions — blocked exclusion
# ===========================================================================

class TestDfToSignalActions:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        records = []
        for r in rows:
            records.append({
                "date":   r.get("date", "2026-05-14"),
                "symbol": r.get("symbol", "AAPL"),
                "score":  r.get("score", 0.62),
                "action": r.get("action", "buy"),
                "reason": r.get("reason", ""),
            })
        return pd.DataFrame(records)

    def test_buy_included(self, demo):
        df = self._make_df([{"symbol": "AAPL", "action": "buy"}])
        actions = demo._df_to_signal_actions(df)
        assert any(a.symbol == "AAPL" and a.action == "buy" for a in actions)

    def test_sell_included(self, demo):
        df = self._make_df([{"symbol": "MSFT", "action": "sell"}])
        actions = demo._df_to_signal_actions(df)
        assert any(a.symbol == "MSFT" and a.action == "sell" for a in actions)

    def test_hold_included(self, demo):
        df = self._make_df([{"symbol": "NVDA", "action": "hold"}])
        actions = demo._df_to_signal_actions(df)
        assert any(a.symbol == "NVDA" and a.action == "hold" for a in actions)

    def test_blocked_excluded(self, demo):
        """blocked rows must be filtered out entirely at this stage."""
        df = self._make_df([
            {"symbol": "AAPL", "action": "buy"},
            {"symbol": "JPM",  "action": "blocked", "reason": "blocked: below SMA200"},
        ])
        actions = demo._df_to_signal_actions(df)
        symbols = [a.symbol for a in actions]
        assert "JPM" not in symbols

    def test_blocked_row_does_not_produce_any_action(self, demo):
        df = self._make_df([{"symbol": "JPM", "action": "blocked"}])
        actions = demo._df_to_signal_actions(df)
        assert actions == []

    def test_empty_action_excluded(self, demo):
        df = self._make_df([{"symbol": "AAPL", "action": ""}])
        actions = demo._df_to_signal_actions(df)
        assert actions == []

    def test_unknown_action_excluded(self, demo):
        df = self._make_df([{"symbol": "AAPL", "action": "maybe"}])
        actions = demo._df_to_signal_actions(df)
        assert actions == []

    def test_mixed_actions_correct_count(self, demo):
        df = self._make_df([
            {"symbol": "AAPL", "action": "buy"},
            {"symbol": "MSFT", "action": "sell"},
            {"symbol": "NVDA", "action": "hold"},
            {"symbol": "JPM",  "action": "blocked"},
            {"symbol": "META", "action": ""},
        ])
        actions = demo._df_to_signal_actions(df)
        assert len(actions) == 3   # buy + sell + hold; blocked and empty excluded


# ===========================================================================
# Integration: blocked rows never reach proposed orders
# ===========================================================================

class TestBlockedNeverProposed:
    def _make_signals_df_with_blocked(self, blocked_symbol: str = "JPM") -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2026-05-14", "symbol": "AAPL", "score": 0.70, "action": "buy",     "reason": ""},
            {"date": "2026-05-14", "symbol": "MSFT", "score": 0.65, "action": "buy",     "reason": ""},
            {"date": "2026-05-14", "symbol": blocked_symbol,
                                                     "score": 0.58, "action": "blocked", "reason": "blocked: below SMA200 and score below 0.62"},
            {"date": "2026-05-14", "symbol": "META", "score": 0.55, "action": "hold",    "reason": ""},
        ])

    def test_blocked_symbol_not_in_signal_actions(self, demo):
        df = self._make_signals_df_with_blocked("JPM")
        actions = demo._df_to_signal_actions(df)
        assert not any(a.symbol == "JPM" for a in actions)

    def test_blocked_symbol_not_in_proposed_orders(self, demo):
        """End-to-end: JPM blocked -> not in proposed orders."""
        from tsml.broker.execution import signals_to_proposed_orders
        from tsml.broker.risk import RiskConfig

        df      = self._make_signals_df_with_blocked("JPM")
        actions = demo._df_to_signal_actions(df)
        orders  = signals_to_proposed_orders(actions, 10_000.0, RiskConfig())

        order_symbols = [o.symbol for o in orders]
        assert "JPM" not in order_symbols

    def test_blocked_symbol_not_a_buy_order(self, demo):
        """JPM-style regression: blocked symbol must never appear as BUY."""
        from tsml.broker.execution import signals_to_proposed_orders
        from tsml.broker.risk import RiskConfig

        df      = self._make_signals_df_with_blocked("JPM")
        actions = demo._df_to_signal_actions(df)
        orders  = signals_to_proposed_orders(actions, 10_000.0, RiskConfig())

        buy_symbols = [o.symbol for o in orders if o.side == "BUY"]
        assert "JPM" not in buy_symbols

    def test_non_blocked_symbols_still_proposed(self, demo):
        """Filtering blocked rows must not accidentally remove buy/hold rows."""
        from tsml.broker.execution import signals_to_proposed_orders
        from tsml.broker.risk import RiskConfig

        df      = self._make_signals_df_with_blocked("JPM")
        actions = demo._df_to_signal_actions(df)
        orders  = signals_to_proposed_orders(actions, 10_000.0, RiskConfig())

        order_symbols = [o.symbol for o in orders]
        assert "AAPL" in order_symbols
        assert "MSFT" in order_symbols

    def test_any_blocked_symbol_excluded(self, demo):
        """Parametric: any symbol with action=blocked is never proposed."""
        from tsml.broker.execution import signals_to_proposed_orders
        from tsml.broker.risk import RiskConfig

        for blocked in ["JPM", "GS", "XOM", "TSLA"]:
            df      = self._make_signals_df_with_blocked(blocked)
            actions = demo._df_to_signal_actions(df)
            orders  = signals_to_proposed_orders(actions, 10_000.0, RiskConfig())
            order_symbols = [o.symbol for o in orders]
            assert blocked not in order_symbols, (
                f"{blocked} was blocked but appeared in proposed orders"
            )
