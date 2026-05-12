"""
run_etoro_demo.py -- eToro demo account order execution.

Reads the most recent signal CSV from signals/, converts BUY/SELL actions
into proposed orders, validates them against the risk rules, prints an
order plan, and optionally submits to the eToro demo account.

IMPORTANT
---------
- Only demo mode is supported.  Real trading is never initiated here.
- Default behaviour is dry-run: no HTTP requests are sent unless
  ``--execute`` is passed AND the eToro API key is set.
- API key is read from the ``ETORO_API_KEY`` environment variable.
  Never hard-code credentials.
- All eToro API endpoint paths in etoro_client.py are marked TODO and
  must be verified against the official eToro API documentation before
  any live demo execution is attempted.

Usage
-----
    # Dry-run (default): build and print the order plan, no submission
    python run_etoro_demo.py

    # Execute against the demo account (requires ETORO_API_KEY)
    python run_etoro_demo.py --execute

Environment variables
---------------------
ETORO_API_KEY         Required for --execute.  Ignored in dry-run.
ETORO_ACCOUNT_MODE    Must be "demo" (default).

Setup
-----
    export ETORO_API_KEY=your_key_here
    export ETORO_ACCOUNT_MODE=demo
    python run_etoro_demo.py --execute
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from tsml.broker.base import BrokerAuthError, BrokerModeError
from tsml.broker.execution import (
    build_execution_plan,
    execute_plan,
    log_orders,
    print_plan,
    signals_to_proposed_orders,
)
from tsml.broker.risk import RiskConfig
from tsml.portfolio.strategy import SignalAction

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Approved trading universe — must match run_weekly_signal.py UNIVERSE.
# Only symbols in this list may be traded.
APPROVED_UNIVERSE: frozenset[str] = frozenset(
    [
        "SPY", "QQQ",
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
        "TSLA", "JPM", "JNJ", "XOM", "V", "GS", "NFLX",
    ]
)

RISK_CONFIG = RiskConfig(
    mode="demo",
    max_positions=5,
    max_position_pct=0.20,
    max_trade_amount_pct=0.20,
    cash_buffer_pct=0.05,
    approved_universe=APPROVED_UNIVERSE,
)

SIGNALS_DIR = Path("signals")


# ===========================================================================
# HELPERS
# ===========================================================================

def _load_latest_signals() -> tuple[pd.DataFrame, Path]:
    """Load the most recent *.csv file from signals/ by filename date."""
    files = sorted(SIGNALS_DIR.glob("*.csv"))
    if not files:
        print("ERROR: No signal files found in signals/.", file=sys.stderr)
        print("       Run run_weekly_signal.py first to generate signals.")
        sys.exit(1)
    latest = files[-1]
    df = pd.read_csv(latest)
    return df, latest


def _df_to_signal_actions(df: pd.DataFrame) -> list[SignalAction]:
    """
    Reconstruct SignalAction objects from the signal CSV.

    Only rows with action in {buy, sell, hold, blocked} are included.
    Rows with empty / NaN action (unranked symbols) are skipped.
    """
    valid = {"buy", "sell", "hold", "blocked"}
    actions = []
    for _, row in df.iterrows():
        action = str(row.get("action", "") or "").strip().lower()
        if action not in valid:
            continue
        reason = str(row.get("reason", "") or "")
        score  = float(row["score"]) if pd.notna(row.get("score")) else 0.0
        actions.append(SignalAction(str(row["symbol"]), action, score, reason))
    return actions


def _stub_account(df: pd.DataFrame) -> tuple[float, float, list[str]]:
    """
    Return placeholder account figures for dry-run mode.

    In dry-run, we cannot call the broker API, so we use a notional
    $10,000 demo account with no open positions.  Replace this with
    actual API calls (client.get_account(), client.get_positions())
    when --execute is used.
    """
    balance   = 10_000.0
    cash      = 10_000.0
    positions = []   # no current holdings assumed in dry-run
    return balance, cash, positions


def _live_account(client) -> tuple[float, float, list[str]]:
    """Fetch real account state from the broker."""
    account   = client.get_account()
    positions = client.get_positions()
    return account.balance, account.cash, [p.symbol for p in positions]


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="eToro demo account order execution (dry-run by default)."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Submit approved orders to the eToro demo account.  "
            "Requires ETORO_API_KEY to be set.  "
            "Without this flag the script prints the plan only."
        ),
    )
    args = parser.parse_args()

    dry_run = not args.execute

    _SEP2 = "=" * 70

    print(_SEP2)
    print("  eToro demo order execution")
    print(_SEP2)
    print(f"  Mode     : {'DRY-RUN (no orders submitted)' if dry_run else 'EXECUTE (demo account)'}")
    print(f"  API mode : {os.environ.get('ETORO_ACCOUNT_MODE', 'demo')}")
    print(_SEP2)
    print()

    # ── Load signals ────────────────────────────────────────────────────
    df, signal_file = _load_latest_signals()
    print(f"Signal file : {signal_file}")
    print(f"Signal date : {df['date'].iloc[0]}")
    print(f"Total rows  : {len(df)}")
    print()

    signals = _df_to_signal_actions(df)
    buys    = [s for s in signals if s.action == "buy"]
    sells   = [s for s in signals if s.action == "sell"]
    print(
        f"Actions from file: {len(buys)} BUY  |  {len(sells)} SELL  |  "
        f"{sum(1 for s in signals if s.action == 'hold')} HOLD  |  "
        f"{sum(1 for s in signals if s.action == 'blocked')} BLOCKED"
    )
    print()

    # ── Get account state ───────────────────────────────────────────────
    client = None

    if not dry_run:
        try:
            from tsml.broker.etoro_client import EtoroClient
            client = EtoroClient()
            balance, cash, open_positions = _live_account(client)
            print(
                f"Account   : balance=${balance:,.2f}  "
                f"cash=${cash:,.2f}  positions={open_positions}"
            )
        except (BrokerAuthError, BrokerModeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        balance, cash, open_positions = _stub_account(df)
        print(
            f"Account   : (stub) balance=${balance:,.2f}  "
            f"cash=${cash:,.2f}  positions={open_positions}"
        )
        print("           (dry-run: using placeholder account values)")
    print()

    # ── Build proposed orders ───────────────────────────────────────────
    proposed = signals_to_proposed_orders(signals, balance, RISK_CONFIG)
    print(f"Proposed orders : {len(proposed)}")

    # ── Validate with risk rules ────────────────────────────────────────
    plan = build_execution_plan(
        proposed,
        risk_config=RISK_CONFIG,
        account_balance=balance,
        account_cash=cash,
        open_positions=open_positions,
    )

    print_plan(plan, dry_run=dry_run)

    # ── Execute ─────────────────────────────────────────────────────────
    if not dry_run and client is not None:
        print("Submitting approved orders to eToro demo account ...")
        plan = execute_plan(plan, client, dry_run=False)
        print("Done.")
    elif not dry_run:
        print("WARNING: --execute passed but no client available.", file=sys.stderr)
    else:
        print(
            "Dry-run complete.  Pass --execute to submit orders to the demo account.\n"
            "Make sure ETORO_API_KEY is set and all TODO endpoint paths in\n"
            "src/tsml/broker/etoro_client.py are verified first."
        )

    # ── Log orders ───────────────────────────────────────────────────────
    if plan.all_records:
        log_path = log_orders(plan, dry_run=dry_run)
        print(f"\nOrder log written: {log_path}")


if __name__ == "__main__":
    main()
