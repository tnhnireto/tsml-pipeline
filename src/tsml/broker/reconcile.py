"""
Broker reconciliation — compare local paper portfolio state with broker demo.

This module contains the reusable reconciliation logic shared between:

* ``scripts/reconcile_broker.py`` — standalone CLI tool
* ``run_etoro_demo.py``           — required gate before any live ``--execute``

Checks
------
1. **Cash** — local approximate cash vs. broker available cash.
   Allowed drift: up to ``CASH_TOLERANCE_ABS`` USD **or**
   ``CASH_TOLERANCE_REL`` relative.  Drift is expected because local SELL
   fills are not priced in dry-run mode.

2. **Held symbols** — the set of locally held symbols must equal the set of
   open positions reported by the broker.  Symbol mismatches always fail.

3. **Position sizes** (informational) — broker quantities are shown in the
   report but do not affect the exit code (sizes are not tracked locally).

Usage
-----
    from tsml.broker.reconcile import format_report, reconcile
    from tsml.portfolio.state import load_state
    from tsml.broker.etoro_client import EtoroClient

    state  = load_state()
    client = EtoroClient()
    result = reconcile(state, client)
    print(format_report(result, state))
    if not result.overall_ok:
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tsml.broker.base import AccountInfo, BrokerClient, PositionInfo
from tsml.portfolio.state import STATE_PATH, PortfolioState

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

CASH_TOLERANCE_ABS: float = 100.0   # USD — allow up to $100 drift
CASH_TOLERANCE_REL: float = 0.10    # allow up to 10 % of broker cash


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ReconcileResult:
    """
    Output of a single :func:`reconcile` call.

    Attributes
    ----------
    local_cash, broker_cash:
        Cash values from local state and broker account respectively.
    cash_diff:
        Absolute difference (always >= 0).
    cash_diff_pct:
        Relative difference as a fraction (0.05 = 5 %).
    cash_ok:
        True if cash difference is within tolerance.
    local_symbols, broker_symbols:
        Symbol sets from each source.
    only_local:
        Held locally but absent from broker (should be empty).
    only_broker:
        Open at broker but absent locally (should be empty).
    matched_symbols:
        Symbols present in both sources.
    positions_ok:
        True if ``only_local`` and ``only_broker`` are both empty.
    overall_ok:
        True if both ``cash_ok`` and ``positions_ok`` are True.
    broker_account:
        Raw ``AccountInfo`` from the broker.
    broker_positions:
        Raw ``PositionInfo`` list from the broker.
    """

    local_cash:       float
    broker_cash:      float
    cash_diff:        float
    cash_diff_pct:    float
    cash_ok:          bool
    local_symbols:    set[str]           = field(default_factory=set)
    broker_symbols:   set[str]           = field(default_factory=set)
    only_local:       set[str]           = field(default_factory=set)
    only_broker:      set[str]           = field(default_factory=set)
    matched_symbols:  set[str]           = field(default_factory=set)
    positions_ok:     bool               = True
    overall_ok:       bool               = True
    broker_account:   AccountInfo | None = None
    broker_positions: list[PositionInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def reconcile(
    state: PortfolioState,
    client: BrokerClient,
    cash_tolerance_abs: float = CASH_TOLERANCE_ABS,
    cash_tolerance_rel: float = CASH_TOLERANCE_REL,
) -> ReconcileResult:
    """
    Compare *state* (local paper portfolio) against the live broker account.

    Calls ``client.get_account()`` and ``client.get_positions()`` to obtain
    the broker-side snapshot.  No orders are placed.

    Parameters
    ----------
    state:
        Local portfolio state loaded from ``data/portfolio_state.json``.
    client:
        ``BrokerClient`` implementation (real or test stub).
    cash_tolerance_abs:
        Max allowed USD cash difference before cash is reported as a mismatch.
    cash_tolerance_rel:
        Max allowed relative cash difference (0.10 = 10 %).

    Returns
    -------
    ReconcileResult

    Raises
    ------
    BrokerError
        Propagated from ``client.get_account()`` or ``client.get_positions()``
        if the broker call fails.
    """
    account   = client.get_account()
    positions = client.get_positions()

    # ── Cash ────────────────────────────────────────────────────────────
    local_cash  = state.cash
    broker_cash = account.cash
    cash_diff   = abs(local_cash - broker_cash)
    cash_diff_pct = (
        cash_diff / broker_cash
        if broker_cash
        else (0.0 if cash_diff == 0.0 else 1.0)
    )
    cash_ok = (cash_diff <= cash_tolerance_abs) or (cash_diff_pct <= cash_tolerance_rel)

    # ── Symbols ──────────────────────────────────────────────────────────
    local_symbols  = set(state.positions.keys())
    broker_symbols = {p.symbol for p in positions}
    only_local     = local_symbols - broker_symbols
    only_broker    = broker_symbols - local_symbols
    matched        = local_symbols & broker_symbols
    positions_ok   = (len(only_local) == 0) and (len(only_broker) == 0)

    return ReconcileResult(
        local_cash=local_cash,
        broker_cash=broker_cash,
        cash_diff=cash_diff,
        cash_diff_pct=cash_diff_pct,
        cash_ok=cash_ok,
        local_symbols=local_symbols,
        broker_symbols=broker_symbols,
        only_local=only_local,
        only_broker=only_broker,
        matched_symbols=matched,
        positions_ok=positions_ok,
        overall_ok=cash_ok and positions_ok,
        broker_account=account,
        broker_positions=positions,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_W  = 68
_S2 = "=" * _W
_S1 = "-" * _W


def _fmt_cash(v: float) -> str:
    return f"${v:>10,.2f}"


def _ok_fail(flag: bool) -> str:
    return "[OK]" if flag else "[MISMATCH]"


def _sym_list(s: set[str]) -> str:
    return ", ".join(sorted(s)) if s else "(none)"


def format_report(
    result: ReconcileResult,
    state: PortfolioState,
    state_path=STATE_PATH,
    cash_tolerance_abs: float = CASH_TOLERANCE_ABS,
    cash_tolerance_rel: float = CASH_TOLERANCE_REL,
) -> str:
    """
    Return a human-readable reconciliation report as a single ASCII string.

    Parameters
    ----------
    result:
        Output of :func:`reconcile`.
    state:
        The local portfolio state (used for metadata such as last dates).
    state_path:
        Path of the state file shown in the report header.
    cash_tolerance_abs, cash_tolerance_rel:
        Tolerances echoed in the report.

    Returns
    -------
    str
        Multi-line ASCII string suitable for ``print()``.
    """
    lines: list[str] = []
    a = lines.append

    account = result.broker_account

    a(_S2)
    a("  Broker Reconciliation Report")
    a(_S2)
    a(f"  Local state file  : {state_path}")
    a(f"  Last signal date  : {state.last_signal_date or '(not set)'}")
    a(f"  Last rebalance    : {state.last_rebalance_date or '(not set)'}")
    if account:
        a(f"  Broker account ID : {account.account_id or '(unknown)'}")
        a(f"  Broker mode       : {account.mode}")
        a(f"  Broker balance    : {_fmt_cash(account.balance)}")
        a(f"  Broker equity     : {_fmt_cash(account.equity)}")
    a(_S2)
    a("")

    # ── Cash ─────────────────────────────────────────────────────────────
    a("Cash")
    a(_S1)
    a(f"  Local (paper)  : {_fmt_cash(result.local_cash)}")
    a(f"  Broker (demo)  : {_fmt_cash(result.broker_cash)}")
    a(f"  Difference     : {_fmt_cash(result.cash_diff)}  ({result.cash_diff_pct * 100:.1f} %)")
    tol_note = (
        f"within ${cash_tolerance_abs:,.0f} abs "
        f"or {cash_tolerance_rel * 100:.0f} % rel"
    )
    a(f"  Tolerance      : {tol_note}")
    a(f"  Status         : {_ok_fail(result.cash_ok)}")
    if not result.cash_ok:
        a("")
        a("  NOTE: Cash divergence is expected when SELL fills are not priced")
        a("  locally.  Investigate if the gap is large or growing unexpectedly.")
    a("")

    # ── Positions ─────────────────────────────────────────────────────────
    a("Positions")
    a(_S1)
    a(f"  Local held     : {_sym_list(result.local_symbols)}")
    a(f"  Broker open    : {_sym_list(result.broker_symbols)}")
    a(f"  Matched        : {_sym_list(result.matched_symbols)}")
    a(f"  Only local     : {_sym_list(result.only_local)}")
    a(f"  Only broker    : {_sym_list(result.only_broker)}")
    a(f"  Status         : {_ok_fail(result.positions_ok)}")
    a("")

    # ── Per-position detail (informational) ──────────────────────────────
    if result.broker_positions:
        a("Position detail (broker-reported quantities)")
        a(_S1)
        a(f"  {'Symbol':<10} {'Qty':>10}  {'Market value':>14}  Local")
        a(f"  {'-'*10} {'-'*10}  {'-'*14}  -----")
        for pos in sorted(result.broker_positions, key=lambda p: p.symbol):
            held = "held" if pos.symbol in result.local_symbols else "(not local)"
            a(
                f"  {pos.symbol:<10} {pos.quantity:>10,.4f}  "
                f"{_fmt_cash(pos.market_value):>14}  {held}"
            )
        a("")

    # ── Summary ─────────────────────────────────────────────────────────
    a(_S2)
    if result.overall_ok:
        a("  RECONCILIATION PASSED")
    else:
        a("  RECONCILIATION FAILED")
        if not result.cash_ok:
            a(
                f"    - Cash mismatch: ${result.cash_diff:,.2f} "
                f"({result.cash_diff_pct * 100:.1f} %)"
            )
        if result.only_local:
            a(f"    - Held locally but NOT at broker: {_sym_list(result.only_local)}")
        if result.only_broker:
            a(f"    - Open at broker but NOT locally: {_sym_list(result.only_broker)}")
    a(_S2)

    return "\n".join(lines)
