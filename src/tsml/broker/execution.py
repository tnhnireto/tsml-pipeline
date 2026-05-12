"""
Execution layer — convert signal actions into validated demo orders.

Workflow
--------
1.  ``signals_to_proposed_orders`` translates a list of ``SignalAction``
    objects (from ``generate_signals``) into ``ProposedOrder`` objects.
    BUY → buy order sized at ``max_trade_amount_pct * balance``.
    SELL → sell order for the full position.
    HOLD / BLOCKED / no-action → skipped (no order generated).

2.  ``build_execution_plan`` runs every proposed order through
    ``validate_order`` (risk.py), tracks running position count and
    remaining cash, and returns an ``ExecutionPlan`` separating approved
    from rejected orders.

3.  ``execute_plan`` submits approved orders to the broker client, or
    returns dry-run results if ``dry_run=True``.

4.  ``log_orders`` appends a JSONL record for each order (proposed,
    approved, or rejected) to ``logs/orders/YYYY-MM-DD.jsonl``.
    API keys and secrets are never written to the log.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from tsml.broker.base import BrokerClient, OrderResult
from tsml.broker.risk import ProposedOrder, RiskConfig, RiskResult, validate_order

if TYPE_CHECKING:
    from tsml.portfolio.strategy import SignalAction

LOGS_DIR = Path("logs") / "orders"

# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------

@dataclass
class OrderRecord:
    """One entry in the execution plan (approved or rejected)."""

    order:       ProposedOrder
    risk_result: RiskResult
    broker_result: OrderResult | None = None   # set after submission

    @property
    def approved(self) -> bool:
        return self.risk_result.approved


@dataclass
class ExecutionPlan:
    """
    Full order plan for one rebalance cycle.

    ``approved`` and ``rejected`` are populated by ``build_execution_plan``.
    ``broker_results`` is populated by ``execute_plan`` after submission.
    """

    approved:         list[OrderRecord] = field(default_factory=list)
    rejected:         list[OrderRecord] = field(default_factory=list)
    account_balance:  float = 0.0
    cash_before:      float = 0.0
    cash_after_est:   float = 0.0   # estimated remaining cash after all buys
    generated_at:     str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def all_records(self) -> list[OrderRecord]:
        return self.approved + self.rejected


# ---------------------------------------------------------------------------
# Step 1: signals → proposed orders
# ---------------------------------------------------------------------------

def signals_to_proposed_orders(
    signals: list[SignalAction],
    account_balance: float,
    risk_config: RiskConfig,
) -> list[ProposedOrder]:
    """
    Convert a list of ``SignalAction`` objects into ``ProposedOrder`` objects.

    - ``"buy"``     → BUY order sized at ``max_trade_amount_pct * balance``
    - ``"sell"``    → SELL order (amount set to 0.0; broker handles full exit)
    - ``"hold"``    → skipped
    - ``"blocked"`` → skipped (already filtered by the risk/signal layer)
    """
    orders: list[ProposedOrder] = []
    trade_amount = risk_config.max_trade_amount_pct * account_balance

    for sig in signals:
        if sig.action == "buy":
            orders.append(
                ProposedOrder(
                    symbol=sig.symbol,
                    side="BUY",
                    amount=round(trade_amount, 2),
                    score=sig.score,
                    signal_reason=sig.reason,
                )
            )
        elif sig.action == "sell":
            orders.append(
                ProposedOrder(
                    symbol=sig.symbol,
                    side="SELL",
                    amount=0.0,   # full-position close; broker resolves actual size
                    score=sig.score,
                    signal_reason=sig.reason,
                )
            )
        # hold, blocked, (no action) → no order

    return orders


# ---------------------------------------------------------------------------
# Step 2: build execution plan with risk validation
# ---------------------------------------------------------------------------

def build_execution_plan(
    proposed: list[ProposedOrder],
    risk_config: RiskConfig,
    account_balance: float,
    account_cash: float,
    open_positions: list[str],
) -> ExecutionPlan:
    """
    Run every proposed order through risk validation and build an
    ``ExecutionPlan``.

    State (running cash and position list) is updated after each approved
    order so subsequent orders see the correct context.
    """
    plan = ExecutionPlan(
        account_balance=account_balance,
        cash_before=account_cash,
        cash_after_est=account_cash,
    )

    # Work on a copy so we don't mutate the caller's list.
    running_cash      = account_cash
    running_positions = list(open_positions)

    for order in proposed:
        result = validate_order(
            order,
            risk_config,
            account_balance=account_balance,
            account_cash=running_cash,
            open_positions=running_positions,
        )

        record = OrderRecord(order=order, risk_result=result)

        if result.approved:
            plan.approved.append(record)
            # Update running state for subsequent orders in this batch.
            if order.side == "BUY":
                running_cash -= order.amount
                if order.symbol not in running_positions:
                    running_positions.append(order.symbol)
            elif order.side == "SELL":
                if order.symbol in running_positions:
                    running_positions.remove(order.symbol)
        else:
            plan.rejected.append(record)

    plan.cash_after_est = running_cash
    return plan


# ---------------------------------------------------------------------------
# Step 3: execute (or dry-run) the plan
# ---------------------------------------------------------------------------

def execute_plan(
    plan: ExecutionPlan,
    client: BrokerClient,
    dry_run: bool = True,
) -> ExecutionPlan:
    """
    Submit all approved orders in ``plan`` to ``client``.

    When ``dry_run=True`` (the default) ``client.place_order`` is called with
    ``dry_run=True``, which means no HTTP POST is sent — see
    :meth:`EtoroClient.place_order`.
    """
    for record in plan.approved:
        o = record.order
        try:
            broker_result = client.place_order(
                symbol=o.symbol,
                side=o.side,
                amount=o.amount,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            broker_result = None
            print(
                f"  [execution] ERROR placing {o.side} {o.symbol}: {exc}",
                file=sys.stderr,
            )
        record.broker_result = broker_result

    return plan


# ---------------------------------------------------------------------------
# Step 4: order logging
# ---------------------------------------------------------------------------

def log_orders(plan: ExecutionPlan, dry_run: bool) -> Path:
    """
    Append all order records in ``plan`` to a JSONL file under ``logs/orders/``.

    The filename is ``YYYY-MM-DD.jsonl`` using the UTC date of plan generation.
    Each line is one JSON object.  API keys and secrets are never written.

    Returns the path of the log file written.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    date_str  = plan.generated_at[:10]   # "YYYY-MM-DD"
    log_path  = LOGS_DIR / f"{date_str}.jsonl"

    with log_path.open("a", encoding="utf-8") as fh:
        for record in plan.all_records:
            o = record.order
            entry = {
                "timestamp":     plan.generated_at,
                "dry_run":       dry_run,
                "type":          "approved" if record.approved else "rejected",
                "symbol":        o.symbol,
                "side":          o.side,
                "amount":        o.amount,
                "score":         o.score,
                "signal_reason": o.signal_reason,
                "risk_approved": record.approved,
                "risk_rule":     record.risk_result.rule,
                "risk_reason":   record.risk_result.reason,
                "broker_status": (
                    record.broker_result.status
                    if record.broker_result else None
                ),
                "broker_order_id": (
                    record.broker_result.order_id
                    if record.broker_result else None
                ),
            }
            fh.write(json.dumps(entry) + "\n")

    return log_path


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_plan(plan: ExecutionPlan, dry_run: bool) -> None:
    """Print the execution plan to stdout in a human-readable format."""
    _SEP  = "-" * 70
    _SEP2 = "=" * 70
    mode  = "DRY-RUN" if dry_run else "LIVE (demo account)"

    print()
    print(_SEP2)
    print(f"  Execution plan  [{mode}]")
    print(_SEP2)
    print(f"  Account balance : {plan.account_balance:,.2f}")
    print(f"  Cash before     : {plan.cash_before:,.2f}")
    print(f"  Cash after (est): {plan.cash_after_est:,.2f}")
    print(_SEP2)

    if plan.approved:
        print(f"  APPROVED ({len(plan.approved)})")
        print(_SEP)
        for rec in plan.approved:
            o = rec.order
            print(f"    {o.side:<4}  {o.symbol:<8}  ${o.amount:>9,.2f}  "
                  f"score: {o.score:.3f}")
        print()

    if plan.rejected:
        print(f"  REJECTED ({len(plan.rejected)})")
        print(_SEP)
        for rec in plan.rejected:
            o   = rec.order
            rr  = rec.risk_result
            print(f"    {o.side:<4}  {o.symbol:<8}  "
                  f"[{rr.rule}]  {rr.reason}")
        print()

    if not plan.approved and not plan.rejected:
        print("  No orders proposed.")

    print(_SEP2)
