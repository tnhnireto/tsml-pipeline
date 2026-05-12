"""
Risk management — per-order validation rules for demo trading.

Every proposed order must pass ``validate_order`` before it is submitted to
the broker.  The function is stateless: it receives the order, the risk
configuration, and a snapshot of the current account state, and returns a
``RiskResult`` with ``approved=True/False`` and a human-readable reason.

Callers (``execution.py``) are responsible for tracking changes in account
state as each order in a batch is approved, so that position-count and
cash-buffer checks remain accurate within a single rebalance cycle.

Rules applied
-------------
1.  Mode must be ``"demo"`` — real trading is not permitted.
2.  No leverage — order leverage must equal 1 (not configurable).
3.  No short selling — SELL is only allowed when the symbol is currently held.
4.  Position count — BUY is only allowed when open positions < max_positions.
5.  Max trade size — order amount must not exceed
    ``max_trade_amount_pct * account.balance``.
6.  Cash buffer — after the trade, remaining cash must be >=
    ``cash_buffer_pct * account.balance``.
7.  Approved universe — if the universe set is non-empty, the symbol must be
    in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    Risk parameters for the demo execution layer.

    Parameters
    ----------
    mode:
        Must be ``"demo"``.  Any other value causes every order to be
        rejected — this is a hard safety gate, not a soft warning.
    max_positions:
        Maximum number of open positions at any time.
    max_position_pct:
        Maximum fraction of account balance a single position may represent.
        (Not enforced per-order; used for reporting only — position sizing
        in execution.py already caps trades at ``max_trade_amount_pct``.)
    max_trade_amount_pct:
        Maximum single-trade notional as a fraction of account balance.
    cash_buffer_pct:
        Minimum cash-to-balance ratio that must be maintained after each trade.
    approved_universe:
        Set of tickers that may be traded.  If empty, all symbols are
        accepted.
    """

    mode:                str            = "demo"
    max_positions:       int            = 5
    max_position_pct:    float          = 0.20
    max_trade_amount_pct: float         = 0.20
    cash_buffer_pct:     float          = 0.05
    approved_universe:   frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """
    Outcome of a single ``validate_order`` call.

    Attributes
    ----------
    approved:
        ``True`` if the order passed all risk checks.
    reason:
        Empty when approved.  Human-readable explanation when rejected.
    rule:
        Short tag identifying which rule fired (e.g. ``"mode"``,
        ``"short_sell"``, ``"max_positions"``).  Empty when approved.
    """

    approved: bool
    reason:   str = ""
    rule:     str = ""

    def __str__(self) -> str:
        if self.approved:
            return "APPROVED"
        return f"REJECTED [{self.rule}]: {self.reason}"


# ---------------------------------------------------------------------------
# Proposed order (input to validate_order)
# ---------------------------------------------------------------------------

@dataclass
class ProposedOrder:
    """
    A single trade proposed by the execution layer.

    Parameters
    ----------
    symbol:
        Ticker string (e.g. ``"AAPL"``).
    side:
        ``"BUY"`` or ``"SELL"``.
    amount:
        USD notional amount.
    score:
        Model score that drove the signal (informational only).
    signal_reason:
        Human-readable signal context (e.g. from ``SignalAction.reason``).
    leverage:
        Must always be 1 (no leverage).  Any other value is rejected by
        ``validate_order``.
    """

    symbol:        str
    side:          str      # "BUY" | "SELL"
    amount:        float    # USD notional
    score:         float    = 0.0
    signal_reason: str      = ""
    leverage:      int      = 1   # must stay at 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_order(
    order: ProposedOrder,
    config: RiskConfig,
    account_balance: float,
    account_cash: float,
    open_positions: list[str],
) -> RiskResult:
    """
    Validate a single proposed order against the risk rules.

    Parameters
    ----------
    order:
        The trade to validate.
    config:
        Risk configuration (thresholds, universe, mode).
    account_balance:
        Total account value in USD (used for percentage-based checks).
    account_cash:
        Available cash in USD (used for cash-buffer check).
    open_positions:
        Symbols of currently open positions.  The caller must keep this
        list up to date as orders are approved within a batch.

    Returns
    -------
    RiskResult
        ``approved=True`` if all rules passed, else ``approved=False`` with
        ``reason`` and ``rule`` filled in.
    """

    def _reject(rule: str, reason: str) -> RiskResult:
        return RiskResult(approved=False, reason=reason, rule=rule)

    # Rule 1: demo mode only
    if config.mode != "demo":
        return _reject(
            "mode",
            f"Account mode is '{config.mode}'.  Only 'demo' mode is supported.  "
            "Real trading is not permitted through this system.",
        )

    # Rule 2: no leverage
    if order.leverage != 1:
        return _reject(
            "leverage",
            f"Leverage {order.leverage}x is not allowed.  Only leverage=1 (no leverage) "
            "is permitted.",
        )

    # Rule 3: no short selling
    if order.side == "SELL" and order.symbol not in open_positions:
        return _reject(
            "short_sell",
            f"Cannot SELL {order.symbol}: symbol is not in current positions.  "
            "Short selling is not permitted.",
        )

    # Rule 4: position count limit (buy only)
    if order.side == "BUY" and len(open_positions) >= config.max_positions:
        return _reject(
            "max_positions",
            f"Cannot open new position in {order.symbol}: already holding "
            f"{len(open_positions)} / {config.max_positions} maximum positions.",
        )

    # Rule 5: max trade size
    if account_balance > 0:
        max_allowed = config.max_trade_amount_pct * account_balance
        if order.amount > max_allowed:
            return _reject(
                "max_trade_size",
                f"Order amount {order.amount:.2f} exceeds "
                f"{config.max_trade_amount_pct * 100:.0f}% of account balance "
                f"({max_allowed:.2f}).",
            )

    # Rule 6: cash buffer (buy only — selling returns cash)
    if order.side == "BUY":
        cash_after = account_cash - order.amount
        min_cash   = config.cash_buffer_pct * account_balance
        if cash_after < min_cash:
            return _reject(
                "cash_buffer",
                f"Insufficient cash: buying {order.amount:.2f} would leave "
                f"{cash_after:.2f} which is below the required "
                f"{config.cash_buffer_pct * 100:.0f}% buffer ({min_cash:.2f}).",
            )

    # Rule 7: approved universe (if non-empty)
    if config.approved_universe and order.symbol not in config.approved_universe:
        return _reject(
            "universe",
            f"{order.symbol} is not in the approved trading universe.",
        )

    return RiskResult(approved=True)
