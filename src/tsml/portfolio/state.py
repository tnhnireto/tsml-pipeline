"""
Portfolio state -- persistent local tracking for paper / demo trading.

The state file (``data/portfolio_state.json``) records which symbols are
currently held, the approximate cash balance, and when signals were last
generated and acted upon.  It is used to:

  * populate ``CURRENT_POSITIONS`` in ``run_weekly_signal.py`` so the
    signal generator knows which symbols to suggest HOLD/SELL for instead
    of issuing fresh BUYs;
  * provide the ``open_positions`` list that the risk engine checks against
    the ``max_positions`` limit in ``run_etoro_demo.py``.

Paper-trading limitations
--------------------------
* Positions are tracked as boolean holdings (share count = 1.0).  Actual
  share quantities are not stored because fills are not confirmed from a
  real broker in dry-run mode.
* Cash is decremented by the USD notional of each approved BUY.  On SELL
  the amount is unknown (broker resolves it), so cash is not incremented --
  the balance will therefore drift downward over time.  This is a known,
  documented limitation of the paper-trading approximation.
* The state file must never contain API keys or secrets.
* Add ``data/portfolio_state.json`` (or ``data/*.json``) to ``.gitignore``.

Typical usage
-------------
Load state (falls back to default if file absent)::

    from tsml.portfolio.state import load_state, STATE_PATH
    state = load_state()
    current_positions = set(state.positions.keys())

Commit state after dry-run with ``--commit-state``::

    from tsml.portfolio.state import apply_orders, save_state
    new_state = apply_orders(state, plan.approved, signal_date="2026-05-14")
    save_state(new_state)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STATE_PATH = Path("data") / "portfolio_state.json"

_DEFAULT_CASH = 10_000.0


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """
    Snapshot of local paper-trading state.

    Attributes
    ----------
    cash:
        Approximate available cash in USD.  Decremented on approved BUY;
        not incremented on SELL (fill price unknown in dry-run mode).
    positions:
        Map of ``symbol -> 1.0`` for each currently held symbol.  The value
        is always ``1.0`` -- a boolean placeholder for "held".
    last_signal_date:
        ISO date string (``YYYY-MM-DD``) of the most recent signal file used.
    last_rebalance_date:
        ISO date string of the last time ``--commit-state`` was applied.
    """

    cash:                float             = _DEFAULT_CASH
    positions:           dict[str, float]  = field(default_factory=dict)
    last_signal_date:    str | None        = None
    last_rebalance_date: str | None        = None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_state(path: Path = STATE_PATH) -> PortfolioState:
    """
    Load state from *path*.

    If the file does not exist or cannot be parsed, a fresh default state
    (cash = 10 000, no positions) is returned silently.  This ensures the
    first-ever run behaves as "all cash, no open positions" without requiring
    manual initialisation.

    Parameters
    ----------
    path:
        Path to the JSON state file.  Defaults to ``data/portfolio_state.json``.

    Returns
    -------
    PortfolioState
    """
    if not path.exists():
        return PortfolioState()

    try:
        raw  = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return PortfolioState()

    return PortfolioState(
        cash=float(data.get("cash", _DEFAULT_CASH)),
        positions={
            str(k): float(v)
            for k, v in data.get("positions", {}).items()
        },
        last_signal_date=data.get("last_signal_date"),
        last_rebalance_date=data.get("last_rebalance_date"),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_state(state: PortfolioState, path: Path = STATE_PATH) -> None:
    """
    Persist *state* to *path* as pretty-printed JSON.

    Creates parent directories if necessary.  Never writes secrets.

    Parameters
    ----------
    state:
        The state to persist.
    path:
        Destination file.  Defaults to ``data/portfolio_state.json``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "cash":                state.cash,
        "positions":           state.positions,
        "last_signal_date":    state.last_signal_date,
        "last_rebalance_date": state.last_rebalance_date,
    }
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Apply orders
# ---------------------------------------------------------------------------

def apply_orders(
    state: PortfolioState,
    approved_records: list,
    signal_date: str,
) -> PortfolioState:
    """
    Return a **new** ``PortfolioState`` with approved orders applied.

    The input *state* is never mutated.

    Parameters
    ----------
    state:
        Current portfolio state.
    approved_records:
        Sequence of ``OrderRecord`` objects from ``ExecutionPlan.approved``.
        Only the ``.order.side``, ``.order.symbol``, and ``.order.amount``
        fields are accessed (avoids a circular import).
    signal_date:
        ISO date string (``YYYY-MM-DD``) of the signal file that drove the
        orders.  Written to ``last_signal_date`` and ``last_rebalance_date``.

    Returns
    -------
    PortfolioState
        A new state with positions and cash updated.

    Notes
    -----
    * BUY:  adds the symbol to positions; decrements ``cash`` by the order's
      USD notional.
    * SELL: removes the symbol from positions; cash is **not** incremented
      because the fill price is unknown in dry-run mode.
    * Cash is floored at 0 to prevent negative values from accounting drift.
    """
    new_cash      = state.cash
    new_positions = dict(state.positions)   # shallow copy -- values are floats

    for record in approved_records:
        o = record.order
        if o.side == "BUY":
            new_positions[o.symbol] = 1.0   # boolean: held
            new_cash -= o.amount
        elif o.side == "SELL":
            new_positions.pop(o.symbol, None)
            # cash not incremented: fill price unknown in dry-run mode

    return PortfolioState(
        cash=max(0.0, new_cash),
        positions=new_positions,
        last_signal_date=signal_date,
        last_rebalance_date=signal_date,
    )
