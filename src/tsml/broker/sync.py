"""
Broker-to-local portfolio state sync.

Build a :class:`~tsml.portfolio.state.PortfolioState` snapshot from a live
broker account (demo only via :class:`~tsml.broker.etoro_client.EtoroClient`).

Used by ``scripts/sync_state_from_broker.py``.  Never places orders.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsml.broker.base import BrokerClient, BrokerError
from tsml.portfolio.state import PortfolioState, STATE_PATH


@dataclass
class SyncResult:
    """
    Outcome of :func:`build_state_from_broker`.

    Attributes
    ----------
    state:
        Portfolio state derived from the broker account.
    broker_cash:
        Cash reported by the broker.
    broker_position_count:
        Number of open positions returned by the broker.
    symbols:
        Sorted list of symbols written into ``state.positions``.
    """

    state: PortfolioState
    broker_cash: float
    broker_position_count: int
    symbols: list[str]


def build_state_from_broker(client: BrokerClient) -> SyncResult:
    """
    Read broker account + positions and build a fresh local state snapshot.

    Parameters
    ----------
    client:
        Authenticated ``BrokerClient`` (demo).

    Returns
    -------
    SyncResult

    Raises
    ------
    BrokerError
        Propagated from ``get_account()`` or ``get_positions()``.
    """
    account = client.get_account()
    positions = client.get_positions()

    pos_map: dict[str, float] = {}
    for pos in positions:
        qty = float(pos.quantity)
        pos_map[pos.symbol] = qty if qty > 0 else 1.0

    state = PortfolioState(
        cash=float(account.cash),
        positions=pos_map,
        last_signal_date=None,
        last_rebalance_date=None,
    )
    symbols = sorted(pos_map.keys())
    return SyncResult(
        state=state,
        broker_cash=float(account.cash),
        broker_position_count=len(positions),
        symbols=symbols,
    )


_W = 68
_S2 = "=" * _W
_S1 = "-" * _W


def _fmt_cash(v: float) -> str:
    return f"${v:>12,.2f}"


def format_sync_report(
    result: SyncResult,
    *,
    state_path=STATE_PATH,
    confirmed: bool,
    written: bool,
) -> str:
    """
    Return a human-readable ASCII sync report.

    Parameters
    ----------
    result:
        Output of :func:`build_state_from_broker`.
    state_path:
        Target state file path shown in the report.
    confirmed:
        True when ``--confirm`` was passed.
    written:
        True when the state file was actually written.
    """
    lines: list[str] = []
    a = lines.append

    a(_S2)
    a("  Broker-to-Local State Sync")
    a(_S2)
    a("")
    a("Broker snapshot")
    a(_S1)
    a(f"  Cash              : {_fmt_cash(result.broker_cash)}")
    a(f"  Open positions    : {result.broker_position_count}")
    sym_list = ", ".join(result.symbols) if result.symbols else "(none)"
    a(f"  Symbols synced    : {sym_list}")
    a("")
    a("Local state to write")
    a(_S1)
    a(f"  cash              : {_fmt_cash(result.state.cash)}")
    a(f"  positions         : {len(result.state.positions)} symbol(s)")
    a(f"  last_signal_date  : {result.state.last_signal_date!r}")
    a(f"  last_rebalance    : {result.state.last_rebalance_date!r}")
    a(f"  path              : {state_path}")
    a("")

    if written:
        a(f"  Action            : WRITTEN")
    elif confirmed:
        a(f"  Action            : CONFIRM requested but not written")
    else:
        a(f"  Action            : DRY-RUN (pass --confirm to write)")
    a("")
    a(_S2)
    if written:
        a("  STATE FILE UPDATED")
    else:
        a("  NO CHANGES MADE")
    a(_S2)
    return "\n".join(lines)
