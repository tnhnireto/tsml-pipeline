"""
Initialize or reset local portfolio state from the eToro demo broker account.

Reads the live demo account via :class:`~tsml.broker.etoro_client.EtoroClient`
and builds a fresh :class:`~tsml.portfolio.state.PortfolioState` for
``data/portfolio_state.json``.

Safety
------
* Never places orders.
* Without ``--confirm``: prints what would be written, exits 0, no file change.
* With ``--confirm``: writes ``data/portfolio_state.json``, exits 0.
* Secrets are never read from or written to the state file.

Usage
-----
    # Preview only (default)
    python scripts/sync_state_from_broker.py

    # Write state file
    python scripts/sync_state_from_broker.py --confirm

Requires
--------
    ETORO_API_KEY, ETORO_USER_KEY, ETORO_ACCOUNT_MODE=demo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/sync_state_from_broker.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsml.broker.base import BrokerAuthError, BrokerError, BrokerClient, BrokerModeError
from tsml.broker.sync import build_state_from_broker, format_sync_report
from tsml.portfolio.state import STATE_PATH, save_state


def main(
    client: BrokerClient | None = None,
    *,
    confirm: bool = False,
    state_path: Path | None = None,
) -> int:
    """
    Sync local portfolio state from the broker demo account.

    Parameters
    ----------
    client:
        Optional broker stub for tests.  When ``None``, creates ``EtoroClient``.
    confirm:
        When True, write ``state_path``.  Otherwise dry-run only.
    state_path:
        Override state file path (defaults to ``data/portfolio_state.json``).

    Returns
    -------
    int
        0 on success (including dry-run), 1 on auth/broker errors.
    """
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    path = state_path or STATE_PATH

    if client is None:
        try:
            from tsml.broker.etoro_client import EtoroClient
            client = EtoroClient()
        except (BrokerAuthError, BrokerModeError) as exc:
            print(f"ERROR: Could not create broker client: {exc}", file=sys.stderr)
            print(
                "  Set ETORO_API_KEY, ETORO_USER_KEY, and ETORO_ACCOUNT_MODE=demo.",
                file=sys.stderr,
            )
            return 1

    try:
        result = build_state_from_broker(client)
    except BrokerError as exc:
        print(f"ERROR: Broker call failed: {exc}", file=sys.stderr)
        return 1

    written = False
    if confirm:
        save_state(result.state, path)
        written = True

    print(format_sync_report(result, state_path=path, confirmed=confirm, written=written))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync data/portfolio_state.json from the eToro demo broker account.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Write data/portfolio_state.json (default is dry-run preview only).",
    )
    args = parser.parse_args()
    sys.exit(main(confirm=args.confirm))
