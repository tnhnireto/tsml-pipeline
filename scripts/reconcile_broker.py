"""
Broker reconciliation CLI — compare local paper state with broker demo account.

Purpose
-------
Before relying on ``run_etoro_demo.py --execute`` for real demo execution,
verify that the locally tracked ``data/portfolio_state.json`` is consistent
with what the broker actually reports.  Any divergence should be investigated
and corrected before live orders are submitted.

The reconciliation logic lives in ``tsml.broker.reconcile`` and is shared with
``run_etoro_demo.py``, which runs reconciliation automatically when
``--execute`` is passed.

Exit codes
----------
0   All critical checks passed (within tolerance).
1   One or more critical checks failed, or broker could not be reached.

Usage
-----
    python scripts/reconcile_broker.py

Requires
--------
    ETORO_API_KEY       -- eToro public API key
    ETORO_USER_KEY      -- demo (virtual) user key
    ETORO_ACCOUNT_MODE  -- must be "demo"

Do not pass ``--execute``.  This script never places orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/reconcile_broker.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsml.broker.base import BrokerAuthError, BrokerError, BrokerModeError
from tsml.broker.reconcile import format_report, reconcile
from tsml.portfolio.state import STATE_PATH, load_state


def main() -> int:
    """
    Load local state, query the broker, run reconciliation, print report.

    Returns
    -------
    int
        0 if reconciliation passed, 1 otherwise.
    """
    # Re-root to project root so relative paths (data/, logs/) resolve correctly
    # when the script is called from any working directory.
    import os
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    state_path = STATE_PATH
    state      = load_state(state_path)

    # ── Create broker client ─────────────────────────────────────────────
    try:
        from tsml.broker.etoro_client import EtoroClient
        client = EtoroClient()
    except (BrokerAuthError, BrokerModeError) as exc:
        print(f"ERROR: Could not create broker client: {exc}", file=sys.stderr)
        print(
            "  Make sure ETORO_API_KEY, ETORO_USER_KEY, and ETORO_ACCOUNT_MODE=demo are set.",
            file=sys.stderr,
        )
        return 1

    # ── Reconcile ─────────────────────────────────────────────────────────
    try:
        result = reconcile(state, client)
    except BrokerError as exc:
        print(f"ERROR: Broker call failed: {exc}", file=sys.stderr)
        print(
            "  Check your ETORO_API_KEY and network connection.",
            file=sys.stderr,
        )
        return 1

    # ── Print report ──────────────────────────────────────────────────────
    print(format_report(result, state, state_path=state_path))

    return 0 if result.overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
