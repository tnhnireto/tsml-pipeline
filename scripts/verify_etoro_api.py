"""
eToro API verification CLI.

Verifies that the configured demo API credentials and endpoints respond
correctly **before** any trading execution is attempted.

Three read-only checks are performed in order:

1. ``get_account()``   — confirms API auth and returns balance / cash.
2. ``get_positions()`` — confirms position-list access.
3. ``get_instrument("AAPL")`` — confirms instrument-lookup access.

No orders are ever placed.

Exit codes
----------
0   All three checks passed.
1   One or more checks failed, or authentication / mode error.

Usage
-----
    python scripts/verify_etoro_api.py
    python scripts/verify_etoro_api.py --diagnose

Requires
--------
    ETORO_API_KEY       -- Public API Key from Settings > Trading (required)
    ETORO_USER_KEY      -- Demo generated user key (required)
    ETORO_ACCOUNT_MODE  -- must be "demo" (default)

Credentials may also be placed in ``.env`` at the project root (never commit).

Typical pre-flight workflow
---------------------------
    1.  python scripts/verify_etoro_api.py       # confirm API works
    2.  python scripts/reconcile_broker.py       # confirm state matches broker
    3.  python run_etoro_demo.py --execute       # execute (reconciliation runs again)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/verify_etoro_api.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsml.broker.base import BrokerAuthError, BrokerClient, BrokerModeError
from tsml.broker.diagnose import diagnose, format_diagnose_report
from tsml.broker.env_loader import load_etoro_env_files
from tsml.broker.verify import format_report, verify


def main(client: BrokerClient | None = None, *, run_diagnose: bool = False) -> int:
    """
    Create a broker client, run all verification checks, print the report.

    Parameters
    ----------
    client:
        Pass a stub ``BrokerClient`` to bypass real ``EtoroClient``
        construction — used in tests.  When ``None`` (the default),
        a real ``EtoroClient`` is created from environment variables.
    run_diagnose:
        When True, run HTTP probes and env hints before the normal checks.

    Returns
    -------
    int
        0 if all checks pass, 1 otherwise.
    """
    _SEP2 = "=" * 68

    load_etoro_env_files()

    print(_SEP2)
    print("  eToro API pre-flight verification")
    print("  Read-only: no orders will be placed")
    print(_SEP2)
    print()

    if run_diagnose and client is None:
        diag = diagnose()
        print(format_diagnose_report(diag))
        print()
        if not diag.env_ok:
            return 1

    if client is None:
        try:
            from tsml.broker.etoro_client import EtoroClient
            client = EtoroClient()
        except BrokerAuthError as exc:
            print(f"ERROR: Authentication failed: {exc}", file=sys.stderr)
            print(
                "  Set ETORO_API_KEY (Public API Key) and ETORO_USER_KEY "
                "(Demo user key) in .env or your shell, then retry.",
                file=sys.stderr,
            )
            print(
                "  Run: python scripts/verify_etoro_api.py --diagnose",
                file=sys.stderr,
            )
            return 1
        except BrokerModeError as exc:
            print(f"ERROR: Mode error: {exc}", file=sys.stderr)
            print("  Set ETORO_ACCOUNT_MODE=demo and retry.", file=sys.stderr)
            return 1

    result = verify(client)
    print(format_report(result))

    if not result.all_ok:
        print(
            "\n  Tip: run  python scripts/verify_etoro_api.py --diagnose  "
            "for per-endpoint HTTP details.",
            file=sys.stderr,
        )

    return 0 if result.all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify eToro demo API connectivity.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print env + HTTP probe details before running checks.",
    )
    args = parser.parse_args()
    sys.exit(main(run_diagnose=args.diagnose))
