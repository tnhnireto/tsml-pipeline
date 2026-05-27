"""
eToro API verification — confirm credentials and endpoints before trading.

This module contains the reusable verification logic shared between:

* ``scripts/verify_etoro_api.py`` — standalone CLI tool
* Any future pre-flight check that wants to validate the broker connection.

Calls three read-only broker methods in order:

1. ``get_account()``       — confirms API auth and returns balance/cash.
2. ``get_positions()``     — confirms position-list access.
3. ``get_instrument()``    — confirms instrument-lookup access (using AAPL).

No orders are ever placed.

Usage
-----
    from tsml.broker.verify import verify, format_report
    from tsml.broker.etoro_client import EtoroClient

    client = EtoroClient()
    result = verify(client)
    print(format_report(result))
    if not result.all_ok:
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tsml.broker.base import BrokerClient, BrokerError

# Symbol used for the instrument-lookup check.
_INSTRUMENT_CHECK_SYMBOL = "AAPL"

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """
    Outcome of one API call.

    Attributes
    ----------
    name:
        Human-readable check name shown in the report.
    ok:
        True if the call succeeded.
    detail:
        Short description of the successful response (e.g. balance figures).
    error:
        Exception message if the call failed; empty string on success.
    """

    name:   str
    ok:     bool
    detail: str = ""
    error:  str = ""


@dataclass
class VerifyResult:
    """
    Aggregated output of :func:`verify`.

    Attributes
    ----------
    checks:
        One :class:`CheckResult` per API call, in the order they were made.
    all_ok:
        True only when every check in *checks* passed.
    """

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.ok)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify(client: BrokerClient) -> VerifyResult:
    """
    Run three read-only API checks against *client*.

    Each check is tried independently — a failure in one does not prevent
    subsequent checks from running.  This produces a complete picture of
    which endpoints are reachable.

    Parameters
    ----------
    client:
        ``BrokerClient`` implementation (real or test stub).
        The client must already be authenticated (constructor succeeded).

    Returns
    -------
    VerifyResult
        Contains one :class:`CheckResult` per check.
    """
    result = VerifyResult()

    # ── 1. Account ───────────────────────────────────────────────────────
    try:
        account = client.get_account()
        result.checks.append(CheckResult(
            name="Account (get_account)",
            ok=True,
            detail=(
                f"balance=${account.balance:,.2f}  "
                f"cash=${account.cash:,.2f}  "
                f"mode={account.mode}"
            ),
        ))
    except BrokerError as exc:
        result.checks.append(CheckResult(
            name="Account (get_account)",
            ok=False,
            error=str(exc),
        ))

    # ── 2. Positions ─────────────────────────────────────────────────────
    try:
        positions = client.get_positions()
        count = len(positions)
        result.checks.append(CheckResult(
            name="Positions (get_positions)",
            ok=True,
            detail=f"{count} open position(s)",
        ))
    except BrokerError as exc:
        result.checks.append(CheckResult(
            name="Positions (get_positions)",
            ok=False,
            error=str(exc),
        ))

    # ── 3. Instrument lookup ─────────────────────────────────────────────
    sym = _INSTRUMENT_CHECK_SYMBOL
    try:
        instrument = client.get_instrument(sym)
        result.checks.append(CheckResult(
            name=f"Instrument lookup (get_instrument {sym})",
            ok=True,
            detail=(
                f"name={instrument.name!r}  "
                f"tradeable={instrument.tradeable}"
            ),
        ))
    except BrokerError as exc:
        result.checks.append(CheckResult(
            name=f"Instrument lookup (get_instrument {sym})",
            ok=False,
            error=str(exc),
        ))

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_W  = 68
_S2 = "=" * _W
_S1 = "-" * _W


def format_report(result: VerifyResult) -> str:
    """
    Return a human-readable ASCII verification report.

    Parameters
    ----------
    result:
        Output of :func:`verify`.

    Returns
    -------
    str
        Multi-line ASCII string suitable for ``print()``.
    """
    lines: list[str] = []
    a = lines.append

    a(_S2)
    a("  eToro API Verification")
    a(_S2)
    a("")

    for i, check in enumerate(result.checks, start=1):
        status = "[OK]" if check.ok else "[FAILED]"
        a(f"Check {i}: {check.name}")
        a(f"  Status : {status}")
        if check.ok and check.detail:
            a(f"  Detail : {check.detail}")
        if not check.ok and check.error:
            a(f"  Error  : {check.error}")
        a("")

    a(_S2)
    total = len(result.checks)
    passed = result.n_passed
    if result.all_ok:
        a(f"  ALL CHECKS PASSED  ({passed}/{total})")
    else:
        a(f"  VERIFICATION FAILED  ({passed}/{total} passed)")
    a(_S2)

    return "\n".join(lines)
