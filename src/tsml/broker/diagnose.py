"""
eToro API connectivity diagnostics — pinpoint auth vs endpoint failures.

Used by ``scripts/verify_etoro_api.py --diagnose``.  Never prints key values.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import requests

from tsml.broker.env_loader import load_etoro_env_files
from tsml.broker.etoro_client import DEFAULT_BASE_URL, EtoroClient

_W = 68
_S2 = "=" * _W


@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


@dataclass
class DiagnoseResult:
    env_files: list[str] = field(default_factory=list)
    env_ok: bool = False
    env_detail: str = ""
    key_hints: list[str] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return self.env_ok and all(p.ok for p in self.probes)


def _mask_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        return "NOT SET"
    return f"set (len {len(val)})"


def _key_shape_hints() -> list[str]:
    hints: list[str] = []
    api = os.environ.get("ETORO_API_KEY", "").strip()
    user = os.environ.get("ETORO_USER_KEY", "").strip()
    if api and user and api == user:
        hints.append(
            "ETORO_API_KEY and ETORO_USER_KEY are identical - they must be "
            "different.  Public API Key (top of API Key Management) vs "
            "Generated User Key (Demo)."
        )
    if api.startswith("eyJ") and not user.startswith("eyJ"):
        hints.append(
            "ETORO_API_KEY looks like a User Key (starts with 'eyJ').  "
            "Use the Public API Key for ETORO_API_KEY instead."
        )
    if user and not user.startswith("eyJ") and api and not api.startswith("eyJ"):
        hints.append(
            "ETORO_USER_KEY does not look like a typical generated User Key.  "
            "Confirm you copied from Generated Keys (Demo), not the Public API Key."
        )
    return hints


def _probe_get(
    name: str,
    url: str,
    headers: dict[str, str],
    params: dict | None = None,
    auth_hint: str = "",
) -> ProbeResult:
    req_headers = {**headers, "x-request-id": str(uuid.uuid4())}
    try:
        resp = requests.get(url, headers=req_headers, params=params, timeout=15)
    except requests.RequestException as exc:
        return ProbeResult(name, False, f"network error: {exc}")

    if resp.status_code == 200:
        return ProbeResult(name, True, f"HTTP {resp.status_code}")

    body = resp.text[:300].replace("\n", " ")
    hint = auth_hint if resp.status_code in (401, 403) else ""
    if resp.status_code == 404:
        hint = "Endpoint path may be wrong - check API version."
    return ProbeResult(name, False, f"HTTP {resp.status_code}: {body}", hint)


def diagnose() -> DiagnoseResult:
    """Run env checks and three HTTP probes (portfolio, search, instruments)."""
    result = DiagnoseResult()

    loaded = load_etoro_env_files()
    result.env_files = [str(p) for p in loaded]

    api_ok = bool(os.environ.get("ETORO_API_KEY", "").strip())
    user_ok = bool(os.environ.get("ETORO_USER_KEY", "").strip())
    mode = os.environ.get("ETORO_ACCOUNT_MODE", "demo").strip().lower()

    parts = [
        f"ETORO_API_KEY={_mask_env('ETORO_API_KEY')}",
        f"ETORO_USER_KEY={_mask_env('ETORO_USER_KEY')}",
        f"ETORO_ACCOUNT_MODE={mode or '(not set, defaults demo)'}",
    ]
    result.env_detail = "  ".join(parts)
    result.env_ok = api_ok and user_ok
    result.key_hints = _key_shape_hints()

    if not result.env_ok:
        if not loaded:
            result.key_hints.append(
                "No .env file found.  Create .env in the project root or set "
                "variables in the same terminal/session that runs the script."
            )
        return result

    api_key = os.environ["ETORO_API_KEY"].strip()
    user_key = os.environ["ETORO_USER_KEY"].strip()
    base = DEFAULT_BASE_URL
    headers = {
        "x-api-key":  api_key,
        "x-user-key": user_key,
        "Accept":     "application/json",
    }
    auth_hint = (
        "Auth rejected.  Use Public API Key as ETORO_API_KEY and a Demo "
        "User Key as ETORO_USER_KEY (Settings > Trading > API Key Management).  "
        "Do not swap them.  Real keys fail on /demo/ endpoints."
    )

    result.probes.append(_probe_get(
        "Portfolio (demo)",
        f"{base}/trading/info/demo/portfolio",
        headers,
        auth_hint=auth_hint,
    ))
    result.probes.append(_probe_get(
        "Instrument search (AAPL)",
        f"{base}/market-data/search",
        headers,
        params={
            "internalSymbolFull": "AAPL",
            "fields": "instrumentId,internalSymbolFull,displayname,isCurrentlyTradable",
            "pageSize": 5,
            "pageNumber": 1,
        },
        auth_hint=auth_hint,
    ))
    result.probes.append(_probe_get(
        "PnL (demo, alternate)",
        f"{base}/trading/info/demo/pnl",
        headers,
        auth_hint=auth_hint,
    ))

    return result


def format_diagnose_report(result: DiagnoseResult) -> str:
    lines: list[str] = []
    a = lines.append

    a(_S2)
    a("  eToro API Diagnostics")
    a(_S2)
    a("")
    a("Environment")
    a("-" * _W)
    a(f"  {result.env_detail}")
    if result.env_files:
        a(f"  Loaded files: {', '.join(result.env_files)}")
    else:
        a("  Loaded files: (none - using process environment only)")
    a(f"  Env OK: {'yes' if result.env_ok else 'NO'}")
    a("")

    if result.key_hints:
        a("Key hints")
        a("-" * _W)
        for hint in result.key_hints:
            a(f"  ! {hint}")
        a("")

    if result.probes:
        a("HTTP probes")
        a("-" * _W)
        for probe in result.probes:
            status = "[OK]" if probe.ok else "[FAILED]"
            a(f"  {probe.name}: {status}")
            a(f"    {probe.detail}")
            if probe.hint:
                a(f"    Hint: {probe.hint}")
        a("")

    a(_S2)
    if result.all_ok:
        a("  ALL PROBES PASSED")
    else:
        a("  DIAGNOSTICS FAILED - see hints above")
    a(_S2)
    return "\n".join(lines)
