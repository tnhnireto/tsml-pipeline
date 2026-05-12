"""
eToro HTTP client skeleton.

IMPORTANT — API endpoint status
--------------------------------
eToro's Public API is available to enrolled partners.  Exact endpoint paths,
request/response schemas, and authentication header names **must be verified
against the official eToro API documentation** before any real calls are made.

Every endpoint path in this file is tagged ``# TODO: verify`` and uses a
plausible placeholder.  The client structure (REST over HTTPS, API-key
authentication, JSON payloads) reflects eToro's publicly described API style,
but no path should be considered confirmed.

Obtaining API access
--------------------
1. Enrol in the eToro Partner / Developer programme.
2. Retrieve your API key from the partner dashboard.
3. Set the environment variable ``ETORO_API_KEY=<your_key>``.
4. Set ``ETORO_ACCOUNT_MODE=demo`` (the only mode supported here).

Environment variables
---------------------
ETORO_API_KEY
    Required.  Your eToro API key.  Never hard-code this value.
ETORO_ACCOUNT_MODE
    ``"demo"`` (default).  ``"real"`` is rejected by this client.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from tsml.broker.base import (
    AccountInfo,
    BrokerAuthError,
    BrokerError,
    BrokerModeError,
    InstrumentInfo,
    OrderResult,
    PositionInfo,
)

# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------
# All paths are placeholders.  Replace with values from official documentation.

_ENDPOINTS: dict[str, str] = {
    "account":    "/v1/account",                    # TODO: verify
    "positions":  "/v1/positions",                  # TODO: verify
    "instrument": "/v1/instruments/{symbol}",       # TODO: verify
    "orders":     "/v1/orders",                     # TODO: verify
}

_SUPPORTED_MODES = frozenset({"demo"})

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class EtoroClient:
    """
    Skeleton HTTP client for the eToro Public API.

    Parameters
    ----------
    base_url:
        Override the API base URL.  Defaults to the eToro production host.
        Set to a mock server URL in tests.
    timeout:
        Per-request timeout in seconds.

    Raises
    ------
    BrokerAuthError
        If ``ETORO_API_KEY`` is not set in the environment.
    BrokerModeError
        If ``ETORO_ACCOUNT_MODE`` is not ``"demo"``.
    """

    # TODO: confirm the correct base URL with eToro documentation.
    DEFAULT_BASE_URL = "https://api.etoro.com"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        api_key = os.environ.get("ETORO_API_KEY", "").strip()
        if not api_key:
            raise BrokerAuthError(
                "ETORO_API_KEY environment variable is not set or is empty.  "
                "Obtain your API key from the eToro partner dashboard and "
                "export it before running this script.\n"
                "    export ETORO_API_KEY=your_key_here"
            )

        mode = os.environ.get("ETORO_ACCOUNT_MODE", "demo").strip().lower()
        if mode not in _SUPPORTED_MODES:
            raise BrokerModeError(
                f"Account mode '{mode}' is not supported.  "
                f"Only {sorted(_SUPPORTED_MODES)} mode(s) are allowed.  "
                "Set ETORO_ACCOUNT_MODE=demo."
            )

        self._api_key  = api_key   # kept private; never logged or printed
        self._mode     = mode
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._timeout  = timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                # TODO: confirm the correct authentication header name.
                # Common candidates: "x-api-key", "Authorization: Bearer <key>",
                # "Authorization: Token <key>".  Check official docs.
                "x-api-key":    api_key,   # TODO: verify header name
                "Content-Type": "application/json",
                "Accept":       "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        """
        Fetch the current account snapshot.

        TODO: verify endpoint path, response schema, and field names.
        """
        data = self._get(_ENDPOINTS["account"])
        return AccountInfo(
            account_id=str(data.get("accountId", "")),   # TODO: verify field
            mode=self._mode,
            balance=float(data.get("balance", 0.0)),     # TODO: verify field
            equity=float(data.get("equity", 0.0)),       # TODO: verify field
            cash=float(data.get("cash", 0.0)),           # TODO: verify field
            currency=str(data.get("currency", "USD")),   # TODO: verify field
            raw=data,
        )

    def get_positions(self) -> list[PositionInfo]:
        """
        Fetch all open positions.

        TODO: verify endpoint path, response schema, pagination, and field names.
        """
        data = self._get(_ENDPOINTS["positions"])
        # TODO: adjust to match the actual response envelope (e.g. "data", "positions", etc.)
        items = data if isinstance(data, list) else data.get("positions", [])
        return [
            PositionInfo(
                symbol=str(p.get("instrumentId", "")),   # TODO: verify field
                quantity=float(p.get("amount", 0.0)),    # TODO: verify field
                market_value=float(p.get("value", 0.0)), # TODO: verify field
                open_price=float(p.get("openRate", 0.0)), # TODO: verify field
                raw=p,
            )
            for p in items
        ]

    def get_instrument(self, symbol: str) -> InstrumentInfo:
        """
        Fetch metadata for one instrument.

        TODO: verify endpoint path, symbol format (ticker vs numeric ID), and field names.
        """
        path = _ENDPOINTS["instrument"].format(symbol=symbol)
        data = self._get(path)
        return InstrumentInfo(
            symbol=str(data.get("instrumentId", symbol)),  # TODO: verify field
            name=str(data.get("instrumentName", "")),       # TODO: verify field
            tradeable=bool(data.get("isTradable", False)),  # TODO: verify field
            leverage_max=float(data.get("leverageMax", 1.0)), # TODO: verify field
            raw=data,
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        dry_run: bool = True,
    ) -> OrderResult:
        """
        Place (or simulate) a market order on the demo account.

        When ``dry_run=True`` (the default) no HTTP request is sent and the
        method returns immediately with status ``"dry_run"``.

        Parameters
        ----------
        symbol:
            Ticker string (e.g. ``"AAPL"``).
        side:
            ``"BUY"`` or ``"SELL"``.
        amount:
            USD notional amount.
        dry_run:
            Must be ``True`` until this method is validated against a live
            demo environment with confirmed endpoint paths.
        """
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"side must be 'BUY' or 'SELL'; got '{side}'.")
        if amount <= 0:
            raise ValueError(f"amount must be positive; got {amount}.")

        if dry_run:
            return OrderResult(
                symbol=symbol,
                side=side,
                amount=amount,
                dry_run=True,
                status="dry_run",
                order_id=None,
                message="Dry-run: no HTTP request sent.",
            )

        # TODO: confirm payload schema, field names, and response envelope.
        payload: dict[str, Any] = {
            "instrumentId": symbol,   # TODO: verify field name; may need numeric ID
            "isBuy":        side == "BUY",  # TODO: verify field name
            "amount":       amount,   # TODO: verify field name and unit (USD vs units)
            "leverage":     1,        # always 1 — no leverage
        }
        data = self._post(_ENDPOINTS["orders"], payload)

        return OrderResult(
            symbol=symbol,
            side=side,
            amount=amount,
            dry_run=False,
            status=str(data.get("status", "submitted")),  # TODO: verify field
            order_id=str(data.get("orderId", "")),        # TODO: verify field
            message=str(data.get("message", "")),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = self._base_url + path
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise BrokerError(
                f"GET {url} returned {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except requests.RequestException as exc:
            raise BrokerError(f"GET {url} failed: {exc}") from exc

    def _post(self, path: str, payload: dict) -> dict:
        url = self._base_url + path
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise BrokerError(
                f"POST {url} returned {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except requests.RequestException as exc:
            raise BrokerError(f"POST {url} failed: {exc}") from exc
