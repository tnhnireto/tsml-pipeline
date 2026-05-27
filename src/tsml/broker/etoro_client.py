"""
eToro HTTP client for the Public API (demo mode only).

Endpoint paths and response mapping follow the official eToro API docs:
https://api-portal.etoro.com/

Environment variables
---------------------
ETORO_API_KEY
    Required.  Public API key from the eToro partner dashboard.
ETORO_USER_KEY
    Required.  User key for the demo (virtual) account environment.
ETORO_ACCOUNT_MODE
    ``"demo"`` (default).  ``"real"`` is rejected by this client.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import requests

from tsml.broker.env_loader import load_etoro_env_files

from tsml.broker.base import (
    AccountInfo,
    BrokerAuthError,
    BrokerError,
    BrokerModeError,
    InstrumentInfo,
    OrderResult,
    PositionInfo,
)

# Documented base URL — paths below are relative to /api/v1.
DEFAULT_BASE_URL = "https://public-api.etoro.com/api/v1"

_ENDPOINTS: dict[str, str] = {
    "portfolio_demo": "/trading/info/demo/portfolio",
    "search":         "/market-data/search",
    "instruments":    "/market-data/instruments",
}

# Fields requested from the search endpoint (``fields`` is required by the API).
_SEARCH_FIELDS = (
    "instrumentId,internalSymbolFull,symbol,displayname,"
    "isCurrentlyTradable,isOpen,isBuyEnabled"
)

_SUPPORTED_MODES = frozenset({"demo"})
_DEFAULT_CACHE_PATH = Path("data/instrument_id_cache.json")


def _field(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first matching key, trying exact and case-variant spellings."""
    for name in names:
        if name in data:
            return data[name]
        lower = name.lower()
        for key, value in data.items():
            if key.lower() == lower:
                return value
    return default


class EtoroClient:
    """
    HTTP client for the eToro Public API (demo account only).

    Parameters
    ----------
    base_url:
        Override the API base URL.  Defaults to the documented production host.
        Set to a mock server URL in tests.
    timeout:
        Per-request timeout in seconds.
    cache_path:
        Optional JSON file for persisting symbol -> instrument ID mappings.
        Defaults to ``data/instrument_id_cache.json``.

    Raises
    ------
    BrokerAuthError
        If ``ETORO_API_KEY`` or ``ETORO_USER_KEY`` is missing.
    BrokerModeError
        If ``ETORO_ACCOUNT_MODE`` is not ``"demo"``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
        cache_path: Path | str | None = _DEFAULT_CACHE_PATH,
    ) -> None:
        load_etoro_env_files()

        api_key = os.environ.get("ETORO_API_KEY", "").strip()
        if not api_key:
            raise BrokerAuthError(
                "ETORO_API_KEY environment variable is not set or is empty.  "
                "Obtain your API key from the eToro partner dashboard and "
                "export it before running this script.\n"
                "    export ETORO_API_KEY=your_key_here"
            )

        user_key = os.environ.get("ETORO_USER_KEY", "").strip()
        if not user_key:
            raise BrokerAuthError(
                "ETORO_USER_KEY environment variable is not set or is empty.  "
                "Generate a demo (virtual) user key from the eToro API portal "
                "and export it before running this script.\n"
                "    export ETORO_USER_KEY=your_user_key_here"
            )

        mode = os.environ.get("ETORO_ACCOUNT_MODE", "demo").strip().lower()
        if mode not in _SUPPORTED_MODES:
            raise BrokerModeError(
                f"Account mode '{mode}' is not supported.  "
                f"Only {sorted(_SUPPORTED_MODES)} mode(s) are allowed.  "
                "Set ETORO_ACCOUNT_MODE=demo."
            )

        self._api_key  = api_key
        self._user_key = user_key
        self._mode     = mode
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout  = timeout
        self._cache_path = Path(cache_path) if cache_path is not None else None

        self._portfolio_cache: dict[str, Any] | None = None
        self._symbol_to_id: dict[str, int] = {}
        self._id_to_symbol: dict[int, str] = {}
        self._load_id_cache()

        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key":    api_key,
                "x-user-key":   user_key,
                "Content-Type": "application/json",
                "Accept":       "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        """
        Fetch the current demo account snapshot from the portfolio endpoint.

        Uses ``GET /trading/info/demo/portfolio`` — ``credit`` is available
        cash; equity is the sum of open manual position amounts.
        """
        portfolio = self._client_portfolio()
        credit = float(_field(portfolio, "credit", default=0.0) or 0.0)
        positions = self._manual_positions(portfolio)
        equity = sum(
            float(_field(p, "amount", default=0.0) or 0.0) for p in positions
        )
        account_id = ""
        if positions:
            account_id = str(_field(positions[0], "CID", default="") or "")
        return AccountInfo(
            account_id=account_id,
            mode=self._mode,
            balance=credit + equity,
            equity=equity,
            cash=credit,
            currency="USD",
            raw=portfolio,
        )

    def get_positions(self) -> list[PositionInfo]:
        """
        Fetch open manual positions from the demo portfolio endpoint.

        Copy-trading mirror positions (``mirrorID != 0``) are excluded.
        Instrument IDs are resolved to ticker symbols for reconciliation.
        """
        portfolio = self._client_portfolio()
        raw_positions = self._manual_positions(portfolio)
        instrument_ids = {
            int(_field(p, "instrumentID", "instrumentId"))
            for p in raw_positions
            if _field(p, "instrumentID", "instrumentId") is not None
        }
        self._ensure_symbols_for_ids(instrument_ids)

        result: list[PositionInfo] = []
        for p in raw_positions:
            iid = _field(p, "instrumentID", "instrumentId")
            symbol = self._id_to_symbol.get(int(iid), str(iid)) if iid else ""
            amount = float(_field(p, "amount", default=0.0) or 0.0)
            units = float(_field(p, "units", default=0.0) or 0.0)
            open_rate = float(_field(p, "openRate", default=0.0) or 0.0)
            result.append(
                PositionInfo(
                    symbol=symbol,
                    quantity=units,
                    market_value=amount,
                    open_price=open_rate,
                    raw=p,
                )
            )
        return result

    def get_instrument(self, symbol: str) -> InstrumentInfo:
        """
        Resolve a ticker symbol via the documented search flow.

        Uses ``GET /market-data/search?internalSymbolFull=<symbol>`` and
        verifies an exact ``internalSymbolFull`` match before returning.
        """
        item = self._search_instrument(symbol)
        iid = _field(item, "instrumentId", "instrumentID")
        if iid is not None:
            key = symbol.upper()
            self._symbol_to_id[key] = int(iid)
            self._id_to_symbol[int(iid)] = key
            self._save_id_cache()
        name = str(
            _field(item, "displayname", "internalInstrumentDisplayName", default="")
            or ""
        )
        tradeable = bool(
            _field(item, "isCurrentlyTradable", default=False)
            or _field(item, "isOpen", default=False)
        )
        return InstrumentInfo(
            symbol=symbol.upper(),
            name=name,
            tradeable=tradeable,
            leverage_max=1.0,
            raw=item,
        )

    def resolve_instrument_id(self, symbol: str) -> int:
        """
        Resolve a ticker symbol to a numeric instrument ID, using cache when available.
        """
        key = symbol.upper()
        if key in self._symbol_to_id:
            return self._symbol_to_id[key]
        item = self._search_instrument(symbol)
        iid = _field(item, "instrumentId", "instrumentID")
        if iid is None:
            raise BrokerError(f"Instrument search for {symbol!r} returned no ID.")
        self._symbol_to_id[key] = int(iid)
        self._id_to_symbol[int(iid)] = key
        self._save_id_cache()
        return int(iid)

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        dry_run: bool = True,
    ) -> OrderResult:
        """
        Place (or simulate) a market order on the demo account.

        When ``dry_run=True`` (the default) no HTTP request is sent.

        Live execution is not implemented until the documented demo order
        endpoint and payload are wired and validated separately.
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

        raise NotImplementedError(
            "Live order placement is not implemented.  "
            "Documented endpoint: POST "
            "/trading/execution/demo/market-open-orders/by-amount"
        )

    # ------------------------------------------------------------------
    # Portfolio / instrument helpers
    # ------------------------------------------------------------------

    def _fetch_portfolio(self) -> dict[str, Any]:
        if self._portfolio_cache is None:
            self._portfolio_cache = self._get(_ENDPOINTS["portfolio_demo"])
        return self._portfolio_cache

    def _client_portfolio(self) -> dict[str, Any]:
        data = self._fetch_portfolio()
        portfolio = data.get("clientPortfolio")
        if not isinstance(portfolio, dict):
            raise BrokerError(
                "Portfolio response missing 'clientPortfolio' object."
            )
        return portfolio

    @staticmethod
    def _manual_positions(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
        positions = portfolio.get("positions", [])
        if not isinstance(positions, list):
            return []
        return [
            p for p in positions
            if isinstance(p, dict) and int(_field(p, "mirrorID", "mirrorId", default=0) or 0) == 0
        ]

    def _search_instrument(self, symbol: str) -> dict[str, Any]:
        """
        Resolve *symbol* via ``GET /market-data/search``.

        Tries the documented guide flow first (``internalSymbolFull`` only),
        then fallbacks with ``fields`` / ``searchText`` per OpenAPI.
        """
        key = symbol.upper()
        last_error = ""
        last_count = 0

        strategies: list[tuple[str, dict[str, Any]]] = [
            ("guide", {"internalSymbolFull": key}),
            ("fields", {
                "internalSymbolFull": key,
                "fields": _SEARCH_FIELDS,
                "pageSize": 10,
                "pageNumber": 1,
            }),
            ("fields_filter", {
                "fields": f"internalSymbolFull={key},{_SEARCH_FIELDS}",
                "pageSize": 10,
                "pageNumber": 1,
            }),
            ("search_text", {
                "searchText": key,
                "fields": _SEARCH_FIELDS,
                "pageSize": 10,
                "pageNumber": 1,
            }),
        ]

        for label, params in strategies:
            try:
                data = self._get(_ENDPOINTS["search"], params=params)
            except BrokerError as exc:
                last_error = str(exc)
                continue

            items = self._search_items(data)
            last_count = len(items)
            trust_single = label in {"guide", "fields", "fields_filter"}
            item = self._pick_search_item(items, key, trust_single=trust_single)
            if item is not None:
                return self._enrich_search_item(item, key)

        if last_error and last_count == 0:
            raise BrokerError(last_error)
        raise BrokerError(
            f"No instrument matching {key!r} found in search results "
            f"(last attempt returned {last_count} item(s))."
        )

    @staticmethod
    def _search_items(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if not isinstance(data, dict):
            return []
        items = data.get("items", [])
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
        return []

    @staticmethod
    def _pick_search_item(
        items: list[dict[str, Any]],
        key: str,
        *,
        trust_single: bool = False,
    ) -> dict[str, Any] | None:
        for item in items:
            if EtoroClient._symbol_matches_item(item, key):
                return item
        if trust_single and len(items) == 1:
            item = items[0]
            iid = _field(item, "instrumentId", "instrumentID")
            has_symbol = any(
                _field(item, f) for f in ("internalSymbolFull", "symbol", "symbolFull")
            )
            # Server-side filter returned one row without symbol fields projected.
            if iid is not None and not has_symbol:
                return item
        return None

    @staticmethod
    def _symbol_matches_item(item: dict[str, Any], key: str) -> bool:
        for field in ("internalSymbolFull", "symbol", "symbolFull"):
            raw = _field(item, field, default="")
            val = str(raw or "").upper()
            if not val:
                continue
            if val == key or val.split(".")[0] == key:
                return True
        return False

    def _enrich_search_item(self, item: dict[str, Any], key: str) -> dict[str, Any]:
        """Fill missing display fields from ``/market-data/instruments`` when needed."""
        iid = _field(item, "instrumentId", "instrumentID")
        has_name = bool(_field(item, "displayname", "internalInstrumentDisplayName"))
        has_tradeable = _field(item, "isCurrentlyTradable") is not None
        if iid is None or (has_name and has_tradeable):
            return item

        try:
            data = self._get(
                _ENDPOINTS["instruments"],
                params={"instrumentIds": str(int(iid))},
            )
        except BrokerError:
            return item

        entries = data.get("instrumentDisplayDatas", [])
        if not isinstance(entries, list) or not entries:
            return item
        meta = entries[0] if isinstance(entries[0], dict) else {}
        symbol_full = str(_field(meta, "symbolFull", default=key) or key).upper()
        enriched = dict(item)
        enriched.setdefault("instrumentId", int(iid))
        enriched.setdefault(
            "displayname",
            _field(meta, "instrumentDisplayName", default=""),
        )
        enriched.setdefault("internalSymbolFull", symbol_full)
        enriched.setdefault("symbol", symbol_full.split(".")[0])
        enriched.setdefault("isCurrentlyTradable", True)
        return enriched

    def _ensure_symbols_for_ids(self, instrument_ids: set[int]) -> None:
        missing = {iid for iid in instrument_ids if iid not in self._id_to_symbol}
        if not missing:
            return
        id_list = ",".join(str(iid) for iid in sorted(missing))
        data = self._get(
            _ENDPOINTS["instruments"],
            params={"instrumentIds": id_list},
        )
        entries = data.get("instrumentDisplayDatas", [])
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            iid = _field(entry, "instrumentID", "instrumentId")
            sym = _field(entry, "symbolFull", "internalSymbolFull")
            if iid is None or not sym:
                continue
            sym_key = str(sym).upper()
            self._id_to_symbol[int(iid)] = sym_key
            self._symbol_to_id[sym_key] = int(iid)
        self._save_id_cache()

    def _load_id_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.is_file():
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        mapping = raw.get("symbol_to_id", raw)
        if not isinstance(mapping, dict):
            return
        for sym, iid in mapping.items():
            try:
                self._symbol_to_id[str(sym).upper()] = int(iid)
                self._id_to_symbol[int(iid)] = str(sym).upper()
            except (TypeError, ValueError):
                continue

    def _save_id_cache(self) -> None:
        if self._cache_path is None or not self._symbol_to_id:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"symbol_to_id": self._symbol_to_id}
            self._cache_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self._base_url + path
        headers = {"x-request-id": str(uuid.uuid4())}
        try:
            resp = self._session.get(
                url, params=params, headers=headers, timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                raise BrokerAuthError(
                    f"GET {url} returned {exc.response.status_code}: "
                    f"{exc.response.text[:200]}\n"
                    "Hint: ETORO_API_KEY must be the Public API Key (top of "
                    "Settings > Trading > API Key Management).  "
                    "ETORO_USER_KEY must be a Demo generated user key.  "
                    "Do not swap them."
                ) from exc
            raise BrokerError(
                f"GET {url} returned {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except requests.RequestException as exc:
            raise BrokerError(f"GET {url} failed: {exc}") from exc
