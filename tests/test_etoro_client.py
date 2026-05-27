"""
Tests for EtoroClient against documented API response shapes.

All tests are offline — HTTP is intercepted via mocked requests.Session.get.
Response fixtures are based on the official OpenAPI examples at
https://api-portal.etoro.com/api-reference/openapi.json
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from tsml.broker.base import BrokerAuthError, BrokerError, BrokerModeError
from tsml.broker.etoro_client import EtoroClient, DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# Documented response fixtures (OpenAPI examples)
# ---------------------------------------------------------------------------

PORTFOLIO_RESPONSE = {
    "clientPortfolio": {
        "credit": 280.35,
        "positions": [
            {
                "positionID": 2150896073,
                "CID": 7765437,
                "openDateTime": "2024-08-01T07:44:26.103Z",
                "openRate": 2020.7784,
                "instrumentID": 1002,
                "isBuy": True,
                "mirrorID": 0,
                "amount": 100.0,
                "units": 0.049485,
            },
            {
                "positionID": 999,
                "CID": 7765437,
                "openRate": 150.0,
                "instrumentID": 1003,
                "isBuy": True,
                "mirrorID": 1841334,
                "amount": 50.0,
                "units": 1.0,
            },
        ],
        "mirrors": [],
        "orders": [],
    }
}

INSTRUMENTS_METADATA_RESPONSE = {
    "instrumentDisplayDatas": [
        {
            "instrumentID": 1002,
            "instrumentDisplayName": "Apple Inc.",
            "symbolFull": "AAPL",
        }
    ]
}

SEARCH_AAPL_RESPONSE = {
    "page": 1,
    "pageSize": 10,
    "totalItems": 1,
    "items": [
        {
            "instrumentId": 1001,
            "internalSymbolFull": "AAPL",
            "displayname": "Apple Inc.",
            "isCurrentlyTradable": True,
            "isOpen": True,
        }
    ],
}


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = str(payload)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        def _raise():
            raise requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = _raise
    return resp


@pytest.fixture()
def demo_env(monkeypatch):
    monkeypatch.setenv("ETORO_API_KEY", "test-api-key")
    monkeypatch.setenv("ETORO_USER_KEY", "test-user-key")
    monkeypatch.setenv("ETORO_ACCOUNT_MODE", "demo")


@pytest.fixture()
def client(demo_env, tmp_path) -> EtoroClient:
    return EtoroClient(
        base_url=DEFAULT_BASE_URL,
        cache_path=tmp_path / "instrument_cache.json",
    )

# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------

class TestEtoroClientConstruction:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setenv("ETORO_USER_KEY", "user")
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "demo")
        with pytest.raises(BrokerAuthError, match="ETORO_API_KEY"):
            EtoroClient(cache_path=None)

    def test_missing_user_key_raises(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "api")
        monkeypatch.delenv("ETORO_USER_KEY", raising=False)
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "demo")
        with pytest.raises(BrokerAuthError, match="ETORO_USER_KEY"):
            EtoroClient(cache_path=None)

    def test_real_mode_rejected(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "api")
        monkeypatch.setenv("ETORO_USER_KEY", "user")
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "real")
        with pytest.raises(BrokerModeError, match="demo"):
            EtoroClient(cache_path=None)

    def test_default_base_url(self, demo_env):
        c = EtoroClient(cache_path=None)
        assert c._base_url == DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# get_account / get_positions — demo portfolio endpoint
# ---------------------------------------------------------------------------

class TestPortfolioEndpoints:
    def test_get_account_uses_demo_portfolio_path(self, client):
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(PORTFOLIO_RESPONSE)
            client.get_account()
        called_url = mock_get.call_args[0][0]
        assert called_url.endswith("/trading/info/demo/portfolio")

    def test_get_account_maps_credit_and_equity(self, client):
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(PORTFOLIO_RESPONSE)
            account = client.get_account()
        assert account.cash == pytest.approx(280.35)
        assert account.equity == pytest.approx(100.0)
        assert account.balance == pytest.approx(380.35)
        assert account.mode == "demo"
        assert account.account_id == "7765437"

    def test_get_positions_excludes_mirror_positions(self, client):
        def side_effect(url, **kwargs):
            if url.endswith("/trading/info/demo/portfolio"):
                return _mock_response(PORTFOLIO_RESPONSE)
            if "/market-data/instruments" in url:
                return _mock_response(INSTRUMENTS_METADATA_RESPONSE)
            raise AssertionError(f"Unexpected URL: {url}")

        with patch.object(client._session, "get", side_effect=side_effect):
            positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == pytest.approx(0.049485)
        assert positions[0].market_value == pytest.approx(100.0)
        assert positions[0].open_price == pytest.approx(2020.7784)

    def test_portfolio_fetched_once_for_account_and_positions(self, client):
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(PORTFOLIO_RESPONSE)
            client.get_account()
            client.get_positions()
        portfolio_calls = [
            c for c in mock_get.call_args_list
            if c[0][0].endswith("/trading/info/demo/portfolio")
        ]
        assert len(portfolio_calls) == 1

    def test_request_includes_auth_headers(self, client):
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(PORTFOLIO_RESPONSE)
            client.get_account()
        headers = mock_get.call_args[1]["headers"]
        assert "x-request-id" in headers
        assert client._session.headers["x-api-key"] == "test-api-key"
        assert client._session.headers["x-user-key"] == "test-user-key"


# ---------------------------------------------------------------------------
# get_instrument — search flow
# ---------------------------------------------------------------------------

class TestInstrumentSearch:
    def test_get_instrument_uses_guide_search_first(self, client):
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(SEARCH_AAPL_RESPONSE)
            client.get_instrument("AAPL")
        first_params = mock_get.call_args_list[0][1]["params"]
        assert first_params == {"internalSymbolFull": "AAPL"}

    def test_get_instrument_returns_metadata(self, client):
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(SEARCH_AAPL_RESPONSE)
            info = client.get_instrument("aapl")
        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.tradeable is True

    def test_get_instrument_requires_exact_symbol_match(self, client):
        payload = {
            "items": [
                {"instrumentId": 999, "internalSymbolFull": "AAP", "symbol": "AAP", "displayname": "Partial"},
            ]
        }
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(payload)
            with pytest.raises(BrokerError, match="matching 'AAPL'"):
                client.get_instrument("AAPL")
        assert mock_get.call_count == 4   # all four search strategies

    def test_get_instrument_trusts_sparse_single_result(self, client):
        """Guide search may return one item without symbol fields projected."""
        def side_effect(url, **kwargs):
            params = kwargs.get("params", {})
            if params == {"internalSymbolFull": "AAPL"}:
                return _mock_response({"items": [{"instrumentId": 1001}]})
            if "/market-data/instruments" in url:
                return _mock_response({
                    "instrumentDisplayDatas": [{
                        "instrumentID": 1001,
                        "instrumentDisplayName": "Apple Inc.",
                        "symbolFull": "AAPL",
                    }]
                })
            return _mock_response({"items": []})

        with patch.object(client._session, "get", side_effect=side_effect):
            info = client.get_instrument("AAPL")
        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."

    def test_get_instrument_accepts_symbol_suffix(self, client):
        payload = {
            "items": [
                {
                    "instrumentId": 1001,
                    "internalSymbolFull": "AAPL.US",
                    "symbol": "AAPL",
                    "displayname": "Apple Inc.",
                    "isCurrentlyTradable": True,
                }
            ]
        }
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value = _mock_response(payload)
            info = client.get_instrument("AAPL")
        assert info.symbol == "AAPL"
        assert info.tradeable is True

    def test_resolve_instrument_id_caches_result(self, client, tmp_path):
        cache_path = tmp_path / "cache.json"
        c = EtoroClient(base_url=DEFAULT_BASE_URL, cache_path=cache_path)
        with patch.object(c._session, "get") as mock_get:
            mock_get.return_value = _mock_response(SEARCH_AAPL_RESPONSE)
            first = c.resolve_instrument_id("AAPL")
            second = c.resolve_instrument_id("AAPL")
        assert first == 1001
        assert second == 1001
        assert mock_get.call_count == 1
        assert cache_path.is_file()


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

ORDER_BUY_RESPONSE = {
    "orderId": "ord-demo-12345",
    "status": "Executed",
    "instrumentId": 1001,
    "amount": 500.0,
}


class TestPlaceOrder:
    def test_dry_run_does_not_call_http(self, client):
        with patch.object(client._session, "get") as mock_get, \
             patch.object(client._session, "post") as mock_post:
            result = client.place_order("AAPL", "BUY", 500.0, dry_run=True)
        mock_get.assert_not_called()
        mock_post.assert_not_called()
        assert result.status == "dry_run"
        assert result.dry_run is True

    def test_live_buy_posts_to_demo_by_amount_endpoint(self, client, monkeypatch):
        monkeypatch.setenv("TSML_MAX_LIVE_ORDER_AMOUNT", "1000")
        with patch.object(client, "resolve_instrument_id", return_value=1001) as mock_resolve:
            with patch.object(client._session, "post") as mock_post:
                mock_post.return_value = _mock_response(ORDER_BUY_RESPONSE)
                result = client.place_order("AAPL", "BUY", 500.0, dry_run=False)

        mock_resolve.assert_called_once_with("AAPL")
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert url.endswith("/trading/execution/demo/market-open-orders/by-amount")
        body = mock_post.call_args.kwargs["json"]
        assert body == {
            "InstrumentId": 1001,
            "IsBuy": True,
            "Leverage": 1,
            "Amount": 500.0,
        }
        headers = mock_post.call_args.kwargs["headers"]
        assert "x-request-id" in headers
        assert client._session.headers["x-api-key"] == "test-api-key"
        assert client._session.headers["x-user-key"] == "test-user-key"
        assert result.order_id == "ord-demo-12345"
        assert result.status == "filled"
        assert result.dry_run is False

    def test_amount_above_max_live_order_amount_rejected(self, client, monkeypatch):
        monkeypatch.setenv("TSML_MAX_LIVE_ORDER_AMOUNT", "1000")
        with patch.object(client._session, "post") as mock_post:
            with pytest.raises(BrokerError, match="TSML_MAX_LIVE_ORDER_AMOUNT"):
                client.place_order("AAPL", "BUY", 1500.0, dry_run=False)
        mock_post.assert_not_called()

    def test_failed_post_raises_broker_error(self, client, monkeypatch):
        monkeypatch.setenv("TSML_MAX_LIVE_ORDER_AMOUNT", "1000")
        with patch.object(client, "resolve_instrument_id", return_value=1001):
            with patch.object(client._session, "post") as mock_post:
                mock_post.return_value = _mock_response({"error": "bad"}, status_code=400)
                with pytest.raises(BrokerError, match="POST"):
                    client.place_order("AAPL", "BUY", 500.0, dry_run=False)

    def test_demo_mode_required_at_construction(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "api")
        monkeypatch.setenv("ETORO_USER_KEY", "user")
        monkeypatch.setenv("ETORO_ACCOUNT_MODE", "real")
        with pytest.raises(BrokerModeError, match="demo"):
            EtoroClient(cache_path=None)
