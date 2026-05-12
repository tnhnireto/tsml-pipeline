"""
Broker abstractions — Protocol and shared data types.

The ``BrokerClient`` Protocol defines the minimal interface every broker
integration must satisfy.  Both the real ``EtoroClient`` and any test double
(stub, fake, mock) must implement these four methods.

Data classes
------------
``AccountInfo``
    Snapshot of the account: balance, equity, cash, mode.
``PositionInfo``
    A single open position: symbol, quantity, market value.
``InstrumentInfo``
    Tradeable instrument metadata: symbol, name, availability.
``OrderResult``
    What the broker returned (or would return in dry-run) after a
    ``place_order`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AccountInfo:
    """Snapshot of the trading account."""

    account_id:  str
    mode:        str    # "demo" | "real"
    balance:     float  # total account value (USD)
    equity:      float  # value of open positions (USD)
    cash:        float  # available cash (USD)
    currency:    str = "USD"
    raw:         dict = field(default_factory=dict, repr=False)


@dataclass
class PositionInfo:
    """A single open position."""

    symbol:       str
    quantity:     float
    market_value: float  # current USD value
    open_price:   float
    raw:          dict = field(default_factory=dict, repr=False)


@dataclass
class InstrumentInfo:
    """Tradeable instrument metadata."""

    symbol:      str
    name:        str
    tradeable:   bool
    leverage_max: float = 1.0   # 1.0 means no leverage available / allowed
    raw:         dict = field(default_factory=dict, repr=False)


@dataclass
class OrderResult:
    """
    Result of a ``place_order`` call.

    When ``dry_run=True`` no HTTP request was made; ``order_id`` will be
    ``None`` and ``status`` will be ``"dry_run"``.
    """

    symbol:   str
    side:     str    # "BUY" | "SELL"
    amount:   float
    dry_run:  bool
    status:   str    # "dry_run" | "submitted" | "filled" | "rejected"
    order_id: str | None = None
    message:  str = ""
    raw:      dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BrokerClient(Protocol):
    """
    Minimal broker interface.

    Implementers
    ------------
    - :class:`tsml.broker.etoro_client.EtoroClient`
    - Any test stub that satisfies the four methods below.

    All methods may raise :class:`BrokerError` on network or API failures.
    """

    def get_account(self) -> AccountInfo:
        """Return a snapshot of the current account state."""
        ...

    def get_positions(self) -> list[PositionInfo]:
        """Return all currently open positions."""
        ...

    def get_instrument(self, symbol: str) -> InstrumentInfo:
        """Return metadata for the named instrument."""
        ...

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        dry_run: bool = True,
    ) -> OrderResult:
        """
        Place (or simulate) a market order.

        Parameters
        ----------
        symbol:
            Ticker string, e.g. ``"AAPL"``.
        side:
            ``"BUY"`` or ``"SELL"``.
        amount:
            USD notional amount.
        dry_run:
            If ``True`` (the default) the call is logged but no HTTP request
            is sent.  This must be ``True`` for all non-demo accounts.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class BrokerError(Exception):
    """Raised when a broker API call fails or returns an unexpected response."""


class BrokerAuthError(BrokerError):
    """Raised when authentication fails (missing or invalid API key)."""


class BrokerModeError(BrokerError):
    """Raised when an unsupported account mode is requested."""
