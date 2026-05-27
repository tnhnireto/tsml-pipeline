"""
Safety limits for live demo order execution.

Environment variables
---------------------
TSML_MAX_LIVE_ORDER_AMOUNT
    Maximum USD notional for a single live demo order (default 1000).
"""

from __future__ import annotations

import os

DEFAULT_MAX_LIVE_ORDER_AMOUNT = 1000.0
ENV_MAX_LIVE_ORDER_AMOUNT = "TSML_MAX_LIVE_ORDER_AMOUNT"


def get_max_live_order_amount() -> float:
    """Return the configured maximum live demo order amount in USD."""
    raw = os.environ.get(ENV_MAX_LIVE_ORDER_AMOUNT, "").strip()
    if not raw:
        return DEFAULT_MAX_LIVE_ORDER_AMOUNT
    value = float(raw)
    if value <= 0:
        raise ValueError(
            f"{ENV_MAX_LIVE_ORDER_AMOUNT} must be positive; got {raw!r}."
        )
    return value
