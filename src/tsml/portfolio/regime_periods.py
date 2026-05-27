"""Historical regime windows for backtest stress testing."""

from __future__ import annotations

REGIME_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020 crash/recovery", "2020-01-01", "2020-12-31"),
    ("2021 bull", "2021-01-01", "2021-12-31"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
    ("2023-2024 AI bull", "2023-01-01", "2024-12-31"),
)
