"""
Portfolio simulator — periodic rebalancing over a date range.

``simulate`` runs a full walk-forward pre-computation for every symbol,
then steps through calendar time, rebalancing on the first trading day of
each week.  At each rebalance date it ranks the universe, generates trading
signals, executes trades (with optional transaction costs), and records the
daily portfolio value.

Leakage guarantee
-----------------
Walk-forward probabilities are pre-computed once per symbol using
``run_walk_forward_proba``, which ensures every OOS probability at date *t*
was computed with data only up to *t*.  The probability matrix is then
**shifted forward by one trading day** so the score used on rebalance date
*d* is the value that was available at the close of the previous trading day.
This replicates the convention used throughout this project: signal at *t*,
execution at *t+1*.

Portfolio mechanics
-------------------
- Positions are **equal-weight** (1/N per held symbol; cash earns 0 %).
- Rebalancing is triggered on the **first trading day of each ISO calendar
  week** (i.e. Monday, or the next available trading day if Monday is a
  holiday).
- Transaction costs are applied as a fraction of **one-way turnover**:
  ``cost = one_way_turnover * portfolio_value * costs_bps * 1e-4``.
  One-way turnover is ``sum(|Δweight|) / 2`` across all symbols, which
  accounts for both entry, exit, and weight changes from position-count
  changes.
- The portfolio value cannot drop below zero.

Output
------
``SimulationResult`` carries:

- ``equity_curve``: daily ``pd.Series`` of portfolio value (starts at
  ``initial_capital``), indexed by the same trading-day DatetimeIndex as
  the price data.
- ``trades_log``: ``pd.DataFrame`` with columns
  ``[date, symbol, action, score]`` — one row per executed trade (buys and
  sells; holds are omitted).

Example
-------
>>> from tsml.models.baselines import CalibratedLogisticRegressionModel
>>> from tsml.portfolio.simulator import simulate
>>> from tsml.validation import WalkForwardSplit
>>>
>>> result = simulate(
...     symbols=["SPY", "QQQ", "MSFT"],
...     model=CalibratedLogisticRegressionModel(),
...     splitter=WalkForwardSplit(n_splits=5, min_train_size=252, test_size=63),
...     start_date="2020-01-01",
...     end_date="2023-12-31",
...     top_n=2,
...     min_score=0.55,
...     costs_bps=5.0,
... )
>>> result.equity_curve.plot(title="Portfolio equity curve")
>>> print(result.trades_log.head())
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.features.benchmarks import load_benchmark_closes
from tsml.pipelines.train import run_walk_forward_proba
from tsml.portfolio.backtest_strategy import generate_backtest_signals
from tsml.portfolio.ranker import compute_context_as_of
from tsml.portfolio.regime_overlay import (
    RegimeOverlayConfig,
    build_regime_target_series,
    compute_target_exposure_as_of,
    deployable_fraction,
    per_position_weight,
)
from tsml.portfolio.strategy import generate_signals
from tsml.validation.splitters import WalkForwardSplit


@dataclass
class SimulationResult:
    """
    Return value of :func:`simulate`.

    Attributes
    ----------
    equity_curve:
        Daily portfolio value indexed by trading-day timestamps.
        Starts at ``initial_capital`` on the first day.
    trades_log:
        One row per executed trade (buys and sells only; holds are omitted).
        Columns: ``date``, ``symbol``, ``action`` (``"buy"``/``"sell"``),
        ``score`` (P(up) used to rank that symbol on that day).
    exposure:
        Daily fraction of portfolio value allocated to equities (0-1).
    turnover_total:
        Cumulative one-way turnover across all rebalances.
    avg_holding_weeks:
        Mean holding period (weekly rebalance periods) for closed positions.
    """

    equity_curve: pd.Series
    trades_log: pd.DataFrame
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    turnover_total: float = 0.0
    avg_holding_weeks: float = 0.0
    rebalance_log: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["date", "turnover", "n_positions", "regime_target"]
        )
    )
    regime_target: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float, name="regime_target")
    )


@dataclass
class SimulationInputs:
    """
    Precomputed walk-forward scores and prices for repeated backtest runs.

    Used by parameter sweeps to avoid re-fitting the model for every grid point.
    Walk-forward probabilities are computed once with strict past-data cutoffs.
    """

    close_map: dict[str, pd.Series]
    proba_map: dict[str, pd.Series]
    spy_close: pd.Series | None = None


def prepare_simulation_inputs(
    symbols: Sequence[str],
    model: Any,
    splitter: WalkForwardSplit,
    *,
    start_date: str,
    end_date: str,
    target: str = "direction",
    loader: DataLoader | None = None,
    feature_set: str = "legacy",
    load_spy: bool = True,
    spy_symbol: str = "SPY",
) -> SimulationInputs:
    """
    Load prices and compute walk-forward probabilities once.

    Parameters
    ----------
    load_spy:
        When ``True``, also load ``spy_symbol`` closes for regime overlays.
    """
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    close_map: dict[str, pd.Series] = {}
    proba_map: dict[str, pd.Series] = {}

    benchmarks = None
    if feature_set == "extended":
        benchmarks = load_benchmark_closes(loader, start_date, end_date)

    for symbol in symbols:
        try:
            df = loader.load(symbol, start_date, end_date)
            probas = run_walk_forward_proba(
                df,
                model,
                splitter,
                target=target,
                feature_set=feature_set,
                benchmarks=benchmarks,
            )
            close_map[symbol] = df["close"]
            proba_map[symbol] = probas
        except Exception as exc:  # noqa: BLE001
            print(
                f"[prepare_simulation_inputs] skipping {symbol}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    spy_close: pd.Series | None = None
    if load_spy:
        try:
            spy_close = loader.load(spy_symbol, start_date, end_date)["close"]
        except Exception as exc:  # noqa: BLE001
            print(
                f"[prepare_simulation_inputs] cannot load {spy_symbol}: {exc}",
                file=sys.stderr,
            )

    return SimulationInputs(
        close_map=close_map,
        proba_map=proba_map,
        spy_close=spy_close,
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def simulate(
    symbols: Sequence[str],
    model: Any,
    splitter: WalkForwardSplit,
    *,
    start_date: str,
    end_date: str,
    target: str = "direction",
    top_n: int = 5,
    min_score: float = 0.55,
    min_score_downtrend: float = 0.62,
    cash_buffer_pct: float = 0.0,
    costs_bps: float = 5.0,
    initial_capital: float = 1.0,
    rebalance_frequency: str = "weekly",
    loader: DataLoader | None = None,
    turnover_control: bool = False,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.52,
    min_hold_weeks: int = 2,
    feature_set: str = "legacy",
    regime_overlay: RegimeOverlayConfig | None = None,
    precomputed: SimulationInputs | None = None,
) -> SimulationResult:
    """
    Simulate periodic portfolio rebalancing driven by walk-forward model scores.

    Parameters
    ----------
    symbols:
        Ticker strings for the universe to trade (e.g. ``["SPY", "QQQ"]``).
    model:
        Any object with ``.fit(X, y)`` and ``.predict_proba(X)`` methods.
        Reused across all symbols; state is overwritten at each fold.
    splitter:
        A configured ``WalkForwardSplit``.  Symbols whose cleaned dataset is
        too small to yield even one valid fold are skipped.
    start_date:
        Inclusive start date for data loading (``"YYYY-MM-DD"``).
    end_date:
        Inclusive end date for data loading (``"YYYY-MM-DD"``).
    target:
        Target type passed to ``make_dataset``.  Must be one of
        ``"direction"``, ``"direction_5d"``, ``"threshold"``, ``"return"``.
        Default ``"direction"``.
    top_n:
        Maximum number of symbols to hold simultaneously.
    min_score:
        Minimum P(up) score for a symbol to be eligible to buy.  Held
        positions that fall below this threshold are sold.
    min_score_downtrend:
        Stricter threshold applied when ``above_sma_200`` is False.
    cash_buffer_pct:
        Minimum cash fraction kept uninvested (e.g. 0.05 = 5 %).
        Selected positions are equal-weighted across the remaining
        ``(1 - cash_buffer_pct)`` of portfolio value.
    costs_bps:
        One-way transaction cost in basis points applied to traded notional
        at each rebalance.  Default 5 bps.
    initial_capital:
        Starting portfolio value (any positive float, or 1.0 for a
        normalised equity curve).
    rebalance_frequency:
        Only ``"weekly"`` is currently supported.
    loader:
        Optional ``DataLoader``.  Defaults to
        ``YFinanceLoader(cache_dir="data/raw")``.
    turnover_control:
        When ``True``, use backtest-only hysteresis / min-hold rules via
        :func:`~tsml.portfolio.backtest_strategy.generate_backtest_signals`.
        Live-style logic (no hysteresis) is used when ``False``.
    buy_threshold:
        Minimum score to open a new position (turnover control only).
    sell_threshold:
        Minimum score to keep an existing position (turnover control only).
    min_hold_weeks:
        Minimum weekly rebalance periods before a discretionary sell
        (turnover control only).  Hard risk exits ignore this floor.
    feature_set:
        ``"legacy"`` or ``"extended"`` feature set for walk-forward scoring.
    regime_overlay:
        Optional :class:`~tsml.portfolio.regime_overlay.RegimeOverlayConfig`.
        When enabled, scales deployable capital by SPY regime (backtest only).
    precomputed:
        Optional pre-loaded prices and walk-forward probabilities.  When
        provided, Phase 1 (data load + model scoring) is skipped so parameter
        sweeps can reuse the same OOS scores across many portfolio settings.

    Returns
    -------
    SimulationResult
        Contains ``equity_curve`` (daily ``pd.Series``) and ``trades_log``
        (``pd.DataFrame``).  If every symbol fails pre-computation, both
        fields are empty.

    Raises
    ------
    ValueError
        If ``rebalance_frequency`` is not ``"weekly"``.
    """
    if rebalance_frequency != "weekly":
        raise ValueError(
            f"rebalance_frequency must be 'weekly'; got '{rebalance_frequency}'."
        )
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")
    if regime_overlay is None:
        regime_overlay = RegimeOverlayConfig(enabled=False)

    # ------------------------------------------------------------------
    # Phase 1: Load prices and compute walk-forward probabilities
    # ------------------------------------------------------------------

    if precomputed is not None:
        close_map = precomputed.close_map
        proba_map = precomputed.proba_map
        spy_close = precomputed.spy_close
    else:
        close_map = {}
        proba_map = {}

        benchmarks = None
        if feature_set == "extended":
            benchmarks = load_benchmark_closes(loader, start_date, end_date)

        spy_close = None
        if regime_overlay.enabled:
            sym = regime_overlay.benchmark_symbol
            try:
                spy_close = loader.load(sym, start_date, end_date)["close"]
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[simulate] regime overlay disabled: cannot load {sym}: {exc}",
                    file=sys.stderr,
                )
                regime_overlay = RegimeOverlayConfig(enabled=False)

        for symbol in symbols:
            try:
                df = loader.load(symbol, start_date, end_date)
                probas = run_walk_forward_proba(
                    df,
                    model,
                    splitter,
                    target=target,
                    feature_set=feature_set,
                    benchmarks=benchmarks,
                )
                close_map[symbol] = df["close"]
                proba_map[symbol] = probas
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[simulate] skipping {symbol}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    if regime_overlay.enabled and spy_close is None and precomputed is not None:
        regime_overlay = RegimeOverlayConfig(enabled=False)

    if not close_map:
        return SimulationResult(
            equity_curve=pd.Series(dtype=float, name="portfolio_value"),
            trades_log=pd.DataFrame(columns=["date", "symbol", "action", "score"]),
            exposure=pd.Series(dtype=float, name="exposure"),
            turnover_total=0.0,
            avg_holding_weeks=0.0,
            rebalance_log=pd.DataFrame(
                columns=["date", "turnover", "n_positions", "regime_target"]
            ),
            regime_target=pd.Series(dtype=float, name="regime_target"),
        )

    # ------------------------------------------------------------------
    # Phase 2: Shared time axis, daily returns, score matrix
    # ------------------------------------------------------------------

    close_df    = pd.DataFrame(close_map)
    trading_days = close_df.index
    daily_rets  = close_df.pct_change()

    # Probability matrix: rows = trading days, columns = symbols.
    # Forward-fill so every row has the most recent known probability.
    # Shift by 1 so the score on day t is the value known at t-1 close.
    proba_df = _build_proba_matrix(proba_map, trading_days)
    score_df = proba_df.shift(1)          # strict past — no lookahead

    # ------------------------------------------------------------------
    # Phase 3: Rebalance dates
    # ------------------------------------------------------------------

    rebalance_dates = _weekly_rebalance_dates(trading_days)

    regime_target_series = (
        build_regime_target_series(spy_close, trading_days, regime_overlay)
        if spy_close is not None and regime_overlay.enabled
        else pd.Series(1.0, index=trading_days, name="regime_target")
    )

    # ------------------------------------------------------------------
    # Phase 4: Day-by-day simulation loop
    # ------------------------------------------------------------------

    portfolio_value: float         = initial_capital
    current_positions: set[str]    = set()
    current_regime_target: float   = 1.0
    position_weeks: dict[str, int] = {}
    holding_periods: list[int]     = []
    equity_rows: list[dict]        = []
    exposure_rows: list[dict]      = []
    trade_rows:  list[dict]        = []
    rebalance_rows: list[dict]     = []
    turnover_total: float          = 0.0

    for date in trading_days:

        regime_target_today = float(regime_target_series.loc[date])

        # ── Rebalance ──────────────────────────────────────────────────
        if date in rebalance_dates:
            ranking = _build_ranking(score_df, date)
            ranking = _enrich_ranking_as_of(ranking, close_map, date, trading_days)

            if not ranking.empty:
                weeks_for_signals = {
                    sym: position_weeks.get(sym, 0)
                    for sym in current_positions
                }
                if turnover_control:
                    signals = generate_backtest_signals(
                        ranking,
                        current_positions,
                        top_n=top_n,
                        min_score_downtrend=min_score_downtrend,
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold,
                        weeks_held=weeks_for_signals,
                        min_hold_weeks=min_hold_weeks,
                    )
                else:
                    signals = generate_signals(
                        ranking,
                        current_positions,
                        top_n=top_n,
                        min_score=min_score,
                        min_score_downtrend=min_score_downtrend,
                    )

                new_positions = {
                    s.symbol for s in signals if s.action in ("buy", "hold")
                }
                new_buys = {s.symbol for s in signals if s.action == "buy"}

                if regime_overlay.enabled and regime_target_today <= 0.0:
                    new_positions = set()
                    new_buys = set()

                one_way_to = _one_way_turnover(
                    current_positions,
                    new_positions,
                    cash_buffer_pct,
                    old_regime_target=current_regime_target,
                    new_regime_target=regime_target_today,
                )
                turnover_total += one_way_to
                rebalance_rows.append(
                    {
                        "date": date,
                        "turnover": one_way_to,
                        "n_positions": len(new_positions),
                        "regime_target": regime_target_today,
                    }
                )
                cost = one_way_to * portfolio_value * costs_bps * 1e-4
                portfolio_value    = max(0.0, portfolio_value - cost)

                old_positions = set(current_positions)
                for sym in old_positions - new_positions:
                    holding_periods.append(weeks_for_signals.get(sym, 0) + 1)
                    score = float(
                        ranking.loc[ranking["symbol"] == sym, "score"].iloc[0]
                    ) if sym in ranking["symbol"].values else float("nan")
                    trade_rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "action": "sell",
                            "score": score,
                        }
                    )
                for sym in new_positions - old_positions:
                    score = float(
                        ranking.loc[ranking["symbol"] == sym, "score"].iloc[0]
                    )
                    trade_rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "action": "buy",
                            "score": score,
                        }
                    )

                position_weeks = _update_position_weeks(
                    new_positions,
                    new_buys,
                    weeks_for_signals,
                )
                current_positions = new_positions
                current_regime_target = regime_target_today

        # ── Daily return ───────────────────────────────────────────────
        if current_positions and current_regime_target > 0.0:
            portfolio_value = _apply_daily_return(
                portfolio_value,
                current_positions,
                daily_rets,
                date,
                cash_buffer_pct=cash_buffer_pct,
                regime_target=current_regime_target,
            )

        exposure_frac = _equity_exposure(
            len(current_positions),
            cash_buffer_pct,
            current_regime_target,
        )
        equity_rows.append({"date": date, "value": portfolio_value})
        exposure_rows.append({"date": date, "exposure": exposure_frac})

    # ------------------------------------------------------------------
    # Phase 5: Assemble output
    # ------------------------------------------------------------------

    equity_curve = (
        pd.DataFrame(equity_rows)
        .set_index("date")["value"]
        .rename("portfolio_value")
    )
    exposure = (
        pd.DataFrame(exposure_rows)
        .set_index("date")["exposure"]
        .rename("exposure")
    )

    trades_log = pd.DataFrame(
        trade_rows if trade_rows else [],
        columns=["date", "symbol", "action", "score"],
    )
    rebalance_log = pd.DataFrame(
        rebalance_rows if rebalance_rows else [],
        columns=["date", "turnover", "n_positions", "regime_target"],
    )

    avg_holding = (
        float(np.mean(holding_periods)) if holding_periods else 0.0
    )

    return SimulationResult(
        equity_curve=equity_curve,
        trades_log=trades_log,
        exposure=exposure,
        turnover_total=turnover_total,
        avg_holding_weeks=avg_holding,
        rebalance_log=rebalance_log,
        regime_target=regime_target_series,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_position_weeks(
    new_positions: set[str],
    new_buys: set[str],
    weeks_for_signals: dict[str, int],
) -> dict[str, int]:
    """Track completed weekly rebalance periods held for each open position."""
    updated: dict[str, int] = {}
    for sym in new_positions:
        if sym in new_buys:
            updated[sym] = 0
        else:
            updated[sym] = weeks_for_signals.get(sym, 0) + 1
    return updated


def _build_proba_matrix(
    proba_map: dict[str, pd.Series],
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Reindex each symbol's OOS probability series to all trading days,
    then forward-fill gaps (dates before the first OOS prediction are NaN).
    """
    if not proba_map:
        return pd.DataFrame(index=trading_days)

    return pd.DataFrame(
        {sym: series.reindex(trading_days).ffill() for sym, series in proba_map.items()},
        index=trading_days,
    )


def _weekly_rebalance_dates(trading_days: pd.DatetimeIndex) -> frozenset[pd.Timestamp]:
    """
    Return the first trading day of each ISO calendar week.

    This is normally Monday; if Monday is a holiday the next available
    trading day is used instead.
    """
    if trading_days.empty:
        return frozenset()

    iso   = trading_days.isocalendar()
    # Composite key: "YYYY-WW"
    keys  = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    first = pd.Series(trading_days, index=keys).groupby(level=0).first()
    return frozenset(first)


def _build_ranking(
    score_df: pd.DataFrame, date: pd.Timestamp
) -> pd.DataFrame:
    """
    Return a ranking DataFrame for ``date`` from the pre-shifted score matrix.

    Symbols whose score is NaN on this date (no probability available yet)
    are excluded.
    """
    if score_df.empty or date not in score_df.index:
        return pd.DataFrame(columns=["symbol", "score"])

    row = score_df.loc[date].dropna()
    if row.empty:
        return pd.DataFrame(columns=["symbol", "score"])

    return (
        pd.DataFrame({"symbol": row.index.astype(str), "score": row.values.astype(float)})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def _prior_trading_day(
    trading_days: pd.DatetimeIndex,
    date: pd.Timestamp,
) -> pd.Timestamp | None:
    """Return the trading day immediately before *date*, or None."""
    loc = trading_days.get_indexer([date], method="pad")[0]
    if loc <= 0:
        return None
    return trading_days[loc - 1]


def _enrich_ranking_as_of(
    ranking: pd.DataFrame,
    close_map: dict[str, pd.Series],
    date: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Add SMA200 context columns using prices only through the prior close.

    Scores on *date* are already lagged by one day; context uses the same
    information cutoff to avoid lookahead.
    """
    if ranking.empty:
        return ranking

    as_of = _prior_trading_day(trading_days, date)
    if as_of is None:
        return ranking

    ctx_rows: list[dict] = []
    for sym in ranking["symbol"]:
        close = close_map.get(sym)
        if close is None:
            ctx_rows.append({"symbol": sym, **_empty_context_row()})
            continue
        ctx = compute_context_as_of(close, as_of)
        ctx_rows.append({"symbol": sym, **ctx})

    ctx_df = pd.DataFrame(ctx_rows)
    return ranking.merge(ctx_df, on="symbol", how="left")


def _empty_context_row() -> dict:
    return {
        "return_20d": float("nan"),
        "return_60d": float("nan"),
        "volatility_20d": float("nan"),
        "price_vs_sma_200": float("nan"),
        "above_sma_200": None,
    }


def _equity_exposure(
    n_positions: int,
    cash_buffer_pct: float,
    regime_target: float = 1.0,
) -> float:
    return deployable_fraction(n_positions, cash_buffer_pct, regime_target)


def _one_way_turnover(
    old_positions: set[str],
    new_positions: set[str],
    cash_buffer_pct: float,
    *,
    old_regime_target: float = 1.0,
    new_regime_target: float = 1.0,
) -> float:
    """One-way turnover fraction for a rebalance with buffer and regime scaling."""
    n_old = len(old_positions)
    n_new = len(new_positions)

    if n_old == 0 and n_new == 0:
        return 0.0

    old_deploy = deployable_fraction(n_old, cash_buffer_pct, old_regime_target)
    new_deploy = deployable_fraction(n_new, cash_buffer_pct, new_regime_target)

    old_w = (
        {sym: old_deploy / n_old for sym in old_positions}
        if n_old else {}
    )
    new_w = (
        {sym: new_deploy / n_new for sym in new_positions}
        if n_new else {}
    )

    old_cash = 1.0 - sum(old_w.values())
    new_cash = 1.0 - sum(new_w.values())

    all_syms = set(old_w) | set(new_w)
    equity_to = sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in all_syms)
    cash_to = abs(new_cash - old_cash)
    return (equity_to + cash_to) / 2.0


def _compute_cost(
    old_positions: set[str],
    new_positions: set[str],
    portfolio_value: float,
    costs_bps: float,
    cash_buffer_pct: float = 0.0,
    old_regime_target: float = 1.0,
    new_regime_target: float = 1.0,
) -> float:
    """
    Transaction cost for moving from ``old_positions`` to ``new_positions``.

    Computed as ``one_way_turnover * portfolio_value * costs_bps * 1e-4``.
    """
    one_way_to = _one_way_turnover(
        old_positions,
        new_positions,
        cash_buffer_pct,
        old_regime_target=old_regime_target,
        new_regime_target=new_regime_target,
    )
    return one_way_to * portfolio_value * costs_bps * 1e-4


def _apply_daily_return(
    portfolio_value: float,
    positions: set[str],
    daily_rets: pd.DataFrame,
    date: pd.Timestamp,
    *,
    cash_buffer_pct: float = 0.0,
    regime_target: float = 1.0,
) -> float:
    """
    Update ``portfolio_value`` for one day's equal-weight portfolio return.

    Symbols missing from the price data or with NaN return are treated as
    contributing zero return for that day (conservative: does not fabricate
    data, does not crash).
    """
    if not positions or date not in daily_rets.index:
        return portfolio_value

    weight = per_position_weight(len(positions), cash_buffer_pct, regime_target)
    port_ret = 0.0

    for sym in positions:
        if sym not in daily_rets.columns:
            continue
        r = daily_rets.loc[date, sym]
        if not np.isnan(r):
            port_ret += weight * r

    return portfolio_value * (1.0 + port_ret)
