"""
Universe ranker — score a list of symbols by model conviction.

``rank_universe`` runs walk-forward probability estimation for every symbol
in the universe, takes the **last out-of-sample probability** as a ranking
score, and returns the universe sorted by that score in descending order.

What the score represents
--------------------------
The score for symbol *s* is P(up) on the most recent out-of-sample date
produced by the walk-forward loop.  It reflects the model's current
conviction that the next trading day will be positive, estimated using only
data that was available at prediction time.

Because it is the final output of a properly leakage-free walk-forward
pipeline, the score is never computed using future data.

Ranking use case
----------------
A common use of this function is to select the top-K symbols from a
candidate universe before entering positions.  Higher score means the model
is more bullish on that symbol relative to the others.  The score is
comparable across symbols only when:

  - the same model and splitter are used for all symbols
  - data availability is similar (comparable OOS window lengths)

Symbols for which training or probability estimation fails are silently
skipped and do not appear in the output.  The reason is written to stderr
so callers can inspect failures without crashing a downstream pipeline.

Example
-------
>>> from tsml.models.baselines import CalibratedLogisticRegressionModel
>>> from tsml.portfolio import rank_universe
>>> from tsml.validation import WalkForwardSplit
>>>
>>> model    = CalibratedLogisticRegressionModel()
>>> splitter = WalkForwardSplit(n_splits=5, min_train_size=252, test_size=63)
>>> ranking  = rank_universe(
...     symbols  = ["SPY", "QQQ", "MSFT", "NVDA", "GOOGL"],
...     model    = model,
...     splitter = splitter,
...     target   = "direction",
...     start    = "2020-01-01",
...     end      = "2023-12-31",
... )
>>> print(ranking)
  symbol     score
0   NVDA  0.623...
1   MSFT  0.581...
...
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tsml.data_loader import YFinanceLoader
from tsml.data_loader.base import DataLoader
from tsml.pipelines.train import run_walk_forward_proba
from tsml.validation.splitters import WalkForwardSplit


def rank_universe(
    symbols: Sequence[str],
    model: Any,
    splitter: WalkForwardSplit,
    target: str = "direction",
    *,
    start: str,
    end: str,
    loader: DataLoader | None = None,
) -> pd.DataFrame:
    """
    Rank a universe of symbols by model conviction (P(up) on the last OOS date).

    For each symbol the function:

    1. Loads OHLCV data for the requested date range.
    2. Builds features and the requested target with ``make_dataset``.
    3. Runs ``run_walk_forward_proba`` to produce out-of-sample P(up) estimates.
    4. Takes ``probas.iloc[-1]`` — the probability on the most recent OOS date
       — as the ranking score.

    Symbols for which any step raises an exception (insufficient data,
    download failure, splitter mismatch) are skipped.  A one-line warning is
    printed to stderr for each skipped symbol.

    Parameters
    ----------
    symbols:
        Sequence of ticker strings to score (e.g. ``["SPY", "QQQ", "AAPL"]``).
    model:
        Any object with ``.fit(X, y)`` and ``.predict_proba(X)`` methods.
        The same instance is reused across symbols; state from one symbol's
        final fold will be overwritten by the next symbol's first fold.
    splitter:
        A configured ``WalkForwardSplit``.  If the cleaned dataset for a
        symbol is too small to satisfy ``splitter``'s requirements, that
        symbol is skipped.
    target:
        Target type passed to ``make_dataset``.  Must be one of
        ``"direction"``, ``"direction_5d"``, ``"threshold"``, ``"return"``.
        Default ``"direction"``.
    start:
        Inclusive start date for data loading, e.g. ``"2020-01-01"``.
    end:
        Inclusive end date for data loading, e.g. ``"2023-12-31"``.
    loader:
        Optional ``DataLoader`` instance.  Defaults to
        ``YFinanceLoader(cache_dir="data/raw")``.

    Returns
    -------
    pd.DataFrame
        Columns: ``symbol`` (str), ``score`` (float).
        Sorted by score descending.  Empty if every symbol failed.
        Index is reset (0, 1, 2, …).

    Notes
    -----
    The score is only comparable across symbols when the same model and
    splitter are used for all of them and their OOS date ranges overlap.
    Symbols with very different data densities (e.g. a newly-listed stock
    vs. a 10-year history) may have scores that are not directly comparable.
    """
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    records: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            df     = loader.load(symbol, start, end)
            probas = run_walk_forward_proba(df, model, splitter, target=target)
            score  = float(probas.iloc[-1])
            records.append({"symbol": symbol, "score": score})

        except Exception as exc:  # noqa: BLE001
            print(
                f"[rank_universe] skipping {symbol}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if not records:
        return pd.DataFrame(columns=["symbol", "score"])

    result = (
        pd.DataFrame(records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# Context enrichment
# ---------------------------------------------------------------------------

def enrich_with_context(
    ranking_df: pd.DataFrame,
    *,
    start: str,
    end: str,
    loader: DataLoader | None = None,
) -> pd.DataFrame:
    """
    Enrich a ranking DataFrame with explanatory market-context columns.

    These columns are derived solely from raw close prices and are intended
    to help a human understand *why* a symbol scored as it did.  They are
    **not** used as model features and are computed independently of the
    walk-forward pipeline.

    Columns added
    -------------
    return_20d      : 20-trading-day price return (float, e.g. 0.082 = +8.2 %)
    return_60d      : 60-trading-day price return (float)
    volatility_20d  : Annualised 20-day daily-return std (float)
    price_vs_sma_200: (close − SMA 200) / SMA 200 (float)
    above_sma_200   : True / False, or None when fewer than 200 rows are available

    Parameters
    ----------
    ranking_df:
        DataFrame as returned by :func:`rank_universe` (must contain a
        ``"symbol"`` column).
    start:
        Inclusive start date for data loading (used only to hit the cache;
        the most recent prices in the loaded range are used for metrics).
    end:
        Inclusive end date for data loading.
    loader:
        Optional ``DataLoader``.  Defaults to
        ``YFinanceLoader(cache_dir="data/raw")``.

    Returns
    -------
    pd.DataFrame
        The input ``ranking_df`` left-joined with the context columns.
        Symbols that fail context computation keep NaN / None values but
        remain in the output.
    """
    if loader is None:
        loader = YFinanceLoader(cache_dir="data/raw")

    ctx_rows: list[dict] = []
    for sym in ranking_df["symbol"]:
        row: dict[str, Any] = {"symbol": sym}
        try:
            df  = loader.load(sym, start, end)
            row.update(_compute_context(df["close"]))
        except Exception:  # noqa: BLE001
            row.update(_empty_context())
        ctx_rows.append(row)

    ctx_df = pd.DataFrame(ctx_rows)
    return ranking_df.merge(ctx_df, on="symbol", how="left")


def _compute_context(close: pd.Series) -> dict:
    """Compute context metrics from a close price series."""
    close = close.dropna()

    def _pct_return(n: int) -> float:
        if len(close) < n + 1:
            return float("nan")
        return float(close.iloc[-1] / close.iloc[-(n + 1)] - 1)

    rets    = close.pct_change().dropna()
    vol_20d = (
        float(rets.iloc[-20:].std() * np.sqrt(252))
        if len(rets) >= 20 else float("nan")
    )

    if len(close) >= 200:
        sma_200 = float(close.iloc[-200:].mean())
        price   = float(close.iloc[-1])
        vs_sma  = float(price / sma_200 - 1)
        abv_sma: bool | None = price > sma_200
    else:
        vs_sma  = float("nan")
        abv_sma = None

    return {
        "return_20d":       _pct_return(20),
        "return_60d":       _pct_return(60),
        "volatility_20d":   vol_20d,
        "price_vs_sma_200": vs_sma,
        "above_sma_200":    abv_sma,
    }


def _empty_context() -> dict:
    return {
        "return_20d":       float("nan"),
        "return_60d":       float("nan"),
        "volatility_20d":   float("nan"),
        "price_vs_sma_200": float("nan"),
        "above_sma_200":    None,
    }
