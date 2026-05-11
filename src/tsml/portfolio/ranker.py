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
