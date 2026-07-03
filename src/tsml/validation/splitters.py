"""
Time-series-aware cross-validation splitter.

Walk-forward validation is the correct way to cross-validate a time
series model.  The key rule: the training set always ends before the
test set begins.

How it works (expanding window)
--------------------------------
Given a dataset of N rows and the parameters below, each fold k
(0-indexed) produces:

    train: rows 0 … (min_train_size + k * test_size - 1)   ← grows
    gap:   rows (train_end) … (train_end + gap - 1)         ← skipped
    test:  rows (train_end + gap) … (train_end + gap + test_size - 1)

With gap=0 the test window starts immediately after training ends.
A gap > 0 adds an embargo so that labels near the train boundary
(which may be partially observed) are not used for evaluation.

Parameters
----------
n_splits      : int   Number of folds to produce.
min_train_size: int   Rows in the first training window (grows after).
test_size     : int   Rows in each test window (constant across folds).
gap           : int   Rows to skip between train end and test start.
                      Default 0.

Minimum dataset size required: min_train_size + n_splits * test_size + gap

Example
-------
>>> splitter = WalkForwardSplit(n_splits=3, min_train_size=100,
...                              test_size=20, gap=1)
>>> for train_idx, test_idx in splitter.split(X):
...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
...     preds = model.predict(X.iloc[test_idx])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class WalkForwardSplit:
    """Expanding-window walk-forward cross-validator for time series."""

    def __init__(
        self,
        n_splits: int,
        min_train_size: int,
        test_size: int,
        gap: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}.")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}.")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}.")

        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, X: pd.DataFrame | np.ndarray
    ) -> "Generator[tuple[np.ndarray, np.ndarray]]":
        """
        Yield (train_indices, test_indices) for each fold.

        Both index arrays are integer positions (suitable for .iloc).

        Parameters
        ----------
        X:
            The feature matrix or any array-like whose length is the
            number of available rows.  Only len(X) is used.

        Raises
        ------
        ValueError
            If the dataset is too short to produce all requested folds.
        """
        n = len(X)
        min_required = self.min_train_size + self.n_splits * self.test_size + self.gap
        if n < min_required:
            raise ValueError(
                f"Not enough data for {self.n_splits} folds. "
                f"Need at least {min_required} rows, got {n}. "
                f"(min_train_size={self.min_train_size}, "
                f"test_size={self.test_size}, gap={self.gap})"
            )

        for k in range(self.n_splits):
            train_end = self.min_train_size + k * self.test_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return the number of folds. Matches the sklearn CV interface."""
        return self.n_splits


def coverage_n_splits(
    start: str,
    end: str,
    *,
    min_train_size: int = 252,
    test_size: int = 63,
    gap: int = 0,
    warmup_rows: int = 70,
) -> int:
    """
    Estimate how many walk-forward folds fit between ``start`` and ``end``.

    A fixed, small ``n_splits`` only produces out-of-sample predictions for
    the first ``n_splits * test_size`` rows after the initial training
    window.  Downstream consumers (e.g. the portfolio simulator) forward-
    fill the last known score, so a backtest that extends past the final
    fold silently trades on stale, frozen probabilities.  Sizing
    ``n_splits`` to the date range keeps out-of-sample coverage honest.

    The row count is estimated from business days, deliberately shaved for
    market holidays (~4%) and the feature warmup (``warmup_rows``; the
    feature pipeline drops ~25-65 leading rows depending on feature set).
    Underestimating is safe — at worst the final ``< test_size`` rows are
    forward-filled — while overestimating would make the splitter raise
    and callers skip symbols entirely.

    Returns at least 1.
    """
    approx_bdays = len(pd.bdate_range(start, end))
    usable = int(approx_bdays * 0.96) - warmup_rows
    return max(1, (usable - min_train_size - gap) // test_size)


@dataclass(frozen=True)
class AdaptiveWalkForwardParams:
    """Shared walk-forward sizing used by coverage estimation and splitting."""

    min_train_size: int = 252
    test_size: int = 63
    gap: int = 0
    warmup_rows: int = 70


def make_adaptive_walk_forward_splitter(
    start: str,
    end: str,
    params: AdaptiveWalkForwardParams | None = None,
) -> WalkForwardSplit:
    """
    Build a walk-forward splitter sized to cover ``start`` through ``end``.

    ``coverage_n_splits`` and ``WalkForwardSplit`` receive the same
    ``min_train_size``, ``test_size``, and ``gap`` from ``params`` so the
    fold count cannot drift from the splitter configuration.
    """
    p = params or AdaptiveWalkForwardParams()
    n_splits = coverage_n_splits(
        start,
        end,
        min_train_size=p.min_train_size,
        test_size=p.test_size,
        gap=p.gap,
        warmup_rows=p.warmup_rows,
    )
    return WalkForwardSplit(
        n_splits=n_splits,
        min_train_size=p.min_train_size,
        test_size=p.test_size,
        gap=p.gap,
    )
