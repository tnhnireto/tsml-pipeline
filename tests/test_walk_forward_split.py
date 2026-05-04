"""
Tests for WalkForwardSplit.

Every test encodes one specific invariant that must hold for
walk-forward cross-validation to be leakage-free.  If any test fails
when you modify the splitter, you have broken a safety guarantee.

Invariants tested
-----------------
1. Train indices always come before test indices (no temporal inversion).
2. Train and test sets are disjoint (no row in both).
3. The gap/embargo is exactly respected.
4. Training window grows with each fold (expanding window).
5. Test window is exactly test_size rows in every fold.
6. Exactly n_splits folds are produced.
7. Too-short datasets raise a clear error.
8. Invalid constructor arguments raise immediately.
9. gap=0 means train and test are adjacent (no skipped rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.validation.splitters import WalkForwardSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _X(n: int = 200) -> pd.DataFrame:
    """Minimal DataFrame — only length matters for the splitter."""
    return pd.DataFrame({"a": np.arange(n)})


def _all_folds(splitter: WalkForwardSplit, X: pd.DataFrame):
    return list(splitter.split(X))


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestTimeOrder:
    """Train must always end before test begins."""

    def test_all_train_before_test(self):
        splitter = WalkForwardSplit(n_splits=4, min_train_size=50, test_size=20)
        for train_idx, test_idx in splitter.split(_X()):
            assert train_idx.max() < test_idx.min(), (
                f"Train bleeds into test: max(train)={train_idx.max()}, "
                f"min(test)={test_idx.min()}"
            )

    def test_train_max_less_than_test_min_with_gap(self):
        splitter = WalkForwardSplit(n_splits=3, min_train_size=40, test_size=10, gap=5)
        for train_idx, test_idx in splitter.split(_X()):
            assert train_idx.max() < test_idx.min()


class TestDisjoint:
    """No row index should appear in both train and test."""

    def test_no_overlap_gap_zero(self):
        splitter = WalkForwardSplit(n_splits=4, min_train_size=50, test_size=20)
        for train_idx, test_idx in splitter.split(_X()):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_no_overlap_with_gap(self):
        splitter = WalkForwardSplit(n_splits=3, min_train_size=40, test_size=10, gap=5)
        for train_idx, test_idx in splitter.split(_X()):
            assert len(set(train_idx) & set(test_idx)) == 0


class TestGap:
    """The embargo gap must be exactly `gap` skipped rows."""

    def test_gap_zero_means_adjacent(self):
        """With gap=0, test starts immediately after the last train row."""
        splitter = WalkForwardSplit(n_splits=3, min_train_size=50, test_size=10, gap=0)
        for train_idx, test_idx in splitter.split(_X()):
            assert test_idx.min() == train_idx.max() + 1, (
                "With gap=0, test[0] must equal train[-1] + 1"
            )

    def test_gap_one_skips_one_row(self):
        splitter = WalkForwardSplit(n_splits=3, min_train_size=50, test_size=10, gap=1)
        for train_idx, test_idx in splitter.split(_X()):
            skipped = test_idx.min() - train_idx.max() - 1
            assert skipped == 1, f"Expected 1 skipped row, got {skipped}"

    def test_gap_five_skips_five_rows(self):
        splitter = WalkForwardSplit(n_splits=3, min_train_size=40, test_size=10, gap=5)
        for train_idx, test_idx in splitter.split(_X()):
            skipped = test_idx.min() - train_idx.max() - 1
            assert skipped == 5, f"Expected 5 skipped rows, got {skipped}"


class TestExpandingWindow:
    """Training set must grow by exactly test_size rows each fold."""

    def test_train_grows_monotonically(self):
        splitter = WalkForwardSplit(n_splits=4, min_train_size=50, test_size=20)
        folds = _all_folds(splitter, _X())
        train_sizes = [len(tr) for tr, _ in folds]
        assert train_sizes == sorted(train_sizes), "Train sizes not monotonically increasing"

    def test_train_grows_by_test_size_each_fold(self):
        splitter = WalkForwardSplit(n_splits=4, min_train_size=50, test_size=20)
        folds = _all_folds(splitter, _X())
        for i in range(1, len(folds)):
            growth = len(folds[i][0]) - len(folds[i - 1][0])
            assert growth == splitter.test_size, (
                f"Fold {i}: expected growth of {splitter.test_size}, got {growth}"
            )

    def test_first_fold_train_size_equals_min_train_size(self):
        splitter = WalkForwardSplit(n_splits=3, min_train_size=60, test_size=15)
        first_train, _ = next(splitter.split(_X()))
        assert len(first_train) == 60


class TestFoldCount:
    def test_exactly_n_splits_folds(self):
        for n in [1, 3, 5]:
            splitter = WalkForwardSplit(n_splits=n, min_train_size=50, test_size=10)
            folds = _all_folds(splitter, _X())
            assert len(folds) == n

    def test_get_n_splits_matches(self):
        splitter = WalkForwardSplit(n_splits=4, min_train_size=50, test_size=20)
        assert splitter.get_n_splits() == 4


class TestTestSize:
    def test_test_size_constant_across_folds(self):
        splitter = WalkForwardSplit(n_splits=4, min_train_size=50, test_size=20)
        for _, test_idx in splitter.split(_X()):
            assert len(test_idx) == 20

    def test_test_size_one(self):
        splitter = WalkForwardSplit(n_splits=5, min_train_size=50, test_size=1)
        for _, test_idx in splitter.split(_X()):
            assert len(test_idx) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_too_short_raises(self):
        # Need 50 + 3*20 + 0 = 110 rows; only give 100.
        splitter = WalkForwardSplit(n_splits=3, min_train_size=50, test_size=20)
        with pytest.raises(ValueError, match="Not enough data"):
            list(splitter.split(_X(n=100)))

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            WalkForwardSplit(n_splits=0, min_train_size=50, test_size=10)

    def test_invalid_min_train_size_raises(self):
        with pytest.raises(ValueError, match="min_train_size"):
            WalkForwardSplit(n_splits=3, min_train_size=0, test_size=10)

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardSplit(n_splits=3, min_train_size=50, test_size=0)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError, match="gap"):
            WalkForwardSplit(n_splits=3, min_train_size=50, test_size=10, gap=-1)


# ---------------------------------------------------------------------------
# Sentinel test: a manual overlap check that would catch a broken splitter
# ---------------------------------------------------------------------------

class TestManualInvariant:
    """
    These tests are written in a way that would FAIL if the splitter
    wrongly returned overlapping or reversed index arrays.
    We verify both the property and that a broken version would be caught.
    """

    def test_reversed_indices_would_fail(self):
        """Show that the test catches time-order violations."""
        splitter = WalkForwardSplit(n_splits=2, min_train_size=50, test_size=20)
        folds = _all_folds(splitter, _X())
        for train_idx, test_idx in folds:
            # This assertion would fail if train and test were swapped.
            assert not (test_idx.max() < train_idx.min()), \
                "Indices appear completely reversed"
            # Correct assertion: train comes first.
            assert train_idx.max() < test_idx.min()

    def test_overlapping_indices_would_fail(self):
        """Verify the disjoint check would catch overlapping ranges."""
        # Construct two overlapping arrays manually to confirm the check works.
        bad_train = np.arange(0, 60)
        bad_test = np.arange(50, 80)   # overlaps 50..59
        overlap = set(bad_train) & set(bad_test)
        assert len(overlap) > 0  # the fake "bad" case does overlap

        # Now confirm the real splitter does NOT have this problem.
        splitter = WalkForwardSplit(n_splits=2, min_train_size=50, test_size=20)
        for train_idx, test_idx in splitter.split(_X()):
            real_overlap = set(train_idx) & set(test_idx)
            assert len(real_overlap) == 0
