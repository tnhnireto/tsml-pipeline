"""
Tests for run_walk_forward.

We use a lightweight stub model so tests run without sklearn and without
depending on any particular model's accuracy.  This isolates the pipeline
logic from the model logic.

Invariants tested
-----------------
1. Output is a pandas Series with a DatetimeIndex.
2. Number of predictions equals n_splits * test_size.
3. All prediction dates come from test folds, not training folds.
4. Prediction dates are in chronological order.
5. No prediction date falls inside a fold's own training window.
6. model.fit is called once per fold.
7. model.predict is called once per fold.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

from tsml.pipelines.train import run_walk_forward
from tsml.validation.splitters import WalkForwardSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame large enough for several folds."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
    close = 300.0 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.5, 2.0, n)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.Index(dates, name="date"),
    )


class _StubModel:
    """
    A model that always predicts 1 and records every call to fit/predict.
    It is a simple stand-in so tests focus on pipeline logic, not model logic.
    """

    def __init__(self):
        self.fit_calls: list[tuple[int, int]] = []   # (train_rows, n_features)
        self.predict_calls: list[int] = []            # test_rows per call

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_StubModel":
        self.fit_calls.append((len(X), X.shape[1]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls.append(len(X))
        return np.ones(len(X), dtype=int)


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_returns_series(self):
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=3, min_train_size=100, test_size=20)
        preds = run_walk_forward(df, _StubModel(), splitter)
        assert isinstance(preds, pd.Series)

    def test_length_equals_n_splits_times_test_size(self):
        df = _ohlcv()
        n_splits, test_size = 3, 20
        splitter = WalkForwardSplit(n_splits=n_splits, min_train_size=100,
                                    test_size=test_size)
        preds = run_walk_forward(df, _StubModel(), splitter)
        assert len(preds) == n_splits * test_size

    def test_index_is_datetimeindex(self):
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=2, min_train_size=100, test_size=20)
        preds = run_walk_forward(df, _StubModel(), splitter)
        assert isinstance(preds.index, pd.DatetimeIndex)

    def test_series_is_named_prediction(self):
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=2, min_train_size=100, test_size=20)
        preds = run_walk_forward(df, _StubModel(), splitter)
        assert preds.name == "prediction"


# ---------------------------------------------------------------------------
# Time-order invariants
# ---------------------------------------------------------------------------

class TestTimeOrder:
    def test_prediction_dates_are_sorted(self):
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=3, min_train_size=100, test_size=20)
        preds = run_walk_forward(df, _StubModel(), splitter)
        assert preds.index.is_monotonic_increasing

    def test_no_duplicate_prediction_dates(self):
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=3, min_train_size=100, test_size=20)
        preds = run_walk_forward(df, _StubModel(), splitter)
        assert not preds.index.duplicated().any()


# ---------------------------------------------------------------------------
# Leakage invariant: train window must not contain any test-fold date
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_prediction_dates_not_in_training_windows(self):
        """
        For each fold k, the test dates must not appear in the training
        window of fold k.  (They DO appear in later folds' training
        windows — that is expected and correct for expanding windows.)
        """
        from tsml.features.pipeline import make_dataset

        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=3, min_train_size=100, test_size=20)

        # Collect per-fold (train_dates, test_dates)
        X, _ = make_dataset(df)
        fold_info = [
            (set(X.iloc[tr].index), set(X.iloc[te].index))
            for tr, te in splitter.split(X)
        ]

        for fold_k, (train_dates, test_dates) in enumerate(fold_info):
            overlap = train_dates & test_dates
            assert len(overlap) == 0, (
                f"Fold {fold_k}: {len(overlap)} test date(s) appear in "
                f"their own training window — data leakage!"
            )

    def test_fit_only_sees_past_data(self):
        """
        The training data passed to model.fit() for fold k must consist
        entirely of rows whose dates precede every test date in fold k.
        """
        from tsml.features.pipeline import make_dataset

        df = _ohlcv()
        n_splits, test_size = 3, 20
        splitter = WalkForwardSplit(n_splits=n_splits, min_train_size=100,
                                    test_size=test_size)
        X, y = make_dataset(df)

        captured_fit_dates: list[pd.DatetimeIndex] = []
        captured_test_dates: list[pd.DatetimeIndex] = []

        class _RecordingModel:
            def fit(self, X_tr, y_tr):
                captured_fit_dates.append(X_tr.index)
                return self
            def predict(self, X_te):
                captured_test_dates.append(X_te.index)
                return np.ones(len(X_te), dtype=int)

        run_walk_forward(df, _RecordingModel(), splitter)

        for fold_k, (train_dates, test_dates) in enumerate(
            zip(captured_fit_dates, captured_test_dates)
        ):
            assert train_dates.max() < test_dates.min(), (
                f"Fold {fold_k}: last training date {train_dates.max()} "
                f">= first test date {test_dates.min()} — time-order violated!"
            )


# ---------------------------------------------------------------------------
# Model call counts
# ---------------------------------------------------------------------------

class TestModelCallCounts:
    def test_fit_called_once_per_fold(self):
        df = _ohlcv()
        n_splits = 4
        splitter = WalkForwardSplit(n_splits=n_splits, min_train_size=100,
                                    test_size=20)
        model = _StubModel()
        run_walk_forward(df, model, splitter)
        assert len(model.fit_calls) == n_splits

    def test_predict_called_once_per_fold(self):
        df = _ohlcv()
        n_splits = 4
        splitter = WalkForwardSplit(n_splits=n_splits, min_train_size=100,
                                    test_size=20)
        model = _StubModel()
        run_walk_forward(df, model, splitter)
        assert len(model.predict_calls) == n_splits

    def test_train_size_grows_each_fold(self):
        """Training set passed to fit() must grow with each fold."""
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=4, min_train_size=100, test_size=20)
        model = _StubModel()
        run_walk_forward(df, model, splitter)
        train_sizes = [rows for rows, _ in model.fit_calls]
        assert train_sizes == sorted(train_sizes), (
            f"Training sizes not monotonically increasing: {train_sizes}"
        )

    def test_predict_always_receives_test_size_rows(self):
        df = _ohlcv()
        test_size = 25
        splitter = WalkForwardSplit(n_splits=3, min_train_size=100,
                                    test_size=test_size)
        model = _StubModel()
        run_walk_forward(df, model, splitter)
        assert all(n == test_size for n in model.predict_calls)


# ---------------------------------------------------------------------------
# Return target
# ---------------------------------------------------------------------------

class TestReturnTarget:
    def test_return_target_works(self):
        df = _ohlcv()
        splitter = WalkForwardSplit(n_splits=2, min_train_size=100, test_size=20)
        preds = run_walk_forward(df, _StubModel(), splitter, target="return")
        assert len(preds) == 2 * 20
