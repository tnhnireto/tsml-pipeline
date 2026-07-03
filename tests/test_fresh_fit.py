"""
Tests for run_full_fit_latest_proba — the fresh-fit live scoring path.

Key guarantees under test:

1. The model is trained on ALL labelled history (not a stale walk-forward
   fold) and predicts the latest feature rows.
2. For forward-horizon targets (direction_5d) the unlabelled tail rows are
   excluded from training but still scored — the returned index must end at
   the most recent trading day.
3. smoothing_window controls how many trailing rows are scored.
4. Insufficient data fails with a clear ValueError (callers skip gracefully).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.features.pipeline import build_features
from tsml.models.baselines import AlwaysLong, LogisticRegressionModel
from tsml.pipelines.train import run_full_fit_latest_proba


def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n))
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class SpyModel:
    """Wraps LogisticRegressionModel and records fit/predict inputs."""

    def __init__(self) -> None:
        self._inner = LogisticRegressionModel()
        self.fit_index: pd.DatetimeIndex | None = None
        self.predict_index: pd.DatetimeIndex | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SpyModel":
        self.fit_index = X.index
        self._inner.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_index = X.index
        return self._inner.predict_proba(X)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutput:
    def test_returns_series_of_length_smoothing_window(self):
        df = _make_ohlcv(600)
        result = run_full_fit_latest_proba(
            df, LogisticRegressionModel(), target="direction_5d", smoothing_window=5
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert result.name == "proba_up"

    def test_probabilities_in_unit_interval(self):
        df = _make_ohlcv(600)
        result = run_full_fit_latest_proba(
            df, LogisticRegressionModel(), target="direction_5d", smoothing_window=5
        )
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_always_long_scores_one(self):
        df = _make_ohlcv(600)
        result = run_full_fit_latest_proba(
            df, AlwaysLong(), target="direction_5d", smoothing_window=3
        )
        assert (result == 1.0).all()

    def test_last_scored_date_is_last_feature_date(self):
        df = _make_ohlcv(600)
        result = run_full_fit_latest_proba(
            df, LogisticRegressionModel(), smoothing_window=1
        )
        expected_last = build_features(df, feature_set="legacy").dropna().index[-1]
        assert result.index[-1] == expected_last
        # For legacy features the last feature row is the last trading day.
        assert result.index[-1] == df.index[-1]


# ---------------------------------------------------------------------------
# Freshness — the property walk-forward scoring lacks
# ---------------------------------------------------------------------------


class TestFreshness:
    def test_model_is_trained_on_all_labelled_history(self):
        """The training window must reach the newest labelled row."""
        df = _make_ohlcv(600)
        spy = SpyModel()
        run_full_fit_latest_proba(df, spy, target="direction", smoothing_window=1)
        features = build_features(df, feature_set="legacy").dropna()
        # direction target: every feature row except the last has a label.
        assert spy.fit_index is not None
        assert spy.fit_index[-1] == features.index[-2]

    def test_direction_5d_trains_without_last_5_labels_but_scores_them(self):
        df = _make_ohlcv(600)
        spy = SpyModel()
        result = run_full_fit_latest_proba(
            df, spy, target="direction_5d", smoothing_window=5
        )
        assert spy.fit_index is not None
        assert spy.predict_index is not None
        # Training stops where labels stop (5 rows before the end).
        features = build_features(df, feature_set="legacy").dropna()
        assert spy.fit_index[-1] == features.index[-6]
        # Prediction still covers the most recent trading days.
        assert spy.predict_index[-1] == features.index[-1]
        assert (result.index == features.index[-5:]).all()

    def test_prediction_dates_are_newer_than_walk_forward_final_fold(self):
        """Fresh-fit training data must extend beyond any walk-forward fold."""
        df = _make_ohlcv(600)
        spy = SpyModel()
        run_full_fit_latest_proba(df, spy, target="direction", smoothing_window=1)

        # Final fold of WalkForwardSplit(n_splits=3, min_train_size=252,
        # test_size=63) trains on the first 252 + 2*63 = 378 rows.
        last_train_end = 252 + 2 * 63
        features = build_features(df, feature_set="legacy").dropna()
        wf_last_train_date = features.index[last_train_end - 1]

        assert spy.fit_index[-1] > wf_last_train_date


# ---------------------------------------------------------------------------
# Validation and graceful failure
# ---------------------------------------------------------------------------


class TestValidation:
    def test_too_little_data_raises_value_error(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match="Not enough labelled rows"):
            run_full_fit_latest_proba(
                df, LogisticRegressionModel(), min_train_rows=252
            )

    def test_smoothing_window_below_1_raises(self):
        df = _make_ohlcv(600)
        with pytest.raises(ValueError, match="smoothing_window"):
            run_full_fit_latest_proba(
                df, LogisticRegressionModel(), smoothing_window=0
            )

    def test_min_train_rows_below_1_raises(self):
        df = _make_ohlcv(600)
        with pytest.raises(ValueError, match="min_train_rows"):
            run_full_fit_latest_proba(
                df, LogisticRegressionModel(), min_train_rows=0
            )


class TestSmoothingHorizonGuard:
    """smoothing_window must not exceed the target's label horizon."""

    def test_direction_5d_smoothing_at_horizon_passes(self):
        df = _make_ohlcv(600)
        result = run_full_fit_latest_proba(
            df, LogisticRegressionModel(), target="direction_5d", smoothing_window=5
        )
        assert len(result) == 5

    def test_direction_5d_smoothing_above_horizon_raises(self):
        df = _make_ohlcv(600)
        with pytest.raises(ValueError, match="smoothing_window"):
            run_full_fit_latest_proba(
                df, LogisticRegressionModel(), target="direction_5d", smoothing_window=6
            )

    def test_direction_smoothing_above_horizon_raises(self):
        df = _make_ohlcv(600)
        with pytest.raises(ValueError, match="smoothing_window"):
            run_full_fit_latest_proba(
                df, LogisticRegressionModel(), target="direction", smoothing_window=2
            )

    def test_direction_smoothing_at_horizon_passes(self):
        df = _make_ohlcv(600)
        result = run_full_fit_latest_proba(
            df, LogisticRegressionModel(), target="direction", smoothing_window=1
        )
        assert len(result) == 1
