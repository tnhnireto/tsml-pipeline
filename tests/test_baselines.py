"""
Tests for baseline models: AlwaysLong, PreviousDirection, LogisticRegressionModel.

Each model is tested for:
  - correct output shape and dtype
  - correct prediction logic (where deterministic)
  - predict_proba shape and valid probability values
  - fit/predict lifecycle (predict before fit should raise for learnable models)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.models.baselines import AlwaysLong, LogisticRegressionModel, PreviousDirection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_X() -> pd.DataFrame:
    """Small feature DataFrame with the columns our models expect."""
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame(
        {
            "return_1d": rng.normal(0, 0.01, n),
            "log_return_1d": rng.normal(0, 0.01, n),
            "return_lag1": rng.normal(0, 0.01, n),
            "rolling_vol_10": np.abs(rng.normal(0.01, 0.005, n)),
            "rsi_14": rng.uniform(30, 70, n),
        }
    )


@pytest.fixture
def simple_y() -> pd.Series:
    """Binary target aligned with simple_X."""
    rng = np.random.default_rng(1)
    return pd.Series(rng.integers(0, 2, 30).astype(float), name="target_direction")


# ---------------------------------------------------------------------------
# AlwaysLong
# ---------------------------------------------------------------------------

class TestAlwaysLong:
    def test_predict_always_returns_ones(self, simple_X):
        model = AlwaysLong().fit(simple_X, None)
        preds = model.predict(simple_X)
        assert (preds == 1).all()

    def test_predict_shape(self, simple_X):
        preds = AlwaysLong().fit(simple_X, None).predict(simple_X)
        assert preds.shape == (len(simple_X),)

    def test_predict_dtype_is_int(self, simple_X):
        preds = AlwaysLong().fit(simple_X, None).predict(simple_X)
        assert np.issubdtype(preds.dtype, np.integer)

    def test_predict_proba_shape(self, simple_X):
        proba = AlwaysLong().fit(simple_X, None).predict_proba(simple_X)
        assert proba.shape == (len(simple_X), 2)

    def test_predict_proba_sums_to_one(self, simple_X):
        proba = AlwaysLong().fit(simple_X, None).predict_proba(simple_X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_column_1_is_one(self, simple_X):
        """AlwaysLong assigns all probability mass to class 1."""
        proba = AlwaysLong().fit(simple_X, None).predict_proba(simple_X)
        assert (proba[:, 1] == 1.0).all()
        assert (proba[:, 0] == 0.0).all()

    def test_fit_ignores_y(self, simple_X):
        """fit() must work even when y is None (no labels needed)."""
        model = AlwaysLong()
        model.fit(simple_X, None)
        preds = model.predict(simple_X)
        assert len(preds) == len(simple_X)


# ---------------------------------------------------------------------------
# PreviousDirection
# ---------------------------------------------------------------------------

class TestPreviousDirection:
    def test_positive_return_predicts_one(self):
        X = pd.DataFrame({"return_1d": [0.01, 0.02, 0.005]})
        model = PreviousDirection().fit(X, None)
        preds = model.predict(X)
        assert (preds == 1).all()

    def test_negative_return_predicts_zero(self):
        X = pd.DataFrame({"return_1d": [-0.01, -0.02, -0.005]})
        model = PreviousDirection().fit(X, None)
        preds = model.predict(X)
        assert (preds == 0).all()

    def test_mixed_returns(self):
        X = pd.DataFrame({"return_1d": [0.01, -0.01, 0.0, 0.02, -0.03]})
        model = PreviousDirection().fit(X, None)
        preds = model.predict(X)
        expected = np.array([1, 0, 0, 1, 0])
        np.testing.assert_array_equal(preds, expected)

    def test_zero_return_predicts_zero(self):
        """return_1d == 0 is not strictly positive → predict 0."""
        X = pd.DataFrame({"return_1d": [0.0]})
        preds = PreviousDirection().fit(X, None).predict(X)
        assert preds[0] == 0

    def test_missing_return_1d_raises_on_fit(self):
        X = pd.DataFrame({"rsi_14": [50.0, 60.0]})
        with pytest.raises(ValueError, match="return_1d"):
            PreviousDirection().fit(X, None)

    def test_missing_return_1d_raises_on_predict(self):
        X_train = pd.DataFrame({"return_1d": [0.01]})
        X_test = pd.DataFrame({"rsi_14": [50.0]})
        model = PreviousDirection().fit(X_train, None)
        with pytest.raises(ValueError, match="return_1d"):
            model.predict(X_test)

    def test_predict_shape(self, simple_X):
        preds = PreviousDirection().fit(simple_X, None).predict(simple_X)
        assert preds.shape == (len(simple_X),)

    def test_predict_proba_shape(self, simple_X):
        proba = PreviousDirection().fit(simple_X, None).predict_proba(simple_X)
        assert proba.shape == (len(simple_X), 2)

    def test_predict_proba_sums_to_one(self, simple_X):
        proba = PreviousDirection().fit(simple_X, None).predict_proba(simple_X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_consistent_with_predict(self, simple_X):
        """The argmax of predict_proba must equal predict."""
        model = PreviousDirection().fit(simple_X, None)
        preds = model.predict(simple_X)
        proba = model.predict_proba(simple_X)
        assert (np.argmax(proba, axis=1) == preds).all()


# ---------------------------------------------------------------------------
# LogisticRegressionModel
# ---------------------------------------------------------------------------

class TestLogisticRegressionModel:
    def test_predict_before_fit_raises(self, simple_X):
        model = LogisticRegressionModel()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(simple_X)

    def test_predict_proba_before_fit_raises(self, simple_X):
        model = LogisticRegressionModel()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict_proba(simple_X)

    def test_fit_returns_self(self, simple_X, simple_y):
        model = LogisticRegressionModel()
        result = model.fit(simple_X, simple_y)
        assert result is model

    def test_predict_shape(self, simple_X, simple_y):
        model = LogisticRegressionModel().fit(simple_X, simple_y)
        preds = model.predict(simple_X)
        assert preds.shape == (len(simple_X),)

    def test_predict_binary_output(self, simple_X, simple_y):
        model = LogisticRegressionModel().fit(simple_X, simple_y)
        preds = model.predict(simple_X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, simple_X, simple_y):
        model = LogisticRegressionModel().fit(simple_X, simple_y)
        proba = model.predict_proba(simple_X)
        assert proba.shape == (len(simple_X), 2)

    def test_predict_proba_rows_sum_to_one(self, simple_X, simple_y):
        model = LogisticRegressionModel().fit(simple_X, simple_y)
        proba = model.predict_proba(simple_X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_values_in_0_1(self, simple_X, simple_y):
        model = LogisticRegressionModel().fit(simple_X, simple_y)
        proba = model.predict_proba(simple_X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_consistent_with_proba(self, simple_X, simple_y):
        model = LogisticRegressionModel().fit(simple_X, simple_y)
        preds = model.predict(simple_X)
        proba = model.predict_proba(simple_X)
        assert (np.argmax(proba, axis=1) == preds).all()

    def test_scaler_fit_on_train_only(self):
        """
        A test for the leakage invariant: the scaler must be fitted only on
        train data, not on test data.  We verify this by checking that the
        scaler mean is close to the training data mean, not the test data mean.
        """
        rng = np.random.default_rng(42)
        n_train = 100

        # Training data: returns centred around +0.01
        X_train = pd.DataFrame({"f": rng.normal(0.01, 0.005, n_train)})
        y_train = pd.Series(rng.integers(0, 2, n_train).astype(float))

        # Test data: completely different distribution
        X_test = pd.DataFrame({"f": rng.normal(5.0, 0.1, 50)})

        model = LogisticRegressionModel().fit(X_train, y_train)

        # The scaler's mean should match the training distribution, not test.
        scaler_mean = model._scaler.mean_[0]
        assert abs(scaler_mean - 0.01) < 0.01, (
            f"Scaler mean {scaler_mean:.4f} doesn't match training mean ~0.01. "
            "It may have been fitted on the wrong data."
        )

    def test_different_C_changes_predictions(self, simple_X, simple_y):
        """Changing regularisation strength should generally change predictions."""
        m1 = LogisticRegressionModel(C=0.001).fit(simple_X, simple_y)
        m2 = LogisticRegressionModel(C=1000.0).fit(simple_X, simple_y)
        # Very different C values almost certainly produce different probabilities.
        assert not np.allclose(
            m1.predict_proba(simple_X), m2.predict_proba(simple_X)
        ), "Extreme C values should differ; if they don't, something is wrong."
