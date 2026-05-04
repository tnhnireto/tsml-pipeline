"""
Baseline models — the first thing to beat before trying anything fancier.

A more complex model is only useful if it outperforms these baselines on
out-of-sample Sharpe.  A model that can't beat AlwaysLong or
PreviousDirection is probably overfit or measuring the wrong thing.

All models share the same three-method interface:
    fit(X, y)          → self
    predict(X)         → np.ndarray of int labels (0 or 1)
    predict_proba(X)   → np.ndarray of shape (n, 2), columns [P(0), P(1)]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# AlwaysLong
# ---------------------------------------------------------------------------


class AlwaysLong:
    """
    Always predicts 1 (market goes up every day).

    This is the weakest possible baseline.  For long-only equity indices
    like SPY it is surprisingly hard to beat on risk-adjusted returns
    because markets have a long-run upward bias.

    If your model underperforms this, it is worse than doing nothing.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlwaysLong":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Degenerate probabilities: P(up) = 1 for every row."""
        n = len(X)
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 1] = 1.0
        return proba


# ---------------------------------------------------------------------------
# PreviousDirection
# ---------------------------------------------------------------------------


class PreviousDirection:
    """
    Predicts tomorrow's direction by repeating today's direction (momentum).

    Logic:
        today's direction   = 1  if  close_t > close_{t-1}  (return_1d > 0)
        predicted direction = today's direction

    This baseline encodes the simplest momentum hypothesis: if the market
    went up today, predict it goes up again tomorrow.

    It requires the 'return_1d' feature column to be present in X.
    That column is always produced by `build_features`.

    No parameters are learned during fit; this model is entirely rule-based.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreviousDirection":
        if "return_1d" not in X.columns:
            raise ValueError(
                "PreviousDirection requires 'return_1d' in X. "
                "Run build_features() first."
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "return_1d" not in X.columns:
            raise ValueError(
                "PreviousDirection requires 'return_1d' in X. "
                "Run build_features() first."
            )
        return (X["return_1d"] > 0).astype(int).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Hard probabilities: 1.0 for the predicted class, 0.0 for the other."""
        preds = self.predict(X)
        n = len(X)
        proba = np.zeros((n, 2), dtype=float)
        proba[np.arange(n), preds] = 1.0
        return proba


# ---------------------------------------------------------------------------
# LogisticRegressionModel
# ---------------------------------------------------------------------------


class LogisticRegressionModel:
    """
    Logistic regression with built-in feature scaling.

    Logistic regression is a strong linear baseline for binary classification.
    Financial features (returns, RSI, SMA ratio) have very different scales,
    so a StandardScaler is always applied before fitting.

    The scaler is fitted on the training X only — never on test or
    future data.  The same fitted scaler is used at predict time.

    Parameters
    ----------
    C:
        Inverse regularisation strength. Smaller = more regularisation.
    max_iter:
        Solver iteration limit.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")


# ---------------------------------------------------------------------------
# CalibratedLogisticRegressionModel
# ---------------------------------------------------------------------------


class CalibratedLogisticRegressionModel:
    """
    Logistic regression with Platt-scaling calibration (sigmoid method).

    CalibratedClassifierCV is fitted entirely on the training fold using
    internal cross-validation, so no future data is ever touched.  The
    resulting predict_proba outputs are better calibrated than raw logistic
    regression probabilities, meaning P(up)=0.60 should more faithfully
    represent a 60 % empirical frequency of up-days.

    Leakage guarantee
    -----------------
    Both the StandardScaler and the CalibratedClassifierCV are fitted
    inside model.fit(X_train, y_train) — strictly on training-fold data.
    The same fitted objects are used at predict/predict_proba time.

    Parameters
    ----------
    C:
        Inverse regularisation strength for the base LogisticRegression.
    max_iter:
        Solver iteration limit.
    method:
        Calibration method — ``"sigmoid"`` (Platt scaling) or
        ``"isotonic"`` (non-parametric, needs more data).
    cv:
        Number of CV folds used internally by CalibratedClassifierCV.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        method: str = "sigmoid",
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        base = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self._model = CalibratedClassifierCV(base, method=method, cv=cv)
        self._is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "CalibratedLogisticRegressionModel":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(self._scaler.transform(X))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")


# ---------------------------------------------------------------------------
# RandomForestModel
# ---------------------------------------------------------------------------


class RandomForestModel:
    """
    Random forest classifier for binary direction prediction.

    Random forests are naturally resistant to overfitting via bagging and
    feature subsampling, and their predict_proba outputs are already
    reasonable without explicit calibration (though still imperfect).

    No feature scaling is applied — tree-based models are scale-invariant.

    Parameters
    ----------
    n_estimators:
        Number of trees.  200 gives stable probability estimates.
    max_depth:
        Maximum tree depth.  Shallow trees (4–5) reduce variance on
        small financial datasets.
    min_samples_leaf:
        Minimum samples required at a leaf.  Higher values smooth
        probability estimates and reduce overfitting.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
