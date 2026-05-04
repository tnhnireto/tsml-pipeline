"""
Walk-forward training pipeline.

`run_walk_forward` is the main entry point.  It wires together every
component built so far:

    OHLCV DataFrame
        → make_dataset  (features + target, NaNs dropped)
        → WalkForwardSplit  (time-ordered folds)
        → for each fold:
              model.fit(X_train, y_train)
              model.predict(X_test)
        → collect predictions into a date-indexed Series

The output is a pandas Series whose index contains only the test-fold
dates.  It is intentionally kept separate from the backtest so each step
can be inspected, saved, or replaced independently.

Leakage guarantees in this function
-------------------------------------
1. `make_dataset` is called once on the full DataFrame to compute
   features, but the *splitter indices* determine what the model sees.
2. `model.fit` only receives rows from the current fold's training
   window.  For `LogisticRegressionModel` this means the StandardScaler
   is also fitted on those rows only.
3. `model.predict` is called on test rows that were never seen during
   `fit`.
4. Predictions are stored and then returned — they are never fed back
   into the model during the loop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tsml.features.pipeline import make_dataset
from tsml.validation.splitters import WalkForwardSplit


def run_walk_forward(
    df: pd.DataFrame,
    model: Any,
    splitter: WalkForwardSplit,
    target: str = "direction",
) -> pd.Series:
    """
    Run walk-forward cross-validation and return all out-of-sample predictions.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame (UTC-indexed, validated by the data loader).
    model:
        Any object with .fit(X, y) and .predict(X) methods.
        The same instance is reused across folds — each fold calls
        fit() again, replacing the previous model state.
    splitter:
        A configured WalkForwardSplit instance.
    target:
        ``"direction"`` (binary, default) or ``"return"`` (regression).

    Returns
    -------
    pd.Series
        Predictions indexed by date, one entry per test-fold row.
        Name is ``"prediction"``.

    Raises
    ------
    ValueError
        If the cleaned dataset is too small for the splitter.
    """
    X, y = make_dataset(df, target=target)

    predictions: dict[pd.Timestamp, Any] = {}

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        model.fit(X_train, y_train)
        fold_preds = model.predict(X_test)

        for date, pred in zip(X_test.index, fold_preds):
            predictions[date] = pred

    if not predictions:
        raise RuntimeError("No predictions were produced. Check splitter parameters.")

    result = pd.Series(predictions, name="prediction")
    result.index.name = "date"
    return result


def run_walk_forward_proba(
    df: pd.DataFrame,
    model: Any,
    splitter: WalkForwardSplit,
    target: str = "direction",
) -> pd.Series:
    """
    Walk-forward cross-validation that returns P(up) probabilities.

    Identical to ``run_walk_forward`` but calls ``model.predict_proba``
    instead of ``model.predict``.  The returned Series contains the
    probability of class 1 (market up) for every out-of-sample date.

    Apply a threshold to convert probabilities to 0/1 signals::

        signals = (probas > threshold).astype(int)

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame.
    model:
        Any object with .fit(X, y) and .predict_proba(X) methods.
        ``predict_proba`` must return an (n, 2) array where column 1
        is P(class=1).
    splitter:
        A configured WalkForwardSplit instance.
    target:
        ``"direction"`` (default) or ``"return"``.

    Returns
    -------
    pd.Series
        P(up) probabilities indexed by date.  Name is ``"proba_up"``.
    """
    X, y = make_dataset(df, target=target)

    probas: dict[pd.Timestamp, float] = {}

    for _, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        model.fit(X_train, y_train)
        fold_probas = model.predict_proba(X_test)[:, 1]

        for date, p in zip(X_test.index, fold_probas):
            probas[date] = p

    if not probas:
        raise RuntimeError("No probabilities were produced. Check splitter parameters.")

    result = pd.Series(probas, name="proba_up")
    result.index.name = "date"
    return result
