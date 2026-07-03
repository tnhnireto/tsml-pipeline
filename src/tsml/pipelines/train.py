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

from tsml.features.pipeline import build_features, make_dataset
from tsml.features.targets import target_label_horizon
from tsml.validation.splitters import WalkForwardSplit


def run_walk_forward(
    df: pd.DataFrame,
    model: Any,
    splitter: WalkForwardSplit,
    target: str = "direction",
    *,
    feature_set: str = "legacy",
    benchmarks: dict[str, pd.Series] | None = None,
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
    X, y = make_dataset(
        df,
        target=target,
        feature_set=feature_set,
        benchmarks=benchmarks,
    )

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
    *,
    feature_set: str = "legacy",
    benchmarks: dict[str, pd.Series] | None = None,
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
    X, y = make_dataset(
        df,
        target=target,
        feature_set=feature_set,
        benchmarks=benchmarks,
    )

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


def run_full_fit_latest_proba(
    df: pd.DataFrame,
    model: Any,
    target: str = "direction",
    *,
    feature_set: str = "legacy",
    benchmarks: dict[str, pd.Series] | None = None,
    smoothing_window: int = 1,
    min_train_rows: int = 252,
) -> pd.Series:
    """
    Fit a fresh model on ALL labelled history and score the latest rows.

    This is the **live-signal** counterpart of ``run_walk_forward_proba``.
    Walk-forward probabilities are ideal for evaluation but the final OOS
    fold's model can be up to ``test_size`` days stale.  For live ranking we
    instead:

    1. Build the full (X, y) training set with ``make_dataset``.  For
       forward-looking targets (e.g. ``direction_5d``) the last rows have no
       label yet and are automatically excluded from training.
    2. Fit ``model`` on that entire labelled history — the freshest possible
       model, trained on data up to today.
    3. Recompute features for **all** rows (labels not required) and return
       P(up) for the last ``smoothing_window`` feature rows, which include
       the most recent trading day.

    Leakage guarantee
    -----------------
  When ``smoothing_window <= target_label_horizon(target)``:

    Training labels only use realised prices; rows whose labels would need
    future data are dropped by ``make_dataset``.  Prediction rows use only
    backward-looking features.  No future information is used anywhere.

    If ``smoothing_window`` exceeds the target's label horizon, some scored
    rows overlap the training set and a ``ValueError`` is raised.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame.
    model:
        Any object with ``.fit(X, y)`` and ``.predict_proba(X)`` methods.
    target:
        Target type passed to ``make_dataset``.
    feature_set:
        ``"legacy"``, ``"extended"`` or ``"extended_v2"``.
    benchmarks:
        SPY/QQQ close series when the feature set requires them.
    smoothing_window:
        Number of most-recent feature rows to score (>= 1).  Callers can
        average the returned probabilities to smooth day-to-day noise.
    min_train_rows:
        Minimum labelled training rows required.  Below this a
        ``ValueError`` is raised so callers can skip the symbol gracefully.

    Returns
    -------
    pd.Series
        P(up) for the last ``smoothing_window`` feature dates, indexed by
        date, name ``"proba_up"``.  The final entry corresponds to the most
        recent trading day in ``df``.
    """
    if smoothing_window < 1:
        raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}.")
    if min_train_rows < 1:
        raise ValueError(f"min_train_rows must be >= 1, got {min_train_rows}.")

    horizon = target_label_horizon(target)
    if smoothing_window > horizon:
        raise ValueError(
            f"smoothing_window ({smoothing_window}) exceeds the label horizon "
            f"for target '{target}' ({horizon}). Scoring would overlap training "
            f"rows whose labels were used in fit()."
        )

    X, y = make_dataset(
        df,
        target=target,
        feature_set=feature_set,
        benchmarks=benchmarks,
    )
    if len(X) < min_train_rows:
        raise ValueError(
            f"Not enough labelled rows to fit a live model: "
            f"got {len(X)}, need at least {min_train_rows}."
        )

    model.fit(X, y)

    # Features exist for rows whose *labels* are still unknown (the forward
    # horizon) — exactly the rows a live prediction must cover.
    features = build_features(
        df, feature_set=feature_set, benchmarks=benchmarks
    ).dropna()
    if len(features) < smoothing_window:
        raise ValueError(
            f"Not enough feature rows to score: got {len(features)}, "
            f"need at least {smoothing_window}."
        )

    X_latest = features.iloc[-smoothing_window:]
    probas = model.predict_proba(X_latest)[:, 1]

    result = pd.Series(probas, index=X_latest.index, name="proba_up")
    result.index.name = "date"
    return result
