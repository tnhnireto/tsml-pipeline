"""
ML classification metrics for binary direction prediction.

All functions accept array-like inputs (pd.Series or np.ndarray).
They are thin wrappers around sklearn, added here so the rest of the
codebase imports from one place and so tests document the expected
behaviour explicitly.

Metrics
-------
accuracy           : fraction of correct predictions
precision          : TP / (TP + FP)  — of predicted "up" days, how many were up?
recall             : TP / (TP + FN)  — of actual "up" days, how many did we catch?
confusion_matrix   : 2×2 matrix [[TN, FP], [FN, TP]]
classification_report : per-class precision / recall / f1 as a dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report as _sklearn_report,
    confusion_matrix as _sklearn_cm,
    precision_score,
    recall_score,
)


# Type alias — both Series and ndarray are accepted.
ArrayLike = "pd.Series | np.ndarray"


def _to_array(x: ArrayLike) -> np.ndarray:
    """
    Convert input to a plain integer numpy array.

    sklearn's classification_report uses label values as dict keys.
    Converting float labels (0.0 / 1.0) to int (0 / 1) keeps the keys
    clean ("0" / "1" instead of "0.0" / "1.0").
    """
    arr = x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x)
    # Convert float arrays that contain only whole numbers to int.
    if np.issubdtype(arr.dtype, np.floating) and np.all(arr == arr.astype(int)):
        return arr.astype(int)
    return arr


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Fraction of predictions that match the true label.

        accuracy = count(y_true == y_pred) / n

    50 % is the floor for a balanced binary problem (coin-flip baseline).
    """
    return float(accuracy_score(_to_array(y_true), _to_array(y_pred)))


def precision(y_true: ArrayLike, y_pred: ArrayLike, pos_label: int = 1) -> float:
    """
    Precision for the positive class (default: class 1 = "up").

        precision = TP / (TP + FP)

    "Of the days we predicted 'up', what fraction actually went up?"
    High precision → few false buy signals.
    """
    return float(
        precision_score(
            _to_array(y_true),
            _to_array(y_pred),
            pos_label=pos_label,
            zero_division=0,
        )
    )


def recall(y_true: ArrayLike, y_pred: ArrayLike, pos_label: int = 1) -> float:
    """
    Recall for the positive class (default: class 1 = "up").

        recall = TP / (TP + FN)

    "Of all the days that actually went up, how many did we predict correctly?"
    High recall → few missed up days.
    """
    return float(
        recall_score(
            _to_array(y_true),
            _to_array(y_pred),
            pos_label=pos_label,
            zero_division=0,
        )
    )


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """
    2×2 confusion matrix.

        [[TN, FP],
         [FN, TP]]

    TN: predicted down, was down
    FP: predicted up,   was down  (false buy signal)
    FN: predicted down, was up    (missed up day)
    TP: predicted up,   was up
    """
    return _sklearn_cm(_to_array(y_true), _to_array(y_pred))


def classification_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """
    Per-class precision, recall, and F1 score as a nested dict.

    Keys: "0", "1", "macro avg", "weighted avg", "accuracy".

    Example
    -------
    >>> report = classification_report(y_true, y_pred)
    >>> report["1"]["precision"]
    0.54
    """
    return _sklearn_report(
        _to_array(y_true),
        _to_array(y_pred),
        output_dict=True,
        zero_division=0,
    )


def summary(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """
    Compute all ML metrics at once and return them as a flat dict.

    Returns
    -------
    dict with keys: accuracy, precision, recall.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
    }
