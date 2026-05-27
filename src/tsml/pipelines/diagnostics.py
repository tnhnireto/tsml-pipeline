"""
Walk-forward diagnostics — probabilities plus per-fold feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tsml.features.pipeline import make_dataset
from tsml.validation.splitters import WalkForwardSplit


@dataclass
class FoldImportance:
    """Feature importance snapshot for one walk-forward fold."""

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    importance: pd.DataFrame


@dataclass
class WalkForwardDiagnostics:
    """Output of :func:`run_walk_forward_diagnostics`."""

    probas: pd.Series
    fold_importances: list[FoldImportance] = field(default_factory=list)


def run_walk_forward_diagnostics(
    df: pd.DataFrame,
    model: Any,
    splitter: WalkForwardSplit,
    target: str = "direction",
    *,
    feature_set: str = "legacy",
    benchmarks: dict[str, pd.Series] | None = None,
) -> WalkForwardDiagnostics:
    """
    Walk-forward OOS probabilities with per-fold feature importance snapshots.

    Identical leakage guarantees to :func:`run_walk_forward_proba`, but
    additionally records ``feature_importance()`` after each fold fit when
    the model supports it.
    """
    X, y = make_dataset(
        df,
        target=target,
        feature_set=feature_set,
        benchmarks=benchmarks,
    )

    probas: dict[pd.Timestamp, float] = {}
    fold_importances: list[FoldImportance] = []

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        model.fit(X_train, y_train)
        fold_probas = model.predict_proba(X_test)[:, 1]

        for date, p in zip(X_test.index, fold_probas):
            probas[date] = p

        importance_fn = getattr(model, "feature_importance", None)
        if importance_fn is not None:
            imp = importance_fn()
            if imp is not None and not imp.empty:
                fold_importances.append(
                    FoldImportance(
                        fold=fold_num,
                        train_end=X_train.index[-1],
                        test_start=X_test.index[0],
                        test_end=X_test.index[-1],
                        importance=imp.copy(),
                    )
                )

    if not probas:
        raise RuntimeError("No probabilities were produced. Check splitter parameters.")

    result = pd.Series(probas, name="proba_up")
    result.index.name = "date"
    return WalkForwardDiagnostics(probas=result, fold_importances=fold_importances)


def aggregate_fold_importance(
    fold_importances: list[FoldImportance],
) -> pd.DataFrame:
    """
    Summarise feature importance across walk-forward folds.

    Returns mean importance, standard deviation, and a stability score
    (1 − coefficient of variation; higher = more stable).
    """
    if not fold_importances:
        return pd.DataFrame(
            columns=["feature", "mean_importance", "std_importance", "stability"]
        )

    frames = [fi.importance.set_index("feature")["importance"] for fi in fold_importances]
    combined = pd.concat(frames, axis=1)
    mean_imp = combined.mean(axis=1)
    std_imp = combined.std(axis=1).fillna(0.0)
    stability = 1.0 - (std_imp / mean_imp.replace(0, float("nan"))).fillna(0.0)

    return (
        pd.DataFrame(
            {
                "feature": mean_imp.index,
                "mean_importance": mean_imp.values,
                "std_importance": std_imp.values,
                "stability": stability.values,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
