"""
Minimal model interface (Protocol).

Any class that implements fit / predict / predict_proba with these
signatures is a valid model in this pipeline.  No inheritance required.

Using typing.Protocol means the type checker will catch mismatches, but
we don't force every model to subclass an ABC.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """
    The interface every model in this pipeline must satisfy.

    fit(X, y)          → self
    predict(X)         → 1-D int array  (class labels)
    predict_proba(X)   → 2-D float array, shape (n, 2), columns [P(0), P(1)]
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...
