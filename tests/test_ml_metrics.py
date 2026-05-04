"""
Tests for ML classification metrics.

All tests use inputs where the correct answer follows directly from the
definition of the metric, so expected values are computed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsml.metrics.ml import (
    accuracy,
    classification_report,
    confusion_matrix,
    precision,
    recall,
    summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s(values: list) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_all_correct(self):
        assert accuracy(_s([0, 1, 0, 1]), _s([0, 1, 0, 1])) == pytest.approx(1.0)

    def test_all_wrong(self):
        assert accuracy(_s([0, 0, 1, 1]), _s([1, 1, 0, 0])) == pytest.approx(0.0)

    def test_half_correct(self):
        assert accuracy(_s([0, 0, 1, 1]), _s([0, 1, 0, 1])) == pytest.approx(0.5)

    def test_three_out_of_four(self):
        assert accuracy(_s([1, 1, 0, 1]), _s([1, 0, 0, 1])) == pytest.approx(0.75)

    def test_accepts_numpy_arrays(self):
        y_t = np.array([0, 1, 1, 0])
        y_p = np.array([0, 1, 0, 0])
        assert accuracy(y_t, y_p) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------

class TestPrecision:
    def test_perfect_precision(self):
        # All predicted 1s are correct.
        y_t = _s([1, 1, 0, 0])
        y_p = _s([1, 1, 0, 0])
        assert precision(y_t, y_p) == pytest.approx(1.0)

    def test_zero_precision(self):
        # All predicted 1s are wrong.
        y_t = _s([0, 0, 0, 0])
        y_p = _s([1, 1, 1, 1])
        assert precision(y_t, y_p) == pytest.approx(0.0)

    def test_half_precision(self):
        # Predict 1 four times; 2 correct.  precision = 2/4 = 0.5
        y_t = _s([1, 0, 1, 0])
        y_p = _s([1, 1, 1, 1])
        assert precision(y_t, y_p) == pytest.approx(0.5)

    def test_no_positive_predictions_returns_zero(self):
        # Model never predicts 1 → precision is 0 (zero_division handled).
        y_t = _s([1, 1, 0, 0])
        y_p = _s([0, 0, 0, 0])
        assert precision(y_t, y_p) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_perfect_recall(self):
        # All actual 1s are captured.
        y_t = _s([1, 1, 0, 0])
        y_p = _s([1, 1, 1, 1])  # also catches 0s, but recall is for 1s
        assert recall(y_t, y_p) == pytest.approx(1.0)

    def test_zero_recall(self):
        # No actual 1s are captured.
        y_t = _s([1, 1, 1, 1])
        y_p = _s([0, 0, 0, 0])
        assert recall(y_t, y_p) == pytest.approx(0.0)

    def test_half_recall(self):
        # 4 actual 1s, 2 caught.  recall = 2/4 = 0.5
        y_t = _s([1, 1, 1, 1])
        y_p = _s([1, 0, 1, 0])
        assert recall(y_t, y_p) == pytest.approx(0.5)

    def test_no_positive_class_returns_zero(self):
        # No actual 1s → recall is 0 (zero_division handled).
        y_t = _s([0, 0, 0, 0])
        y_p = _s([0, 0, 0, 0])
        assert recall(y_t, y_p) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_shape(self):
        cm = confusion_matrix(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        assert cm.shape == (2, 2)

    def test_known_values(self):
        # y_true=[0, 1, 0, 1], y_pred=[0, 1, 1, 0]
        # TN=1 (0→0), FP=1 (0→1), FN=1 (1→0), TP=1 (1→1)
        # Expected matrix: [[TN, FP], [FN, TP]] = [[1, 1], [1, 1]]
        cm = confusion_matrix(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP

    def test_perfect_predictions(self):
        y = _s([0, 1, 0, 1, 1])
        cm = confusion_matrix(y, y)
        assert cm[0, 1] == 0  # no FP
        assert cm[1, 0] == 0  # no FN

    def test_all_wrong(self):
        y_t = _s([0, 0, 1, 1])
        y_p = _s([1, 1, 0, 0])
        cm = confusion_matrix(y_t, y_p)
        assert cm[0, 0] == 0  # no TN
        assert cm[1, 1] == 0  # no TP


# ---------------------------------------------------------------------------
# classification_report
# ---------------------------------------------------------------------------

class TestClassificationReport:
    def test_returns_dict(self):
        report = classification_report(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        assert isinstance(report, dict)

    def test_has_expected_keys(self):
        report = classification_report(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        # Float labels are converted to int internally, so keys are "0" / "1".
        assert "0" in report
        assert "1" in report
        assert "accuracy" in report

    def test_per_class_has_precision_recall_f1(self):
        report = classification_report(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        for cls in ["0", "1"]:  # int keys after _to_array conversion
            assert "precision" in report[cls]
            assert "recall" in report[cls]
            assert "f1-score" in report[cls]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_returns_dict(self):
        result = summary(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        assert isinstance(result, dict)

    def test_has_correct_keys(self):
        result = summary(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        assert set(result.keys()) == {"accuracy", "precision", "recall"}

    def test_all_values_are_floats(self):
        result = summary(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_values_in_0_1(self):
        result = summary(_s([0, 1, 0, 1]), _s([0, 1, 1, 0]))
        for v in result.values():
            assert 0.0 <= v <= 1.0
