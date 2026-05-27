"""Feature importance formatting for model debug output."""

from __future__ import annotations

from typing import Any


def format_feature_importance(model: Any, *, top_n: int = 15) -> str:
    """
    Format ``feature_importance()`` output from a fitted model.

    Returns a multi-line string suitable for stdout.  If the model does not
    expose feature importance, returns a short placeholder message.
    """
    importance_fn = getattr(model, "feature_importance", None)
    if importance_fn is None:
        return "  Feature importance not available for this model."

    df = importance_fn()
    if df is None or df.empty:
        return "  Feature importance not available (model not fitted or no coefs)."

    lines = ["  Top features (mean |coef| across calibration folds):"]
    for _, row in df.head(top_n).iterrows():
        lines.append(f"    {row['feature']:<28} {row['importance']:.4f}")
    return "\n".join(lines)
