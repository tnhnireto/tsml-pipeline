"""Guard-rail tests for live-script configuration.

run_weekly_signal.py and demo.py execute their pipelines at import time,
so these tests parse the source with ``ast`` instead of importing.  They
pin the Phase-1 live-signal configuration:

* 5-day target with gap >= horizon (no label leakage at fold boundaries),
* extended_v2 stationary feature set,
* fresh-fit scoring with 5-day smoothing,
* 2017 start date for longer warmup.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module_constants(path: Path) -> dict[str, object]:
    """Return module-level constant assignments (literals only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except ValueError:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = literal
    return out


@pytest.fixture(scope="module")
def weekly_cfg() -> dict[str, object]:
    return _module_constants(ROOT / "run_weekly_signal.py")


@pytest.fixture(scope="module")
def demo_source() -> str:
    return (ROOT / "demo.py").read_text(encoding="utf-8")


class TestWeeklySignalConfig:
    def test_target_is_direction_5d(self, weekly_cfg):
        assert weekly_cfg["TARGET"] == "direction_5d"

    def test_gap_covers_label_horizon(self, weekly_cfg):
        assert weekly_cfg["GAP"] >= 5, (
            "direction_5d labels look 5 days ahead; gap must be >= 5 "
            "to prevent train/test label leakage"
        )

    def test_feature_set_is_extended_v2(self, weekly_cfg):
        assert weekly_cfg["FEATURE_SET"] == "extended_v2"

    def test_scoring_is_fresh_fit(self, weekly_cfg):
        assert weekly_cfg["SCORING"] == "fresh_fit"

    def test_smoothing_window_default(self, weekly_cfg):
        assert weekly_cfg["SMOOTHING_WINDOW"] == 5

    def test_start_date_supports_warmup(self, weekly_cfg):
        assert weekly_cfg["START_DATE"] == "2017-01-01"

    def test_thresholds_unchanged(self, weekly_cfg):
        assert weekly_cfg["MIN_SCORE"] == 0.55
        assert weekly_cfg["MIN_SCORE_DOWNTREND"] == 0.62

    def test_valid_config_values(self, weekly_cfg):
        """Config strings must be values the library accepts."""
        from tsml.features.pipeline import _VALID_FEATURE_SETS, _VALID_TARGETS
        from tsml.portfolio.ranker import _VALID_SCORING

        assert weekly_cfg["TARGET"] in _VALID_TARGETS
        assert weekly_cfg["FEATURE_SET"] in _VALID_FEATURE_SETS
        assert weekly_cfg["SCORING"] in _VALID_SCORING
        assert int(weekly_cfg["SMOOTHING_WINDOW"]) >= 1


class TestDemoGap:
    def test_make_splitter_accepts_gap(self, demo_source):
        tree = ast.parse(demo_source)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "make_splitter"
        )
        assert "gap" in [a.arg for a in fn.args.args], (
            "make_splitter must take a per-config gap so multi-day targets "
            "get an embargo covering their label horizon"
        )

    def test_call_site_scales_gap_with_holding_period(self, demo_source):
        tree = ast.parse(demo_source)
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "make_splitter"
        ]
        assert calls, "expected a make_splitter call in demo.py"
        for call in calls:
            gap_kw = [k for k in call.keywords if k.arg == "gap"]
            assert gap_kw, "make_splitter call must pass an explicit gap"
            src = ast.unparse(gap_kw[0].value)
            assert "holding_period" in src, (
                f"gap must scale with the config's holding period, got: {src}"
            )
