"""
Tests for scripts/weekly_job.py.

The module is imported via importlib so it can live outside the installed
package tree.  subprocess.run is patched throughout — no child processes
are spawned and no network calls are made.
"""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load scripts/weekly_job.py as a module
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def wj():
    """Return the weekly_job module object."""
    spec = importlib.util.spec_from_file_location(
        "weekly_job",
        _SCRIPTS_DIR / "weekly_job.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper: build a mock subprocess.run that succeeds or fails by step index
# ---------------------------------------------------------------------------

def _mock_run_factory(fail_at: int | None = None):
    """
    Return a mock subprocess.run callable.

    Parameters
    ----------
    fail_at:
        0-based index of the step that should fail.
        ``None`` means all steps succeed.
    """
    call_log: list[list] = []

    def _run(cmd, **kwargs):
        call_log.append(list(cmd))
        idx = len(call_log) - 1
        if fail_at is not None and idx == fail_at:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                output="",
                stderr="simulated failure",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    _run.calls = call_log   # expose for assertions
    return _run


# ---------------------------------------------------------------------------
# Shared fixture: valid environment (all checks pass)
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_env(wj, monkeypatch, tmp_path):
    """
    Patch PROJECT_ROOT, set ETORO_API_KEY, activate a fake venv, and create
    required directories so that _validate_environment() returns no errors.
    """
    monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ETORO_API_KEY", "test-key-abc")
    # Simulate an active venv by making sys.prefix differ from sys.base_prefix
    monkeypatch.setattr(wj.sys, "prefix", str(tmp_path / ".venv"))
    for name in wj.REQUIRED_DIRS:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return tmp_path


# ===========================================================================
# 1. STEPS definition
# ===========================================================================

class TestStepsDefinition:
    def test_four_steps(self, wj):
        assert len(wj.STEPS) == 4

    def test_no_execute_flag_in_any_step(self, wj):
        """--execute must never appear in any command."""
        for step in wj.STEPS:
            assert "--execute" not in step["cmd"], (
                f"--execute found in step '{step['name']}': {step['cmd']}"
            )

    def test_no_execute_flag_as_substring(self, wj):
        """Guard against variations like '--execute=true'."""
        for step in wj.STEPS:
            for token in step["cmd"]:
                assert "execute" not in token.lower() or token == sys.executable, (
                    f"'execute' substring found in token '{token}' for step '{step['name']}'"
                )

    def test_sys_executable_is_first_token(self, wj):
        """Every step must use sys.executable, not a hard-coded 'python'."""
        for step in wj.STEPS:
            assert step["cmd"][0] == sys.executable, (
                f"Step '{step['name']}' uses '{step['cmd'][0]}' instead of sys.executable"
            )

    def test_step_order_weekly_signal_first(self, wj):
        assert "signal" in wj.STEPS[0]["name"].lower() or "weekly" in wj.STEPS[0]["name"].lower()

    def test_step_order_etoro_second(self, wj):
        assert "etoro" in wj.STEPS[1]["name"].lower() or "demo" in wj.STEPS[1]["name"].lower()

    def test_step_order_signal_analysis_third(self, wj):
        name = wj.STEPS[2]["name"].lower()
        assert "signal" in name or "analysis" in name or "analyse" in name

    def test_step_order_portfolio_last(self, wj):
        assert "portfolio" in wj.STEPS[3]["name"].lower()

    def test_etoro_script_is_run_etoro_demo(self, wj):
        """The eToro step must use run_etoro_demo.py, not some other script."""
        etoro_step = wj.STEPS[1]
        script = etoro_step["cmd"][1]
        assert "etoro" in script.lower()
        assert "demo" in script.lower()

    def test_weekly_signal_script_name(self, wj):
        script = wj.STEPS[0]["cmd"][1]
        assert "weekly" in script.lower() and "signal" in script.lower()


# ===========================================================================
# 2. run_step
# ===========================================================================

class TestRunStep:
    def test_returns_true_on_success(self, wj, monkeypatch):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        buf = io.StringIO()
        ok  = wj.run_step(wj.STEPS[0], buf)
        assert ok is True

    def test_returns_false_on_failure(self, wj, monkeypatch):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory(fail_at=0))
        buf = io.StringIO()
        ok  = wj.run_step(wj.STEPS[0], buf)
        assert ok is False

    def test_success_writes_step_name_to_log(self, wj, monkeypatch):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        buf = io.StringIO()
        wj.run_step(wj.STEPS[0], buf)
        assert wj.STEPS[0]["name"] in buf.getvalue()

    def test_failure_writes_failed_to_log(self, wj, monkeypatch):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory(fail_at=0))
        buf = io.StringIO()
        wj.run_step(wj.STEPS[0], buf)
        assert "FAILED" in buf.getvalue()

    def test_stdout_captured_in_log(self, wj, monkeypatch):
        def _run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="hello stdout\n", stderr="")
        monkeypatch.setattr(wj.subprocess, "run", _run)
        buf = io.StringIO()
        wj.run_step(wj.STEPS[0], buf)
        assert "hello stdout" in buf.getvalue()

    def test_stderr_captured_in_log_on_failure(self, wj, monkeypatch):
        def _run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")
        monkeypatch.setattr(wj.subprocess, "run", _run)
        buf = io.StringIO()
        wj.run_step(wj.STEPS[0], buf)
        assert "boom" in buf.getvalue()


# ===========================================================================
# 3. Environment validation
# ===========================================================================

class TestValidateEnvironment:
    def test_all_valid_returns_empty_list(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ETORO_API_KEY", "test-key")
        monkeypatch.setattr(wj.sys, "prefix", str(tmp_path / ".venv"))
        for name in wj.REQUIRED_DIRS:
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        assert wj._validate_environment() == []

    def test_missing_api_key_returns_error(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setattr(wj.sys, "prefix", str(tmp_path / ".venv"))
        for name in wj.REQUIRED_DIRS:
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        errors = wj._validate_environment()
        assert any("ETORO_API_KEY" in e for e in errors)

    def test_whitespace_api_key_returns_error(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ETORO_API_KEY", "   ")
        monkeypatch.setattr(wj.sys, "prefix", str(tmp_path / ".venv"))
        for name in wj.REQUIRED_DIRS:
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        errors = wj._validate_environment()
        assert any("ETORO_API_KEY" in e for e in errors)

    def test_no_venv_returns_error(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ETORO_API_KEY", "test-key")
        # Make prefix == base_prefix to simulate no active venv
        monkeypatch.setattr(wj.sys, "prefix", sys.base_prefix)
        for name in wj.REQUIRED_DIRS:
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        errors = wj._validate_environment()
        assert any("virtual environment" in e.lower() for e in errors)

    def test_missing_signals_dir_returns_error(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ETORO_API_KEY", "test-key")
        monkeypatch.setattr(wj.sys, "prefix", str(tmp_path / ".venv"))
        # Create only logs/, leave signals/ absent
        (tmp_path / "logs").mkdir()
        errors = wj._validate_environment()
        assert any("signals" in e for e in errors)

    def test_missing_logs_dir_returns_error(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ETORO_API_KEY", "test-key")
        monkeypatch.setattr(wj.sys, "prefix", str(tmp_path / ".venv"))
        # Create only signals/, leave logs/ absent
        (tmp_path / "signals").mkdir()
        errors = wj._validate_environment()
        assert any("logs" in e for e in errors)

    def test_multiple_failures_reported(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setattr(wj.sys, "prefix", sys.base_prefix)
        # No required dirs created
        errors = wj._validate_environment()
        assert len(errors) >= 3   # api key + venv + 2 dirs

    def test_main_exits_with_one_on_validation_failure(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setattr(wj.sys, "prefix", sys.base_prefix)
        rc = wj.main()
        assert rc == 1

    def test_main_prints_error_on_validation_failure(self, wj, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setattr(wj.sys, "prefix", sys.base_prefix)
        wj.main()
        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "ETORO_API_KEY" in out

    def test_validation_failure_runs_no_steps(self, wj, monkeypatch, tmp_path):
        monkeypatch.setattr(wj, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.setattr(wj.sys, "prefix", sys.base_prefix)
        mock = _mock_run_factory()
        monkeypatch.setattr(wj.subprocess, "run", mock)
        wj.main()
        assert len(mock.calls) == 0


# ===========================================================================
# 4. main() — orchestration
# ===========================================================================

class TestMain:
    def test_returns_zero_when_all_succeed(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        assert wj.main() == 0

    def test_returns_one_when_first_step_fails(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory(fail_at=0))
        assert wj.main() == 1

    def test_returns_one_when_last_step_fails(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory(fail_at=3))
        assert wj.main() == 1

    def test_failure_stops_later_steps(self, wj, monkeypatch, valid_env):
        """When step 1 fails, steps 2-4 must not be executed."""
        mock = _mock_run_factory(fail_at=0)
        monkeypatch.setattr(wj.subprocess, "run", mock)
        wj.main()
        assert len(mock.calls) == 1

    def test_second_step_failure_stops_at_two(self, wj, monkeypatch, valid_env):
        mock = _mock_run_factory(fail_at=1)
        monkeypatch.setattr(wj.subprocess, "run", mock)
        wj.main()
        assert len(mock.calls) == 2

    def test_all_steps_run_on_success(self, wj, monkeypatch, valid_env):
        mock = _mock_run_factory()
        monkeypatch.setattr(wj.subprocess, "run", mock)
        wj.main()
        assert len(mock.calls) == len(wj.STEPS)


# ===========================================================================
# 5. Log file creation
# ===========================================================================

class TestLogging:
    def test_logs_jobs_dir_created(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        assert (valid_env / "logs" / "jobs").is_dir()

    def test_log_file_created(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        log_files = list((valid_env / "logs" / "jobs").glob("*.log"))
        assert len(log_files) == 1

    def test_log_filename_contains_date(self, wj, monkeypatch, valid_env):
        import re
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        log_file = next((valid_env / "logs" / "jobs").glob("*.log"))
        assert re.search(r"\d{4}-\d{2}-\d{2}", log_file.stem)

    def test_log_file_contains_step_names(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        log_file = next((valid_env / "logs" / "jobs").glob("*.log"))
        content  = log_file.read_text(encoding="utf-8")
        for step in wj.STEPS:
            assert step["name"] in content

    def test_log_file_appended_on_second_run(self, wj, monkeypatch, valid_env):
        """Running twice appends to the same daily log, not overwriting it."""
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        wj.main()
        log_file = next((valid_env / "logs" / "jobs").glob("*.log"))
        content  = log_file.read_text(encoding="utf-8")
        assert content.count("WEEKLY JOB STARTED") == 2

    def test_log_contains_success_on_full_run(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        log_file = next((valid_env / "logs" / "jobs").glob("*.log"))
        assert "SUCCESS" in log_file.read_text(encoding="utf-8")

    def test_log_contains_failed_on_step_failure(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory(fail_at=0))
        wj.main()
        log_file = next((valid_env / "logs" / "jobs").glob("*.log"))
        assert "FAILED" in log_file.read_text(encoding="utf-8")


# ===========================================================================
# 6. File lock
# ===========================================================================

class TestFileLock:
    def _lock(self, root: Path) -> Path:
        return root / "logs" / "jobs" / ".weekly_job.lock"

    def test_lock_removed_after_success(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory())
        wj.main()
        assert not self._lock(valid_env).exists()

    def test_lock_removed_after_step_failure(self, wj, monkeypatch, valid_env):
        monkeypatch.setattr(wj.subprocess, "run", _mock_run_factory(fail_at=0))
        wj.main()
        assert not self._lock(valid_env).exists()

    def test_existing_lock_returns_nonzero(self, wj, monkeypatch, valid_env):
        jobs_dir = valid_env / "logs" / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock(valid_env).touch()
        assert wj.main() != 0

    def test_existing_lock_runs_no_steps(self, wj, monkeypatch, valid_env):
        jobs_dir = valid_env / "logs" / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock(valid_env).touch()
        mock = _mock_run_factory()
        monkeypatch.setattr(wj.subprocess, "run", mock)
        wj.main()
        assert len(mock.calls) == 0

    def test_existing_lock_prints_already_running(self, wj, monkeypatch, valid_env, capsys):
        jobs_dir = valid_env / "logs" / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock(valid_env).touch()
        wj.main()
        assert "already running" in capsys.readouterr().out.lower()
