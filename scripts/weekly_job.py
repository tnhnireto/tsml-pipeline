"""
Weekly trading-test workflow runner.

Runs the full pipeline in the correct order, captures output from each step,
writes a time-stamped log to logs/jobs/, and stops immediately on the first
failure so broken state is never silently propagated to later steps.

Steps
-----
1. run_weekly_signal.py    — rank universe, generate signal CSV
2. run_etoro_demo.py       — build dry-run execution plan (never --execute)
3. analyze_signals.py      — post-process signal CSVs, forward-return analysis
4. scripts/analyze_portfolio.py — rebuild equity curve from order logs, print stats

Usage
-----
    python scripts/weekly_job.py

The script re-roots itself to the project root, so it can be invoked from
any working directory.

Log files
---------
    logs/jobs/YYYY-MM-DD_weekly_job.log   (appended on each run)

Exit codes
----------
    0  All steps succeeded.
    1  Environment validation failed, job already running, or a step failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

# Project root = parent of the directory that contains this script.
# Resolving here means the script always runs relative to the project root,
# regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
# Each entry has:
#   name  — label shown in the console summary and log
#   cmd   — argv list passed to subprocess.run (relative paths, cwd=PROJECT_ROOT)
#
# IMPORTANT: --execute is deliberately absent from the eToro step.
# This script must never submit real orders.
# ---------------------------------------------------------------------------

STEPS: list[dict] = [
    {
        "name": "Weekly Signal",
        "cmd":  [sys.executable, "run_weekly_signal.py"],
    },
    {
        "name": "eToro Demo (dry-run)",
        "cmd":  [sys.executable, "run_etoro_demo.py"],
    },
    {
        "name": "Signal Analysis",
        "cmd":  [sys.executable, "analyze_signals.py"],
    },
    {
        "name": "Portfolio Analysis",
        "cmd":  [sys.executable, "scripts/analyze_portfolio.py"],
    },
]


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(step: dict, log_fh: StringIO) -> bool:
    """
    Execute one workflow step.

    Runs ``step["cmd"]`` as a subprocess with ``cwd=PROJECT_ROOT``,
    captures stdout and stderr, and writes them to ``log_fh`` together with
    a UTC timestamp header.

    Parameters
    ----------
    step:
        Dict with keys ``"name"`` (str) and ``"cmd"`` (list[str]).
    log_fh:
        Open, writable file-like object for the log.

    Returns
    -------
    bool
        ``True`` on success (exit code 0), ``False`` on failure.
    """
    name = step["name"]
    cmd  = step["cmd"]
    ts   = _utc_now()

    _log(log_fh, "")
    _log(log_fh, "=" * 60)
    _log(log_fh, f"[{ts}]  STEP: {name}")
    _log(log_fh, f"CMD:  {' '.join(str(c) for c in cmd)}")
    _log(log_fh, "=" * 60)

    print(f"  [{ts}]  {name} ...", flush=True)

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            _log(log_fh, result.stdout.rstrip())
        if result.stderr:
            _log(log_fh, "[stderr]")
            _log(log_fh, result.stderr.rstrip())
        _log(log_fh, f"\n\u2714  {name}  SUCCESS")
        print(f"  [{_utc_now()}]    \u2714  OK")
        return True

    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            _log(log_fh, exc.stdout.rstrip())
        _log(log_fh, "[stderr]")
        _log(log_fh, (exc.stderr or "").rstrip())
        _log(log_fh, f"\n\u2718  {name}  FAILED  (exit {exc.returncode})")
        print(f"  [{_utc_now()}]    \u2718  FAILED  (exit {exc.returncode})")
        # Echo the last few stderr lines to the console for quick diagnosis.
        if exc.stderr:
            for line in exc.stderr.strip().splitlines()[-5:]:
                print(f"      {line}")
        return False


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

# Directories that must exist before the workflow can run.
REQUIRED_DIRS: list[str] = ["signals", "logs"]


def _validate_environment() -> list[str]:
    """
    Check that the environment is ready to run the weekly workflow.

    Returns a list of human-readable error strings.  An empty list means
    all checks passed.

    Checks
    ------
    1. ``ETORO_API_KEY`` is set and non-empty in the environment.
    2. A virtual environment is active (``sys.prefix != sys.base_prefix``).
    3. Each directory in ``REQUIRED_DIRS`` exists under ``PROJECT_ROOT``.
    """
    errors: list[str] = []

    # 1. API key
    if not os.environ.get("ETORO_API_KEY", "").strip():
        errors.append(
            "ETORO_API_KEY is not set. "
            "Export it before running: export ETORO_API_KEY=<your-key>"
        )

    # 2. Virtual environment
    if sys.prefix == sys.base_prefix:
        errors.append(
            "No virtual environment is active "
            f"(sys.prefix: {sys.prefix}). "
            "Activate .venv first: source .venv/bin/activate"
        )

    # 3. Required directories
    for name in REQUIRED_DIRS:
        path = PROJECT_ROOT / name
        if not path.is_dir():
            errors.append(f"Required directory missing: {path}")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Run the weekly workflow.

    Returns 0 on full success, 1 if any step failed, 1 if already running.
    """
    # ── Environment validation ───────────────────────────────────────────────
    errors = _validate_environment()
    if errors:
        print()
        print("Environment check failed — cannot start weekly job.")
        for err in errors:
            print(f"  ERROR: {err}")
        print()
        return 1

    logs_dir = PROJECT_ROOT / "logs" / "jobs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Lock ────────────────────────────────────────────────────────────────
    lock_file = logs_dir / ".weekly_job.lock"
    if lock_file.exists():
        print(f"Job already running (lock: {lock_file})")
        return 1

    lock_file.touch()
    try:
        return _run_workflow(logs_dir)
    finally:
        lock_file.unlink(missing_ok=True)


def _run_workflow(logs_dir: Path) -> int:
    """Execute all steps and return the exit code. Called only when lock is held."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = logs_dir / f"{date_str}_weekly_job.log"

    started_at = _utc_now()

    print()
    print("=" * 60)
    print(f"  Weekly job started  [{started_at}]")
    print(f"  Root:  {PROJECT_ROOT}")
    print(f"  Log:   {log_path}")
    print(f"  Steps: {len(STEPS)}")
    print("=" * 60)

    with log_path.open("a", encoding="utf-8") as log_fh:
        _log(log_fh, "=" * 60)
        _log(log_fh, f"WEEKLY JOB STARTED  [{started_at}]")
        _log(log_fh, f"Python:  {sys.executable}")
        _log(log_fh, f"Root:    {PROJECT_ROOT}")
        _log(log_fh, "=" * 60)

        results: list[tuple[str, bool]] = []

        for step in STEPS:
            ok = run_step(step, log_fh)
            results.append((step["name"], ok))
            if not ok:
                _log(log_fh, "\nWorkflow stopped — see failure above.\n")
                break

        finished_at = _utc_now()
        all_ok      = len(results) == len(STEPS) and all(ok for _, ok in results)
        final       = "SUCCESS" if all_ok else "FAILED"

        # Console summary
        skipped = [
            s["name"] for s in STEPS
            if s["name"] not in {n for n, _ in results}
        ]
        print()
        print("─" * 44)
        print("  Summary")
        print("─" * 44)
        for name, ok in results:
            tag = "\u2714 OK     " if ok else "\u2718 FAILED "
            print(f"  {tag}  {name}")
        for name in skipped:
            print(f"  \u25cb SKIPPED  {name}")
        print("─" * 44)
        print(f"  {final}  [{finished_at}]")
        print(f"  Log: {log_path}")
        print()

        # Log summary
        _log(log_fh, "")
        _log(log_fh, "=" * 60)
        _log(log_fh, f"WEEKLY JOB {final}  [{finished_at}]")
        _log(log_fh, "=" * 60)

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(fh, text: str) -> None:
    fh.write(text + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
