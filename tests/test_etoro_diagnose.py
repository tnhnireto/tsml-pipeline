"""Tests for eToro env loading and diagnostics."""

from __future__ import annotations

import os

import pytest

from tsml.broker.diagnose import diagnose, format_diagnose_report
from tsml.broker.env_loader import load_etoro_env_files


class TestEnvLoader:
    def test_loads_from_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "ETORO_API_KEY=from-file\nETORO_USER_KEY=user-from-file\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.delenv("ETORO_USER_KEY", raising=False)

        # Patch project root lookup to tmp_path
        import tsml.broker.env_loader as el
        monkeypatch.setattr(el, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(el, "_ENV_FILES", (tmp_path / ".env",))

        loaded = load_etoro_env_files()
        assert loaded == [tmp_path / ".env"]
        assert os.environ["ETORO_API_KEY"] == "from-file"
        assert os.environ["ETORO_USER_KEY"] == "user-from-file"

    def test_shell_env_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("ETORO_API_KEY=file-key\n", encoding="utf-8")
        monkeypatch.setenv("ETORO_API_KEY", "shell-key")

        import tsml.broker.env_loader as el
        monkeypatch.setattr(el, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(el, "_ENV_FILES", (tmp_path / ".env",))

        load_etoro_env_files()
        assert os.environ["ETORO_API_KEY"] == "shell-key"


class TestDiagnose:
    def test_detects_missing_env(self, monkeypatch):
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.delenv("ETORO_USER_KEY", raising=False)
        result = diagnose()
        assert result.env_ok is False
        report = format_diagnose_report(result)
        report.encode("ascii")

    def test_detects_identical_keys(self, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "same-key-value")
        monkeypatch.setenv("ETORO_USER_KEY", "same-key-value")
        result = diagnose()
        assert any("identical" in h for h in result.key_hints)

    def test_report_is_ascii_safe(self, monkeypatch):
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        format_diagnose_report(diagnose()).encode("ascii")
