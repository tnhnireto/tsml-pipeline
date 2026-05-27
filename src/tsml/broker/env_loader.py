"""
Load eToro credentials from local env files (optional).

Shell environment variables always take precedence.  Files are never committed
(see ``.gitignore`` for ``.env``).
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: src/tsml/broker/env_loader.py -> parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ENV_FILES = (
    _PROJECT_ROOT / ".env",
    _PROJECT_ROOT / "data" / "etoro.env",
)


def load_etoro_env_files() -> list[Path]:
    """
    Populate ``os.environ`` from ``.env`` and ``data/etoro.env`` if present.

    Returns
    -------
    list[Path]
        Files that were found and parsed (whether or not they set new vars).
    """
    loaded: list[Path] = []
    for path in _ENV_FILES:
        if not path.is_file():
            continue
        loaded.append(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return loaded
