"""
Project-wide configuration loaded from a YAML file.

Usage:
    config = Config.from_yaml("configs/example.yaml")
    print(config.symbol)
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    symbol: str = "SPY"
    start: str = "2010-01-01"
    end: str = "2023-12-31"
    # Boundaries that define the three non-overlapping splits.
    train_end: str = "2020-12-31"
    val_end: str = "2021-12-31"


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    raw_data_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls(**(raw or {}))
