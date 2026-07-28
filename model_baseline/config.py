"""Config loading for model_baseline (YAML → dataclass)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "baseline"
    num_agents: int = 7
    num_features: int = 4
    hidden_dim: int = 64
    num_frames: Optional[int] = None


@dataclass
class DataPaths:
    data_train: str = "data/processed/data/events_data_train.csv"
    label_train: str = "data/processed/labels/events_labels_train.csv"
    data_val: str = "data/processed/data/events_data_val.csv"
    label_val: str = "data/processed/labels/events_labels_val.csv"
    future_train: Optional[str] = None
    future_val: Optional[str] = None


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1.0e-3
    max_epochs: int = 20
    seed: int = 42
    device: str = "cpu"


@dataclass
class EvalConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )


@dataclass
class BaselineRuntimeConfig:
    """Runtime knobs; typically loaded from configs/model_baseline.yaml."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataPaths = field(default_factory=DataPaths)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str | Path) -> BaselineRuntimeConfig:
    """Load YAML config into BaselineRuntimeConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    m = raw.get("model", {}) or {}
    d = raw.get("data", {}) or {}
    t = raw.get("train", {}) or {}
    e = raw.get("eval", {}) or {}

    return BaselineRuntimeConfig(
        model=ModelConfig(
            name=m.get("name", "baseline"),
            num_agents=int(m.get("num_agents", 7)),
            num_features=int(m.get("num_features", 4)),
            hidden_dim=int(m.get("hidden_dim", 64)),
            num_frames=m.get("num_frames"),
        ),
        data=DataPaths(
            data_train=d.get("data_train", DataPaths.data_train),
            label_train=d.get("label_train", DataPaths.label_train),
            data_val=d.get("data_val", DataPaths.data_val),
            label_val=d.get("label_val", DataPaths.label_val),
            future_train=d.get("future_train"),
            future_val=d.get("future_val"),
        ),
        train=TrainConfig(
            batch_size=int(t.get("batch_size", 32)),
            num_workers=int(t.get("num_workers", 0)),
            lr=float(t.get("lr", 1.0e-3)),
            max_epochs=int(t.get("max_epochs", 20)),
            seed=int(t.get("seed", 42)),
            device=str(t.get("device", "cpu")),
        ),
        eval=EvalConfig(
            metrics=list(e.get("metrics", ["accuracy", "precision", "recall", "f1"])),
        ),
    )
