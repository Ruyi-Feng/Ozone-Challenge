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
    # Transformer-only knobs (ignored by LSTM baseline)
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1


@dataclass
class DataPaths:
    data_train: str = "data/processed/data/events_data_train.csv"
    label_train: str = "data/processed/labels/events_labels_train.csv"
    data_val: str = "data/processed/data/events_data_val.csv"
    label_val: str = "data/processed/labels/events_labels_val.csv"
    future_train: Optional[str] = None
    future_val: Optional[str] = None


@dataclass
class MaskingConfig:
    """Masked surrogate training (prerequisite for Winter-Shapley attribution).

    enabled=False (default) keeps the ORIGINAL training behaviour untouched:
    no time/channel masks are passed to the model and padding frames stay
    visible, exactly as before this option existed.

    When enabled, each sample sees the full input with probability p_full
    (padding frames blocked via ~valid_mask), otherwise a random
    nested-permutation coalition prefix — the same distribution the Winter
    MC estimator walks at explanation time (model_baseline/masking.py).
    """

    enabled: bool = False
    p_full: float = 0.4
    seg_len_frames: tuple[int, ...] = (5, 10, 20)
    hierarchies: tuple[str, ...] = ("agent_major", "time_major")
    order_modes: tuple[str, ...] = ("uniform", "chrono", "reverse")
    require_valid: bool = True
    val_seed: int = 1234
    val_selection: str = "mixture"  # "mixture" | "full"


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1.0e-3
    max_epochs: int = 20
    seed: int = 42
    device: str = "cpu"
    masking: MaskingConfig = field(default_factory=MaskingConfig)


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


def _load_masking(mk: dict[str, Any]) -> MaskingConfig:
    return MaskingConfig(
        enabled=bool(mk.get("enabled", False)),
        p_full=float(mk.get("p_full", 0.4)),
        seg_len_frames=tuple(int(v) for v in mk.get("seg_len_frames", (5, 10, 20))),
        hierarchies=tuple(str(v) for v in mk.get("hierarchies", ("agent_major", "time_major"))),
        order_modes=tuple(str(v) for v in mk.get("order_modes", ("uniform", "chrono", "reverse"))),
        require_valid=bool(mk.get("require_valid", True)),
        val_seed=int(mk.get("val_seed", 1234)),
        val_selection=str(mk.get("val_selection", "mixture")),
    )


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
            n_layers=int(m.get("n_layers", 2)),
            n_heads=int(m.get("n_heads", 4)),
            dropout=float(m.get("dropout", 0.1)),
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
            masking=_load_masking(t.get("masking", {}) or {}),
        ),
        eval=EvalConfig(
            metrics=list(e.get("metrics", ["accuracy", "precision", "recall", "f1"])),
        ),
    )
