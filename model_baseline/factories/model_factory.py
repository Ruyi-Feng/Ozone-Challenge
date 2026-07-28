"""Model factory: model_name → network.

Both data and model factories share the same registry key (`model.name`).
"""

from __future__ import annotations

from typing import Any

from model_baseline.config import ModelConfig
from model_baseline.models.baseline import BaselineConflictModel

# Registry: model_name → Model class
_MODEL_REGISTRY: dict[str, type] = {
    "baseline": BaselineConflictModel,
}


def register_model(name: str, cls: type) -> None:
    """Register / override a model class for a model_name."""
    _MODEL_REGISTRY[name] = cls


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


def build_model(model_name: str, cfg: ModelConfig | None = None, **kwargs: Any):
    """
    Build the network that matches `model_name`.

    Parameters
    ----------
    model_name : str
        Same key as data factory (e.g. \"baseline\").
    cfg : ModelConfig, optional
        Hyperparams (num_agents, num_features, hidden_dim, ...).
    """
    if model_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model for model_name={model_name!r}. "
            f"Available: {available_models()}"
        )
    cls = _MODEL_REGISTRY[model_name]
    cfg = cfg or ModelConfig(name=model_name)
    return cls(cfg=cfg, **kwargs)
