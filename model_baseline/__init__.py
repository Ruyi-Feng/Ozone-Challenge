"""Model baseline package: factories + training stubs."""

from model_baseline.config import BaselineRuntimeConfig, load_config
from model_baseline.factories import build_dataset, build_model

__all__ = [
    "BaselineRuntimeConfig",
    "load_config",
    "build_dataset",
    "build_model",
]
