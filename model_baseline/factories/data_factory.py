"""Data factory: model_name → Dataset form.

Both data and model factories share the same registry key (`model.name`).
"""

from __future__ import annotations

from typing import Any

from model_baseline.datasets.baseline import BaselineConflictDataset
from model_baseline.datasets.binary_baseline import BinaryBaselineDataset

# Registry: model_name → Dataset class (or builder)
_DATASET_REGISTRY: dict[str, type] = {
    "baseline": BaselineConflictDataset,
    "baseline_binary": BinaryBaselineDataset,
}


def register_dataset(name: str, cls: type) -> None:
    """Register / override a dataset class for a model_name."""
    _DATASET_REGISTRY[name] = cls


def available_datasets() -> list[str]:
    return sorted(_DATASET_REGISTRY.keys())


def build_dataset(
    model_name: str,
    *,
    data_path: str,
    label_path: str,
    split: str = "train",
    future_path: str | None = None,
    **kwargs: Any,
):
    """
    Build the Dataset that matches `model_name`.

    Parameters
    ----------
    model_name : str
        Same key as model factory (e.g. \"baseline\").
    data_path, label_path : str
        Processed CSV paths (events_data_*.csv / events_labels_*.csv).
    split : str
        Logical split tag (\"train\" | \"val\" | \"test\"); dataset may ignore.
    future_path : optional
        Future traj CSV if a dataset form needs it.
    """
    if model_name not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset for model_name={model_name!r}. "
            f"Available: {available_datasets()}"
        )
    cls = _DATASET_REGISTRY[model_name]
    return cls(
        data_path=data_path,
        label_path=label_path,
        split=split,
        future_path=future_path,
        **kwargs,
    )
