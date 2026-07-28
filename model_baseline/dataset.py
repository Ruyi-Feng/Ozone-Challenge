"""Backward-compatible entry.

Prefer:
  from model_baseline.factories import build_dataset
"""

from model_baseline.datasets.baseline import BaselineConflictDataset
from model_baseline.factories.data_factory import build_dataset

__all__ = ["BaselineConflictDataset", "build_dataset"]
