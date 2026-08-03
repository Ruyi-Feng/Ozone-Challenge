"""Baseline dataset — multi-agent history → conflict labels.

Loads pre-computed tensor cache (memmap .npy files) for instant random access.

Sample contract (consumed by BaselineConflictModel):
  __getitem__ → dict with:
    - "x":           FloatTensor [A, T, F]  multi-agent history
    - "agent_mask":  BoolTensor  [A]        True if slot has a vehicle
    - "is_conflict": FloatTensor  scalar    0.0 / 1.0
    - "target_idx":  LongTensor   scalar    neighbor slot index in {0..5}, or -1
    - "event_id":    int                     for debugging

Agent axis A is ordered as:
  [ego, front, rear, left_front, left_rear, right_front, right_rear]
Features F: [dx, dy, heading_rel, speed]  relative to ego @ t₀.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore


class BaselineConflictDataset(Dataset):
    """PyTorch Dataset backed by memory-mapped tensor cache.

    Each sample is one event: 8 s of history for ego + up to 6 neighbours,
    represented as a fixed-size [A=7, T, F=4] tensor with an agent_mask.

    Parameters
    ----------
    data_path : str
        Path prefix for ``{prefix}_x.npy``, ``{prefix}_mask.npy``, etc.
    label_path : str
        Unused (labels are embedded in the ``_y.npy`` cache).
        Kept for factory compatibility.
    split : str
        Logical split tag (printed in logs).
    """

    def __init__(
        self,
        data_path: str,
        label_path: str = "",
        split: str = "train",
        future_path: Optional[str] = None,
        num_agents: int = 7,
        num_features: int = 4,
        num_frames: Optional[int] = None,
        fps: float = 10.0,
        history_sec: float = 8.0,
        **kwargs: Any,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for BaselineConflictDataset.")

        self.split = split
        self.num_agents = num_agents
        self.num_features = num_features
        self.num_frames = num_frames or int(round(fps * history_sec))

        # ── memory-map the pre-computed tensor cache ──────────────────────
        prefix = data_path  # e.g. "data/processed/train"
        self._x = np.load(f"{prefix}_x.npy", mmap_mode="r")
        self._mask = np.load(f"{prefix}_mask.npy", mmap_mode="r")
        self._y = np.load(f"{prefix}_y.npy", mmap_mode="r")
        self._target = np.load(f"{prefix}_target.npy", mmap_mode="r")
        self._event_ids = np.load(f"{prefix}_eid.npy", mmap_mode="r")

        # Build event_id → index for debugging
        self._eid_to_idx = {int(eid): i for i, eid in enumerate(self._event_ids)}

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._x)

    # ------------------------------------------------------------------
    # Get-item
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": torch.from_numpy(self._x[index].copy()),
            "agent_mask": torch.from_numpy(self._mask[index].copy()),
            "is_conflict": torch.tensor(float(self._y[index]), dtype=torch.float32),
            "target_idx": torch.tensor(int(self._target[index]), dtype=torch.long),
            "event_id": int(self._event_ids[index]),
        }
