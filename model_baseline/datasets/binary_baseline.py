"""Memory-efficient dataset backed by flat binary file + in-memory CSV index.

Reads event tensors on-demand with f.seek() + f.read() + np.frombuffer(),
keeping only the index (~3 MB for 127k events) in memory.

Sample contract (same as BaselineConflictDataset):
  __getitem__ → dict with:
    - "x":           FloatTensor [A, T, F]  multi-agent history
    - "agent_mask":  BoolTensor  [A]        True if slot has a vehicle
    - "is_conflict": FloatTensor  scalar    0.0 / 1.0
    - "target_idx":  LongTensor   scalar    neighbor slot index in {0..5}, or -1
    - "event_id":    int                     for debugging
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore


class BinaryBaselineDataset(Dataset):
    """PyTorch Dataset backed by a flat binary file + in-memory CSV index.

    The binary file contains concatenated float32 tensors, one per event:
    each event is [num_agents, num_frames, num_features] = 8960 bytes.

    The index CSV columns:
      event_id, byte_offset, byte_length, is_conflict, target_idx,
      agent_mask, scene_id, conflict_target_role

    Multi-worker safe: each DataLoader worker opens its own file handle via
    ``threading.local()`` lazy initialisation.

    Parameters
    ----------
    data_path : str
        Path prefix for ``{prefix}_data.bin`` and ``{prefix}_index.csv``.
    label_path : str
        Unused (kept for factory compatibility).  All metadata is in the
        index CSV loaded from ``{data_path}_index.csv``.
    split : str
        Logical split tag for logging.
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
            raise ImportError("PyTorch is required for BinaryBaselineDataset.")

        self.split = split
        self.num_agents = num_agents
        self.num_features = num_features
        self.num_frames = num_frames or int(round(fps * history_sec))

        # ── Paths ───────────────────────────────────────────────────────
        prefix = data_path  # e.g. "data/processed/train"
        self._bin_path = f"{prefix}_data.bin"
        index_path = f"{prefix}_index.csv"

        if not Path(self._bin_path).exists():
            raise FileNotFoundError(
                f"Binary data file not found: {self._bin_path}. "
                f"Run build_binary_cache.py first."
            )

        # ── Load index into memory ──────────────────────────────────────
        self._index = pd.read_csv(index_path)
        self._record_size = num_agents * self.num_frames * num_features * 4

        # Verify record size consistency
        if len(self._index) > 0:
            first_len = int(self._index.iloc[0]["byte_length"])
            if first_len != self._record_size:
                print(
                    f"  WARNING: index record size {first_len} != "
                    f"expected {self._record_size}"
                )

        # ── Per-worker file handle (thread-local, re-opened after fork) ─
        self._local = threading.local()

    # ------------------------------------------------------------------
    # File handle management
    # ------------------------------------------------------------------

    def _get_handle(self):
        """Return a file handle for the current thread / worker process.

        Each worker opens its own handle on first access.  This is safe
        with both ``fork`` (CPython reinitialises thread-local storage)
        and ``spawn`` (fresh process, no shared state).
        """
        if not hasattr(self._local, "fh") or self._local.fh is None:
            self._local.fh = open(self._bin_path, "rb")
        return self._local.fh

    def __getstate__(self):
        """Exclude file handles from pickling (needed for spawn mp)."""
        state = dict(self.__dict__)
        state["_local"] = threading.local()
        return state

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Get-item
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._index.iloc[index]
        fh = self._get_handle()
        fh.seek(int(row["byte_offset"]))
        raw = fh.read(int(row["byte_length"]))

        x = (
            np.frombuffer(raw, dtype=np.float32)
            .copy()
            .reshape(self.num_agents, self.num_frames, self.num_features)
        )

        # Decode agent_mask from packed uint8
        mask_bits = int(row["agent_mask"])
        agent_mask = np.array(
            [(mask_bits >> j) & 1 for j in range(self.num_agents)],
            dtype=bool,
        )

        return {
            "x": torch.from_numpy(x),
            "agent_mask": torch.from_numpy(agent_mask),
            "is_conflict": torch.tensor(
                float(row["is_conflict"]), dtype=torch.float32
            ),
            "target_idx": torch.tensor(
                int(row["target_idx"]), dtype=torch.long
            ),
            "event_id": int(row["event_id"]),
        }
