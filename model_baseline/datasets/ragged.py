"""Ragged (variable-agent) dataset + batch-level dynamic padding.

Each event stores only its real agents (1..max_agents). The DataLoader
collate function pads to the batch max, never to a global 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from data_processing.core.variable_agents import (
    MAX_AGENTS,
    PAD_ROLE_ID,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore


def ragged_collate_fn(
    batch: list[dict[str, Any]],
    *,
    max_agents: int = MAX_AGENTS,
    pad_role_id: int = PAD_ROLE_ID,
) -> dict[str, Any]:
    """Pad a list of variable-A samples to A_batch = max(A_i) <= max_agents."""
    if torch is None:
        raise ImportError("PyTorch is required for ragged_collate_fn.")
    if not batch:
        raise ValueError("empty batch")

    lengths = [int(sample["x"].shape[0]) for sample in batch]
    if any(a < 1 for a in lengths):
        raise ValueError(f"agent count must be >= 1, got {lengths}")
    a_batch = max(lengths)
    if a_batch > max_agents:
        raise ValueError(
            f"A_batch={a_batch} exceeds max_agents={max_agents} "
            f"(per-sample lengths={lengths})"
        )

    t = int(batch[0]["x"].shape[1])
    f = int(batch[0]["x"].shape[2])
    b = len(batch)
    x = torch.zeros(b, a_batch, t, f, dtype=batch[0]["x"].dtype)
    valid_mask = torch.zeros(b, a_batch, t, dtype=torch.bool)
    role_ids = torch.full((b, a_batch), pad_role_id, dtype=torch.long)
    agent_ids = torch.full((b, a_batch), -1, dtype=torch.long)
    agent_mask = torch.zeros(b, a_batch, dtype=torch.bool)
    y = torch.empty(b, dtype=torch.float32)
    target_idx = torch.empty(b, dtype=torch.long)
    event_id = torch.empty(b, dtype=torch.long)

    for i, sample in enumerate(batch):
        a_i = lengths[i]
        x[i, :a_i] = sample["x"]
        valid_mask[i, :a_i] = sample["valid_mask"]
        role_ids[i, :a_i] = sample["role_ids"]
        agent_ids[i, :a_i] = sample["agent_ids"]
        agent_mask[i, :a_i] = sample["agent_mask"]
        if not bool(agent_mask[i, 0]):
            raise ValueError("ego agent_mask must be True")
        y[i] = sample["y"] if "y" in sample else sample["is_conflict"]
        target_idx[i] = sample["target_idx"]
        event_id[i] = int(sample["event_id"])

    return {
        "x": x,
        "valid_mask": valid_mask,
        "role_ids": role_ids,
        "agent_ids": agent_ids,
        "agent_mask": agent_mask,
        "y": y,
        "is_conflict": y,
        "target_idx": target_idx,
        "event_id": event_id,
    }


class RaggedBaselineConflictDataset(Dataset):
    """Memory-mapped ragged event cache.

    ``data_path`` is a split prefix such as ``.../ragged10/train``.
    """

    collate_fn = staticmethod(ragged_collate_fn)

    def __init__(
        self,
        data_path: str,
        label_path: str = "",
        split: str = "train",
        future_path: Optional[str] = None,
        num_agents: int = MAX_AGENTS,
        max_agents: Optional[int] = None,
        num_features: int = 4,
        num_frames: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for RaggedBaselineConflictDataset.")

        self.split = split
        self.max_agents = int(max_agents or num_agents or MAX_AGENTS)
        self.num_agents = self.max_agents
        self.num_features = num_features
        self.num_frames = int(num_frames) if num_frames else 80
        prefix = data_path

        self._x = np.load(f"{prefix}_tracks_x.npy", mmap_mode="r")
        self._valid = np.load(f"{prefix}_tracks_valid.npy", mmap_mode="r")
        self._role = np.load(f"{prefix}_tracks_role.npy", mmap_mode="r")
        self._agent_id = np.load(f"{prefix}_tracks_agent_id.npy", mmap_mode="r")
        self._offsets = np.load(f"{prefix}_event_offsets.npy", mmap_mode="r")
        self._event_ids = np.load(f"{prefix}_event_eid.npy", mmap_mode="r")
        self._y = np.load(f"{prefix}_event_y.npy", mmap_mode="r")
        self._target = np.load(f"{prefix}_event_target.npy", mmap_mode="r")
        count_path = Path(f"{prefix}_event_agent_count.npy")
        if count_path.exists():
            self._count = np.load(str(count_path), mmap_mode="r")
        else:
            self._count = np.diff(self._offsets).astype(np.int64)
        self.has_valid_mask = True
        self._eid_to_idx = {int(eid): i for i, eid in enumerate(self._event_ids)}
        self.collate_fn = make_collate_fn(max_agents=self.max_agents)
        self._labels = None
        if label_path and Path(label_path).exists():
            import pandas as pd

            labels_df = pd.read_csv(label_path)
            self._labels = {
                int(row["Event_id"]): row for _, row in labels_df.iterrows()
            }

        if int(self._offsets[-1]) != len(self._x):
            raise ValueError(
                f"{prefix}: offsets[-1]={self._offsets[-1]} != tracks {len(self._x)}"
            )
        if np.any(self._count > self.max_agents) or np.any(self._count < 1):
            raise ValueError(
                f"{prefix}: agent counts must be in [1, {self.max_agents}]"
            )

    def __len__(self) -> int:
        return len(self._event_ids)

    def _slice(self, index: int) -> tuple[int, int]:
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        a_i = end - start
        if not 1 <= a_i <= self.max_agents:
            raise ValueError(
                f"event {int(self._event_ids[index])}: A={a_i} "
                f"outside [1, {self.max_agents}]"
            )
        return start, end

    def __getitem__(self, index: int) -> dict[str, Any]:
        start, end = self._slice(index)
        x = np.asarray(self._x[start:end], dtype=np.float32).copy()
        valid = np.asarray(self._valid[start:end], dtype=bool).copy()
        role_ids = np.asarray(self._role[start:end], dtype=np.int64).copy()
        agent_ids = np.asarray(self._agent_id[start:end], dtype=np.int64).copy()
        a_i = x.shape[0]
        agent_mask = np.ones(a_i, dtype=bool)
        y = float(self._y[index])
        return {
            "x": torch.from_numpy(x),
            "valid_mask": torch.from_numpy(valid),
            "role_ids": torch.from_numpy(role_ids),
            "agent_ids": torch.from_numpy(agent_ids),
            "agent_mask": torch.from_numpy(agent_mask),
            "y": torch.tensor(y, dtype=torch.float32),
            "is_conflict": torch.tensor(y, dtype=torch.float32),
            "target_idx": torch.tensor(int(self._target[index]), dtype=torch.long),
            "event_id": int(self._event_ids[index]),
        }

    def get_metadata(self, index: int) -> dict[str, Any]:
        eid = int(self._event_ids[index])
        meta = {
            "event_id": eid,
            "scene_id": "",
            "conflict_target_role": "",
            "is_conflict": int(self._y[index]),
            "target_idx": int(self._target[index]),
        }
        if self._labels is not None and eid in self._labels:
            row = self._labels[eid]
            scene = row.get("scene_id", "")
            role = row.get("conflict_target_role", "")
            meta["scene_id"] = "" if scene != scene else str(scene)
            meta["conflict_target_role"] = "" if role != role else str(role)
        return meta


def make_collate_fn(
    max_agents: int = MAX_AGENTS,
    pad_role_id: int = PAD_ROLE_ID,
):
    def _fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return ragged_collate_fn(
            batch, max_agents=max_agents, pad_role_id=pad_role_id
        )

    return _fn
