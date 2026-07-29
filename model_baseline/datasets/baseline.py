"""Baseline dataset — multi-agent history → conflict labels.

Loads processed event CSVs and builds fixed-slot [A, T, F] tensors with
ego-relative features.  Variable neighbor count is handled via agent_mask.

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

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore


SLOT_ORDER: tuple[str, ...] = (
    "ego",
    "front",
    "rear",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
)
SLOT_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SLOT_ORDER)}

# Neighbor slots only (indices 1..6), mapped to model target index 0..5
# slot_idx=1 (front) → target=0, …, slot_idx=6 (right_rear) → target=5
SLOT_TO_TARGET: dict[int, int] = {i: i - 1 for i in range(1, len(SLOT_ORDER))}


def _normalize_angle_diff(a: float, b: float) -> float:
    """a - b, wrapped to [-180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class BaselineConflictDataset(Dataset):
    """PyTorch Dataset for multi-agent conflict prediction.

    Each sample is one event: 8 s of history for ego + up to 6 neighbours,
    represented as a fixed-size [A=7, T, F=4] tensor with an agent_mask.

    Parameters
    ----------
    data_path : str
        Path to *events_data_{split}.csv*.
    label_path : str
        Path to *events_labels_{split}.csv*.
    split : str
        Logical split tag (ignored by the dataset; passed through for logging).
    future_path : str or None
        Unused; kept for factory compatibility.
    num_agents : int
        Fixed agent-slot count (default 7: ego + 6 neighbours).
    num_features : int
        Feature dimension (default 4: dx, dy, heading_rel, speed).
    num_frames : int or None
        History length in frames. Inferred as ``fps × history_sec`` when None.
    fps : float
        Sampling rate (Hz). Default 10.
    history_sec : float
        History window in seconds. Default 8.
    """

    def __init__(
        self,
        data_path: str,
        label_path: str,
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

        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.num_agents = num_agents
        self.num_features = num_features
        self.num_frames = num_frames or int(round(fps * history_sec))
        self.fps = fps
        self.history_sec = history_sec

        # ------------------------------------------------------------------
        # Load tables
        # ------------------------------------------------------------------
        self.data_df = pd.read_csv(data_path)
        self.labels_df = pd.read_csv(label_path)

        # ------------------------------------------------------------------
        # Build labels lookup and event index
        # ------------------------------------------------------------------
        self.labels: dict[int, pd.Series] = {}
        for _, row in self.labels_df.iterrows():
            self.labels[int(row["Event_id"])] = row

        self.event_ids = sorted(self.data_df["Event_id"].unique())
        self.event_ids = [int(e) for e in self.event_ids]

        # ------------------------------------------------------------------
        # Pre-group data by Event_id
        # ------------------------------------------------------------------
        self._by_event: dict[int, pd.DataFrame] = {}
        for eid in self.event_ids:
            self._by_event[eid] = self.data_df[self.data_df["Event_id"] == eid]

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.event_ids)

    # ------------------------------------------------------------------
    # Get-item
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        event_id = self.event_ids[index]
        evt = self._by_event[event_id]
        label = self.labels[event_id]

        t0: float = float(label["t0"])
        is_conflict: int = int(label["is_conflict"])
        conflict_target_id: int = int(label["conflict_target_id"])

        # ---- 1. carId → role mapping ----------------------------------
        car_roles: dict[int, str] = {}
        for cid, role in zip(evt["carId"], evt["role"]):
            cid_i = int(cid)
            if cid_i not in car_roles:
                car_roles[cid_i] = role

        # Identify ego
        ego_id: Optional[int] = None  # noqa: F821
        for cid, role in car_roles.items():
            if role == "ego":
                ego_id = cid
                break

        if ego_id is None:
            raise ValueError(f"Event {event_id}: no ego vehicle found")

        # ---- 2. Slot assignment ---------------------------------------
        # slot_cars: slot_idx (0-6) → carId
        slot_cars: dict[int, int] = {}
        # Count frames per car for tie-breaking
        frame_counts = evt.groupby("carId").size()

        for cid, role in car_roles.items():
            slot_idx = SLOT_TO_IDX.get(role)
            if slot_idx is None:
                continue
            if slot_idx not in slot_cars:
                slot_cars[slot_idx] = cid
            else:
                existing = slot_cars[slot_idx]
                # Conflict target wins the slot
                if cid == conflict_target_id:
                    slot_cars[slot_idx] = cid
                elif existing != conflict_target_id:
                    # Keep the car with more frames
                    if frame_counts.get(cid, 0) > frame_counts.get(existing, 0):
                        slot_cars[slot_idx] = cid

        # ---- 3. Ego reference state at t₀ ----------------------------
        ego_rows = evt[evt["carId"] == ego_id]
        # frame at or just before t₀
        ego_at_t0 = ego_rows[ego_rows["frameNum"] <= t0]
        if ego_at_t0.empty:
            # fallback: closest frame
            ego_rows_sorted = ego_rows.copy()
            ego_rows_sorted["_dist"] = np.abs(ego_rows_sorted["frameNum"] - t0)
            ego_at_t0 = ego_rows_sorted.loc[[ego_rows_sorted["_dist"].idxmin()]]
        ego_ref = ego_at_t0.iloc[-1]  # latest frame ≤ t₀
        ego_x0 = float(ego_ref["carCenterXm"])
        ego_y0 = float(ego_ref["carCenterYm"])
        ego_h0 = float(ego_ref["heading"])

        # ---- 4. Frame axis --------------------------------------------
        all_frames = sorted(evt["frameNum"].unique())
        # Only keep frames ≤ t₀ (history window)
        history_frames = [f for f in all_frames if f <= t0]
        # Take the last T frames (pad on the left if fewer)
        if len(history_frames) >= self.num_frames:
            frame_window = history_frames[-self.num_frames :]
        else:
            frame_window = history_frames
        T_actual = len(frame_window)

        # ---- 5. Build tensors -----------------------------------------
        x = torch.zeros(self.num_agents, self.num_frames, self.num_features)
        agent_mask = torch.zeros(self.num_agents, dtype=torch.bool)

        for slot_idx in range(self.num_agents):
            car_id = slot_cars.get(slot_idx)
            if car_id is None:
                continue
            agent_mask[slot_idx] = True
            car_rows = evt[evt["carId"] == car_id]
            # Index rows by frameNum for fast lookup
            car_by_frame = {}
            for _, r in car_rows.iterrows():
                car_by_frame[int(r["frameNum"])] = r

            for t_idx, fn in enumerate(frame_window):
                # left-pad: offset if T_actual < num_frames
                out_t = self.num_frames - T_actual + t_idx
                row = car_by_frame.get(int(fn))
                if row is None:
                    continue
                x[slot_idx, out_t, 0] = float(row["carCenterXm"]) - ego_x0
                x[slot_idx, out_t, 1] = float(row["carCenterYm"]) - ego_y0
                x[slot_idx, out_t, 2] = _normalize_angle_diff(
                    float(row["heading"]), ego_h0
                )
                x[slot_idx, out_t, 3] = float(row["speed"])

        # ---- 6. Target index ------------------------------------------
        target_idx = -1
        if is_conflict and conflict_target_id != -1:
            for slot_idx, cid in slot_cars.items():
                if cid == conflict_target_id and slot_idx in SLOT_TO_TARGET:
                    target_idx = SLOT_TO_TARGET[slot_idx]
                    break

        return {
            "x": x,
            "agent_mask": agent_mask,
            "is_conflict": torch.tensor(float(is_conflict), dtype=torch.float32),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "event_id": event_id,
        }
