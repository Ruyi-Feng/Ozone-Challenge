"""Column contracts and intermediate data structures for the processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Raw CSV (standardized input) — extend when the real format is locked
# ---------------------------------------------------------------------------

# NBDT standard trajectory format — minimum required columns for the pipeline
RAW_REQUIRED_COLUMNS: list[str] = [
    "frameNum",
    "carId",
    "carCenterXm",
    "carCenterYm",
    "heading",
    "speed",
    "objClass",
]

# Columns expected in the OBB data for 2D_TTC calculation (optional but recommended)
RAW_OBB_COLUMNS: list[str] = [
    "boundingBox1Xm",
    "boundingBox1Ym",
    "boundingBox2Xm",
    "boundingBox2Ym",
    "boundingBox3Xm",
    "boundingBox3Ym",
    "boundingBox4Xm",
    "boundingBox4Ym",
]

# ---------------------------------------------------------------------------
# Processed outputs
# ---------------------------------------------------------------------------

DATA_COLUMNS: list[str] = [
    "Event_id",
    "scene_id",
    "frameNum",
    "t_rel",
    "carId",
    "role",  # ego | front | rear | left_front | left_rear | right_front | right_rear
    "carCenterXm",
    "carCenterYm",
    "heading",
    "speed",
]

LABEL_COLUMNS: list[str] = [
    "Event_id",
    "scene_id",
    "t0",
    "is_conflict",
    "conflict_target_id",
    "conflict_target_role",
    "t_conflict",
]

FUTURE_TRAJ_COLUMNS: list[str] = [
    "Event_id",
    "scene_id",
    "carId",
    "frameNum",
    "t_rel",
    "carCenterXm",
    "carCenterYm",
    "heading",
    "speed",
    "role",
]

NEIGHBOR_SLOTS: tuple[str, ...] = (
    "front",
    "rear",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
)


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Raise if required raw columns are missing."""
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}")


# ---------------------------------------------------------------------------
# Intermediate objects passed between pipeline stages
# ---------------------------------------------------------------------------


@dataclass
class ConflictCandidate:
    """Output of conflict / non-conflict detection."""

    scene_id: str
    ego_id: Any
    is_conflict: bool
    t_conflict: Optional[float] = None  # absolute timestamp; None for non-conflict
    conflict_target_id: Optional[Any] = None
    meta: dict = field(default_factory=dict)


@dataclass
class WindowedEventCandidate:
    """Candidate after history/future window validation."""

    scene_id: str
    ego_id: Any
    is_conflict: bool
    t0: float  # start of future 3s interval; history is [t0 - history_sec, t0]
    t_end: float  # t0 + future_sec
    t_conflict: Optional[float] = None
    conflict_target_id: Optional[Any] = None
    history_ok: bool = False
    future_ok: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class TrackedNeighborhoodEvent:
    """Windowed event with ego + persistent neighbor trajectories attached."""

    window: WindowedEventCandidate
    # long-format trajectory rows for [t0 - history, t0] (and optional full duration)
    history_tracks: pd.DataFrame
    # long-format future rows for (t0, t0 + future]
    future_tracks: pd.DataFrame
    # track_id -> slot name for neighbors used in this event
    neighbor_roles: dict = field(default_factory=dict)
    conflict_target_role: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Runtime knobs; typically loaded from configs/data_processing.yaml."""

    history_sec: float = 8.0
    future_sec: float = 3.0
    fps: float = 25.0
    conflict_ttc_threshold: float = 1.5  # 2D_TTC below this → conflict
    # Future heading-ray / TTC-extrapolated meeting point must lie within
    # this distance of BOTH current vehicle centres, otherwise the TTC is
    # treated as invalid (far-ahead false positives).
    max_intersection_distance_m: float = 50.0
    max_distance_m: float = 200.0
    neighbor_slots: tuple[str, ...] = NEIGHBOR_SLOTS
    train_val_split_ratio: float | None = None  # None → no split; 0.8 → 80/20 train/val
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim/candidates"
    data_out: str = "data/processed/data/events_data.csv"
    label_out: str = "data/processed/labels/events_labels.csv"
    future_traj_out: str = "data/processed/labels/events_future_traj.csv"
    # ── rebalance ──
    rebalance_enabled: bool = False
    rebalance_target_ratio: float = 1.0  # target conflict:non-conflict ratio
    rebalance_balance_conflict_types: bool = False
    rebalance_per_scene: bool = True
    rebalance_seed: int = 42
    # Optional pilot-only cap applied after history/future validation.
    # Zero keeps every event (the production/default behaviour).
    max_events_per_class_per_scene: int = 0
    # Drop near-duplicate windows of the same ego (t0 is in frames).
    # Zero disables the gap filter.
    min_t0_gap_sec: float = 0.0
    # After the gap filter, keep at most this many windows per
    # (scene, ego, class). Zero disables the per-ego cap.
    max_events_per_ego_per_class: int = 0
