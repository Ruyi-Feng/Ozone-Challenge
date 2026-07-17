"""Stage 4: assign Event_id and export data / label CSV tables."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from data_processing.io.schema import (
    DATA_COLUMNS,
    FUTURE_TRAJ_COLUMNS,
    LABEL_COLUMNS,
    ProcessingConfig,
    TrackedNeighborhoodEvent,
)
from data_processing.io.writers import (
    write_events_data,
    write_events_labels,
    write_future_traj,
)


def assign_event_ids(
    events: List[TrackedNeighborhoodEvent],
    *,
    start_id: int = 0,
) -> List[Tuple[str, TrackedNeighborhoodEvent]]:
    """Attach Event_id to each tracked event.

    Returns list of (event_id, event).  ID is an integer string: ``"0"``, ``"1"``, ...
    """
    return [(str(start_id + i), evt) for i, evt in enumerate(events)]


def _add_common_columns(
    df: pd.DataFrame,
    event_id: str,
    cfg: ProcessingConfig,
    roles: Dict,
    ego_id,
    t0: float,
) -> pd.DataFrame:
    """Add Event_id, scene_id, t_rel, role to a track DataFrame."""
    if df.empty:
        return pd.DataFrame(columns=DATA_COLUMNS)
    df = df.copy()
    df["Event_id"] = event_id
    df["t_rel"] = (df["frameNum"] - t0) / cfg.fps  # seconds, negative for history
    df["role"] = df["carId"].apply(
        lambda cid: "ego" if cid == ego_id else roles.get(cid, "other")
    )
    # Keep only columns in DATA_COLUMNS (plus original ones that match)
    out_cols = [c for c in DATA_COLUMNS if c in df.columns]
    for c in ["carCenterXm", "carCenterYm", "heading", "speed", "frameNum", "carId", "role", "t_rel", "Event_id", "scene_id"]:
        if c not in out_cols and c in df.columns:
            out_cols.append(c)
    # Ensure scene_id
    if "scene_id" not in df.columns:
        scene = "scene"
        if "scene_id" in df.columns:
            pass  # already handled
        else:
            df["scene_id"] = "scene"
    return df[DATA_COLUMNS] if all(c in df.columns for c in DATA_COLUMNS) else df


def build_event_data_rows(
    event_id: str,
    event: TrackedNeighborhoodEvent,
    cfg: ProcessingConfig,
) -> pd.DataFrame:
    """Build long-format history rows for one event (ego + neighbors).

    Only rows within the history window [t0 - history_sec, t0] are kept.
    """
    df = event.history_tracks.copy()
    if df.empty:
        return pd.DataFrame(columns=DATA_COLUMNS)

    window = event.window
    df = df[df["frameNum"] <= window.t0]  # clamp to t0
    return _add_common_columns(
        df, event_id, cfg, event.neighbor_roles,
        window.ego_id, window.t0,
    )


def build_event_label_row(
    event_id: str,
    event: TrackedNeighborhoodEvent,
    cfg: ProcessingConfig,
) -> dict:
    """Build one label dict: is_conflict, conflict_target_id, t0, etc."""
    w = event.window
    return {
        "Event_id": event_id,
        "scene_id": w.scene_id,
        "t0": w.t0,
        "is_conflict": int(w.is_conflict),
        "conflict_target_id": w.conflict_target_id if w.conflict_target_id is not None else -1,
        "conflict_target_role": event.conflict_target_role or "",
        "t_conflict": w.t_conflict if w.t_conflict is not None else "",
    }


def build_future_traj_rows(
    event_id: str,
    event: TrackedNeighborhoodEvent,
    cfg: ProcessingConfig,
) -> pd.DataFrame:
    """Build future 3s trajectory rows for one event."""
    df = event.future_tracks.copy()
    if df.empty:
        return pd.DataFrame(columns=FUTURE_TRAJ_COLUMNS)

    window = event.window
    df = df[df["frameNum"] > window.t0]  # strictly after t0
    df = _add_common_columns(
        df, event_id, cfg, event.neighbor_roles,
        window.ego_id, window.t0,
    )
    # Subset to FUTURE_TRAJ_COLUMNS
    keep = [c for c in FUTURE_TRAJ_COLUMNS if c in df.columns]
    return df[keep]


def events_to_tables(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert tracked events into (data_df, label_df, future_traj_df)."""
    labeled = assign_event_ids(events)
    data_parts, label_rows, future_parts = [], [], []
    for eid, evt in labeled:
        data_parts.append(build_event_data_rows(eid, evt, cfg))
        label_rows.append(build_event_label_row(eid, evt, cfg))
        future_parts.append(build_future_traj_rows(eid, evt, cfg))

    data_df = pd.concat(data_parts, ignore_index=True) if data_parts else pd.DataFrame(columns=DATA_COLUMNS)
    label_df = pd.DataFrame(label_rows, columns=LABEL_COLUMNS) if label_rows else pd.DataFrame(columns=LABEL_COLUMNS)
    future_df = pd.concat(future_parts, ignore_index=True) if future_parts else pd.DataFrame(columns=FUTURE_TRAJ_COLUMNS)
    return data_df, label_df, future_df


def export_events(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> Tuple[str, str, str]:
    """Assign Event_ids, build tables, write CSV files.

    Returns
    -------
    (data_path, label_path, future_traj_path)
    """
    data_df, label_df, future_df = events_to_tables(events, cfg)
    data_path = str(write_events_data(data_df, cfg.data_out))
    label_path = str(write_events_labels(label_df, cfg.label_out))
    future_path = str(write_future_traj(future_df, cfg.future_traj_out))
    return data_path, label_path, future_path
