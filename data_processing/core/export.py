"""Stage 4: assign Event_id and export data / label CSV tables."""

from __future__ import annotations

from pathlib import Path
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


def _suffix_path(path: str, suffix: str) -> str:
    """Insert *suffix* before the file extension: ``data/foo.csv`` → ``data/foo_train.csv``."""
    p = Path(path)
    return str(p.parent / f"{p.stem}_{suffix}{p.suffix}")


def export_events(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> Tuple[str, str, str]:
    """Assign Event_ids, build tables, write CSV files.

    When ``cfg.train_val_split_ratio`` is set (0 < ratio < 1), events are
    split by ego_id before export and separate train / val CSV files are
    written.  Otherwise a single set of files is produced.

    Returns
    -------
    (data_path, label_path, future_traj_path)  — train paths when split is active.
    """
    ratio = cfg.train_val_split_ratio
    if ratio is not None and 0.0 < ratio < 1.0:
        return _export_with_split(events, cfg, ratio)

    data_df, label_df, future_df = events_to_tables(events, cfg)
    data_path = str(write_events_data(data_df, cfg.data_out))
    label_path = str(write_events_labels(label_df, cfg.label_out))
    future_path = str(write_future_traj(future_df, cfg.future_traj_out))
    return data_path, label_path, future_path


def _export_with_split(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
    train_ratio: float,
) -> Tuple[str, str, str]:
    """Split by ego, then export train and val tables."""
    from data_processing.core.split import split_by_ego

    train_events, val_events = split_by_ego(events, train_ratio)

    # --- Train ---
    data_train, label_train, future_train = events_to_tables(train_events, cfg)
    train_data = str(write_events_data(data_train, _suffix_path(cfg.data_out, "train")))
    train_label = str(write_events_labels(label_train, _suffix_path(cfg.label_out, "train")))
    train_future = str(write_future_traj(future_train, _suffix_path(cfg.future_traj_out, "train")))

    # --- Val ---
    data_val, label_val, future_val = events_to_tables(val_events, cfg)
    val_data = str(write_events_data(data_val, _suffix_path(cfg.data_out, "val")))
    val_label = str(write_events_labels(label_val, _suffix_path(cfg.label_out, "val")))
    val_future = str(write_future_traj(future_val, _suffix_path(cfg.future_traj_out, "val")))

    n_train = len({e.window.ego_id for e in train_events})
    n_val = len({e.window.ego_id for e in val_events})
    print(f"Split: {len(train_events)} train events ({n_train} egos), "
          f"{len(val_events)} val events ({n_val} egos)")

    return train_data, train_label, train_future
