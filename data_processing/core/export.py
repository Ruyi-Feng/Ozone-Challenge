"""Stage 4: assign Event_id and export data / label CSV tables."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from data_processing.io.schema import ProcessingConfig, TrackedNeighborhoodEvent
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
    """
    Attach Event_id to each tracked event.

    Returns list of (event_id, event). ID scheme can be integer or '{scene}_{seq}'.
    """
    raise NotImplementedError


def build_event_data_rows(
    event_id: str,
    event: TrackedNeighborhoodEvent,
    cfg: ProcessingConfig,
) -> pd.DataFrame:
    """
    Build long-format history rows for one event (ego + neighbors).

    Keeps raw kinematic columns and adds Event_id, role, t_rel.
    """
    raise NotImplementedError


def build_event_label_row(
    event_id: str,
    event: TrackedNeighborhoodEvent,
    cfg: ProcessingConfig,
) -> dict:
    """
    Build one label dict: is_conflict, conflict_target_id, t0, etc.
    """
    raise NotImplementedError


def build_future_traj_rows(
    event_id: str,
    event: TrackedNeighborhoodEvent,
    cfg: ProcessingConfig,
) -> pd.DataFrame:
    """Build future 3s trajectory rows for one event."""
    raise NotImplementedError


def events_to_tables(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert tracked events into (data_df, label_df, future_traj_df).
    """
    raise NotImplementedError


def export_events(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> Tuple[str, str, str]:
    """
    Assign Event_ids, build tables, write CSV files.

    Returns
    -------
    (data_path, label_path, future_traj_path)
    """
    data_df, label_df, future_df = events_to_tables(events, cfg)
    data_path = str(write_events_data(data_df, cfg.data_out))
    label_path = str(write_events_labels(label_df, cfg.label_out))
    future_path = str(write_future_traj(future_df, cfg.future_traj_out))
    return data_path, label_path, future_path
