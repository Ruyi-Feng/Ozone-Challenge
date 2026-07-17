"""Stage 3: neighbor slot filtering and persistent trajectory collection."""

from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd

from data_processing.io.schema import (
    ProcessingConfig,
    TrackedNeighborhoodEvent,
    WindowedEventCandidate,
)


def find_entered_neighbor_ids(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
) -> Set:
    """
    Find track_ids that ever enter ego's neighbor region during the target duration.

    Duration at least covers [t0 - history_sec, t0]; config may extend to future.
    Vehicles farther than max_distance_m are excluded.
    """
    raise NotImplementedError


def assign_neighbor_roles(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    neighbor_ids: Set,
    cfg: ProcessingConfig,
) -> Dict:
    """
    Assign each neighbor track_id to a slot:
    front / rear / left_front / left_rear / right_front / right_rear.
    """
    raise NotImplementedError


def collect_persistent_tracks(
    raw_df: pd.DataFrame,
    *,
    scene_id: str,
    track_ids: Set,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """
    Continuously collect trajectories for all given track_ids over [t_start, t_end].

    Once a vehicle entered the neighborhood during the duration, keep collecting
    for the full interval (not only frames while inside the slot).
    """
    raise NotImplementedError


def extract_neighbors_for_event(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
) -> TrackedNeighborhoodEvent:
    """Full neighbor pipeline for a single windowed event."""
    raise NotImplementedError


def extract_neighbors_for_events(
    raw_df: pd.DataFrame,
    windows: List[WindowedEventCandidate],
    cfg: ProcessingConfig,
) -> List[TrackedNeighborhoodEvent]:
    """Run neighbor extraction for all windowed events."""
    return [extract_neighbors_for_event(raw_df, w, cfg) for w in windows]
