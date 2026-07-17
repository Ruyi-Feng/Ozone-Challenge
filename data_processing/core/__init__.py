"""Core stage APIs."""

from data_processing.core.conflict_detect import (
    detect_all_candidates,
    detect_conflicts,
    sample_non_conflicts,
)
from data_processing.core.export import (
    assign_event_ids,
    build_event_data_rows,
    build_event_label_row,
    build_future_traj_rows,
    events_to_tables,
    export_events,
)
from data_processing.core.neighbor_filter import (
    assign_neighbor_roles,
    collect_persistent_tracks,
    extract_neighbors_for_event,
    extract_neighbors_for_events,
    find_entered_neighbor_ids,
)
from data_processing.core.trajectory_window import (
    build_windowed_candidate,
    build_windowed_events,
    propose_t0_for_conflict,
    propose_t0_for_non_conflict,
    validate_future_interval,
    validate_history_length,
)

__all__ = [
    "detect_conflicts",
    "sample_non_conflicts",
    "detect_all_candidates",
    "propose_t0_for_conflict",
    "propose_t0_for_non_conflict",
    "validate_history_length",
    "validate_future_interval",
    "build_windowed_candidate",
    "build_windowed_events",
    "find_entered_neighbor_ids",
    "assign_neighbor_roles",
    "collect_persistent_tracks",
    "extract_neighbors_for_event",
    "extract_neighbors_for_events",
    "assign_event_ids",
    "build_event_data_rows",
    "build_event_label_row",
    "build_future_traj_rows",
    "events_to_tables",
    "export_events",
]
