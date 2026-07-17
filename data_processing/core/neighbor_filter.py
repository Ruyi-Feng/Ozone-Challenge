"""Stage 3: neighbor slot filtering and persistent trajectory collection."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd

from data_processing.io.schema import (
    ProcessingConfig,
    TrackedNeighborhoodEvent,
    WindowedEventCandidate,
)
from data_processing.utils.geometry import (
    classify_neighbor_slot,
    compute_distance,
    compute_relative_pose,
    is_within_max_distance,
)


def _sec2frames(sec: float, fps: float) -> int:
    return int(round(sec * fps))


def _frame_rows(
    raw_df: pd.DataFrame,
    scene_id: str,
    frame_num: int,
) -> pd.DataFrame:
    """Return all rows for a given *frame_num* (optionally filtered by scene)."""
    mask = raw_df["frameNum"] == frame_num
    if "scene_id" in raw_df.columns:
        mask = mask & (raw_df["scene_id"] == scene_id)
    return raw_df[mask]


def _track_rows(
    raw_df: pd.DataFrame,
    scene_id: str,
    track_ids: Set,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """Return all rows for *track_ids* within [t_start, t_end]."""
    mask = raw_df["carId"].isin(track_ids)
    mask &= (raw_df["frameNum"] >= t_start) & (raw_df["frameNum"] <= t_end)
    if "scene_id" in raw_df.columns:
        mask &= raw_df["scene_id"] == scene_id
    return raw_df[mask].sort_values(["carId", "frameNum"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_entered_neighbor_ids(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
) -> Set:
    """Find track_ids that ever enter ego's neighbor region during the history window.

    Scans every frame in [t0 - history_frames, t0]; a target vehicle qualifies
    if at any frame it is within *max_distance_m* AND falls into one of the
    configured neighbor slots.
    """
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)
    t_start = int(window.t0 - history_frames)
    t_end = int(window.t0)
    scene = window.scene_id

    entered: Set = set()

    for fnum in range(t_start, t_end + 1):
        frame_df = _frame_rows(raw_df, scene, fnum)
        ego_rows = frame_df[frame_df["carId"] == window.ego_id]
        if ego_rows.empty:
            continue
        ego = ego_rows.iloc[0]
        ego_x, ego_y = float(ego["carCenterXm"]), float(ego["carCenterYm"])
        ego_heading = float(ego["heading"])

        for _, other in frame_df.iterrows():
            if other["carId"] == window.ego_id:
                continue
            dist = compute_distance(ego_x, ego_y,
                                    float(other["carCenterXm"]),
                                    float(other["carCenterYm"]))
            if not is_within_max_distance(dist, cfg.max_distance_m):
                continue
            dx, dy = compute_relative_pose(ego_x, ego_y, ego_heading,
                                           float(other["carCenterXm"]),
                                           float(other["carCenterYm"]))
            slot = classify_neighbor_slot(dx, dy, slots=cfg.neighbor_slots)
            if slot is not None:
                entered.add(other["carId"])

    return entered


def assign_neighbor_roles(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    neighbor_ids: Set,
    cfg: ProcessingConfig,
) -> Dict:
    """Assign each neighbor track_id to a slot.

    Scans the same history frames as *find_entered_neighbor_ids* and for
    each neighbor picks the slot that appears most frequently (mode).
    Ties are broken by distance — the slot from the closest frame wins.
    """
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)
    t_start = int(window.t0 - history_frames)
    t_end = int(window.t0)
    scene = window.scene_id

    slot_votes: Dict = {nid: {} for nid in neighbor_ids}
    slot_best_dist: Dict = {nid: (None, float("inf")) for nid in neighbor_ids}

    for fnum in range(t_start, t_end + 1):
        frame_df = _frame_rows(raw_df, scene, fnum)
        ego_rows = frame_df[frame_df["carId"] == window.ego_id]
        if ego_rows.empty:
            continue
        ego = ego_rows.iloc[0]
        ego_x, ego_y = float(ego["carCenterXm"]), float(ego["carCenterYm"])
        ego_heading = float(ego["heading"])

        for nid in neighbor_ids:
            other_rows = frame_df[frame_df["carId"] == nid]
            if other_rows.empty:
                continue
            other = other_rows.iloc[0]
            dist = compute_distance(ego_x, ego_y,
                                    float(other["carCenterXm"]),
                                    float(other["carCenterYm"]))
            dx, dy = compute_relative_pose(ego_x, ego_y, ego_heading,
                                           float(other["carCenterXm"]),
                                           float(other["carCenterYm"]))
            slot = classify_neighbor_slot(dx, dy, slots=cfg.neighbor_slots)
            if slot is not None:
                slot_votes[nid][slot] = slot_votes[nid].get(slot, 0) + 1
                if dist < slot_best_dist[nid][1]:
                    slot_best_dist[nid] = (slot, dist)

    roles: Dict = {}
    for nid in neighbor_ids:
        votes = slot_votes[nid]
        if not votes:
            # fallback: best-distance slot
            roles[nid] = slot_best_dist[nid][0] if slot_best_dist[nid][0] else "other"
        else:
            roles[nid] = max(votes, key=votes.get)  # type: ignore[arg-type]

    return roles


def collect_persistent_tracks(
    raw_df: pd.DataFrame,
    *,
    scene_id: str,
    track_ids: Set,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """Continuously collect trajectories for *track_ids* over [t_start, t_end].

    Once a vehicle entered the neighborhood, keep collecting for the full
    interval (not only frames while inside the slot).
    """
    return _track_rows(raw_df, scene_id, track_ids, t_start, t_end)


def extract_neighbors_for_event(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
) -> TrackedNeighborhoodEvent:
    """Full neighbor pipeline for a single windowed event."""
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)
    future_frames = _sec2frames(cfg.future_sec, cfg.fps)
    scene = window.scene_id
    t0 = window.t0

    # 1. Find which vehicles ever entered ego's neighborhood
    neighbor_ids = find_entered_neighbor_ids(raw_df, window, cfg)

    # 2. Assign roles
    roles = assign_neighbor_roles(raw_df, window, neighbor_ids, cfg)

    # 3. Determine conflict target role (if applicable)
    conflict_target_role: Optional[str] = None  # noqa: F821
    if window.is_conflict and window.conflict_target_id is not None:
        tgt = window.conflict_target_id
        if tgt in roles:
            conflict_target_role = roles[tgt]
        elif tgt in neighbor_ids:
            # target entered but didn't get a role → check at conflict frame
            conflict_target_role = "front"  # default for unresolved

    # 4. Collect persistent tracks for ego + all neighbors
    all_ids = neighbor_ids | {window.ego_id}
    if window.conflict_target_id is not None:
        all_ids.add(window.conflict_target_id)

    history_tracks = collect_persistent_tracks(
        raw_df,
        scene_id=scene,
        track_ids=all_ids,
        t_start=float(t0 - history_frames),
        t_end=float(t0),
    )
    future_tracks = collect_persistent_tracks(
        raw_df,
        scene_id=scene,
        track_ids=all_ids,
        t_start=float(t0),
        t_end=float(t0 + future_frames),
    )

    return TrackedNeighborhoodEvent(
        window=window,
        history_tracks=history_tracks,
        future_tracks=future_tracks,
        neighbor_roles=roles,
        conflict_target_role=conflict_target_role,
    )


def extract_neighbors_for_events(
    raw_df: pd.DataFrame,
    windows: List[WindowedEventCandidate],
    cfg: ProcessingConfig,
) -> List[TrackedNeighborhoodEvent]:
    """Run neighbor extraction for all windowed events."""
    return [extract_neighbors_for_event(raw_df, w, cfg) for w in windows]
