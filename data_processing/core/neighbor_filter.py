"""Stage 3: neighbor slot filtering and persistent trajectory collection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from data_processing.io.schema import (
    ProcessingConfig,
    TrackedNeighborhoodEvent,
    WindowedEventCandidate,
)
from data_processing.utils.geometry import (
    DEFAULT_LATERAL_THRESHOLD_M,
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
    *,
    frame_index: dict | None = None,
) -> pd.DataFrame:
    """Return all rows for a given *frame_num* (optionally filtered by scene)."""
    if frame_index is not None:
        frame_df = frame_index.get(frame_num)
        if frame_df is None or frame_df.empty:
            return pd.DataFrame()
        if "scene_id" in frame_df.columns:
            frame_df = frame_df[frame_df["scene_id"] == scene_id]
        return frame_df

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
    *,
    car_index: dict | None = None,
) -> pd.DataFrame:
    """Return all rows for *track_ids* within [t_start, t_end]."""
    if car_index is not None:
        parts: list[pd.DataFrame] = []
        for tid in track_ids:
            car_df = car_index.get(tid)
            if car_df is None or car_df.empty:
                continue
            car_df = car_df[
                (car_df["frameNum"] >= t_start)
                & (car_df["frameNum"] <= t_end)
            ]
            if "scene_id" in car_df.columns:
                car_df = car_df[car_df["scene_id"] == scene_id]
            if not car_df.empty:
                parts.append(car_df)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).sort_values(["carId", "frameNum"])

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
    *,
    frame_index: dict | None = None,
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
        frame_df = _frame_rows(raw_df, scene, fnum, frame_index=frame_index)
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
    *,
    frame_index: dict | None = None,
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
        frame_df = _frame_rows(raw_df, scene, fnum, frame_index=frame_index)
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
    car_index: dict | None = None,
) -> pd.DataFrame:
    """Continuously collect trajectories for *track_ids* over [t_start, t_end].

    Once a vehicle entered the neighborhood, keep collecting for the full
    interval (not only frames while inside the slot).
    """
    return _track_rows(
        raw_df, scene_id, track_ids, t_start, t_end, car_index=car_index,
    )


def _find_neighbors_and_assign_roles(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
    *,
    frame_index: dict | None = None,
) -> Tuple[Set, Dict]:
    """Single-pass combine of find_entered_neighbor_ids + assign_neighbor_roles.

    Iterates the history window **once**, collecting both the set of neighbor
    IDs and per-slot vote counts.  Per-frame geometry is vectorised with numpy
    so the Python-level loop only runs over vehicles that qualify.
    """
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)
    t_start = int(window.t0 - history_frames)
    t_end = int(window.t0)
    scene = window.scene_id
    ego_id = window.ego_id

    # nid → {slot: count}
    slot_votes: Dict[Any, Dict[str, int]] = {}
    # nid → (best_slot, best_dist)  — tiebreaker
    slot_best_dist: Dict[Any, Tuple[Optional[str], float]] = {}

    for fnum in range(t_start, t_end + 1):
        frame_df = _frame_rows(raw_df, scene, fnum, frame_index=frame_index)
        if frame_df.empty:
            continue

        ego_rows = frame_df[frame_df["carId"] == ego_id]
        if ego_rows.empty:
            continue
        ego = ego_rows.iloc[0]
        ego_x = float(ego["carCenterXm"])
        ego_y = float(ego["carCenterYm"])
        ego_heading = float(ego["heading"])

        # All other vehicles in this frame
        others = frame_df[frame_df["carId"] != ego_id]
        if others.empty:
            continue

        # --- vectorised geometry for the whole frame ---
        other_x = others["carCenterXm"].to_numpy(dtype=np.float64)
        other_y = others["carCenterYm"].to_numpy(dtype=np.float64)
        other_ids = others["carId"].values

        dx = other_x - ego_x
        dy = other_y - ego_y
        dists = np.hypot(dx, dy)

        # relative pose in ego frame
        theta = np.radians(ego_heading)
        long_arr = dx * np.cos(theta) + dy * np.sin(theta)   # longitudinal
        lat_arr = -dx * np.sin(theta) + dy * np.cos(theta)    # lateral

        # --- classify only vehicles within range ---
        for k in range(len(other_ids)):
            d = float(dists[k])
            if d > cfg.max_distance_m:
                continue

            slot = classify_neighbor_slot(
                float(long_arr[k]), float(lat_arr[k]), slots=cfg.neighbor_slots,
            )
            if slot is None:
                continue

            nid = other_ids[k]
            if nid not in slot_votes:
                slot_votes[nid] = {}
                slot_best_dist[nid] = (slot, d)
            slot_votes[nid][slot] = slot_votes[nid].get(slot, 0) + 1
            if d < slot_best_dist[nid][1]:
                slot_best_dist[nid] = (slot, d)

    # --- resolve roles: majority vote, tiebroken by closest frame ---
    neighbor_ids: Set = set(slot_votes.keys())
    roles: Dict[Any, str] = {}
    for nid in neighbor_ids:
        votes = slot_votes[nid]
        if not votes:
            roles[nid] = slot_best_dist.get(nid, (None,))[0] or "other"
        else:
            roles[nid] = max(votes, key=votes.get)  # type: ignore[arg-type]

    return neighbor_ids, roles


# ---------------------------------------------------------------------------
# Global frame-neighbor pre-computation cache
# ---------------------------------------------------------------------------

# Type alias for the cache: (scene_id, frame_num) → {ego_id: [(nbr_id, slot, dist), …]}
FrameNeighborCache = Dict[Tuple[str, int], Dict[Any, List[Tuple[Any, str, float]]]]


def _precompute_frame_neighbors(
    frame_index: dict,
    cfg: ProcessingConfig,
) -> FrameNeighborCache:
    """Pre-compute per-frame neighbor-slot assignments for ALL vehicles.

    Uses numpy broadcasting to compute pairwise geometry once per frame,
    then builds a lookup table.  Stage 3 can then answer "which vehicles
    are in ego's neighbor slots at frame *f*" in O(1) without recomputing
    distances / headings / slot classification.
    """
    from tqdm import tqdm

    cache: FrameNeighborCache = {}
    lat_thresh = DEFAULT_LATERAL_THRESHOLD_M
    slots_cfg = cfg.neighbor_slots

    # Flatten scene × frame into a single iterable so the progress bar
    # shows the true number of scene-frame units (not just unique frameNum,
    # which undercounts multi-scene datasets by N_scenes×).
    scene_frames: list[tuple[str, int, pd.DataFrame]] = []
    for fnum, frame_df in frame_index.items():
        if len(frame_df) < 2:
            continue
        if "scene_id" in frame_df.columns:
            for scene, scene_df in frame_df.groupby("scene_id"):
                scene_frames.append((str(scene), int(fnum), scene_df))
        else:
            scene_frames.append(("scene", int(fnum), frame_df))

    for scene, fnum, scene_df in tqdm(
        scene_frames,
        desc="  precomputing frame neighbors",
        unit="frm",
    ):
        scene_key = (str(scene), int(fnum))

        # Guard against rare duplicate carId-in-frame rows (NGSIM data
        # occasionally has multiple rows for the same vehicle in one frame).
        scene_df = scene_df.drop_duplicates(subset="carId", keep="first")

        n_s = len(scene_df)
        if n_s < 2:
            cache[scene_key] = {}
            continue

        x = scene_df["carCenterXm"].to_numpy(dtype=np.float64)
        y = scene_df["carCenterYm"].to_numpy(dtype=np.float64)
        headings = scene_df["heading"].to_numpy(dtype=np.float64)
        car_ids = scene_df["carId"].values

        # --- pairwise geometry (same pattern as _find_front_pairs) ---
        dx = x[np.newaxis, :] - x[:, np.newaxis]   # [i,j] = x_j - x_i
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dist = np.hypot(dx, dy)

        valid = (dist > 0.0) & (dist <= cfg.max_distance_m)
        if not np.any(valid):
            cache[scene_key] = {}
            continue

        theta = np.radians(headings)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        long_mat = dx * cos_t[:, np.newaxis] + dy * sin_t[:, np.newaxis]
        lat_mat = -dx * sin_t[:, np.newaxis] + dy * cos_t[:, np.newaxis]

        # --- classify every valid pair into a neighbor slot ---
        ego_idx, tgt_idx = np.where(valid)
        frame_map: Dict[Any, List[Tuple[Any, str, float]]] = {}

        for i, j in zip(ego_idx, tgt_idx):
            slot = classify_neighbor_slot(
                float(long_mat[i, j]), float(lat_mat[i, j]),
                lateral_threshold=lat_thresh, slots=slots_cfg,
            )
            if slot is None:
                continue
            ego_id = car_ids[i]
            nbr_id = car_ids[j]
            d = float(dist[i, j])
            frame_map.setdefault(ego_id, []).append((nbr_id, slot, d))

        cache[scene_key] = frame_map

    return cache


def _find_neighbors_from_cache(
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
    cache: FrameNeighborCache,
) -> Tuple[Set, Dict]:
    """Look up pre-computed frame data to find neighbors and assign roles.

    This is the fast-path replacement for ``_find_neighbors_and_assign_roles``
    — no geometry, just dict lookups and vote counting.
    """
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)
    t_start = int(window.t0 - history_frames)
    t_end = int(window.t0)
    scene = window.scene_id
    ego_id = window.ego_id

    slot_votes: Dict[Any, Dict[str, int]] = {}
    slot_best_dist: Dict[Any, Tuple[Optional[str], float]] = {}

    for fnum in range(t_start, t_end + 1):
        frame_map = cache.get((scene, fnum))
        if frame_map is None:
            continue
        ego_neighbors = frame_map.get(ego_id)
        if ego_neighbors is None:
            continue

        for nbr_id, slot, dist in ego_neighbors:
            if nbr_id not in slot_votes:
                slot_votes[nbr_id] = {}
                slot_best_dist[nbr_id] = (slot, dist)
            slot_votes[nbr_id][slot] = slot_votes[nbr_id].get(slot, 0) + 1
            if dist < slot_best_dist[nbr_id][1]:
                slot_best_dist[nbr_id] = (slot, dist)

    neighbor_ids: Set = set(slot_votes.keys())
    roles: Dict[Any, str] = {}
    for nid in neighbor_ids:
        votes = slot_votes[nid]
        if not votes:
            roles[nid] = slot_best_dist.get(nid, (None,))[0] or "other"
        else:
            roles[nid] = max(votes, key=votes.get)  # type: ignore[arg-type]

    return neighbor_ids, roles


def extract_neighbors_for_event(
    raw_df: pd.DataFrame,
    window: WindowedEventCandidate,
    cfg: ProcessingConfig,
    *,
    frame_index: dict | None = None,
    car_index: dict | None = None,
    neighbor_cache: FrameNeighborCache | None = None,
) -> TrackedNeighborhoodEvent:
    """Full neighbor pipeline for a single windowed event.

    When *neighbor_cache* is supplied the geometry-heavy frame scans are
    replaced with O(1) cache lookups.
    """
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)
    future_frames = _sec2frames(cfg.future_sec, cfg.fps)
    scene = window.scene_id
    t0 = window.t0

    # 1 + 2. Find neighbors and assign roles (cache-assisted fast path)
    if neighbor_cache is not None:
        neighbor_ids, roles = _find_neighbors_from_cache(
            window, cfg, neighbor_cache,
        )
    else:
        neighbor_ids, roles = _find_neighbors_and_assign_roles(
            raw_df, window, cfg, frame_index=frame_index,
        )

    # 3. Determine conflict target role (if applicable)
    conflict_target_role: Optional[str] = None  # noqa: F821
    if window.is_conflict and window.conflict_target_id is not None:
        tgt = window.conflict_target_id
        if tgt in roles:
            conflict_target_role = roles[tgt]
        elif tgt in neighbor_ids:
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
        car_index=car_index,
    )
    future_tracks = collect_persistent_tracks(
        raw_df,
        scene_id=scene,
        track_ids=all_ids,
        t_start=float(t0),
        t_end=float(t0 + future_frames),
        car_index=car_index,
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
    *,
    frame_index: dict | None = None,
    car_index: dict | None = None,
    neighbor_cache: dict | None = None,
) -> List[TrackedNeighborhoodEvent]:
    """Run neighbor extraction for all windowed events.

    Pre-computes a global frame-neighbor cache so that the per-event loop
    only does O(1) dict lookups instead of recomputing pairwise geometry.

    Pass *neighbor_cache* to reuse a previously computed cache across
    multiple calls (e.g. chunked processing).
    """
    from tqdm import tqdm

    if frame_index is None:
        # Fallback: no pre-computation possible
        results: List[TrackedNeighborhoodEvent] = []
        for w in tqdm(windows, desc="  extracting neighbors", unit="evt"):
            results.append(extract_neighbors_for_event(
                raw_df, w, cfg,
                frame_index=frame_index, car_index=car_index,
            ))
        return results

    # --- Pre-compute once (or reuse), then process every event ---
    if neighbor_cache is None:
        neighbor_cache = _precompute_frame_neighbors(frame_index, cfg)

    results: List[TrackedNeighborhoodEvent] = []
    for w in tqdm(windows, desc="  extracting neighbors", unit="evt"):
        results.append(extract_neighbors_for_event(
            raw_df, w, cfg,
            frame_index=frame_index, car_index=car_index,
            neighbor_cache=neighbor_cache,
        ))
    return results
