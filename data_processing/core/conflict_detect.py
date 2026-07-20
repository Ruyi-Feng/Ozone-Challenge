"""Stage 1: detect conflict and non-conflict sample candidates.

Conflict detection is based on 2D_TTC (Time-To-Collision) computed from
OBB (Oriented Bounding Box) geometry.  A pair (ego, target) is in conflict
when 2D_TTC drops below *conflict_ttc_threshold* for one or more consecutive
frames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from data_processing.io.schema import ConflictCandidate, ProcessingConfig
from data_processing.utils.geometry import (
    classify_neighbor_slot,
    compute_distance,
    compute_relative_pose,
)

# ---------------------------------------------------------------------------
# 2D_TTC helpers
# ---------------------------------------------------------------------------

# Per-frame angular velocity cache: carId -> (prev_heading_rad, dt)
_VELOCITY_CACHE: Dict[Any, Tuple[float, float]] = {}


def _speed_vector(speed: float, heading_rad: float) -> Tuple[float, float]:
    """Return (vx, vy) in m/s from scalar speed and heading (radians)."""
    return (speed * math.sin(heading_rad), speed * math.cos(heading_rad))


def _angular_velocity(
    car_id: Any, heading_rad: float, dt: float
) -> float:
    """Compute angular velocity ω (rad/s) from consecutive heading readings."""
    prev = _VELOCITY_CACHE.get(car_id)
    _VELOCITY_CACHE[car_id] = (heading_rad, dt)
    if prev is None:
        return 0.0
    prev_heading, _ = prev
    return (heading_rad - prev_heading) / dt


def _clear_velocity_cache() -> None:
    _VELOCITY_CACHE.clear()


def _get_obb_corners(row: pd.Series) -> List[Tuple[float, float]]:
    """Extract OBB corners in meters from a raw-data row.

    Corner ordering (observed in NBDT / CitySim data):
      0  front-right
      1  front-left
      2  rear-left
      3  rear-right
    """
    return [
        (row["boundingBox1Xm"], row["boundingBox1Ym"]),
        (row["boundingBox2Xm"], row["boundingBox2Ym"]),
        (row["boundingBox3Xm"], row["boundingBox3Ym"]),
        (row["boundingBox4Xm"], row["boundingBox4Ym"]),
    ]


def _point_to_segment_closest(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    """Closest point on segment AB to point P, and squared distance."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    t = (apx * abx + apy * aby) / max(abx * abx + aby * aby, 1e-16)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    d2 = (px - cx) ** 2 + (py - cy) ** 2
    return cx, cy, d2


def _obb_closest_points(
    corners_a: List[Tuple[float, float]],
    corners_b: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Find the closest pair of points between two OBB rectangles.

    Returns
    -------
    (p_a, p_b, distance)  where p_a is on OBB A, p_b on OBB B.
    """
    best = (None, None, float("inf"))  # type: ignore

    for corners_this, corners_other in [(corners_a, corners_b), (corners_b, corners_a)]:
        n = len(corners_this)
        for i in range(n):
            ax, ay = corners_this[i]
            bx, by = corners_this[(i + 1) % n]
            for cx, cy in corners_other:
                px, py, d2 = _point_to_segment_closest(cx, cy, ax, ay, bx, by)
                if d2 < best[2]:
                    best = ((ax, ay) if corners_this is corners_a else (cx, cy),
                            (cx, cy) if corners_this is corners_a else (ax, ay),
                            d2)

    p_a, p_b, d2 = best
    return (p_a, p_b, math.sqrt(d2))  # type: ignore[return-value]


def _is_front_to_rear_contact(
    p_ego: Tuple[float, float],
    ego_corners: List[Tuple[float, float]],
    ego_heading: float,
    p_tgt: Tuple[float, float],
    tgt_corners: List[Tuple[float, float]],
    tgt_heading: float,
) -> bool:
    """Check whether the closest-approach pair involves ego's front half
    and target's rear half.

    Projects each closest point onto its vehicle's heading axis and compares
    with the OBB longitudinal midpoint.  This replaces the NBDT strip-
    intersection check which fails when vehicle headings are (near-)parallel
    — the most common rear-end scenario.
    """
    ego_hx = math.sin(ego_heading)
    ego_hy = math.cos(ego_heading)
    tgt_hx = math.sin(tgt_heading)
    tgt_hy = math.cos(tgt_heading)

    ego_proj = p_ego[0] * ego_hx + p_ego[1] * ego_hy
    tgt_proj = p_tgt[0] * tgt_hx + p_tgt[1] * tgt_hy

    ego_cp = [c[0] * ego_hx + c[1] * ego_hy for c in ego_corners]
    tgt_cp = [c[0] * tgt_hx + c[1] * tgt_hy for c in tgt_corners]

    ego_mid = (min(ego_cp) + max(ego_cp)) / 2.0
    tgt_mid = (min(tgt_cp) + max(tgt_cp)) / 2.0

    # ego closest point on front half, target closest point on rear half
    return ego_proj >= ego_mid - 1e-6 and tgt_proj <= tgt_mid + 1e-6


def _compute_2d_ttc(
    ego_row: pd.Series,
    target_row: pd.Series,
    dt: float,
) -> Optional[float]:
    """Compute 2D_TTC for an ego-target pair at a single frame.

    Algorithm (matching NBDT *ssm.py* `_compute_2d_ttc_kernel`):

    1. Find closest points *p_ego*, *p_target* on the two OBBs.
    2. Check whether the rear strip of ego intersects the forward strip of target
       (required for a rear-end collision geometry).
    3. Project velocity vectors onto the closing direction *p_target → p_ego*.
    4. Add angular velocity correction.
    5. TTC = distance / effective_closing_speed  (zero-acceleration model).

    Returns
    -------
    TTC in seconds, or None when no collision geometry / vehicles are separating.
    """
    ego_corners = _get_obb_corners(ego_row)
    tgt_corners = _get_obb_corners(target_row)

    # 1. Closest points
    p_ego, p_tgt, dist = _obb_closest_points(ego_corners, tgt_corners)
    if dist < 1e-6:
        return 0.0  # already colliding

    # 2. Front-rear contact check
    #    ego is the *following* vehicle (behind), target is *ahead*.
    #    Verify that the closest OBB points involve ego's front half
    #    and target's rear half — works for both parallel and angled headings.
    ego_hr = math.radians(ego_row["heading"])
    tgt_hr = math.radians(target_row["heading"])
    if not _is_front_to_rear_contact(
        p_ego, ego_corners, ego_hr,
        p_tgt, tgt_corners, tgt_hr,
    ):
        return None  # no front→rear collision geometry

    # 3. Closing speed along p_ego → p_tgt direction
    ego_vx, ego_vy = _speed_vector(ego_row["speed"], ego_hr)
    tgt_vx, tgt_vy = _speed_vector(target_row["speed"], tgt_hr)

    # Direction unit vector from ego to target (closing direction)
    cx = p_tgt[0] - p_ego[0]
    cy = p_tgt[1] - p_ego[1]
    norm = math.hypot(cx, cy)
    if norm < 1e-6:
        return None
    ux, uy = cx / norm, cy / norm

    v_ego_proj = ego_vx * ux + ego_vy * uy
    v_tgt_proj = tgt_vx * ux + tgt_vy * uy
    v_rel = v_ego_proj - v_tgt_proj  # positive → ego closing on target

    # 4. Angular velocity correction
    w_ego = _angular_velocity(ego_row["carId"], ego_hr, dt)
    w_tgt = _angular_velocity(target_row["carId"], tgt_hr, dt)
    v_corr = (w_ego + w_tgt) * dist  # angular contribution

    closing = v_rel + v_corr

    if closing <= 0:
        return None  # separating or static

    # 5. TTC under zero-acceleration assumption
    return dist / closing


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

_FRONT_SLOTS = {"front", "left_front", "right_front"}


def _find_front_pairs(
    frame_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[Tuple[Any, Any]]:
    """For one frame, return (ego_id, target_id) pairs where target is ahead of ego.

    A pair is formed when *target* falls into one of the front slots
    (front / left_front / right_front) relative to *ego*.
    """
    pairs: List[Tuple[Any, Any]] = []
    rows = frame_df.to_dict("records")
    n = len(rows)
    for i in range(n):
        ego = rows[i]
        for j in range(n):
            if i == j:
                continue
            target = rows[j]
            dist = compute_distance(
                ego["carCenterXm"], ego["carCenterYm"],
                target["carCenterXm"], target["carCenterYm"],
            )
            if not (dist <= cfg.max_distance_m):
                continue
            dx, dy = compute_relative_pose(
                ego["carCenterXm"], ego["carCenterYm"], ego["heading"],
                target["carCenterXm"], target["carCenterYm"],
            )
            slot = classify_neighbor_slot(dx, dy, slots=cfg.neighbor_slots)
            if slot in _FRONT_SLOTS:
                pairs.append((ego["carId"], target["carId"]))
    return pairs


# ---------------------------------------------------------------------------
# Conflict extraction
# ---------------------------------------------------------------------------

def _group_into_events(
    conflict_frames: List[int],
    min_duration_frames: int = 1,
) -> List[Dict[str, Any]]:
    """Group consecutive conflict frames into events.

    Returns list of dicts with keys: *start_frame*, *end_frame*, *t_conflict*.
    *t_conflict* is the middle frame of the event.
    """
    if not conflict_frames:
        return []
    events: List[Dict[str, Any]] = []
    batch = [conflict_frames[0]]
    for f in conflict_frames[1:]:
        if f == batch[-1] + 1:
            batch.append(f)
        else:
            if len(batch) >= min_duration_frames:
                events.append({
                    "start_frame": batch[0],
                    "end_frame": batch[-1],
                    "t_conflict": float(batch[len(batch) // 2]),
                })
            batch = [f]
    if len(batch) >= min_duration_frames:
        events.append({
            "start_frame": batch[0],
            "end_frame": batch[-1],
            "t_conflict": float(batch[len(batch) // 2]),
        })
    return events


# ---------------------------------------------------------------------------
# OBB column guard
# ---------------------------------------------------------------------------

def _has_obb_data(df: pd.DataFrame) -> bool:
    """Return True when all 8 OBB meter columns are present."""
    from data_processing.io.schema import RAW_OBB_COLUMNS

    return all(c in df.columns for c in RAW_OBB_COLUMNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_conflicts(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[ConflictCandidate]:
    """Detect conflict events from standardized trajectories.

    Workflow
    --------
    1. Group raw data by *frameNum*.
    2. Per frame, build (ego, target) pairs where target is in a front slot.
    3. For each pair, compute 2D_TTC using OBB geometry.
    4. When 2D_TTC < *conflict_ttc_threshold*, record a conflict frame.
    5. Group consecutive conflict frames into events and emit ConflictCandidate.

    Returns
    -------
    List[ConflictCandidate] with ``is_conflict=True``.
    """
    _clear_velocity_cache()
    dt = 1.0 / cfg.fps

    if not _has_obb_data(raw_df):
        raise ValueError(
            "OBB corner columns (boundingBox1Xm..4Ym) missing from raw data. "
            "2D_TTC requires OBB geometry."
        )

    # Derive scene_id from dataframe if present, else use fallback
    scene_id: str = "scene"
    if "scene_id" in raw_df.columns:
        vals = raw_df["scene_id"].unique()
        scene_id = str(vals[0]) if len(vals) > 0 else "scene"

    # Sort by frame
    df = raw_df.sort_values(["frameNum", "carId"]).reset_index(drop=True)
    frames = df.groupby("frameNum")

    # Per-pair 2D_TTC time series:  (ego, target) → [(frameNum, 2D_TTC), ...]
    pair_series: Dict[Tuple[Any, Any], List[Tuple[int, Optional[float]]]] = {}

    for frame_num, frame_df in frames:
        front_pairs = _find_front_pairs(frame_df, cfg)
        for ego_id, tgt_id in front_pairs:
            key = (ego_id, tgt_id)
            try:
                ego_row = frame_df.loc[frame_df["carId"] == ego_id].iloc[0]
                tgt_row = frame_df.loc[frame_df["carId"] == tgt_id].iloc[0]
            except IndexError:
                continue
            ttc = _compute_2d_ttc(ego_row, tgt_row, dt)
            pair_series.setdefault(key, []).append((int(frame_num), ttc))

    # Extract continuous conflict intervals
    candidates: List[ConflictCandidate] = []
    for (ego_id, tgt_id), records in pair_series.items():
        conflict_frames = [
            fnum for fnum, ttc in records
            if ttc is not None and ttc < cfg.conflict_ttc_threshold
        ]
        events = _group_into_events(conflict_frames)
        for evt in events:
            candidates.append(ConflictCandidate(
                scene_id=scene_id,
                ego_id=ego_id,
                is_conflict=True,
                t_conflict=evt["t_conflict"],
                conflict_target_id=tgt_id,
                meta={
                    "start_frame": evt["start_frame"],
                    "end_frame": evt["end_frame"],
                },
            ))

    return candidates


def sample_non_conflicts(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
    *,
    exclude: Optional[List[ConflictCandidate]] = None,
) -> List[ConflictCandidate]:
    """Sample non-conflict candidates under the same windowing assumptions.

    Strategy
    --------
    For each ego vehicle, randomly sample *t0* points in safe intervals
    (where no conflict is active for that ego).  We sample roughly the same
    number of non-conflict candidates as there are conflict candidates for
    the same ego, clamped by ``max_non_conflict_per_ego``.

    *exclude* lists existing conflict candidates; their ego / time
    neighbourhoods are avoided.
    """
    _clear_velocity_cache()
    dt = 1.0 / cfg.fps
    scene_id: str = "scene"
    if "scene_id" in raw_df.columns:
        vals = raw_df["scene_id"].unique()
        scene_id = str(vals[0]) if len(vals) > 0 else "scene"

    # Build exclusion set: (ego_id, frame) pairs near known conflicts
    exclude_set: Set[Tuple[Any, int]] = set()
    conflict_ego_frames: Dict[Any, Set[int]] = {}
    if exclude is not None:
        margin_frames = int(cfg.history_sec * cfg.fps)
        for c in exclude:
            t_c = int(c.t_conflict) if c.t_conflict is not None else -1
            conflict_ego_frames.setdefault(c.ego_id, set()).add(t_c)
            for offset in range(-margin_frames, margin_frames + 1):
                exclude_set.add((c.ego_id, t_c + offset))

    df = raw_df.sort_values(["frameNum", "carId"]).reset_index(drop=True)
    ego_ids = df["carId"].unique()
    all_frames = sorted(df["frameNum"].unique())

    # Count how many conflict samples per ego (to match roughly)
    conflict_count: Dict[Any, int] = {}
    if exclude is not None:
        for c in exclude:
            conflict_count[c.ego_id] = conflict_count.get(c.ego_id, 0) + 1

    non_conflicts: List[ConflictCandidate] = []
    max_per_ego = 20  # cap to avoid explosion

    for ego_id in ego_ids:
        ego_mask = df["carId"] == ego_id
        ego_frames = sorted(df.loc[ego_mask, "frameNum"].unique())
        if len(ego_frames) < 2:
            continue

        n_target = min(
            conflict_count.get(ego_id, max(3, len(ego_frames) // 100)),
            max_per_ego,
        )
        safe_frames = [
            f for f in ego_frames
            if (ego_id, f) not in exclude_set
        ]
        if len(safe_frames) < n_target:
            continue

        # Simple random sampling (fixed seed for reproducibility)
        import random
        rng = random.Random(hash(ego_id) & 0x7FFFFFFF)
        sampled = rng.sample(safe_frames, n_target)
        for fnum in sampled:
            non_conflicts.append(ConflictCandidate(
                scene_id=scene_id,
                ego_id=ego_id,
                is_conflict=False,
                t_conflict=None,
                conflict_target_id=None,
                meta={"sample_frame": fnum},
            ))

    return non_conflicts


def detect_all_candidates(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[ConflictCandidate]:
    """Run conflict detection then non-conflict sampling; return merged candidate list."""
    conflicts = detect_conflicts(raw_df, cfg)
    non_conflicts = sample_non_conflicts(raw_df, cfg, exclude=conflicts)
    return conflicts + non_conflicts
