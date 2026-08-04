"""Stage 1: detect conflict and non-conflict sample candidates.

Conflict detection is based on 2D_TTC (Time-To-Collision) computed from
OBB (Oriented Bounding Box) geometry, following the NBDT *ssm.py*
reference implementation (``_compute_2d_ttc_bbox`` and helpers).

A pair (ego, target) is in conflict when 2D_TTC drops below
*conflict_ttc_threshold* for one or more consecutive frames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from data_processing.io.schema import ConflictCandidate, ProcessingConfig

# ---------------------------------------------------------------------------
# Per-frame caches (cleared per detect_conflicts / sample_non_conflicts call)
# ---------------------------------------------------------------------------

# carId -> (prev_heading_deg, dt)   for angular velocity (deg/s)
_VELOCITY_CACHE: Dict[Any, Tuple[float, float]] = {}

# carId -> (prev_speed_m_s, dt)     for scalar acceleration (m/s²)
_ACCEL_CACHE: Dict[Any, Tuple[float, float]] = {}

_EPS = 1e-6

# ---------------------------------------------------------------------------
# Geometry helpers — aligned with NBDT GeometryHelper
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


def _order_rect_points(points: List[Point]) -> List[Point]:
    """Sort 4 OBB corners into counter-clockwise order (NBDT GeometryHelper)."""
    cx = sum(p[0] for p in points) / 4.0
    cy = sum(p[1] for p in points) / 4.0
    sorted_pts = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    a, b, c, d = sorted_pts
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross > 0:
        return [a, d, c, b]
    return sorted_pts


def _line_segment_intersection(
    a1: Point, a2: Point, b1: Point, b2: Point,
) -> Optional[Point]:
    """Intersection of two line segments.  None when parallel or disjoint."""
    dx1, dy1 = a2[0] - a1[0], a2[1] - a1[1]
    dx2, dy2 = b2[0] - b1[0], b2[1] - b1[1]
    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0:
        return None
    dx = b1[0] - a1[0]
    dy = b1[1] - a1[1]
    s = (dx * dy2 - dy * dx2) / denom
    t = (dx * dy1 - dy * dx1) / denom
    if 0 <= s <= 1 and 0 <= t <= 1:
        return (a1[0] + s * dx1, a1[1] + s * dy1)
    return None


def _is_inside_rect(point: Point, rect: List[Point]) -> bool:
    """Point-in-convex-polygon via half-plane test (edges CCW)."""
    for i in range(4):
        ax, ay = rect[i]
        bx, by = rect[(i + 1) % 4]
        edge_dx = bx - ax
        edge_dy = by - ay
        # outward normal (rotate edge 90° CW for CCW polygon → inward)
        nx, ny = edge_dy, -edge_dx
        ap_x = point[0] - ax
        ap_y = point[1] - ay
        if nx * ap_x + ny * ap_y < 0:
            return False
    return True


def _rectangles_intersect(rect1: List[Point], rect2: List[Point]) -> bool:
    """Separating Axis Theorem for two convex quads."""
    axes: List[Tuple[float, float]] = []
    for rect in (rect1, rect2):
        for i in range(4):
            ax, ay = rect[i]
            bx, by = rect[(i + 1) % 4]
            ex, ey = bx - ax, by - ay
            length = math.hypot(ex, ey)
            if length == 0:
                continue
            axes.append((ey / length, -ex / length))

    for nx, ny in axes:
        def _proj(r):
            return [p[0] * nx + p[1] * ny for p in r]
        p1 = _proj(rect1)
        p2 = _proj(rect2)
        if max(p1) < min(p2) or max(p2) < min(p1):
            return False
    return True


def _calculate_nearest_points_ordered(
    rect1: List[Point], rect2: List[Point],
) -> Tuple[Point, Point, float]:
    """NBDT ``_calculate_nearest_points_ordered`` — OBB closest points + distance.

    Returns (p_on_rect1, p_on_rect2, distance_m).
    """
    # --- Intersecting case ---
    if _rectangles_intersect(rect1, rect2):
        for p in rect1:
            if _is_inside_rect(p, rect2):
                return (p, p, 0.0)
        for p in rect2:
            if _is_inside_rect(p, rect1):
                return (p, p, 0.0)
        for i in range(4):
            a1, a2 = rect1[i], rect1[(i + 1) % 4]
            for j in range(4):
                b1, b2 = rect2[j], rect2[(j + 1) % 4]
                hit = _line_segment_intersection(a1, a2, b1, b2)
                if hit:
                    return (hit, hit, 0.0)
        c1 = (sum(p[0] for p in rect1) / 4.0, sum(p[1] for p in rect1) / 4.0)
        c2 = (sum(p[0] for p in rect2) / 4.0, sum(p[1] for p in rect2) / 4.0)
        return (c1, c2, 0.0)

    # --- Separated case: vertex→edge ---
    min_dist = float("inf")
    best_pa: Point = (0.0, 0.0)
    best_pb: Point = (0.0, 0.0)

    for _pass, (r_vert, r_edge) in enumerate(
        [(rect1, rect2), (rect2, rect1)]
    ):
        for vi in range(4):
            px, py = r_vert[vi]
            for ej in range(4):
                ax, ay = r_edge[ej]
                bx, by = r_edge[(ej + 1) & 3]
                ex, ey = bx - ax, by - ay
                seg_len_sq = ex * ex + ey * ey
                if seg_len_sq == 0.0:
                    cx, cy = ax, ay
                else:
                    t = max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / seg_len_sq))
                    cx, cy = ax + t * ex, ay + t * ey
                d = math.hypot(px - cx, py - cy)
                if d < min_dist:
                    min_dist = d
                    if _pass == 0:
                        best_pa, best_pb = (px, py), (cx, cy)
                    else:
                        best_pa, best_pb = (cx, cy), (px, py)
    return best_pa, best_pb, min_dist


def _calculate_nearest_points(
    veh1: List[Point], veh2: List[Point],
) -> Tuple[Point, Point, float]:
    """NBDT ``calculate_nearest_points`` — order corners then delegate."""
    return _calculate_nearest_points_ordered(
        _order_rect_points(veh1), _order_rect_points(veh2),
    )


# ---------------------------------------------------------------------------
# Rear-forward strip intersection — NBDT InstantSSMCalculator helpers
# ---------------------------------------------------------------------------

def _rear_edge_from_ordered_bbox(
    ordered_bbox: List[Point], heading_deg: float,
) -> Tuple[Point, Point]:
    """Return the rear edge (two vertices) of a CCW-ordered OBB.

    The rear edge is the one whose midpoint projects *least* along heading.
    """
    rad = math.radians(heading_deg)
    fx, fy = math.cos(rad), math.sin(rad)
    cx = sum(p[0] for p in ordered_bbox) / 4.0
    cy = sum(p[1] for p in ordered_bbox) / 4.0
    best = None
    min_proj = float("inf")
    n = len(ordered_bbox)
    for i in range(n):
        ax, ay = ordered_bbox[i]
        bx, by = ordered_bbox[(i + 1) % n]
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        proj = (mx - cx) * fx + (my - cy) * fy
        if proj < min_proj:
            min_proj = proj
            best = ((ax, ay), (bx, by))
    if best is None:
        return ordered_bbox[0], ordered_bbox[1]
    return best[0], best[1]


def _point_in_forward_strip(
    q: Point, r0: Point, r1: Point, dx: float, dy: float,
) -> bool:
    """Is point *q* inside the forward strip from rear edge [r0,r1] along (dx,dy)?"""
    wx, wy = r1[0] - r0[0], r1[1] - r0[1]
    det = wx * dy - wy * dx
    if abs(det) < 1e-12:
        return False
    vx, vy = q[0] - r0[0], q[1] - r0[1]
    a = (vx * dy - vy * dx) / det
    t = (wx * vy - wy * vx) / det
    return -1e-9 <= a <= 1.0 + 1e-9 and t >= -1e-9


def _ray_intersect_segment(
    ox: float, oy: float, dx: float, dy: float,
    ax: float, ay: float, bx: float, by: float,
) -> bool:
    """Ray from (ox,oy) along (dx,dy) hits segment (a,b)?"""
    wx, wy = bx - ax, by - ay
    cross_dw = dx * wy - dy * wx
    if abs(cross_dw) < 1e-12:
        return False
    t = ((ax - ox) * wy - (ay - oy) * wx) / cross_dw
    u = ((ax - ox) * dy - (ay - oy) * dx) / cross_dw
    return t >= -1e-9 and 0.0 <= u <= 1.0


def _two_rays_intersect_forward(
    o1x: float, o1y: float, d1x: float, d1y: float,
    o2x: float, o2y: float, d2x: float, d2y: float,
) -> bool:
    """Two forward rays intersect (both t ≥ 0)?"""
    cross_d = d1x * d2y - d1y * d2x
    if abs(cross_d) < 1e-12:
        return False
    dx, dy = o2x - o1x, o2y - o1y
    t = (dx * d2y - dy * d2x) / cross_d
    s = (dx * d1y - dy * d1x) / cross_d
    return t >= -1e-9 and s >= -1e-9


def _rear_forward_strips_intersect(
    veh1_corners: List[Point],
    veh2_corners: List[Point],
    heading1_deg: float,
    heading2_deg: float,
) -> bool:
    """NBDT ``_rear_forward_strips_intersect``.

    Checks whether the forward strip emanating from the rear edge of either
    vehicle intersects the other vehicle — the geometric precondition for
    a potential rear-end collision.

    Four independent checks (any True → intersect):
    1. Rear-edge vertex inside other vehicle's forward strip
    2. Forward ray from rear-edge vertex hits other vehicle's rear-edge segment
    3. Two forward rays (one from each vehicle's rear-edge vertex) intersect
    4. The two rear-edge segments directly intersect
    """
    o1 = _order_rect_points(veh1_corners)
    o2 = _order_rect_points(veh2_corners)

    r0a, r1a = _rear_edge_from_ordered_bbox(o1, heading1_deg)
    r0b, r1b = _rear_edge_from_ordered_bbox(o2, heading2_deg)

    t1 = math.radians(heading1_deg)
    t2 = math.radians(heading2_deg)
    dax, day = math.cos(t1), math.sin(t1)
    dbx, dby = math.cos(t2), math.sin(t2)

    # 1. Vertex-in-strip
    if _point_in_forward_strip(r0a, r0b, r1b, dbx, dby):
        return True
    if _point_in_forward_strip(r1a, r0b, r1b, dbx, dby):
        return True
    if _point_in_forward_strip(r0b, r0a, r1a, dax, day):
        return True
    if _point_in_forward_strip(r1b, r0a, r1a, dax, day):
        return True

    # 2. Ray↔segment
    for ox, oy in (r0a, r1a):
        if _ray_intersect_segment(ox, oy, dax, day, r0b[0], r0b[1], r1b[0], r1b[1]):
            return True
        # 3. Two rays
        if _two_rays_intersect_forward(ox, oy, dax, day, r0b[0], r0b[1], dbx, dby):
            return True
        if _two_rays_intersect_forward(ox, oy, dax, day, r1b[0], r1b[1], dbx, dby):
            return True
    for ox, oy in (r0b, r1b):
        if _ray_intersect_segment(ox, oy, dbx, dby, r0a[0], r0a[1], r1a[0], r1a[1]):
            return True
        if _two_rays_intersect_forward(ox, oy, dbx, dby, r0a[0], r0a[1], dax, day):
            return True
        if _two_rays_intersect_forward(ox, oy, dbx, dby, r1a[0], r1a[1], dax, day):
            return True

    # 4. Direct segment intersection
    return _line_segment_intersection(r0a, r1a, r0b, r1b) is not None


# ---------------------------------------------------------------------------
# 2D_TTC kernel — NBDT _compute_2d_ttc_kernel + _project_to_line
# ---------------------------------------------------------------------------

def _project_to_line(
    speed: float, accel: float, heading_deg: float, alpha: float,
) -> Tuple[float, float]:
    """Project scalar speed & acceleration onto direction *alpha* (radians)."""
    delta = math.radians(heading_deg) - alpha
    return speed * math.cos(delta), accel * math.cos(delta)


def _compute_2d_ttc_kernel(
    point1: Point,
    point2: Point,
    distance: float,
    param1: Tuple[float, float, float, float],  # (speed, accel, heading_deg, ω_deg_s)
    param2: Tuple[float, float, float, float],
) -> Optional[float]:
    """NBDT ``_compute_2d_ttc_kernel`` — closed-form 2D_TTC.

    Solves  distance = closing·t + ½·a_rel·t²  for the smallest positive root.
    Returns None when vehicles are not closing.
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    alpha = math.atan2(dy, dx)

    v1, a1 = _project_to_line(param1[0], param1[1], param1[2], alpha)
    v2, a2 = _project_to_line(param2[0], param2[1], param2[2], alpha)
    v_rel = v1 - v2
    a_rel = a1 - a2

    # Angular velocity contribution (deg/s → rad/s)
    w = math.radians(param1[3] + param2[3])
    closing = v_rel + w * distance

    if closing <= _EPS and a_rel <= _EPS:
        return None

    if abs(a_rel) < _EPS:
        if closing <= 0:
            return None
        t = distance / closing
        return t if t > 0 else None

    disc = closing**2 + 2 * a_rel * distance
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-closing - sqrt_disc) / a_rel
    t2 = (-closing + sqrt_disc) / a_rel
    candidates = [t for t in (t1, t2) if t > _EPS]
    if not candidates:
        return None
    return min(candidates)


# ---------------------------------------------------------------------------
# Row-level helpers
# ---------------------------------------------------------------------------

def _get_obb_corners(row: pd.Series) -> List[Point]:
    """Extract OBB corners in meters from a raw-data row.

    Corner ordering (NBDT / CitySim data):
      0  front-right
      1  front-left
      2  rear-left
      3  rear-right
    """
    return [
        (float(row["boundingBox1Xm"]), float(row["boundingBox1Ym"])),
        (float(row["boundingBox2Xm"]), float(row["boundingBox2Ym"])),
        (float(row["boundingBox3Xm"]), float(row["boundingBox3Ym"])),
        (float(row["boundingBox4Xm"]), float(row["boundingBox4Ym"])),
    ]


def _scalar_acceleration(car_id: Any, speed: float, dt: float) -> float:
    """Compute scalar acceleration (m/s²) from consecutive speed readings."""
    prev = _ACCEL_CACHE.get(car_id)
    _ACCEL_CACHE[car_id] = (speed, dt)
    if prev is None:
        return 0.0
    prev_speed, _ = prev
    return (speed - prev_speed) / dt


def _angular_velocity(car_id: Any, heading_deg: float, dt: float) -> float:
    """Compute angular velocity (deg/s) from consecutive heading readings."""
    prev = _VELOCITY_CACHE.get(car_id)
    _VELOCITY_CACHE[car_id] = (heading_deg, dt)
    if prev is None:
        return 0.0
    prev_heading, _ = prev
    diff = (heading_deg - prev_heading + 180.0) % 360.0 - 180.0
    return diff / dt  # deg/s


def _clear_caches() -> None:
    _VELOCITY_CACHE.clear()
    _ACCEL_CACHE.clear()


# ---------------------------------------------------------------------------
# 2D_TTC entry point
# ---------------------------------------------------------------------------

def _compute_2d_ttc(
    ego_row: pd.Series,
    target_row: pd.Series,
    dt: float,
) -> Optional[float]:
    """Compute 2D_TTC for an ego-target pair at a single frame.

    Algorithm (NBDT ``_compute_2d_ttc_bbox``):
    1. ``calculate_nearest_points`` → p_ego, p_tgt, distance
    2. ``_rear_forward_strips_intersect`` → geometric precondition
    3. ``_compute_2d_ttc_kernel`` → quadratic TTC

    Returns TTC in seconds, or None.
    """
    ego_corners = _get_obb_corners(ego_row)
    tgt_corners = _get_obb_corners(target_row)

    # 1. Nearest points + distance
    p_ego, p_tgt, dist = _calculate_nearest_points(ego_corners, tgt_corners)
    if dist <= 0:
        return 0.0  # already colliding / intersecting

    # 2. Rear-forward strip intersection (NBDT standard)
    ego_hdg = float(ego_row["heading"])
    tgt_hdg = float(target_row["heading"])
    if not _rear_forward_strips_intersect(ego_corners, tgt_corners, ego_hdg, tgt_hdg):
        return None

    # 3. Build params: (speed, accel, heading_deg, angular_velocity_deg_per_dt)
    ego_speed = float(ego_row["speed"])
    tgt_speed = float(target_row["speed"])
    ego_accel = _scalar_acceleration(ego_row["carId"], ego_speed, dt)
    tgt_accel = _scalar_acceleration(target_row["carId"], tgt_speed, dt)
    ego_w = _angular_velocity(ego_row["carId"], ego_hdg, dt)
    tgt_w = _angular_velocity(target_row["carId"], tgt_hdg, dt)

    param1 = (ego_speed, ego_accel, ego_hdg, ego_w)
    param2 = (tgt_speed, tgt_accel, tgt_hdg, tgt_w)

    return _compute_2d_ttc_kernel(p_ego, p_tgt, dist, param1, param2)


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

_FRONT_SLOTS = {"front", "left_front", "right_front"}


def _find_front_pairs(
    frame_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[Tuple[Any, Any]]:
    """For one frame, return (ego_id, target_id) pairs where target is ahead of ego.

    When *frame_df* contains vehicles from multiple scenes (multi-file merge),
    pairs are formed within each scene independently to avoid cross-scene
    contamination.

    Vectorised with numpy broadcasting — O(N²) in C, not Python.
    A pair is formed when *target* falls into one of the front slots
    (front / left_front / right_front) relative to *ego* and is within
    *max_distance_m*.
    """
    if "scene_id" in frame_df.columns:
        pairs: List[Tuple[Any, Any]] = []
        for _, scene_df in frame_df.groupby("scene_id"):
            pairs.extend(_find_front_pairs_single(scene_df, cfg))
        return pairs
    return _find_front_pairs_single(frame_df, cfg)


def _find_front_pairs_single(
    frame_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[Tuple[Any, Any]]:
    """Vectorised pair building for a single scene (no cross-scene pairs)."""
    n = len(frame_df)
    if n < 2:
        return []

    # --- extract columns as numpy arrays ---
    x = frame_df["carCenterXm"].to_numpy(dtype=np.float64)
    y = frame_df["carCenterYm"].to_numpy(dtype=np.float64)
    heading = frame_df["heading"].to_numpy(dtype=np.float64)
    car_ids = frame_df["carId"].values

    # --- pairwise dx, dy: dx[i, j] = x_j - x_i ---
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]

    # --- distance matrix ---
    dist = np.hypot(dx, dy)

    # --- mask: i != j, within max_distance ---
    valid = (dist > 0.0) & (dist <= cfg.max_distance_m)
    if not np.any(valid):
        return []

    # --- rotate (dx, dy) into each ego's local frame ---
    theta = np.radians(heading)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # longitudinal[i,j]: how far ahead target j is from ego i
    long_mat = dx * cos_t[:, np.newaxis] + dy * sin_t[:, np.newaxis]

    # "front slot" ⇔ target is ahead of ego (longitudinal ≥ 0).
    # All three front variants (front / left_front / right_front) share this
    # condition; the lateral split only matters for slot naming in Stage 3.
    valid &= long_mat >= 0.0

    if not np.any(valid):
        return []

    # --- extract (ego_idx, tgt_idx) pairs ---
    ego_idx, tgt_idx = np.where(valid)
    return [(car_ids[i], car_ids[j]) for i, j in zip(ego_idx, tgt_idx)]


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

    When *raw_df* spans multiple scenes (``scene_id`` column present),
    each scene is processed independently to avoid cross-scene vehicle
    pairing.  Caches (velocity / acceleration) are cleared between scenes.

    Workflow (per scene)
    --------------------
    1. Group raw data by *frameNum*.
    2. Per frame, build (ego, target) pairs where target is in a front slot.
    3. For each pair, compute 2D_TTC using OBB geometry (NBDT standard).
    4. When 2D_TTC < *conflict_ttc_threshold*, record a conflict frame.
    5. Group consecutive conflict frames into events and emit ConflictCandidate.

    Returns
    -------
    List[ConflictCandidate] with ``is_conflict=True``.
    """
    if not _has_obb_data(raw_df):
        raise ValueError(
            "OBB corner columns (boundingBox1Xm..4Ym) missing from raw data. "
            "2D_TTC requires OBB geometry."
        )

    if "scene_id" in raw_df.columns:
        all_candidates: List[ConflictCandidate] = []
        scene_names = sorted(raw_df["scene_id"].unique())
        for scene_id in scene_names:
            _clear_caches()
            scene_df = raw_df[raw_df["scene_id"] == scene_id]
            candidates = _detect_conflicts_one_scene(
                scene_df, cfg, scene_id=str(scene_id),
            )
            all_candidates.extend(candidates)
        return all_candidates

    return _detect_conflicts_one_scene(raw_df, cfg, "scene")


def _detect_conflicts_one_scene(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
    scene_id: str,
) -> List[ConflictCandidate]:
    """Core detection logic for a single scene.  See ``detect_conflicts``."""
    _clear_caches()
    dt = 1.0 / cfg.fps

    # Sort by frame — critical for cache correctness (per-frame sequential)
    df = raw_df.sort_values(["frameNum", "carId"]).reset_index(drop=True)
    frames = df.groupby("frameNum")

    from tqdm import tqdm

    # Per-pair 2D_TTC time series:  (ego, target) → [(frameNum, 2D_TTC), ...]
    pair_series: Dict[Tuple[Any, Any], List[Tuple[int, Optional[float]]]] = {}

    n_frames_total = len(frames)
    print(f"  Stage 1/4: conflict detection — {n_frames_total} frames, "
          f"fps={cfg.fps}, TTC<{cfg.conflict_ttc_threshold}s"
          f"{' (scene ' + scene_id + ')' if scene_id != 'scene' else ''}")

    for frame_num, frame_df in tqdm(
        frames, total=n_frames_total, desc="  scanning frames", unit="frm"
    ):
        front_pairs = _find_front_pairs(frame_df, cfg)

        # Build O(1) carId → positional-index lookup once per frame
        # (avoids repeated boolean-mask scans in the hot pair loop below).
        car_ids = frame_df["carId"].values
        car_pos: dict = {cid: i for i, cid in enumerate(car_ids)}

        for ego_id, tgt_id in front_pairs:
            key = (ego_id, tgt_id)
            ego_pos = car_pos.get(ego_id)
            tgt_pos = car_pos.get(tgt_id)
            if ego_pos is None or tgt_pos is None:
                continue
            ego_row = frame_df.iloc[ego_pos]
            tgt_row = frame_df.iloc[tgt_pos]
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

    When *raw_df* spans multiple scenes each scene is sampled independently
    so that ego_ids from different scenes do not leak.

    Strategy
    --------
    Per scene, sample non-conflict frames to match the number of conflict
    candidates × *cfg.rebalance_target_ratio*.  When a scene does not have
    enough safe frames to reach the target, all available safe frames are
    sampled and the caller should downsample conflicts later to achieve
    the desired ratio.

    *exclude* lists existing conflict candidates; their ego / time
    neighbourhoods are avoided.
    """
    if "scene_id" in raw_df.columns:
        all_non: List[ConflictCandidate] = []
        for scene_id in sorted(raw_df["scene_id"].unique()):
            scene_df = raw_df[raw_df["scene_id"] == scene_id]
            scene_exclude = (
                [c for c in exclude if c.scene_id == scene_id]
                if exclude else None
            )
            all_non.extend(_sample_non_conflicts_one_scene(
                scene_df, cfg, exclude=scene_exclude, scene_id=str(scene_id),
            ))
        return all_non

    return _sample_non_conflicts_one_scene(raw_df, cfg, exclude=exclude, scene_id="scene")


def _sample_non_conflicts_one_scene(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
    *,
    exclude: Optional[List[ConflictCandidate]] = None,
    scene_id: str = "scene",
) -> List[ConflictCandidate]:
    """Core non-conflict sampling for a single scene.

    Targets ``n_conflicts × target_ratio`` non-conflict samples.  Safe frames
    are allocated across egos proportionally to each ego's frame count.
    When total safe frames are insufficient the function samples everything
    available and warns — the caller is responsible for downsampling conflicts.
    """
    _clear_caches()

    # Build exclusion set and per-ego conflict counts
    exclude_set: Set[Tuple[Any, int]] = set()
    conflict_count: Dict[Any, int] = {}
    if exclude is not None:
        margin_frames = int(cfg.history_sec * cfg.fps)
        for c in exclude:
            t_c = int(c.t_conflict) if c.t_conflict is not None else -1
            conflict_count[c.ego_id] = conflict_count.get(c.ego_id, 0) + 1
            for offset in range(-margin_frames, margin_frames + 1):
                exclude_set.add((c.ego_id, t_c + offset))

    df = raw_df.sort_values(["frameNum", "carId"]).reset_index(drop=True)
    ego_ids = df["carId"].unique()

    # ── per-ego: collect safe frames ────────────────────────────────────
    ego_safe_frames: Dict[Any, List[int]] = {}
    ego_total_frames: Dict[Any, int] = {}
    total_safe = 0
    total_conflicts = sum(conflict_count.values())

    for ego_id in ego_ids:
        ego_mask = df["carId"] == ego_id
        ego_frames = sorted(df.loc[ego_mask, "frameNum"].unique())
        n_frames = len(ego_frames)
        if n_frames < 2:
            continue
        ego_total_frames[ego_id] = n_frames
        safe = [f for f in ego_frames if (ego_id, f) not in exclude_set]
        ego_safe_frames[ego_id] = safe
        total_safe += len(safe)

    # ── target: match conflicts × ratio, capped by available safe frames ─
    target_ratio = getattr(cfg, "rebalance_target_ratio", 1.0)
    n_target = int(total_conflicts * target_ratio)
    n_sample = min(n_target, total_safe)

    if total_safe < n_target:
        print(f"  [{scene_id}] WARNING: only {total_safe} safe frames available "
              f"for {total_conflicts} conflicts (target {n_target}). "
              f"Non-conflict sampling saturated — conflicts should be downsampled "
              f"in post-processing.")

    if n_sample == 0:
        return []

    # ── proportional allocation across egos ─────────────────────────────
    import random
    seed = getattr(cfg, "rebalance_seed", 42)

    # Give each ego a share of n_sample proportional to its safe-frames count
    allocated: Dict[Any, int] = {}
    remaining = n_sample
    for ego_id in ego_ids:
        safe = ego_safe_frames.get(ego_id)
        if not safe:
            allocated[ego_id] = 0
            continue
        # Round-robin: allocate proportionally, floor to available
        share = max(1, int(n_sample * len(safe) / max(total_safe, 1)))
        share = min(share, len(safe))
        allocated[ego_id] = share

    # Adjust to hit exactly n_sample (top-up from egos with spare capacity)
    total_allocated = sum(allocated.values())
    ego_order = sorted(ego_ids, key=lambda e: len(ego_safe_frames.get(e, [])), reverse=True)
    i = 0
    while total_allocated < n_sample and i < len(ego_order) * 2:
        ego_id = ego_order[i % len(ego_order)]
        safe = ego_safe_frames.get(ego_id, [])
        if allocated.get(ego_id, 0) < len(safe):
            allocated[ego_id] = allocated.get(ego_id, 0) + 1
            total_allocated += 1
        i += 1
    while total_allocated > n_sample and i < len(ego_order) * 2:
        ego_id = ego_order[-(i % len(ego_order)) - 1]
        if allocated.get(ego_id, 0) > 0:
            allocated[ego_id] = allocated[ego_id] - 1
            total_allocated -= 1
        i += 1

    # ── sample ──────────────────────────────────────────────────────────
    non_conflicts: List[ConflictCandidate] = []
    for ego_id in ego_ids:
        n = allocated.get(ego_id, 0)
        safe = ego_safe_frames.get(ego_id, [])
        if n <= 0 or not safe:
            continue
        rng = random.Random(hash(f"{ego_id}_{scene_id}_{seed}") & 0x7FFFFFFF)
        for fnum in rng.sample(safe, min(n, len(safe))):
            non_conflicts.append(ConflictCandidate(
                scene_id=scene_id,
                ego_id=ego_id,
                is_conflict=False,
                t_conflict=None,
                conflict_target_id=None,
                meta={"sample_frame": fnum},
            ))

    return non_conflicts


def _downsample_conflicts_per_scene(
    conflicts: List[ConflictCandidate],
    n_target: int,
    seed: int = 42,
) -> List[ConflictCandidate]:
    """Randomly downsample conflicts to *n_target* with deterministic seed."""
    if len(conflicts) <= n_target:
        return conflicts
    import random
    rng = random.Random(seed)
    return rng.sample(conflicts, n_target)


def detect_all_candidates(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[ConflictCandidate]:
    """Run conflict detection then non-conflict sampling; return merged candidate list.

    Per-scene rebalancing (when ``cfg.rebalance_enabled``):
    - For each scene, conflict count is capped so that the conflict:non-conflict
      ratio does not exceed *cfg.rebalance_target_ratio*.
    - This means in scenes with very few safe non-conflict frames (e.g.
      Peachtree), conflicts are downsampled to restore balance.
    """
    conflicts = detect_conflicts(raw_df, cfg)
    non_conflicts = sample_non_conflicts(raw_df, cfg, exclude=conflicts)

    if not cfg.rebalance_enabled:
        return conflicts + non_conflicts

    # ── per-scene rebalance: downsample conflicts if needed ──────────────
    target_ratio = cfg.rebalance_target_ratio
    seed = cfg.rebalance_seed

    # Group by scene_id
    from collections import defaultdict
    scene_conflicts: Dict[str, List[ConflictCandidate]] = defaultdict(list)
    scene_non: Dict[str, List[ConflictCandidate]] = defaultdict(list)
    for c in conflicts:
        scene_conflicts[c.scene_id].append(c)
    for n in non_conflicts:
        scene_non[n.scene_id].append(n)

    balanced_conflicts: List[ConflictCandidate] = []
    for scene_id in sorted(set(list(scene_conflicts) + list(scene_non))):
        n_c = len(scene_conflicts.get(scene_id, []))
        n_n = len(scene_non.get(scene_id, []))
        if n_n == 0:
            balanced_conflicts.extend(scene_conflicts[scene_id])
            continue
        # Cap conflicts: at most n_non * target_ratio
        max_c = max(n_n, int(n_n * target_ratio))
        if n_c > max_c:
            print(f"  [{scene_id}] downsampling conflicts: {n_c} → {max_c} "
                  f"(non-conflicts: {n_n}, target ratio: {target_ratio})")
            balanced_conflicts.extend(
                _downsample_conflicts_per_scene(
                    scene_conflicts[scene_id], max_c, seed=seed,
                )
            )
        else:
            balanced_conflicts.extend(scene_conflicts[scene_id])

    return balanced_conflicts + non_conflicts
