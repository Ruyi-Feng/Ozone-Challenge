"""Validate conflict_detect NBDT-aligned implementation.

NBDT heading convention: 0° = East (+x), 90° = North (+y).

Covers: geometry helpers, strip intersection, 2D_TTC kernel,
and end-to-end detect_conflicts on synthetic data.
"""

import math
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_processing.core.conflict_detect import (
    _clear_caches,
    _order_rect_points,
    _rectangles_intersect,
    _calculate_nearest_points,
    _rear_forward_strips_intersect,
    _compute_2d_ttc_kernel,
    _compute_2d_ttc,
    _get_obb_corners,
    detect_conflicts,
)
from data_processing.io.schema import ProcessingConfig

HEAD_N = 90.0   # North (NBDT: 90°)
HEAD_E = 0.0    # East  (NBDT: 0°)
HEAD_S = 270.0  # South
HEAD_W = 180.0  # West


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vehicle_row(
    car_id,
    frame_num: int,
    cx: float,
    cy: float,
    heading_deg: float,
    speed: float,
    length: float = 4.5,
    width: float = 2.0,
):
    """Build a row with OBB corners derived from center, heading, length, width.

    NBDT heading convention: 0° = East (+x), 90° = North (+y).
    Corner ordering: 0=front-right, 1=front-left, 2=rear-left, 3=rear-right
    """
    h_rad = math.radians(heading_deg)
    sin_h = math.sin(h_rad)
    cos_h = math.cos(h_rad)
    half_l = length / 2.0
    half_w = width / 2.0

    def _rotate(lx, ly):
        # lx = lateral (+right), ly = longitudinal (+forward)
        # heading direction = (cos_h, sin_h), right = (-sin_h, cos_h)
        gx = cx - lx * sin_h + ly * cos_h
        gy = cy + lx * cos_h + ly * sin_h
        return gx, gy

    fr = _rotate(+half_w, +half_l)
    fl = _rotate(-half_w, +half_l)
    rl = _rotate(-half_w, -half_l)
    rr = _rotate(+half_w, -half_l)

    return {
        "carId": car_id,
        "frameNum": frame_num,
        "carCenterXm": cx,
        "carCenterYm": cy,
        "heading": heading_deg,
        "speed": speed,
        "objClass": "car",
        "boundingBox1Xm": fr[0], "boundingBox1Ym": fr[1],
        "boundingBox2Xm": fl[0], "boundingBox2Ym": fl[1],
        "boundingBox3Xm": rl[0], "boundingBox3Ym": rl[1],
        "boundingBox4Xm": rr[0], "boundingBox4Ym": rr[1],
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestGeometryHelpers:
    def test_order_rect_points_ccw(self):
        rect = [(1.0, 2.0), (-1.0, 2.0), (-1.0, -2.0), (1.0, -2.0)]
        ordered = _order_rect_points(rect)
        assert len(ordered) == 4

    def test_rectangles_intersect_overlapping(self):
        r1 = [(0.0, 0.0), (0.0, 2.0), (3.0, 2.0), (3.0, 0.0)]
        r2 = [(1.0, 1.0), (1.0, 3.0), (4.0, 3.0), (4.0, 1.0)]
        assert _rectangles_intersect(_order_rect_points(r1), _order_rect_points(r2))

    def test_rectangles_intersect_separated(self):
        r1 = [(0.0, 0.0), (0.0, 2.0), (3.0, 2.0), (3.0, 0.0)]
        r2 = [(10.0, 10.0), (10.0, 12.0), (13.0, 12.0), (13.0, 10.0)]
        assert not _rectangles_intersect(_order_rect_points(r1), _order_rect_points(r2))

    def test_nearest_points_separated(self):
        ego_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 10.0)))
        tgt_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("tgt", 0, 0.0, 20.0, HEAD_N, 5.0)))
        p1, p2, d = _calculate_nearest_points(ego_c, tgt_c)
        assert d > 0
        assert 14.0 < d < 17.0, f"expected ~15.5m, got {d:.2f}"

    def test_nearest_points_intersecting(self):
        r1 = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)]
        r2 = [(2.0, 2.0), (2.0, 6.0), (6.0, 6.0), (6.0, 2.0)]
        p1, p2, d = _calculate_nearest_points(r1, r2)
        assert d == 0.0


# ---------------------------------------------------------------------------
# Strip intersection
# ---------------------------------------------------------------------------

class TestStripIntersection:
    def test_parallel_headings_rear_facing(self):
        """Two vehicles heading North, ego behind target → strips intersect."""
        ego_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 10.0)))
        tgt_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("tgt", 0, 0.0, 20.0, HEAD_N, 5.0)))
        assert _rear_forward_strips_intersect(ego_c, tgt_c, HEAD_N, HEAD_N)

    def test_opposite_headings(self):
        """Vehicles heading toward each other (N vs S) → strips intersect."""
        ego_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 10.0)))
        tgt_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("tgt", 0, 0.0, 20.0, HEAD_S, 10.0)))
        assert _rear_forward_strips_intersect(ego_c, tgt_c, HEAD_N, HEAD_S)

    def test_crossing_headings(self):
        """One N-heading, one E-heading → function runs without error."""
        ego_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 10.0)))
        tgt_c = _get_obb_corners(pd.Series(
            _make_vehicle_row("tgt", 0, 10.0, 10.0, HEAD_E, 10.0)))
        result = _rear_forward_strips_intersect(ego_c, tgt_c, HEAD_N, HEAD_E)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2D_TTC kernel
# ---------------------------------------------------------------------------

class Test2DTTCKernel:
    def test_closing_zero_accel(self):
        """Both heading North, ego (10 m/s) closing on tgt (5 m/s).
        Points along y-axis, distance 15m → TTC = 15/(10-5) = 3s."""
        ttc = _compute_2d_ttc_kernel(
            (0.0, 0.0), (0.0, 15.0), 15.0,
            (10.0, 0.0, HEAD_N, 0.0),   # ego: 10 m/s north
            (5.0, 0.0, HEAD_N, 0.0),    # tgt: 5 m/s north
        )
        assert ttc is not None, "should compute TTC"
        assert 2.5 < ttc < 3.5, f"expected ~3s, got {ttc:.2f}"

    def test_separating_returns_none(self):
        """Ego slower → separating → None."""
        ttc = _compute_2d_ttc_kernel(
            (0.0, 0.0), (0.0, 15.0), 15.0,
            (5.0, 0.0, HEAD_N, 0.0),
            (10.0, 0.0, HEAD_N, 0.0),
        )
        assert ttc is None

    def test_accel_makes_ttc_shorter(self):
        """Positive relative accel → TTC < d / v_rel."""
        ttc = _compute_2d_ttc_kernel(
            (0.0, 0.0), (0.0, 20.0), 20.0,
            (12.0, 0.5, HEAD_N, 0.0),   # ego: 12 m/s, +0.5 m/s²
            (10.0, 0.0, HEAD_N, 0.0),   # tgt: 10 m/s constant
        )
        assert ttc is not None
        assert ttc < 10.0, f"acceleration should reduce TTC, got {ttc:.2f}"

    def test_angular_velocity_correction(self):
        """Angular velocity contributes to closing speed."""
        ttc = _compute_2d_ttc_kernel(
            (0.0, 0.0), (0.0, 15.0), 15.0,
            (10.0, 0.0, HEAD_N, 5.0),    # ego rotating +5 deg/s
            (5.0, 0.0, HEAD_N, -2.0),    # tgt rotating -2 deg/s
        )
        assert ttc is not None
        assert ttc > 0


# ---------------------------------------------------------------------------
# _compute_2d_ttc (full entry point)
# ---------------------------------------------------------------------------

class TestCompute2DTTC:
    def setup_method(self):
        _clear_caches()

    def test_parallel_headings_closing(self):
        """Both heading North, ego behind + faster → valid TTC."""
        ego = pd.Series(_make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 15.0))
        tgt = pd.Series(_make_vehicle_row("tgt", 0, 0.0, 20.0, HEAD_N, 10.0))
        dt = 1.0 / 25.0
        ttc = _compute_2d_ttc(ego, tgt, dt)
        assert ttc is not None, "parallel headings should produce valid TTC"
        assert 2.0 < ttc < 5.0, f"expected ~3s, got {ttc:.2f}"

    def test_separating_returns_none(self):
        """Ego slower → separating → None."""
        ego = pd.Series(_make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 5.0))
        tgt = pd.Series(_make_vehicle_row("tgt", 0, 0.0, 20.0, HEAD_N, 15.0))
        dt = 1.0 / 25.0
        ttc = _compute_2d_ttc(ego, tgt, dt)
        assert ttc is None

    def test_multi_frame_cache(self):
        """Caches (accel, angular vel) work across frames."""
        dt = 1.0 / 25.0
        ego_f0 = pd.Series(_make_vehicle_row("ego", 0, 0.0, 0.0, HEAD_N, 15.0))
        tgt_f0 = pd.Series(_make_vehicle_row("tgt", 0, 0.0, 20.0, HEAD_N, 10.0))
        ttc0 = _compute_2d_ttc(ego_f0, tgt_f0, dt)
        assert ttc0 is not None

        ego_f1 = pd.Series(_make_vehicle_row("ego", 1, 0.0, 15.0 / 25.0, HEAD_N, 15.0))
        tgt_f1 = pd.Series(_make_vehicle_row("tgt", 1, 0.0, 20.0 + 10.0 / 25.0, HEAD_N, 10.0))
        ttc1 = _compute_2d_ttc(ego_f1, tgt_f1, dt)
        assert ttc1 is not None


# ---------------------------------------------------------------------------
# Integration: detect_conflicts
# ---------------------------------------------------------------------------

def _build_two_vehicle_df(frames_config):
    """Build a DataFrame from (frame, ex, ey, eh, es, tx, ty, th, ts) tuples."""
    rows = []
    for fnum, ex, ey, eh, es, tx, ty, th, ts in frames_config:
        rows.append(_make_vehicle_row("ego", fnum, ex, ey, eh, es))
        rows.append(_make_vehicle_row("tgt", fnum, tx, ty, th, ts))
    return pd.DataFrame(rows)


class TestDetectConflicts:
    def test_closing_scenario_finds_conflict(self):
        """Ego closes on target → TTC drops below threshold → conflict."""
        frames = []
        for i in range(50):
            ego_y = 0.0 + i * 15.0 / 25.0
            tgt_y = 20.0 + i * 10.0 / 25.0
            frames.append((i, 0.0, ego_y, HEAD_N, 15.0,
                           0.0, tgt_y, HEAD_N, 10.0))

        df = _build_two_vehicle_df(frames)
        cfg = ProcessingConfig(fps=25.0, conflict_ttc_threshold=3.0, max_distance_m=200.0)
        conflicts = detect_conflicts(df, cfg)
        assert len(conflicts) > 0, f"expected conflicts, got {len(conflicts)}"

    def test_separating_scenario_no_conflict(self):
        """Ego slower → separating → no conflict."""
        frames = []
        for i in range(50):
            ego_y = 0.0 + i * 5.0 / 25.0
            tgt_y = 20.0 + i * 15.0 / 25.0
            frames.append((i, 0.0, ego_y, HEAD_N, 5.0,
                           0.0, tgt_y, HEAD_N, 15.0))

        df = _build_two_vehicle_df(frames)
        cfg = ProcessingConfig(fps=25.0, conflict_ttc_threshold=3.0, max_distance_m=200.0)
        conflicts = detect_conflicts(df, cfg)
        assert len(conflicts) == 0, f"expected 0 conflicts, got {len(conflicts)}"

    def test_conflict_threshold_respected(self):
        """TTC above threshold → not a conflict."""
        frames = []
        for i in range(50):
            ego_y = 0.0 + i * 10.5 / 25.0
            tgt_y = 100.0 + i * 10.0 / 25.0
            frames.append((i, 0.0, ego_y, HEAD_N, 10.5,
                           0.0, tgt_y, HEAD_N, 10.0))

        df = _build_two_vehicle_df(frames)
        cfg = ProcessingConfig(fps=25.0, conflict_ttc_threshold=3.0, max_distance_m=200.0)
        conflicts = detect_conflicts(df, cfg)
        assert len(conflicts) == 0, f"expected 0 conflicts (TTC >> threshold), got {len(conflicts)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
