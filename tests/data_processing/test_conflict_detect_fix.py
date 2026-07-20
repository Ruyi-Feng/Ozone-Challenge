"""Validate that _compute_2d_ttc works for parallel-heading rear-end scenarios.

The old strip-intersection check failed when ego and target had identical
(or near-identical) headings because the front/rear edges were parallel.
This test reproduces that scenario and confirms the fix.
"""

import math
import sys
import os

import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_processing.core.conflict_detect import (
    _clear_velocity_cache,
    _compute_2d_ttc,
    _is_front_to_rear_contact,
    _get_obb_corners,
    _obb_closest_points,
    _find_front_pairs,
    detect_conflicts,
)
from data_processing.io.schema import ProcessingConfig


# ---------------------------------------------------------------------------
# Helpers: build a single-frame row for one vehicle
# ---------------------------------------------------------------------------

def _make_vehicle_row(
    car_id,
    frame_num: int,
    cx: float,
    cy: float,
    heading_deg: float,
    speed: float,
    length: float = 4.5,  # m
    width: float = 2.0,   # m
):
    """Build a row with OBB corners derived from center, heading, length, width.

    heading_deg = 0 → pointing North (+y).
    Corner ordering (observed in NBDT/CitySim):
      0  front-right
      1  front-left
      2  rear-left
      3  rear-right
    """
    h_rad = math.radians(heading_deg)
    sin_h = math.sin(h_rad)
    cos_h = math.cos(h_rad)

    half_l = length / 2.0
    half_w = width / 2.0

    # Local coords: x → lateral (right +), y → longitudinal (front +)
    # front-right:  (+w/2, +l/2)
    # front-left:   (-w/2, +l/2)
    # rear-left:    (-w/2, -l/2)
    # rear-right:   (+w/2, -l/2)

    def _rotate(lx, ly):
        # rotate local (lx,ly) by heading_rad into global (gx,gy)
        gx = cx + lx * cos_h - ly * sin_h
        gy = cy + lx * sin_h + ly * cos_h
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
# Unit: _is_front_to_rear_contact
# ---------------------------------------------------------------------------

class TestFrontToRearContact:
    def test_ego_behind_target_same_heading_returns_true(self):
        """Ego at (0,0), target at (0,20), both heading 0° (north).
        Closest OBB points should be ego-front and target-rear."""
        ego_row = _make_vehicle_row("ego", 0, 0.0, 0.0, heading_deg=0.0, speed=15.0)
        tgt_row = _make_vehicle_row("tgt", 0, 0.0, 20.0, heading_deg=0.0, speed=10.0)

        ego_c = _get_obb_corners(ego_row)
        tgt_c = _get_obb_corners(tgt_row)
        p_ego, p_tgt, dist = _obb_closest_points(ego_c, tgt_c)

        ego_hr = math.radians(ego_row["heading"])
        tgt_hr = math.radians(tgt_row["heading"])

        assert _is_front_to_rear_contact(p_ego, ego_c, ego_hr, p_tgt, tgt_c, tgt_hr)

    def test_ego_ahead_of_target_returns_false(self):
        """Ego ahead of target → contact should be rear-to-front, not front-to-rear."""
        ego_row = _make_vehicle_row("ego", 0, 0.0, 20.0, heading_deg=0.0, speed=10.0)
        tgt_row = _make_vehicle_row("tgt", 0, 0.0, 0.0, heading_deg=0.0, speed=15.0)

        ego_c = _get_obb_corners(ego_row)
        tgt_c = _get_obb_corners(tgt_row)
        p_ego, p_tgt, dist = _obb_closest_points(ego_c, tgt_c)

        ego_hr = math.radians(ego_row["heading"])
        tgt_hr = math.radians(tgt_row["heading"])

        assert not _is_front_to_rear_contact(p_ego, ego_c, ego_hr, p_tgt, tgt_c, tgt_hr)

    def test_side_by_side_returns_false(self):
        """Vehicles side by side → closest points are lateral, not front/rear."""
        ego_row = _make_vehicle_row("ego", 0, 0.0, 0.0, heading_deg=0.0, speed=10.0)
        tgt_row = _make_vehicle_row("tgt", 0, 10.0, 0.0, heading_deg=0.0, speed=10.0)

        ego_c = _get_obb_corners(ego_row)
        tgt_c = _get_obb_corners(tgt_row)
        p_ego, p_tgt, dist = _obb_closest_points(ego_c, tgt_c)

        ego_hr = math.radians(ego_row["heading"])
        tgt_hr = math.radians(tgt_row["heading"])

        assert not _is_front_to_rear_contact(p_ego, ego_c, ego_hr, p_tgt, tgt_c, tgt_hr)


# ---------------------------------------------------------------------------
# Unit: _compute_2d_ttc
# ---------------------------------------------------------------------------

class TestCompute2DTTC:
    def setup_method(self):
        _clear_velocity_cache()

    def test_parallel_headings_closing_returns_ttc(self):
        """Same-heading rear-end: ego faster, closing on target → valid TTC."""
        ego = _make_vehicle_row("ego", 0, 0.0, 0.0, heading_deg=0.0, speed=15.0)
        tgt = _make_vehicle_row("tgt", 0, 0.0, 20.0, heading_deg=0.0, speed=10.0)

        ego_s = pd.Series(ego)
        tgt_s = pd.Series(tgt)
        dt = 1.0 / 25.0

        ttc = _compute_2d_ttc(ego_s, tgt_s, dt)
        assert ttc is not None, "parallel headings should produce a valid TTC"
        # Closing speed ≈ 5 m/s, distance ≈ 15.5 m → TTC ≈ 3.1 s
        assert 2.0 < ttc < 5.0, f"expected TTC ~3s, got {ttc:.2f}"

    def test_separating_returns_none(self):
        """Ego slower than target → separating → None."""
        ego = _make_vehicle_row("ego", 0, 0.0, 0.0, heading_deg=0.0, speed=5.0)
        tgt = _make_vehicle_row("tgt", 0, 0.0, 20.0, heading_deg=0.0, speed=15.0)

        ego_s = pd.Series(ego)
        tgt_s = pd.Series(tgt)
        dt = 1.0 / 25.0

        ttc = _compute_2d_ttc(ego_s, tgt_s, dt)
        assert ttc is None, "separating vehicles should return None"

    def test_target_not_in_front_returns_none(self):
        """Target behind ego → not front-to-rear → None."""
        ego = _make_vehicle_row("ego", 0, 0.0, 20.0, heading_deg=0.0, speed=15.0)
        tgt = _make_vehicle_row("tgt", 0, 0.0, 0.0, heading_deg=0.0, speed=10.0)

        ego_s = pd.Series(ego)
        tgt_s = pd.Series(tgt)
        dt = 1.0 / 25.0

        ttc = _compute_2d_ttc(ego_s, tgt_s, dt)
        assert ttc is None, "target behind ego should not count as front→rear geometry"

    def test_angled_headings_closing_returns_ttc(self):
        """Slight heading difference → should still work."""
        ego = _make_vehicle_row("ego", 0, 0.0, 0.0, heading_deg=5.0, speed=15.0)
        tgt = _make_vehicle_row("tgt", 0, 2.0, 20.0, heading_deg=0.0, speed=10.0)

        ego_s = pd.Series(ego)
        tgt_s = pd.Series(tgt)
        dt = 1.0 / 25.0

        ttc = _compute_2d_ttc(ego_s, tgt_s, dt)
        assert ttc is not None, "angled headings should still produce valid TTC"

    def test_multi_frame_angular_velocity_cache(self):
        """Angular velocity cache works across frames."""
        ego_f0 = pd.Series(_make_vehicle_row("ego", 0, 0.0, 0.0, heading_deg=0.0, speed=15.0))
        tgt_f0 = pd.Series(_make_vehicle_row("tgt", 0, 0.0, 20.0, heading_deg=0.0, speed=10.0))
        dt = 1.0 / 25.0

        # First frame — cache initialised
        ttc0 = _compute_2d_ttc(ego_f0, tgt_f0, dt)
        assert ttc0 is not None

        # Second frame — heading unchanged, angular velocity should be 0
        ego_f1 = pd.Series(_make_vehicle_row("ego", 1, 0.0, 0.0 + 15.0/25.0, heading_deg=0.0, speed=15.0))
        tgt_f1 = pd.Series(_make_vehicle_row("tgt", 1, 0.0, 20.0 + 10.0/25.0, heading_deg=0.0, speed=10.0))
        ttc1 = _compute_2d_ttc(ego_f1, tgt_f1, dt)
        assert ttc1 is not None


# ---------------------------------------------------------------------------
# Integration: detect_conflicts with synthetic multi-frame data
# ---------------------------------------------------------------------------

def _build_two_vehicle_df(frames_config):
    """Build a DataFrame from a list of (frame, ego_x, ego_y, ego_h, ego_s, tgt_x, tgt_y, tgt_h, tgt_s)."""
    rows = []
    for fnum, ex, ey, eh, es, tx, ty, th, ts in frames_config:
        rows.append(_make_vehicle_row("ego", fnum, ex, ey, heading_deg=eh, speed=es))
        rows.append(_make_vehicle_row("tgt", fnum, tx, ty, heading_deg=th, speed=ts))
    return pd.DataFrame(rows)


class TestDetectConflicts:
    def test_simple_closing_scenario_finds_conflict(self):
        """Ego closes on target over multiple frames → conflict event found."""
        frames = []
        for i in range(10):
            ego_y = 0.0 + i * 15.0 / 25.0   # 15 m/s
            tgt_y = 20.0 + i * 10.0 / 25.0  # 10 m/s
            frames.append((i, 0.0, ego_y, 0.0, 15.0, 0.0, tgt_y, 0.0, 10.0))

        df = _build_two_vehicle_df(frames)
        cfg = ProcessingConfig(fps=25.0, conflict_ttc_threshold=3.0, max_distance_m=200.0)

        conflicts = detect_conflicts(df, cfg)
        assert len(conflicts) > 0, f"expected at least 1 conflict, got {len(conflicts)}"

    def test_separating_scenario_finds_no_conflict(self):
        """Ego slower than target → no conflict."""
        frames = []
        for i in range(10):
            ego_y = 0.0 + i * 5.0 / 25.0
            tgt_y = 20.0 + i * 15.0 / 25.0
            frames.append((i, 0.0, ego_y, 0.0, 5.0, 0.0, tgt_y, 0.0, 15.0))

        df = _build_two_vehicle_df(frames)
        cfg = ProcessingConfig(fps=25.0, conflict_ttc_threshold=3.0, max_distance_m=200.0)

        conflicts = detect_conflicts(df, cfg)
        assert len(conflicts) == 0, f"expected 0 conflicts, got {len(conflicts)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
