"""Tests for Stage-2 window post-processing: min-sample-gap + rebalance."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_processing.core.trajectory_window import (
    enforce_min_sample_gap,
    rebalance_windows_by_class,
)
from data_processing.io.schema import WindowedEventCandidate


def _w(scene, ego, is_conflict, t0):
    return WindowedEventCandidate(
        scene_id=scene,
        ego_id=ego,
        is_conflict=is_conflict,
        t0=float(t0),
        t_end=float(t0) + 30.0,
    )


def test_enforce_min_sample_gap_spacing():
    # ego "a": t0 at 0, 50, 110, 120 → keep 0 and 110 (120 too close to 110)
    windows = [
        _w("s", "a", True, 0),
        _w("s", "a", True, 50),
        _w("s", "a", True, 110),
        _w("s", "a", True, 120),
    ]
    kept = enforce_min_sample_gap(windows, min_gap_sec=11.0, fps=10.0)
    assert sorted(w.t0 for w in kept) == [0.0, 110.0]


def test_enforce_min_sample_gap_independent_across_egos():
    # different egos are independent → both kept despite close t0
    windows = [
        _w("s", "a", True, 0),
        _w("s", "b", True, 5),
    ]
    kept = enforce_min_sample_gap(windows, min_gap_sec=11.0, fps=10.0)
    assert len(kept) == 2


def test_rebalance_downsamples_over_represented():
    # 3 conflict + 7 non-conflict, ratio 1.0 → 3:3
    windows = [_w("s", f"c{i}", True, i * 200) for i in range(3)]
    windows += [_w("s", f"n{i}", False, i * 200) for i in range(7)]
    out = rebalance_windows_by_class(windows, ratio=1.0, seed=42)
    c = sum(1 for w in out if w.is_conflict)
    n = sum(1 for w in out if not w.is_conflict)
    assert c == 3 and n == 3


def test_rebalance_passes_missing_class_through():
    # only conflicts → untouched
    windows = [_w("s", f"c{i}", True, i * 200) for i in range(5)]
    out = rebalance_windows_by_class(windows, ratio=1.0, seed=42)
    assert len(out) == 5
