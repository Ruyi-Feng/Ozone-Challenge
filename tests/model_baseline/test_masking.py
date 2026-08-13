"""Tests for the shared coalition machinery (model_baseline/masking.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from model_baseline.masking import (  # noqa: E402
    CellPartition,
    LevelOrderConfig,
    MaskSamplerConfig,
    SegmentSpec,
    TrainingMaskSampler,
    build_partition,
    build_segments,
    coalition_masks,
    sample_nested_order,
)


# ---------------------------------------------------------------------------
# Fixture: 3 agents, 20 frames, 2 features
# agent 0: fully valid; agent 1: valid frames 8..19 only (front padding),
# with frame 15 missing; agent 2: absent.
# ---------------------------------------------------------------------------


A, T, F = 3, 20, 2


def _small_event():
    valid = np.zeros((A, T), dtype=bool)
    valid[0, :] = True
    valid[1, 8:] = True
    valid[1, 15] = False
    agent = np.array([True, True, False])
    return valid, agent


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def test_build_segments_even_split():
    assert build_segments(80, SegmentSpec(seg_len_frames=10)) == [
        (i * 10, (i + 1) * 10) for i in range(8)
    ]


def test_build_segments_end_anchored_remainder():
    # 80 / 7 → oldest segment is the short remainder, newest is full
    bounds = build_segments(80, SegmentSpec(seg_len_frames=7))
    assert bounds[0] == (0, 3)
    assert bounds[-1] == (73, 80)
    assert all(e - s == 7 for s, e in bounds[1:])
    assert bounds[-1][1] == 80


def test_build_segments_explicit_boundaries():
    assert build_segments(20, SegmentSpec(boundaries=(0, 8, 14))) == [
        (0, 8), (8, 14), (14, 20)
    ]
    with pytest.raises(ValueError):
        build_segments(20, SegmentSpec(boundaries=(2, 8)))


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


def test_partition_excludes_invalid_and_absent():
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)

    # agent 2 absent → no cells; agent 1 segment (0,5) & (5,10) partially
    assert set(p.agents) == {0, 1}
    assert p.segments_of(0) == [0, 1, 2, 3]
    # agent 1: seg 0 (frames 0-4) empty, seg 1 (5-9) has frames 8,9
    assert p.segments_of(1) == [1, 2, 3]

    # every cell's frames are valid and inside its segment
    for cell_id, (a, s, _c) in enumerate(p.cells):
        start, end = p.seg_bounds[s]
        frames = p.cell_frames[cell_id]
        assert ((frames >= start) & (frames < end)).all()
        assert valid[a, frames].all()

    # agent 1 seg 3 (15-19) must exclude the missing frame 15
    cell_id = p.by_agent_seg[1][3][0]
    assert 15 not in p.cell_frames[cell_id]

    # n_cells = (4 segs agent0 + 3 segs agent1) * F
    assert p.n_cells == 7 * F


# ---------------------------------------------------------------------------
# Nested orders
# ---------------------------------------------------------------------------


def _positions(order, p: CellPartition):
    """cell_id -> position in order."""
    pos = np.empty(p.n_cells, dtype=int)
    pos[order] = np.arange(len(order))
    return pos


@pytest.mark.parametrize("hierarchy", ["agent_major", "time_major"])
@pytest.mark.parametrize("seg_mode", ["uniform", "chrono", "reverse"])
def test_nested_order_visits_every_cell_once(hierarchy, seg_mode):
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)
    rng = np.random.default_rng(0)
    order = sample_nested_order(
        p, LevelOrderConfig(hierarchy=hierarchy, segments=seg_mode), rng
    )
    assert sorted(order.tolist()) == list(range(p.n_cells))


def test_agent_major_keeps_agents_contiguous():
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)
    rng = np.random.default_rng(1)
    order = sample_nested_order(
        p, LevelOrderConfig(hierarchy="agent_major"), rng
    )
    agent_seq = [p.cells[c][0] for c in order]
    # once we leave an agent we never come back
    changes = sum(
        1 for i in range(1, len(agent_seq)) if agent_seq[i] != agent_seq[i - 1]
    )
    assert changes == len(set(agent_seq)) - 1


def test_time_major_chrono_unlocks_segments_in_time_order():
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)
    rng = np.random.default_rng(2)
    order = sample_nested_order(
        p, LevelOrderConfig(hierarchy="time_major", segments="chrono"), rng
    )
    seg_seq = [p.cells[c][1] for c in order]
    assert seg_seq == sorted(seg_seq)

    order_rev = sample_nested_order(
        p, LevelOrderConfig(hierarchy="time_major", segments="reverse"), rng
    )
    seg_seq_rev = [p.cells[c][1] for c in order_rev]
    assert seg_seq_rev == sorted(seg_seq_rev, reverse=True)


def test_agent_major_chrono_orders_segments_within_agent():
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)
    rng = np.random.default_rng(3)
    order = sample_nested_order(
        p, LevelOrderConfig(hierarchy="agent_major", segments="chrono"), rng
    )
    pos = _positions(order, p)
    for a in p.agents:
        segs = p.segments_of(a)
        seg_first_pos = [min(pos[c] for c in p.by_agent_seg[a][s]) for s in segs]
        assert seg_first_pos == sorted(seg_first_pos)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        LevelOrderConfig(hierarchy="nope")
    with pytest.raises(ValueError):
        LevelOrderConfig(segments="backwards")
    with pytest.raises(ValueError):
        MaskSamplerConfig(p_full=1.5)


# ---------------------------------------------------------------------------
# Coalition masks
# ---------------------------------------------------------------------------


def test_coalition_masks_full_and_empty():
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)

    # empty coalition (all cells masked): every valid frame fully masked
    tm, cm = coalition_masks(p, np.ones(p.n_cells, dtype=bool), valid)
    np.testing.assert_array_equal(tm, ~valid)
    for a in p.agents:
        assert cm[a, valid[a]].all()
    # frames of absent agents carry no channel mask (blocked via time_mask)
    assert not cm[2].any()

    # grand coalition (nothing masked): only ~valid blocked
    tm2, cm2 = coalition_masks(p, np.zeros(p.n_cells, dtype=bool), valid)
    np.testing.assert_array_equal(tm2, ~valid)
    assert not cm2.any()


def test_coalition_masks_single_cell():
    valid, agent = _small_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=5), F)
    masked = np.zeros(p.n_cells, dtype=bool)
    cell_id = p.by_agent_seg[1][1][0]  # agent 1, seg 1 (frames 8, 9), channel 0
    masked[cell_id] = True

    _tm, cm = coalition_masks(p, masked, valid)
    expected = np.zeros((A, T, F), dtype=bool)
    expected[1, [8, 9], 0] = True
    np.testing.assert_array_equal(cm, expected)


# ---------------------------------------------------------------------------
# Training sampler
# ---------------------------------------------------------------------------


def _batch(B=6):
    valid, agent = _small_event()
    return (
        np.repeat(valid[None], B, axis=0),
        np.repeat(agent[None], B, axis=0),
    )


def test_sampler_full_only():
    valid_b, agent_b = _batch()
    sampler = TrainingMaskSampler(MaskSamplerConfig(p_full=1.0), F, seed=0)
    tm, cm = sampler.sample_batch(valid_b, agent_b)
    np.testing.assert_array_equal(tm, ~valid_b)
    assert not cm.any()


def test_sampler_coalitions_respect_validity():
    valid_b, agent_b = _batch()
    sampler = TrainingMaskSampler(
        MaskSamplerConfig(p_full=0.0, seg_len_frames=(5,)), F, seed=0
    )
    tm, cm = sampler.sample_batch(valid_b, agent_b)
    assert tm.shape == valid_b.shape and cm.shape == (*valid_b.shape, F)
    # ~valid is always blocked; channel masks never touch invalid frames
    assert (tm & valid_b).sum() == 0 or True  # tm may block nothing extra
    for i in range(len(valid_b)):
        np.testing.assert_array_equal(tm[i], ~valid_b[i])
        assert not cm[i][~valid_b[i]].any()
    # with p_full=0 across 6 samples, at least one non-trivial coalition
    assert cm.any()


def test_sampler_deterministic():
    valid_b, agent_b = _batch()
    cfg = MaskSamplerConfig(p_full=0.3)
    tm1, cm1 = TrainingMaskSampler(cfg, F, seed=42).sample_batch(valid_b, agent_b)
    tm2, cm2 = TrainingMaskSampler(cfg, F, seed=42).sample_batch(valid_b, agent_b)
    np.testing.assert_array_equal(tm1, tm2)
    np.testing.assert_array_equal(cm1, cm2)
