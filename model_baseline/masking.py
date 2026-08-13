"""Shared coalition machinery for masked training and Winter-Shapley attribution.

Players of the cooperative game are (agent, time-segment, channel) cells over
the VALID frames of one event.  The same partition / nested-permutation code
drives BOTH the masked surrogate training (train.py) and the attribution MC
engine (model_baseline/attribution/), so the training-time coalition
distribution matches the explanation-time one exactly — the prerequisite for
v(S) being one well-defined function on both sides.

Winter level structures (nested permutations unlock level by level):
  - hierarchy="agent_major": agent > segment > channel.  One agent's whole
    history unlocks before the next agent's.  Used for the symmetric
    "evidence weight" readout (agent-level Shapley falls out via aggregation).
  - hierarchy="time_major": segment > agent > channel.  The scene unlocks
    time-slice by time-slice.  With segment order "chrono" this yields the
    chain-respecting asymmetric readout: coalitions are always temporally
    contiguous scene prefixes ("risk revelation timeline").

Segment order distribution: "uniform" (symmetric Winter), "chrono"
(ascending time, asymmetric), "reverse" (descending time).  Agent and
channel levels always permute uniformly.

Invalid frames (front zero-padding, missing frames, absent agents — see
build_tensor_cache.py) are NEVER players: they are permanently blocked via
time_mask in every coalition, in training and explanation alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

SEGMENT_ORDER_MODES = ("uniform", "chrono", "reverse")
HIERARCHIES = ("agent_major", "time_major")


# ---------------------------------------------------------------------------
# Segment cutting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentSpec:
    """How the frame axis is cut into segments.

    seg_len_frames : fixed segment length; boundaries are anchored at the END
        of the axis (= t0), so the most recent segment is always full and a
        shorter remainder, if any, sits at the oldest end.
    boundaries : explicit ascending segment START frame indices (first must be
        0); overrides seg_len_frames.  Use for semantic segmentation.
    """

    seg_len_frames: int = 5
    boundaries: Optional[tuple[int, ...]] = None


def build_segments(num_frames: int, spec: SegmentSpec) -> list[tuple[int, int]]:
    """Return ascending [(start, end), ...) frame ranges covering 0..num_frames."""
    if spec.boundaries is not None:
        starts = list(spec.boundaries)
        if not starts or starts[0] != 0 or sorted(starts) != starts:
            raise ValueError(
                f"boundaries must be ascending and start at 0, got {starts}"
            )
        if starts[-1] >= num_frames:
            raise ValueError(
                f"boundaries {starts} exceed num_frames={num_frames}"
            )
        ends = starts[1:] + [num_frames]
        return list(zip(starts, ends))

    L = int(spec.seg_len_frames)
    if L <= 0:
        raise ValueError(f"seg_len_frames must be positive, got {L}")
    # anchor at the end: walk backwards from num_frames
    bounds: list[tuple[int, int]] = []
    end = num_frames
    while end > 0:
        start = max(0, end - L)
        bounds.append((start, end))
        end = start
    bounds.reverse()
    return bounds


# ---------------------------------------------------------------------------
# Cell partition
# ---------------------------------------------------------------------------


@dataclass
class CellPartition:
    """Player set of one event: cells over valid frames.

    cells[cell_id] = (agent, seg_idx, channel); cell_frames[cell_id] = the
    valid frame indices the cell covers.  Cell ids are dense and assigned in
    sorted (agent, seg_idx, channel) order — deterministic for a given
    (valid_mask, agent_mask, SegmentSpec).
    """

    num_agents: int
    num_frames: int
    num_features: int
    seg_bounds: list[tuple[int, int]]
    cells: list[tuple[int, int, int]] = field(default_factory=list)
    cell_frames: list[np.ndarray] = field(default_factory=list)
    # agent -> seg_idx -> [cell_id, ...] (channels in ascending order)
    by_agent_seg: dict[int, dict[int, list[int]]] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def agents(self) -> list[int]:
        return sorted(self.by_agent_seg.keys())

    def segments_of(self, agent: int) -> list[int]:
        return sorted(self.by_agent_seg[agent].keys())

    def agents_in_segment(self, seg_idx: int) -> list[int]:
        return sorted(
            a for a, segs in self.by_agent_seg.items() if seg_idx in segs
        )


def build_partition(
    valid_mask: np.ndarray,
    agent_mask: np.ndarray,
    spec: SegmentSpec,
    num_features: int = 4,
) -> CellPartition:
    """Build the cell partition for one event.

    valid_mask : [A, T] bool — True = real observation
    agent_mask : [A] bool — True = slot occupied

    Absent agents, segments with zero valid frames, and (trivially) invalid
    frames produce no cells: they are context, not players.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    agent_mask = np.asarray(agent_mask, dtype=bool)
    A, T = valid_mask.shape
    seg_bounds = build_segments(T, spec)

    p = CellPartition(
        num_agents=A, num_frames=T, num_features=num_features,
        seg_bounds=seg_bounds,
    )
    for a in range(A):
        if not agent_mask[a]:
            continue
        for s, (start, end) in enumerate(seg_bounds):
            frames = np.nonzero(valid_mask[a, start:end])[0] + start
            if frames.size == 0:
                continue
            for c in range(num_features):
                cell_id = len(p.cells)
                p.cells.append((a, s, c))
                p.cell_frames.append(frames)
                p.by_agent_seg.setdefault(a, {}).setdefault(s, []).append(cell_id)
    return p


# ---------------------------------------------------------------------------
# Nested permutation sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelOrderConfig:
    """Level structure + per-level order distribution of the Winter game."""

    hierarchy: str = "agent_major"   # "agent_major" | "time_major"
    segments: str = "uniform"        # "uniform" | "chrono" | "reverse"

    def __post_init__(self) -> None:
        if self.hierarchy not in HIERARCHIES:
            raise ValueError(
                f"hierarchy must be one of {HIERARCHIES}, got {self.hierarchy!r}"
            )
        if self.segments not in SEGMENT_ORDER_MODES:
            raise ValueError(
                f"segments order must be one of {SEGMENT_ORDER_MODES}, "
                f"got {self.segments!r}"
            )


def _order_segments(
    seg_indices: list[int], mode: str, rng: np.random.Generator
) -> list[int]:
    if mode == "chrono":
        return seg_indices
    if mode == "reverse":
        return seg_indices[::-1]
    return [seg_indices[i] for i in rng.permutation(len(seg_indices))]


def _shuffled(items: list[int], rng: np.random.Generator) -> list[int]:
    return [items[i] for i in rng.permutation(len(items))]


def sample_nested_order(
    p: CellPartition,
    orders: LevelOrderConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one nested permutation → cell ids in unmask order (len n_cells)."""
    out: list[int] = []
    if orders.hierarchy == "agent_major":
        for a in _shuffled(p.agents, rng):
            for s in _order_segments(p.segments_of(a), orders.segments, rng):
                out.extend(_shuffled(p.by_agent_seg[a][s], rng))
    else:  # time_major
        seg_with_cells = sorted(
            {s for segs in p.by_agent_seg.values() for s in segs}
        )
        for s in _order_segments(seg_with_cells, orders.segments, rng):
            for a in _shuffled(p.agents_in_segment(s), rng):
                out.extend(_shuffled(p.by_agent_seg[a][s], rng))
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Coalition → model masks
# ---------------------------------------------------------------------------


def coalition_masks(
    p: CellPartition,
    masked_cells: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a coalition state into (time_mask, channel_mask) for the model.

    masked_cells : [n_cells] bool — True = cell is OUTSIDE the coalition
    valid_mask   : [A, T] bool

    Returns
    -------
    time_mask    : [A, T] bool — True = frame blocked.  Always includes
                   ~valid (padding is permanently unobservable).
    channel_mask : [A, T, F] bool — True = channel masked.  A frame whose
                   channels are all masked is additionally attention-blocked
                   by the model itself (TransformerConflictModel._apply_masks).
    """
    masked_cells = np.asarray(masked_cells, dtype=bool)
    if masked_cells.shape != (p.n_cells,):
        raise ValueError(
            f"masked_cells shape {masked_cells.shape} != ({p.n_cells},)"
        )
    valid_mask = np.asarray(valid_mask, dtype=bool)

    time_mask = ~valid_mask
    channel_mask = np.zeros(
        (p.num_agents, p.num_frames, p.num_features), dtype=bool
    )
    for cell_id in np.nonzero(masked_cells)[0]:
        a, _s, c = p.cells[cell_id]
        channel_mask[a, p.cell_frames[cell_id], c] = True
    return time_mask, channel_mask


# ---------------------------------------------------------------------------
# Training-time coalition sampler (surrogate value function, doc §5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskSamplerConfig:
    """Mixture over full input and nested-permutation coalition prefixes.

    The seg_len / hierarchy / order-mode mixtures deliberately cover every
    game the attribution side may run, so no single cut is baked into the
    surrogate.
    """

    p_full: float = 0.4
    seg_len_frames: tuple[int, ...] = (5, 10, 20)
    hierarchies: tuple[str, ...] = ("agent_major", "time_major")
    order_modes: tuple[str, ...] = ("uniform", "chrono", "reverse")

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_full <= 1.0:
            raise ValueError(f"p_full must be in [0, 1], got {self.p_full}")
        for h in self.hierarchies:
            if h not in HIERARCHIES:
                raise ValueError(f"unknown hierarchy {h!r}")
        for m in self.order_modes:
            if m not in SEGMENT_ORDER_MODES:
                raise ValueError(f"unknown order mode {m!r}")


class TrainingMaskSampler:
    """Draws per-sample (time_mask, channel_mask) batches for masked training.

    With probability p_full a sample sees the full input (only ~valid
    blocked); otherwise a random nested-permutation prefix is visible and the
    remaining cells are masked — the same coalition distribution the Winter
    MC estimator walks at explanation time.
    """

    def __init__(
        self,
        cfg: MaskSamplerConfig,
        num_features: int = 4,
        seed: int | np.random.SeedSequence = 0,
    ) -> None:
        self.cfg = cfg
        self.num_features = num_features
        self.rng = np.random.default_rng(seed)

    def sample_batch(
        self,
        valid: np.ndarray,
        agent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """valid [B, A, T] bool, agent [B, A] bool → (time [B,A,T], channel [B,A,T,F])."""
        valid = np.asarray(valid, dtype=bool)
        agent = np.asarray(agent, dtype=bool)
        B, A, T = valid.shape
        time_out = np.zeros((B, A, T), dtype=bool)
        chan_out = np.zeros((B, A, T, self.num_features), dtype=bool)

        for i in range(B):
            if self.rng.random() < self.cfg.p_full:
                time_out[i] = ~valid[i]
                continue

            spec = SegmentSpec(
                seg_len_frames=int(self.rng.choice(self.cfg.seg_len_frames))
            )
            p = build_partition(
                valid[i], agent[i], spec, num_features=self.num_features
            )
            if p.n_cells == 0:
                time_out[i] = ~valid[i]
                continue

            orders = LevelOrderConfig(
                hierarchy=str(self.rng.choice(self.cfg.hierarchies)),
                segments=str(self.rng.choice(self.cfg.order_modes)),
            )
            order = sample_nested_order(p, orders, self.rng)
            k = int(self.rng.integers(0, p.n_cells + 1))  # visible prefix size
            masked = np.ones(p.n_cells, dtype=bool)
            masked[order[:k]] = False
            time_out[i], chan_out[i] = coalition_masks(p, masked, valid[i])

        return time_out, chan_out
