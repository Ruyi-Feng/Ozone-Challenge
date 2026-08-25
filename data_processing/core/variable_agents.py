"""History-only variable-agent selection for ragged Cross-Agent caches.

Selection never reads conflict_target_id, t_conflict, or post-t0 rows.
Overflow truncation is a capacity cap (max 9 neighbors), not a slot layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

ROLE_NAMES: tuple[str, ...] = (
    "ego",
    "front",
    "rear",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
)
ROLE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ROLE_NAMES)}
ID_TO_ROLE: dict[int, str] = {i: name for name, i in ROLE_TO_ID.items()}
ALLOWED_NEIGHBOR_ROLES: tuple[str, ...] = ROLE_NAMES[1:]
NUM_ROLES: int = len(ROLE_NAMES)
PAD_ROLE_ID: int = NUM_ROLES  # embedding padding index
MAX_AGENTS: int = 10
MAX_NEIGHBORS: int = MAX_AGENTS - 1
NUM_FRAMES: int = 80
NUM_FEATURES: int = 4


def _normalize_angle_diff(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


@dataclass(frozen=True)
class AgentCandidate:
    car_id: int
    role: str
    role_id: int
    distance_at_t0: float
    history_length: int


@dataclass
class SelectedAgents:
    ego: AgentCandidate
    neighbors: list[AgentCandidate]
    n_candidates: int
    n_overflow: int
    n_other_excluded: int
    dropped_car_ids: tuple[int, ...] = ()

    @property
    def agents(self) -> list[AgentCandidate]:
        return [self.ego, *self.neighbors]

    @property
    def agent_count(self) -> int:
        return 1 + len(self.neighbors)

    @property
    def agent_ids(self) -> list[int]:
        return [a.car_id for a in self.agents]


@dataclass
class RaggedEventTensors:
    x: np.ndarray
    valid: np.ndarray
    role_ids: np.ndarray
    agent_ids: np.ndarray
    is_conflict: int
    target_idx: int
    selected: SelectedAgents
    extras: dict = field(default_factory=dict)


def _last_history_state(history: pd.DataFrame) -> pd.Series:
    return history.sort_values("frameNum").iloc[-1]


def _car_role(history: pd.DataFrame) -> str:
    """Role at the last t<=t0 observation; first row is only a fallback."""
    if "role" not in history.columns:
        return ""
    state = _last_history_state(history)
    return str(state["role"])


def select_variable_agents(
    evt: pd.DataFrame,
    t0: float,
    *,
    max_neighbors: int = MAX_NEIGHBORS,
    allowed_roles: Sequence[str] = ALLOWED_NEIGHBOR_ROLES,
) -> SelectedAgents:
    """Select ego + up to ``max_neighbors`` history-only neighbor agents.

    Parameters
    ----------
    evt : event table (may contain future rows; they are ignored)
    t0 : prediction time. Only ``frameNum <= t0`` is read.
    """
    if max_neighbors < 0:
        raise ValueError(f"max_neighbors must be >= 0, got {max_neighbors}")

    allowed = set(allowed_roles)
    history = evt.loc[evt["frameNum"] <= t0]
    if history.empty:
        raise ValueError("no history at or before t0")

    grouped = {
        int(car_id): car_hist for car_id, car_hist in history.groupby("carId", sort=False)
    }

    n_other_excluded = 0
    ego_id: int | None = None
    for car_id, car_hist in grouped.items():
        if _car_role(car_hist) == "ego":
            ego_id = car_id
            break
    if ego_id is None:
        raise ValueError("no ego vehicle")

    ego_hist = grouped[ego_id]
    ego_ref = _last_history_state(ego_hist)
    ego_x0 = float(ego_ref["carCenterXm"])
    ego_y0 = float(ego_ref["carCenterYm"])
    ego = AgentCandidate(
        car_id=ego_id,
        role="ego",
        role_id=int(ROLE_TO_ID["ego"]),
        distance_at_t0=0.0,
        history_length=int(len(ego_hist)),
    )

    filled: list[AgentCandidate] = []
    for car_id, car_hist in grouped.items():
        if car_id == ego_id:
            continue
        role = _car_role(car_hist)
        if role == "other":
            n_other_excluded += 1
            continue
        if role not in allowed:
            continue
        state = _last_history_state(car_hist)
        dist = float(
            np.hypot(
                float(state["carCenterXm"]) - ego_x0,
                float(state["carCenterYm"]) - ego_y0,
            )
        )
        filled.append(
            AgentCandidate(
                car_id=car_id,
                role=role,
                role_id=int(ROLE_TO_ID[role]),
                distance_at_t0=dist,
                history_length=int(len(car_hist)),
            )
        )

    filled.sort(
        key=lambda c: (
            c.distance_at_t0,
            -c.history_length,
            c.role_id,
            c.car_id,
        )
    )
    n_candidates = len(filled)
    dropped: tuple[int, ...] = ()
    n_overflow = 0
    if n_candidates > max_neighbors:
        n_overflow = n_candidates - max_neighbors
        dropped = tuple(c.car_id for c in filled[max_neighbors:])
        filled = filled[:max_neighbors]

    return SelectedAgents(
        ego=ego,
        neighbors=filled,
        n_candidates=n_candidates,
        n_overflow=n_overflow,
        n_other_excluded=n_other_excluded,
        dropped_car_ids=dropped,
    )


def _history_frame_window(
    evt: pd.DataFrame,
    t0: float,
    num_frames: int,
) -> list[int]:
    frames = sorted({int(f) for f in evt.loc[evt["frameNum"] <= t0, "frameNum"]})
    if len(frames) >= num_frames:
        return frames[-num_frames:]
    return frames


def build_ragged_event(
    evt: pd.DataFrame,
    label: pd.Series,
    *,
    max_agents: int = MAX_AGENTS,
    num_frames: int = NUM_FRAMES,
    allowed_roles: Sequence[str] = ALLOWED_NEIGHBOR_ROLES,
    selected: SelectedAgents | None = None,
) -> RaggedEventTensors:
    """Build ragged tensors for one event.

    Agent selection uses only t<=t0 history. ``conflict_target_id`` is read
    afterwards solely to set ``target_idx``.
    """
    t0 = float(label["t0"])
    max_neighbors = max_agents - 1
    if selected is None:
        selected = select_variable_agents(
            evt,
            t0,
            max_neighbors=max_neighbors,
            allowed_roles=allowed_roles,
        )
    if selected.agent_count < 1 or selected.agent_count > max_agents:
        raise ValueError(
            f"agent count {selected.agent_count} outside [1, {max_agents}]"
        )
    if selected.agents[0].role != "ego":
        raise ValueError("ego must occupy local index 0")

    ego_hist = evt.loc[
        (evt["carId"] == selected.ego.car_id) & (evt["frameNum"] <= t0)
    ]
    if ego_hist.empty:
        raise ValueError("ego has no observation at or before t0")
    ego_ref = _last_history_state(ego_hist)
    ego_x0 = float(ego_ref["carCenterXm"])
    ego_y0 = float(ego_ref["carCenterYm"])
    ego_h0 = float(ego_ref["heading"])

    frame_window = _history_frame_window(evt, t0, num_frames)
    t_actual = len(frame_window)
    a = selected.agent_count
    x = np.zeros((a, num_frames, NUM_FEATURES), dtype=np.float32)
    valid = np.zeros((a, num_frames), dtype=bool)
    role_ids = np.array([c.role_id for c in selected.agents], dtype=np.int64)
    agent_ids = np.array([c.car_id for c in selected.agents], dtype=np.int64)

    for local_idx, cand in enumerate(selected.agents):
        car_rows = evt.loc[evt["carId"] == cand.car_id]
        by_frame: dict[int, pd.Series] = {}
        for _, row in car_rows.iterrows():
            fn = int(row["frameNum"])
            if fn <= t0:
                by_frame[fn] = row
        for t_idx, fn in enumerate(frame_window):
            row = by_frame.get(int(fn))
            if row is None:
                continue
            out_t = num_frames - t_actual + t_idx
            x[local_idx, out_t, 0] = float(row["carCenterXm"]) - ego_x0
            x[local_idx, out_t, 1] = float(row["carCenterYm"]) - ego_y0
            x[local_idx, out_t, 2] = _normalize_angle_diff(
                float(row["heading"]), ego_h0
            )
            x[local_idx, out_t, 3] = float(row["speed"])
            valid[local_idx, out_t] = True

    is_conflict = int(label["is_conflict"])
    conflict_target_id = int(label.get("conflict_target_id", -1))
    target_idx = -1
    if is_conflict and conflict_target_id != -1:
        for local_idx, cand in enumerate(selected.agents):
            if cand.car_id != conflict_target_id:
                continue
            if local_idx == 0:
                break
            if not bool(valid[local_idx].any()):
                break
            target_idx = local_idx - 1
            break

    return RaggedEventTensors(
        x=x,
        valid=valid,
        role_ids=role_ids,
        agent_ids=agent_ids,
        is_conflict=is_conflict,
        target_idx=target_idx,
        selected=selected,
        extras={
            "n_candidates": selected.n_candidates,
            "n_overflow": selected.n_overflow,
            "n_other_excluded": selected.n_other_excluded,
        },
    )


def validate_ragged_arrays(
    tracks_x: np.ndarray,
    tracks_valid: np.ndarray,
    tracks_role: np.ndarray,
    tracks_agent_id: np.ndarray,
    event_offsets: np.ndarray,
    event_eid: np.ndarray,
    event_y: np.ndarray,
    event_target: np.ndarray,
    event_agent_count: np.ndarray,
    *,
    max_agents: int = MAX_AGENTS,
) -> None:
    """Raise if a ragged cache violates the layout contract."""
    n = len(event_eid)
    if event_offsets.shape != (n + 1,):
        raise ValueError(
            f"event_offsets length {event_offsets.shape} != {(n + 1,)}"
        )
    if not np.all(np.diff(event_offsets) >= 1):
        raise ValueError("event_offsets must be strictly increasing")
    if int(event_offsets[0]) != 0:
        raise ValueError("event_offsets[0] must be 0")
    if int(event_offsets[-1]) != len(tracks_x):
        raise ValueError("event_offsets[-1] must equal total agent count")
    if len(event_y) != n or len(event_target) != n or len(event_agent_count) != n:
        raise ValueError("event-level arrays must have length N")
    if tracks_x.shape[0] != tracks_valid.shape[0]:
        raise ValueError("tracks_x / tracks_valid length mismatch")
    if tracks_x.shape[0] != len(tracks_role) or tracks_x.shape[0] != len(tracks_agent_id):
        raise ValueError("track-level arrays must share the agent axis")

    for i in range(n):
        start = int(event_offsets[i])
        end = int(event_offsets[i + 1])
        a_i = end - start
        if not 1 <= a_i <= max_agents:
            raise ValueError(f"event {int(event_eid[i])}: A={a_i} not in [1, {max_agents}]")
        if int(event_agent_count[i]) != a_i:
            raise ValueError(
                f"event {int(event_eid[i])}: agent_count={event_agent_count[i]} != {a_i}"
            )
        if int(tracks_role[start]) != ROLE_TO_ID["ego"]:
            raise ValueError(f"event {int(event_eid[i])}: first agent is not ego")
        ids = tracks_agent_id[start:end]
        if len(np.unique(ids)) != a_i:
            raise ValueError(f"event {int(event_eid[i])}: duplicate agent_id")
        tgt = int(event_target[i])
        if tgt >= a_i - 1:
            raise ValueError(
                f"event {int(event_eid[i])}: target_idx={tgt} >= n_neighbors={a_i - 1}"
            )
