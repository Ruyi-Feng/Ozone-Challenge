"""Stage 2: validate and build 8s history + 3s future windows."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from data_processing.io.schema import (
    ConflictCandidate,
    ProcessingConfig,
    WindowedEventCandidate,
)
from data_processing.utils.time_utils import has_continuous_coverage


def _sec2frames(sec: float, fps: float) -> int:
    return int(round(sec * fps))


def _ego_frames(
    raw_df: pd.DataFrame,
    scene_id: str,
    ego_id,
    *,
    car_index: dict | None = None,
) -> pd.DataFrame:
    """Return ego's trajectory rows sorted by frameNum."""
    if car_index is not None:
        car_df = car_index.get(ego_id)
        if car_df is None or car_df.empty:
            return pd.DataFrame()
        if "scene_id" in car_df.columns:
            car_df = car_df[car_df["scene_id"] == scene_id]
        return car_df.sort_values("frameNum")

    mask = (raw_df["scene_id"] == scene_id) if "scene_id" in raw_df.columns else pd.Series(True, index=raw_df.index)
    ego_df = raw_df[mask & (raw_df["carId"] == ego_id)].sort_values("frameNum")
    return ego_df


def _has_scene_col(df: pd.DataFrame) -> bool:
    return "scene_id" in df.columns


def propose_t0_for_conflict(
    candidate: ConflictCandidate,
    cfg: ProcessingConfig,
) -> Optional[float]:
    """Propose t0 such that t_conflict lies in [t0, t0 + future_frames].

    Randomly samples a t0 that keeps the conflict moment inside the
    future window while allowing 8s of history before t0.

    Returns *t0* as a frame number (float), or None if impossible.
    """
    if candidate.t_conflict is None:
        return None
    future_frames = _sec2frames(cfg.future_sec, cfg.fps)
    history_frames = _sec2frames(cfg.history_sec, cfg.fps)

    t_conflict = candidate.t_conflict
    # t0 must be <= t_conflict and t0 >= t_conflict - future_frames
    lo = t_conflict - future_frames
    hi = t_conflict
    if lo > hi:
        return None

    # Also need at least *history_frames* of track before t0
    # (exact validation done later — here we just check feasibility)
    # We'll just pick within [lo, hi]; validation filters later.
    import random
    t0 = lo + random.random() * (hi - lo)
    return float(t0)


def propose_t0_for_non_conflict(
    candidate: ConflictCandidate,
    cfg: ProcessingConfig,
) -> Optional[float]:
    """Propose t0 for a non-conflict sample.

    Uses the *sample_frame* stored in candidate.meta as the approximate
    t0, then validates upstream.
    """
    sample_frame = candidate.meta.get("sample_frame")
    if sample_frame is not None:
        return float(sample_frame)
    return None


def validate_history_length(
    raw_df: pd.DataFrame,
    *,
    scene_id: str,
    ego_id,
    t0: float,
    history_sec: float,
    fps: float = 25.0,
    car_index: dict | None = None,
) -> bool:
    """Return True if ego history in [t0 - history_frames, t0] is sufficiently covered.

    Requires continuous coverage with no gap larger than one frame interval.
    """
    history_frames = _sec2frames(history_sec, fps)
    t_start = t0 - history_frames

    ego_df = _ego_frames(raw_df, scene_id, ego_id, car_index=car_index)
    if ego_df.empty:
        return False

    # Convert frame numbers to seconds — has_continuous_coverage expects
    # timestamps in the same unit as max_gap_sec.
    timestamps = [f / fps for f in ego_df["frameNum"].tolist()]
    max_gap = (1.0 / fps) * 2.1  # tolerate ~2 frame gaps

    return has_continuous_coverage(
        timestamps, float(t_start) / fps, float(t0) / fps, max_gap_sec=max_gap
    )


def validate_future_interval(
    raw_df: pd.DataFrame,
    *,
    scene_id: str,
    ego_id,
    t0: float,
    future_sec: float,
    t_conflict: Optional[float],
    is_conflict: bool,
    fps: float = 25.0,
    car_index: dict | None = None,
) -> bool:
    """Validate future window (t0, t0 + future_frames].

    For conflict samples: require t_conflict inside the interval.
    For non-conflict samples: just require ego track coverage in the window.
    """
    future_frames = _sec2frames(future_sec, fps)
    t_end = t0 + future_frames

    ego_df = _ego_frames(raw_df, scene_id, ego_id, car_index=car_index)
    if ego_df.empty:
        return False

    # Check ego has data in future window
    future_data = ego_df[(ego_df["frameNum"] > t0) & (ego_df["frameNum"] <= t_end)]
    if future_data.empty:
        return False

    # For conflict: t_conflict must fall in (t0, t0 + future_frames]
    if is_conflict and t_conflict is not None:
        if not (t0 < t_conflict <= t_end):
            return False

    return True


def build_windowed_candidate(
    raw_df: pd.DataFrame,
    candidate: ConflictCandidate,
    cfg: ProcessingConfig,
    *,
    car_index: dict | None = None,
) -> Optional[WindowedEventCandidate]:
    """Build one WindowedEventCandidate if t0 proposal and checks pass; else None."""
    # Propose t0
    if candidate.is_conflict:
        t0 = propose_t0_for_conflict(candidate, cfg)
    else:
        t0 = propose_t0_for_non_conflict(candidate, cfg)

    if t0 is None:
        return None

    scene_id = candidate.scene_id

    # Validate history
    history_ok = validate_history_length(
        raw_df,
        scene_id=scene_id,
        ego_id=candidate.ego_id,
        t0=t0,
        history_sec=cfg.history_sec,
        fps=cfg.fps,
        car_index=car_index,
    )

    # Validate future
    future_ok = validate_future_interval(
        raw_df,
        scene_id=scene_id,
        ego_id=candidate.ego_id,
        t0=t0,
        future_sec=cfg.future_sec,
        t_conflict=candidate.t_conflict,
        is_conflict=candidate.is_conflict,
        fps=cfg.fps,
        car_index=car_index,
    )

    return WindowedEventCandidate(
        scene_id=scene_id,
        ego_id=candidate.ego_id,
        is_conflict=candidate.is_conflict,
        t0=float(t0),
        t_end=float(t0 + _sec2frames(cfg.future_sec, cfg.fps)),
        t_conflict=candidate.t_conflict,
        conflict_target_id=candidate.conflict_target_id,
        history_ok=history_ok,
        future_ok=future_ok,
    )


def build_windowed_events(
    raw_df: pd.DataFrame,
    candidates: List[ConflictCandidate],
    cfg: ProcessingConfig,
    *,
    car_index: dict | None = None,
) -> List[WindowedEventCandidate]:
    """Validate all candidates and keep only those with valid 8s + 3s windows."""
    from tqdm import tqdm

    windowed: List[WindowedEventCandidate] = []
    for cand in tqdm(candidates, desc="  validating windows", unit="cand"):
        built = build_windowed_candidate(raw_df, cand, cfg, car_index=car_index)
        if built is not None and built.history_ok and built.future_ok:
            windowed.append(built)
    return windowed
