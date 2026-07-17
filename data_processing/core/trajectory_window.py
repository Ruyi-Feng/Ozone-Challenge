"""Stage 2: validate and build 8s history + 3s future windows."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from data_processing.io.schema import (
    ConflictCandidate,
    ProcessingConfig,
    WindowedEventCandidate,
)


def propose_t0_for_conflict(
    candidate: ConflictCandidate,
    cfg: ProcessingConfig,
) -> Optional[float]:
    """
    Propose t0 such that t_conflict lies in [t0, t0 + future_sec].

    Conflict is allowed to appear randomly inside the 3s future interval;
    t0 is the start of that interval, and history is taken as [t0 - 8s, t0].
    """
    raise NotImplementedError


def propose_t0_for_non_conflict(
    candidate: ConflictCandidate,
    cfg: ProcessingConfig,
) -> Optional[float]:
    """Propose t0 for a non-conflict sample (same history/future lengths)."""
    raise NotImplementedError


def validate_history_length(
    raw_df: pd.DataFrame,
    *,
    scene_id: str,
    ego_id,
    t0: float,
    history_sec: float,
) -> bool:
    """Return True if ego history in [t0 - history_sec, t0] meets length/coverage requirements."""
    raise NotImplementedError


def validate_future_interval(
    raw_df: pd.DataFrame,
    *,
    scene_id: str,
    ego_id,
    t0: float,
    future_sec: float,
    t_conflict: Optional[float],
    is_conflict: bool,
) -> bool:
    """
    Validate future window [t0, t0 + future_sec].

    For conflict samples, require t_conflict inside the interval when provided.
    """
    raise NotImplementedError


def build_windowed_candidate(
    raw_df: pd.DataFrame,
    candidate: ConflictCandidate,
    cfg: ProcessingConfig,
) -> Optional[WindowedEventCandidate]:
    """
    Build one WindowedEventCandidate if t0 proposal and length checks pass; else None.
    """
    raise NotImplementedError


def build_windowed_events(
    raw_df: pd.DataFrame,
    candidates: List[ConflictCandidate],
    cfg: ProcessingConfig,
) -> List[WindowedEventCandidate]:
    """Validate all candidates and keep only those with valid 8s + 3s windows."""
    windowed: List[WindowedEventCandidate] = []
    for cand in candidates:
        built = build_windowed_candidate(raw_df, cand, cfg)
        if built is not None and built.history_ok and built.future_ok:
            windowed.append(built)
    return windowed
