"""Stage 1: detect conflict and non-conflict sample candidates."""

from __future__ import annotations

from typing import List

import pandas as pd

from data_processing.io.schema import ConflictCandidate, ProcessingConfig


def detect_conflicts(raw_df: pd.DataFrame, cfg: ProcessingConfig) -> List[ConflictCandidate]:
    """
    Detect conflict events from standardized trajectories.

    Typical responsibilities
    -----------------------
    - Locate conflict moments / intervals involving ego.
    - Record conflict_target_id and t_conflict.
    - Emit ConflictCandidate with is_conflict=True.
    """
    raise NotImplementedError


def sample_non_conflicts(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
    *,
    exclude: List[ConflictCandidate] | None = None,
) -> List[ConflictCandidate]:
    """
    Sample non-conflict candidates under the same windowing assumptions.

    exclude
        Existing conflict candidates (and neighborhoods in time) to avoid overlap.
    """
    raise NotImplementedError


def detect_all_candidates(raw_df: pd.DataFrame, cfg: ProcessingConfig) -> List[ConflictCandidate]:
    """Run conflict detection then non-conflict sampling; return merged candidate list."""
    conflicts = detect_conflicts(raw_df, cfg)
    non_conflicts = sample_non_conflicts(raw_df, cfg, exclude=conflicts)
    return conflicts + non_conflicts
