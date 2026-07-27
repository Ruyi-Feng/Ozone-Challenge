"""Time helpers: alignment, second/frame conversion, relative time."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def sec_to_frames(seconds: float, fps: float) -> int:
    """Convert duration in seconds to frame count."""
    return int(round(seconds * fps))


def frames_to_sec(frames: int, fps: float) -> float:
    """Convert frame count to seconds."""
    return frames / fps


def get_track_time_span(track_df: pd.DataFrame, time_col: str = "timestamp") -> Tuple[float, float]:
    """Return (t_min, t_max) for a single-track dataframe."""
    return float(track_df[time_col].min()), float(track_df[time_col].max())


def has_continuous_coverage(
    timestamps: Iterable[float],
    t_start: float,
    t_end: float,
    *,
    max_gap_sec: float,
) -> bool:
    """
    Check whether timestamps cover [t_start, t_end] without gaps larger than max_gap_sec.

    The coverage check:
    1. The earliest timestamp must be <= t_start.
    2. The latest timestamp must be >= t_end.
    3. Sorted timestamps within [t_start, t_end] must have no gap > max_gap_sec.
    """
    ts_sorted = sorted(timestamps)
    if not ts_sorted:
        return False
    # Check bounds
    if ts_sorted[0] > t_start:
        return False
    if ts_sorted[-1] < t_end:
        return False
    # Check gaps within the window
    for i in range(1, len(ts_sorted)):
        if ts_sorted[i] > t_end:
            break
        if ts_sorted[i - 1] >= t_start:
            gap = ts_sorted[i] - ts_sorted[i - 1]
            if gap > max_gap_sec:
                return False
    return True


def to_relative_time(
    timestamps: pd.Series,
    t0: float,
) -> pd.Series:
    """Convert absolute timestamps to t_rel = t - t0 (history negative, future positive)."""
    return timestamps - t0


def window_bounds(t0: float, history_sec: float, future_sec: float) -> Tuple[float, float, float]:
    """
    Return (history_start, t0, future_end).

    History: [t0 - history_sec, t0]
    Future:  (t0, t0 + future_sec]
    """
    return t0 - history_sec, t0, t0 + future_sec
