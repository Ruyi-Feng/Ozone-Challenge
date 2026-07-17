"""Writers for processed data / label CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def _ensure_parent(path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_events_data(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write history (and neighbor) trajectory table."""
    out = _ensure_parent(path)
    df.to_csv(out, index=False)
    return out


def write_events_labels(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write per-event label table."""
    out = _ensure_parent(path)
    df.to_csv(out, index=False)
    return out


def write_future_traj(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write future 3s trajectory table."""
    out = _ensure_parent(path)
    df.to_csv(out, index=False)
    return out


def write_interim_candidates(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Optional dump of intermediate candidates for debugging."""
    out = _ensure_parent(path)
    df.to_csv(out, index=False)
    return out
