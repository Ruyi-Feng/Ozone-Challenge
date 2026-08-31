"""I/O readers for raw data and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import yaml

from data_processing.io.schema import ProcessingConfig, validate_raw_schema


def load_config(path: Union[str, Path]) -> ProcessingConfig:
    """Load YAML config into ProcessingConfig."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    time_cfg = raw.get("time", {})
    neighbor_cfg = raw.get("neighbor", {})
    io_cfg = raw.get("io", {})

    split_cfg = raw.get("split", {})
    train_val_test_ratio = split_cfg.get("train_val_test_ratio", None)
    test_only = split_cfg.get("test_only", False)

    rebalance_cfg = raw.get("rebalance", {})

    return ProcessingConfig(
        history_sec=float(time_cfg.get("history_sec", 8.0)),
        future_sec=float(time_cfg.get("future_sec", 3.0)),
        fps=float(time_cfg.get("fps", 25.0)),
        conflict_ttc_threshold=float(
            time_cfg.get("conflict_ttc_threshold", 3.0)
        ),
        min_sample_gap_sec=float(time_cfg.get("min_sample_gap_sec", 0.0)),
        min_duration_frames=int(time_cfg.get("min_duration_frames", 1)),
        max_distance_m=float(neighbor_cfg.get("max_distance_m", 200.0)),
        neighbor_slots=tuple(
            neighbor_cfg.get(
                "slots",
                list(ProcessingConfig.neighbor_slots),
            )
        ),
        train_val_test_ratio=(
            tuple(float(x) for x in train_val_test_ratio)
            if train_val_test_ratio is not None else None
        ),
        test_only=bool(test_only),
        event_id_offset=int(io_cfg.get("event_id_offset", 0)),
        raw_dir=str(io_cfg.get("raw_dir", "data/raw")),
        interim_dir=str(io_cfg.get("interim_dir", "data/interim/candidates")),
        data_out=str(io_cfg.get("data_out", "data/processed/data/events_data.csv")),
        label_out=str(io_cfg.get("label_out", "data/processed/labels/events_labels.csv")),
        future_traj_out=str(
            io_cfg.get(
                "future_traj_out",
                "data/processed/labels/events_future_traj.csv",
            )
        ),
        rebalance_enabled=bool(rebalance_cfg.get("enabled", False)),
        rebalance_target_ratio=float(rebalance_cfg.get("target_ratio", 1.0)),
        rebalance_balance_conflict_types=bool(
            rebalance_cfg.get("balance_conflict_types", False)
        ),
        rebalance_per_scene=bool(rebalance_cfg.get("per_scene", True)),
        rebalance_seed=int(rebalance_cfg.get("seed", 42)),
    )


# Columns that identify a CSV as NGSIM native format (rather than standardised)
_NGSIM_MARKER_COLUMNS = {"Vehicle_ID", "Local_X", "Global_X", "v_Vel"}


def load_ngsim_csv(
    path: Union[str, Path],
    *,
    validate: bool = True,
    scene_id: str = "peachtree",
) -> pd.DataFrame:
    """Read an NGSIM native 10-fps CSV and return a standardised DataFrame."""
    from data_processing.io.ngsim_converter import convert_ngsim_to_standard

    df = convert_ngsim_to_standard(path, scene_id=scene_id)
    if validate:
        validate_raw_schema(df)
    return df


def _is_ngsim_format(df: pd.DataFrame) -> bool:
    """Return True when *df* looks like NGSIM native data."""
    cols = set(df.columns)
    return _NGSIM_MARKER_COLUMNS.issubset(cols)


def load_raw_csv(path: Union[str, Path], *, validate: bool = True) -> pd.DataFrame:
    """Read a standardised raw trajectory CSV (auto-detects NGSIM format)."""
    df = pd.read_csv(path)
    if _is_ngsim_format(df):
        return load_ngsim_csv(path, validate=validate)
    if validate:
        validate_raw_schema(df)
    return df


def list_raw_csv_files(raw_dir: Union[str, Path]) -> list[Path]:
    """List all CSV files under the raw directory."""
    root = Path(raw_dir)
    return sorted(root.glob("*.csv"))
