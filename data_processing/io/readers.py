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
    train_val_ratio = split_cfg.get("train_val_ratio", None)

    return ProcessingConfig(
        history_sec=float(time_cfg.get("history_sec", 8.0)),
        future_sec=float(time_cfg.get("future_sec", 3.0)),
        fps=float(time_cfg.get("fps", 25.0)),
        conflict_ttc_threshold=float(
            time_cfg.get("conflict_ttc_threshold", 3.0)
        ),
        max_distance_m=float(neighbor_cfg.get("max_distance_m", 200.0)),
        neighbor_slots=tuple(
            neighbor_cfg.get(
                "slots",
                list(ProcessingConfig.neighbor_slots),
            )
        ),
        train_val_split_ratio=float(train_val_ratio) if train_val_ratio is not None else None,
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
    )


def load_raw_csv(path: Union[str, Path], *, validate: bool = True) -> pd.DataFrame:
    """Read a standardized raw trajectory CSV."""
    df = pd.read_csv(path)
    if validate:
        validate_raw_schema(df)
    return df


def list_raw_csv_files(raw_dir: Union[str, Path]) -> list[Path]:
    """List all CSV files under the raw directory."""
    root = Path(raw_dir)
    return sorted(root.glob("*.csv"))
