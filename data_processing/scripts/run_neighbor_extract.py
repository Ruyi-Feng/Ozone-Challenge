"""CLI: extract persistent neighbor trajectories for windowed events."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_processing.core.conflict_detect import detect_all_candidates
from data_processing.core.export import export_events
from data_processing.core.neighbor_filter import extract_neighbors_for_events
from data_processing.core.trajectory_window import build_windowed_events
from data_processing.io.readers import load_config, load_raw_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter neighbors (6 slots, 200m) and collect persistent tracks"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    parser.add_argument("--raw", type=Path, required=True, help="Raw standardized CSV")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Also run export to processed data/label CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw_df = load_raw_csv(args.raw)

    candidates = detect_all_candidates(raw_df, cfg)
    windows = build_windowed_events(raw_df, candidates, cfg)
    tracked = extract_neighbors_for_events(raw_df, windows, cfg)

    print(f"tracked neighborhood events: {len(tracked)}")
    if args.export:
        data_path, label_path, future_path = export_events(tracked, cfg)
        print(f"data:         {data_path}")
        print(f"labels:       {label_path}")
        print(f"future_traj:  {future_path}")


if __name__ == "__main__":
    main()
