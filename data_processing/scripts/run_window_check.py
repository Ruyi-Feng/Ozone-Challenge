"""CLI: validate 8s history + 3s future windows for candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_processing.core.conflict_detect import detect_all_candidates
from data_processing.core.trajectory_window import build_windowed_events
from data_processing.io.readers import load_config, load_raw_csv
from data_processing.io.writers import write_interim_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trajectory windows for candidates")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    parser.add_argument("--raw", type=Path, required=True, help="Raw standardized CSV")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional precomputed candidates CSV; if omitted, run detection first",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output windowed candidates CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw_df = load_raw_csv(args.raw)

    if args.candidates is not None:
        # Loading structured candidates from CSV can be wired later;
        # for now re-detect to keep the script callable.
        _ = pd.read_csv(args.candidates)
        candidates = detect_all_candidates(raw_df, cfg)
    else:
        candidates = detect_all_candidates(raw_df, cfg)

    windows = build_windowed_events(raw_df, candidates, cfg)
    rows = [
        {
            "scene_id": w.scene_id,
            "ego_id": w.ego_id,
            "is_conflict": w.is_conflict,
            "t0": w.t0,
            "t_end": w.t_end,
            "t_conflict": w.t_conflict,
            "conflict_target_id": w.conflict_target_id,
            "history_ok": w.history_ok,
            "future_ok": w.future_ok,
        }
        for w in windows
    ]
    out = args.out or (Path(cfg.interim_dir) / f"{args.raw.stem}_windows.csv")
    write_interim_candidates(pd.DataFrame(rows), out)
    print(f"windowed: {len(windows)} / candidates {len(candidates)} -> {out}")


if __name__ == "__main__":
    main()
