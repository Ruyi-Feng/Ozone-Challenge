"""CLI: run conflict / non-conflict detection only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_processing.core.conflict_detect import detect_all_candidates
from data_processing.io.readers import load_config, load_raw_csv
from data_processing.io.writers import write_interim_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect conflict / non-conflict candidates")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    parser.add_argument("--raw", type=Path, required=True, help="Raw standardized CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output candidates CSV (default: interim_dir/<stem>_candidates.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw_df = load_raw_csv(args.raw)

    candidates = detect_all_candidates(raw_df, cfg)
    rows = [
        {
            "scene_id": c.scene_id,
            "ego_id": c.ego_id,
            "is_conflict": c.is_conflict,
            "t_conflict": c.t_conflict,
            "conflict_target_id": c.conflict_target_id,
        }
        for c in candidates
    ]
    out = args.out or (Path(cfg.interim_dir) / f"{args.raw.stem}_candidates.csv")
    write_interim_candidates(pd.DataFrame(rows), out)
    print(f"candidates: {len(candidates)} -> {out}")


if __name__ == "__main__":
    main()
