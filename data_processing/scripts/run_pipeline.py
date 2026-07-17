"""CLI: run full data-processing pipeline (raw -> processed data + labels)."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_processing.pipeline import load_pipeline_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ozone data-processing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_processing.yaml"),
        help="Path to data_processing YAML config",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional single raw CSV path (overrides config raw_dir listing)",
    )
    parser.add_argument(
        "--dump-interim",
        action="store_true",
        help="Write intermediate conflict candidates under interim_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config(args.config)
    data_path, label_path, future_path = run_pipeline(
        cfg,
        raw_path=args.raw,
        dump_interim=args.dump_interim,
    )
    print(f"data:         {data_path}")
    print(f"labels:       {label_path}")
    print(f"future_traj:  {future_path}")


if __name__ == "__main__":
    main()
