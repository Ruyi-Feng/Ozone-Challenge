"""Prepare highway datasets for Ozone-Challenge pipeline.

- NGSIM I-80 / US-101: 10 Hz native → copy directly to data/raw_highway/
- highD: 25 Hz → 10 Hz downsampling → data/raw_highway/

Preserves laneId (real values, unlike intersection datasets where laneId=-1).
Skips .baiduyun.uploading.cfg temp files.

Usage (from repo root):
  python data_processing/scripts/downsample_highway.py \
      --src /path/to/Ozone-NGSIM-highD \
      --out data/raw_highway
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _downsample_25_to_10(df: pd.DataFrame) -> pd.DataFrame:
    """Keep frames where original frameNum % 5 ∈ {0, 2}, then renumber globally."""
    mask = (df["frameNum"] % 5 == 0) | (df["frameNum"] % 5 == 2)
    df = df.loc[mask].copy()
    kept_frames = sorted(df["frameNum"].unique())
    frame_map = {old: new for new, old in enumerate(kept_frames)}
    df["frameNum"] = df["frameNum"].map(frame_map)
    return df


def _copy_csv(src: Path, dst: Path, label: str) -> None:
    """Copy a CSV file directly (for native 10 Hz data)."""
    shutil.copy2(src, dst)
    n_rows = len(pd.read_csv(dst))
    n_cars = pd.read_csv(dst)["carId"].nunique()
    print(f"  {label}: {n_rows} rows, {n_cars} cars → {dst}")


def _downsample_and_save(src: Path, dst: Path, label: str) -> None:
    """Downsample 25→10 Hz and save."""
    df = pd.read_csv(src)
    df = _downsample_25_to_10(df)
    df.to_csv(dst, index=False)
    n_rows = len(df)
    n_cars = df["carId"].nunique()
    print(f"  {label}: {n_rows} rows, {n_cars} cars → {dst}")


def process_ngsim(src_dir: Path, out_dir: Path) -> list[Path]:
    """Copy NGSIM I-80 and US-101 CSVs (already 10 Hz native)."""
    out_paths: list[Path] = []

    # I-80
    i80_files = [
        ("ngsim_i80_standard_0400-0415.csv", "NGSIM_I80_0400-0415.csv"),
        ("ngsim_i80_standard_0500-0515.csv", "NGSIM_I80_0500-0515.csv"),
        ("ngsim_i80_standard_0515-0530.csv", "NGSIM_I80_0515-0530.csv"),
    ]
    for src_name, dst_name in i80_files:
        src = src_dir / src_name
        if src.exists():
            dst = out_dir / dst_name
            _copy_csv(src, dst, dst_name)
            out_paths.append(dst)
        else:
            print(f"  SKIP: {src} not found")

    # US-101
    us101_files = [
        ("ngsim_us101_standard_0750am-0805am.csv", "NGSIM_US101_0750am-0805am.csv"),
        ("ngsim_us101_standard_0805am-0820am.csv", "NGSIM_US101_0805am-0820am.csv"),
        ("ngsim_us101_standard_0820am-0835am.csv", "NGSIM_US101_0820am-0835am.csv"),
    ]
    for src_name, dst_name in us101_files:
        src = src_dir / src_name
        if src.exists():
            dst = out_dir / dst_name
            _copy_csv(src, dst, dst_name)
            out_paths.append(dst)
        else:
            print(f"  SKIP: {src} not found")

    return out_paths


def process_highd(src_dir: Path, out_dir: Path) -> list[Path]:
    """Downsample highD 25→10 Hz."""
    out_paths: list[Path] = []

    highd_dir = src_dir / "highd_standard_csv" / "highd_standard_csv"
    if not highd_dir.exists():
        print(f"  SKIP: highD directory not found: {highd_dir}")
        return out_paths

    csv_files = sorted(
        p for p in highd_dir.glob("*.csv")
        if not p.name.endswith(".baiduyun.uploading.cfg")
    )

    for fp in csv_files:
        # "01_highd_standard.csv" → "highD_01.csv"
        num = fp.stem.split("_")[0]  # "01"
        dst_name = f"highD_{num}.csv"
        dst = out_dir / dst_name
        _downsample_and_save(fp, dst, dst_name)
        out_paths.append(dst)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare highway datasets (NGSIM + highD) for pipeline"
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to Ozone-NGSIM-highD directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("--- NGSIM I-80 / US-101 (native 10 Hz, direct copy) ---")
    ngsim_paths = process_ngsim(args.src, args.out)

    print("\n--- highD (25 Hz → 10 Hz downsampling) ---")
    highd_paths = process_highd(args.src, args.out)

    total = len(ngsim_paths) + len(highd_paths)
    print(f"\nDone: {total} files written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
