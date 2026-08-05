"""Downsample 25fps standard-format data to 10fps and copy to data/raw/.

25fps → 10fps: keep every 5th and 3rd frame (frameNum % 5 ∈ {0, 2}).
NGSIM (native 10fps): copy as-is with optional renumbering.

Output naming: ``{dataset}_{recording}.csv`` → placed in ``data/raw/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ── source directories ──────────────────────────────────────────────────
BASE = Path(r"D:\研究实习\暑期实习\交叉口数据处理")

IN_DIR = BASE / "inD" / "nbdt_standard"
CITYSIM_DIR = BASE / "citysim" / "citysim_standard"
NGSIM_DIR = BASE / "NGSIM" / "nbdt_standard"

# ── output directory ────────────────────────────────────────────────────
OUT_DIR = Path("data/raw")


def _downsample_25_to_10(df: pd.DataFrame) -> pd.DataFrame:
    """Keep frames where original frameNum % 5 ∈ {0, 2}, then renumber globally.

    Global renumbering preserves temporal alignment: all cars that share the
    same original frameNum get the same new frameNum.
    """
    mask = (df["frameNum"] % 5 == 0) | (df["frameNum"] % 5 == 2)
    df = df.loc[mask].copy()
    # Build global mapping: original frameNum → new sequential frameNum
    kept_frames = sorted(df["frameNum"].unique())
    frame_map = {old: new for new, old in enumerate(kept_frames)}
    df["frameNum"] = df["frameNum"].map(frame_map)
    return df


def _rename_columns_for_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Drop laneId if present (not all datasets have it) to match NGSIM columns."""
    if "laneId" in df.columns:
        df = df.drop(columns=["laneId"])
    return df


def process_inD() -> list[Path]:
    """Downsample all inD recordings (25fps → 10fps)."""
    files = sorted(IN_DIR.glob("*_tracks.csv"))
    out_paths = []
    for fp in files:
        rec_name = fp.stem.replace("_tracks", "")  # "00", "01", ...
        df = pd.read_csv(fp)
        df = _downsample_25_to_10(df)
        df = _rename_columns_for_consistency(df)
        out_path = OUT_DIR / f"inD_{rec_name}.csv"
        df.to_csv(out_path, index=False)
        out_paths.append(out_path)
        n_rows = len(df)
        n_cars = df["carId"].nunique()
        print(f"  inD_{rec_name}: {n_rows} rows, {n_cars} cars → {out_path}")
    return out_paths


def process_citysim() -> list[Path]:
    """Downsample all CitySim recordings (25fps → 10fps)."""
    out_paths = []
    for intersection in sorted(CITYSIM_DIR.iterdir()):
        if not intersection.is_dir():
            continue
        int_name = intersection.name  # "IntersectionA", "IntersectionD", "IntersectionE"
        # Extract letter after "Intersection": A, D, E
        short = int_name.replace("Intersection", "").strip()
        for fp in sorted(intersection.glob("*.csv")):
            rec_num = fp.stem.split("-")[-1]  # "01", "02", ...
            df = pd.read_csv(fp)
            df = _downsample_25_to_10(df)
            df = _rename_columns_for_consistency(df)
            out_path = OUT_DIR / f"CitySim_{short}{rec_num}.csv"
            df.to_csv(out_path, index=False)
            out_paths.append(out_path)
            n_rows = len(df)
            n_cars = df["carId"].nunique()
            print(f"  CitySim_{short}{rec_num}: {n_rows} rows, {n_cars} cars → {out_path}")
    return out_paths


def process_ngsim() -> list[Path]:
    """Copy NGSIM Peachtree (native 10fps, already sequential frameNum)."""
    fp = NGSIM_DIR / "peachtree.csv"
    df = pd.read_csv(fp)
    df = _rename_columns_for_consistency(df)
    out_path = OUT_DIR / "NGSIM_Peachtree.csv"
    df.to_csv(out_path, index=False)
    n_rows = len(df)
    n_cars = df["carId"].nunique()
    print(f"  NGSIM_Peachtree: {n_rows} rows, {n_cars} cars → {out_path}")
    return [out_path]


# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old files
    for old in OUT_DIR.glob("*.csv"):
        old.unlink()
        print(f"  removed old: {old}")

    print("\n--- inD (25fps → 10fps) ---")
    inD_paths = process_inD()

    print("\n--- CitySim (25fps → 10fps) ---")
    citysim_paths = process_citysim()

    print("\n--- NGSIM (native 10fps) ---")
    ngsim_paths = process_ngsim()

    total = len(inD_paths) + len(citysim_paths) + len(ngsim_paths)
    print(f"\nDone: {total} files written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
