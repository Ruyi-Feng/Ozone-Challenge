"""Convert freewayD (Hong Kong real freeway) raw CSVs to NBDT-standard format.

Input format (freewayD-XX_final.csv):
  frameNUM, carID, boundingBox1X..4Y, carCenterX, carCenterY,
  carCenterLat, carCenterLon, boundingBox1Lat..4Lon, laneNumber

Pipeline-required columns (data_processing/io/schema.py RAW_REQUIRED_COLUMNS):
  frameNum, carId, carCenterXm, carCenterYm, heading, speed, objClass
  (+ laneId optional)

Steps:
  1. Rename columns (frameNUM -> frameNum, carID -> carId, laneNumber -> laneId)
  2. GPS (Lat/Lon) -> local meters (equirectangular, reference = first row)
  3. heading/speed from smoothed (5-frame) position deltas @ 30 Hz
  4. Downsample 30 Hz -> 10 Hz (keep frameNum % 3 == 0, one of every 3 frames)
  5. objClass = 0 (car); laneId kept from laneNumber

Usage (from repo root):
  python data_processing/scripts/convert_freewayd.py \
      --src "/path/to/Freeway D/Freeway D/Trajectories" \
      --out data/raw_heldout
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0
DT = 1.0 / FPS
SMOOTH_WIN = 6          # frames for heading/speed smoothing (0.2 s)
LAT2M = 110540.0        # meters per degree latitude
LON2M = 111320.0        # meters per degree longitude at equator

KEEP_COLUMNS = [
    "frameNum", "carId", "carCenterXm", "carCenterYm",
    "heading", "speed", "objClass", "laneId",
]


def convert_file(src: Path, dst: Path) -> None:
    print(f"  {src.name} ...")
    df = pd.read_csv(src)

    # 1. rename to pipeline column names
    df = df.rename(columns={
        "frameNUM": "frameNum",
        "carID": "carId",
        "laneNumber": "laneId",
    })

    # 2. GPS -> local meters (equirectangular projection)
    lat0 = float(df["carCenterLat"].iloc[0])
    lon0 = float(df["carCenterLon"].iloc[0])
    cos_lat = math.cos(math.radians(lat0))
    df["carCenterXm"] = (df["carCenterLon"] - lon0) * LON2M * cos_lat
    df["carCenterYm"] = (df["carCenterLat"] - lat0) * LAT2M

    # 3. sort by car + frame, derive heading/speed over a smoothed window
    df = df.sort_values(["carId", "frameNum"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("carId", sort=False)

    dx = g["carCenterXm"].shift(-SMOOTH_WIN) - df["carCenterXm"]
    dy = g["carCenterYm"].shift(-SMOOTH_WIN) - df["carCenterYm"]

    df["heading"] = np.degrees(np.arctan2(dy, dx)) % 360.0
    df["speed"] = np.sqrt(dx**2 + dy**2) / (SMOOTH_WIN * DT)

    # tail NaNs (last SMOOTH_WIN frames of each car) -> backfill
    df["heading"] = g["heading"].bfill().fillna(0.0)
    df["speed"] = g["speed"].bfill().fillna(0.0)

    # 4. objClass (freewayD has no class info -> all cars)
    df["objClass"] = 0

    # 5. downsample 30 Hz -> 10 Hz (keep 1 of every 3 frames)
    mask = df["frameNum"] % 3 == 0
    df = df.loc[mask].copy()
    kept = sorted(df["frameNum"].unique())
    frame_map = {old: new for new, old in enumerate(kept)}
    df["frameNum"] = df["frameNum"].map(frame_map)

    # 6. keep required columns + save
    df = df[KEEP_COLUMNS]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"    -> {dst.name}: {len(df)} rows, "
          f"{df['carId'].nunique()} cars, {df['frameNum'].nunique()} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Path to freewayD Trajectories directory",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw_heldout"),
        help="Output directory (default: data/raw_heldout)",
    )
    args = parser.parse_args()

    files = sorted(args.src.glob("freewayD-*_final.csv"))
    if not files:
        raise FileNotFoundError(f"No freewayD-*_final.csv under {args.src}")

    for fp in files:
        # freewayD-02_final.csv -> freewayD_02.csv
        num = fp.stem.split("-")[1].split("_")[0]
        convert_file(fp, args.out / f"freewayD_{num}.csv")

    print(f"\nDone: {len(files)} files -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
