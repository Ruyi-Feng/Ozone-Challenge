"""Fix NGSIM Peachtree carId reuse: split shared carIds into unique IDs.

NGSIM source data reuses carId across different vehicles in the same
recording.  This script detects (carId, frameNum) duplicates, clusters
each duplicate set by spatial position across frames, and assigns a
unique carId suffix per cluster.

    carId=2  →  carId=2_0  (vehicle A, tracked at ~111,557)
                carId=2_1  (vehicle B, tracked at ~189,24)

Usage (from repo root):
  python data_processing/scripts/fix_ngsim_carid.py

Modifies data/raw/NGSIM_Peachtree.csv in-place (backup at .bak).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

INPUT = Path("data/raw/NGSIM_Peachtree.csv")

# Maximum distance (meters) for matching a vehicle to its track in the
# next frame.  NGSIM has up to ~17 m/s × 0.1s = 1.7m between frames.
MAX_MATCH_DISTANCE_M = 5.0


def _fix_one_carid(df_car: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Resolve duplicate rows for a single (potentially shared) carId.

    Uses frame-to-frame minimum-distance matching to track each physical
    vehicle independently, then assigns ``carId_N`` suffixes.
    """
    frames = sorted(df_car["frameNum"].unique())
    dup_frames = [
        f for f in frames
        if len(df_car[df_car["frameNum"] == f]) > 1
    ]
    if not dup_frames:
        # No duplicates — single vehicle, keep as-is
        return df_car

    # ── Build tracks by frame-to-frame matching ────────────────────────
    # active_tracks: list of dicts, each dict is {frameNum: row_index}
    tracks: list[dict] = []

    for fn in frames:
        rows_f = df_car[df_car["frameNum"] == fn]
        n_this = len(rows_f)
        positions = rows_f[["carCenterXm", "carCenterYm"]].to_numpy(dtype=np.float64)

        if not tracks:
            # First frame — each row starts a new track
            for k in range(n_this):
                tracks.append({fn: rows_f.index[k]})
            continue

        # Predict where each track should be in this frame
        n_tracks = len(tracks)
        # Build cost matrix: track i → row j distance
        cost = np.full((n_tracks, n_this), 1e9)
        for i, tr in enumerate(tracks):
            last_fn = max(tr.keys())
            last_row = df_car.loc[tr[last_fn]]
            last_pos = np.array(
                [last_row["carCenterXm"], last_row["carCenterYm"]],
                dtype=np.float64,
            )
            for j in range(n_this):
                d = float(np.linalg.norm(positions[j] - last_pos))
                if d <= max_dist:
                    cost[i, j] = d

        # Hungarian algorithm: best matching
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_rows: set[int] = set()
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < 1e8:  # valid match
                tracks[i][fn] = rows_f.index[j]
                matched_rows.add(j)

        # Unmatched rows → new tracks
        for j in range(n_this):
            if j not in matched_rows:
                tracks.append({fn: rows_f.index[j]})

    # ── Assign new carIds ──────────────────────────────────────────────
    if len(tracks) <= 1:
        return df_car

    result = df_car.copy()
    base_id = int(df_car["carId"].iloc[0])
    for tidx, tr in enumerate(tracks):
        new_id = f"{base_id}_{tidx}"
        for row_idx in tr.values():
            result.at[row_idx, "carId"] = new_id

    return result


def main() -> None:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found")
        return

    df = pd.read_csv(INPUT)
    print(f"Loaded: {len(df):,} rows, {df['carId'].nunique()} unique carIds")

    # Find carIds with duplicates (multiple rows per frameNum)
    dup_per_car = df.groupby("carId")["frameNum"].apply(
        lambda x: int(x.duplicated().sum())
    )
    problem_cars = dup_per_car[dup_per_car > 0]
    n_problem = len(problem_cars)
    n_dup_rows = int(problem_cars.sum())
    print(f"carIds with duplicate frames: {n_problem}")
    print(f"Duplicate rows: {n_dup_rows} ({n_dup_rows/len(df)*100:.1f}%)")

    if n_problem == 0:
        print("No issues found.")
        return

    # ── Backup ─────────────────────────────────────────────────────────
    backup = INPUT.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy2(INPUT, backup)
        print(f"Backup: {backup}")

    # ── Fix each problematic carId ─────────────────────────────────────
    unique_orig = df["carId"].nunique()

    all_parts = []
    for cid in sorted(problem_cars.index):
        df_car = df[df["carId"] == cid]
        fixed = _fix_one_carid(df_car, MAX_MATCH_DISTANCE_M)
        all_parts.append(fixed)

    # Put back the clean carIds
    clean_mask = ~df["carId"].isin(problem_cars.index)
    all_parts.insert(0, df[clean_mask])

    result = pd.concat(all_parts, ignore_index=True)
    result = result.sort_values(["frameNum", "carId"]).reset_index(drop=True)

    unique_new = result["carId"].nunique()
    print(f"carIds: {unique_orig} → {unique_new} (+{unique_new - unique_orig})")
    print(f"Rows: {len(result):,} (was {len(df):,})")

    result.to_csv(INPUT, index=False)
    print(f"Saved: {INPUT}")


if __name__ == "__main__":
    main()
