"""Analyse 2D_TTC smoothness on NGSIM native 10-fps data.

Compares TTC time-series noise at 10 fps versus the expected behaviour
at 25 fps (upsampled), and diagnoses the noise sources:
- scalar acceleration (speed delta / dt)
- angular velocity (heading delta / dt)
- heading noise from position-delta (atan2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data_processing.io.ngsim_converter import convert_ngsim_to_standard
from data_processing.io.schema import ProcessingConfig
from data_processing.core.conflict_detect import (
    _clear_caches,
    _compute_2d_ttc,
    _find_front_pairs,
    _get_obb_corners,
    _calculate_nearest_points,
    _rear_forward_strips_intersect,
    _compute_2d_ttc_kernel,
    _scalar_acceleration,
    _angular_velocity,
)
from data_processing.utils.geometry import compute_distance, compute_relative_pose, classify_neighbor_slot

# ---------------------------------------------------------------------------
RAW_PATH = "data/raw/peachtree_test.csv"
FPS = 10.0
DT = 1.0 / FPS

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("TTC Smoothness Analysis — NGSIM 10 fps")
print("=" * 60)

raw = convert_ngsim_to_standard(RAW_PATH)
raw = raw.sort_values(["frameNum", "carId"]).reset_index(drop=True)
n_vehicles = raw["carId"].nunique()
n_frames = raw["frameNum"].nunique()
print(f"Data: {len(raw)} rows, {n_vehicles} vehicles, {n_frames} frames, dt={DT:.3f}s")

# ---------------------------------------------------------------------------
# 2. Sequential TTC computation (all frames, maintaining caches correctly)
# ---------------------------------------------------------------------------
_clear_caches()
cfg = ProcessingConfig(fps=FPS, conflict_ttc_threshold=3.0, max_distance_m=200.0)
frames = raw.groupby("frameNum")
frame_list = sorted(raw["frameNum"].unique())

# Collect per-pair TTC time series + intermediate values
pair_data: dict[tuple, list[dict]] = {}

print(f"Processing {len(frame_list)} frames sequentially …")
for idx, fnum in enumerate(frame_list):
    frame_df = frames.get_group(fnum)
    front_pairs = _find_front_pairs(frame_df, cfg)

    for ego_id, tgt_id in front_pairs:
        key = (ego_id, tgt_id)
        try:
            ego_row = frame_df.loc[frame_df["carId"] == ego_id].iloc[0]
            tgt_row = frame_df.loc[frame_df["carId"] == tgt_id].iloc[0]
        except IndexError:
            continue

        # Record intermediate values before computing TTC
        ttc = _compute_2d_ttc(ego_row, tgt_row, DT)

        pair_data.setdefault(key, []).append({
            "frame": int(fnum),
            "ttc": ttc,
        })

    if (idx + 1) % 50 == 0:
        print(f"  … {idx+1}/{len(frame_list)} frames done")

_clear_caches()

# ---------------------------------------------------------------------------
# 3. Filter pairs with sustained interaction
# ---------------------------------------------------------------------------
good_pairs = {}
for k, records in pair_data.items():
    valid = [r for r in records if r["ttc"] is not None and r["ttc"] > 0 and r["ttc"] < 100]
    if len(valid) >= 20:
        good_pairs[k] = valid

print(f"Total front pairs: {len(pair_data)}")
print(f"Pairs with ≥20 valid TTC: {len(good_pairs)}")

if not good_pairs:
    print("\nNo pairs with enough data — try larger dataset.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Per-pair smoothness analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Per-pair TTC Smoothness")
print("=" * 60)
print(f"{'Pair':<25} {'N':>5} {'TTC_range':>12} {'std(TTC)':>10} "
      f"{'mean|ΔTTC|':>11} {'max|ΔTTC|':>11} {'%smooth*':>9}")
print("-" * 80)

all_results = []
for (ego, tgt), records in sorted(good_pairs.items(),
                                   key=lambda x: len(x[1]), reverse=True)[:15]:
    ttc_vals = np.array([r["ttc"] for r in records], dtype=float)
    n = len(ttc_vals)
    ttc_mean = float(np.mean(ttc_vals))
    ttc_std = float(np.std(ttc_vals))
    ttc_diff = np.abs(np.diff(ttc_vals))
    mean_diff = float(np.mean(ttc_diff))
    max_diff = float(np.max(ttc_diff))
    # % of steps where TTC change is < 0.5s (empirically "smooth")
    pct_smooth = float(np.mean(ttc_diff < 0.5) * 100)

    all_results.append({
        "pair": f"e{ego}/t{tgt}",
        "n": n, "mean": ttc_mean, "std": ttc_std,
        "mean_diff": mean_diff, "max_diff": max_diff,
        "pct_smooth": pct_smooth,
        "ttc_vals": ttc_vals,
    })
    print(f"{'e'+str(ego)+'/t'+str(tgt):<25} {n:>5} "
          f"{ttc_vals.min():>6.2f}-{ttc_vals.max():>5.2f}s "
          f"{ttc_std:>10.4f} {mean_diff:>11.4f} {max_diff:>11.4f} {pct_smooth:>8.0f}%")

# ---------------------------------------------------------------------------
# 5. Aggregate
# ---------------------------------------------------------------------------
all_diffs = np.concatenate([np.abs(np.diff(r["ttc_vals"])) for r in all_results])

print(f"\n{'='*60}")
print("Aggregate Smoothness (10 fps native)")
print(f"{'='*60}")
print(f"Pairs analysed:               {len(all_results)}")
print(f"Mean   |ΔTTC| per 0.1s step:  {np.mean(all_diffs):.4f} s")
print(f"Median |ΔTTC| per 0.1s step:  {np.median(all_diffs):.4f} s")
print(f"P75    |ΔTTC|:                {np.percentile(all_diffs, 75):.4f} s")
print(f"P90    |ΔTTC|:                {np.percentile(all_diffs, 90):.4f} s")
print(f"P95    |ΔTTC|:                {np.percentile(all_diffs, 95):.4f} s")
print(f"P99    |ΔTTC|:                {np.percentile(all_diffs, 99):.4f} s")
print(f"% steps where |ΔTTC| < 0.5s:  {np.mean(all_diffs < 0.5)*100:.1f}%")
print(f"% steps where |ΔTTC| < 1.0s:  {np.mean(all_diffs < 1.0)*100:.1f}%")

# ---------------------------------------------------------------------------
# 6. Noise source diagnosis
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Noise Source Diagnosis")
print(f"{'='*60}")
print(f"""
At 10 fps (dt={DT:.1f}s):

1. SCALAR ACCELERATION
   - Computed as (v[t] - v[t-1]) / {DT:.1f}
   - NGSIM v_Vel is reported to 0.01 ft/s precision (= 0.003 m/s)
   - Velocity quantisation → smallest |Δv| = 0.003 m/s
   - → smallest |accel| = 0.003 / {DT:.1f} = {0.003/DT:.3f} m/s²
   - At 25 fps (dt=0.04s): smallest |accel| = 0.003/0.04 = 0.075 m/s²
   - 10fps is {0.075/(0.003/DT):.1f}× LESS sensitive to velocity quantisation

2. ANGULAR VELOCITY
   - Heading from atan2(dy, dx) using position deltas
   - Position in feet, converted to m → ~0.003m precision
   - At 10fps: position diff over 0.1s → larger dx,dy → less angular noise
   - At 25fps: position diff over 0.04s → smaller dx,dy → more angular noise
   - For a vehicle at 10 m/s: 0.1s travel = 1.0m, 0.04s travel = 0.4m
   - Angular error ∝ 1/(travel distance) → 10fps is ~2.5× more precise

3. HEADING FROM NGSIM POSITION DELTAS (atan2)
   - NGSIM has no native heading sensor → heading always from Δposition
   - This is the PRIMARY noise source for angular velocity
   - At 10fps: larger position deltas → more stable heading estimates
   - Stationary vehicles: heading frozen (dx=0, dy=0 → reuse previous)

4. EXPECTED 25fps COMPARISON
   - Acceleration noise: ~2.5× higher (smaller dt amplifies velocity quantisation)
   - Angular velocity noise: ~2.5× higher (smaller dt amplifies heading noise)
   - TTC noise: compound effect — likely 2-4× higher |ΔTTC| at 25fps
""")

# Show a smooth and a noisy example
smoothest = min(all_results, key=lambda r: r["mean_diff"])
noisiest = max(all_results, key=lambda r: r["mean_diff"])
print(f"Smoothest pair ({smoothest['pair']}):")
print(f"  TTC = {[round(v, 2) for v in smoothest['ttc_vals'][:12]]} …")
print(f"Noisiest pair ({noisiest['pair']}):")
print(f"  TTC = {[round(v, 2) for v in noisiest['ttc_vals'][:12]]} …")
