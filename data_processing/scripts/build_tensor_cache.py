"""Pre-compute all event tensors and save as memory-mapped numpy arrays.

Reads Parquet per-event via the BaselineConflictDataset tensor logic, then
writes fixed-size [N, A, T, F] float32 arrays + labels via numpy memmap.
The resulting dataset loads in < 1 second via mmap with zero-copy indexing.

Produces:
  data/processed/train_x.npy        float32 [N, 7, 80, 4]
  data/processed/train_mask.npy     bool     [N, 7]
  data/processed/train_y.npy        float32 [N]        (is_conflict)
  data/processed/train_target.npy   int64    [N]        (target_idx)
  data/processed/train_valid.npy    bool     [N, 7, 80] (True = real observation)
  (same for val_*)

valid.npy marks frames actually written from parquet rows.  False frames are
either front zero-padding (history shorter than 80 frames), interior missing
frames, or empty agent slots.  They are indistinguishable from real data in
x.npy (an ego frame can legitimately be all-zero), so this mask is the ONLY
trustworthy source of frame validity — never infer it from x values.

Usage (from repo root):
  python data_processing/scripts/build_tensor_cache.py               # full build
  python data_processing/scripts/build_tensor_cache.py --valid-only  # only add
      {prefix}_valid.npy next to an EXISTING cache, preserving its exact row
      order (taken from {prefix}_eid.npy) without rewriting x/mask/y/target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# ── config ────────────────────────────────────────────────────────────────

TASKS = [
    # ── intersection only ──
    {
        "parquet": "data/processed/train_events.parquet",
        "labels": "data/processed/labels/events_labels_train.csv",
        "valid": "data/processed/train_events.valid_events",
        "out_prefix": "data/processed/train",
        "name": "train",
    },
    {
        "parquet": "data/processed/val_events.parquet",
        "labels": "data/processed/labels/events_labels_val.csv",
        "valid": "data/processed/val_events.valid_events",
        "out_prefix": "data/processed/val",
        "name": "val",
    },
    # ── highway + intersection combined ──
    {
        "parquet": "data/processed/train_combined_events.parquet",
        "labels": "data/processed/labels/events_labels_combined_train.csv",
        "valid": "data/processed/train_combined_events.valid_events",
        "out_prefix": "data/processed/combined_train",
        "name": "combined_train",
    },
    {
        "parquet": "data/processed/val_combined_events.parquet",
        "labels": "data/processed/labels/events_labels_combined_val.csv",
        "valid": "data/processed/val_combined_events.valid_events",
        "out_prefix": "data/processed/combined_val",
        "name": "combined_val",
    },
]

NUM_AGENTS = 7
NUM_FRAMES = 80
NUM_FEATURES = 4

SLOT_ORDER = (
    "ego", "front", "rear",
    "left_front", "left_rear", "right_front", "right_rear",
)
SLOT_TO_IDX = {s: i for i, s in enumerate(SLOT_ORDER)}
SLOT_TO_TARGET = {i: i - 1 for i in range(1, len(SLOT_ORDER))}


def _normalize_angle_diff(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _load_valid_set(path: str) -> set[int] | None:
    """Optional Event_id filter, one integer per line.

    The .valid_events file is a one-off artifact from the training machine
    (nothing in this repo generates it).  Missing file → None = keep every
    labelled event.
    """
    if not Path(path).exists():
        return None
    with open(path) as f:
        return {int(line.strip()) for line in f if line.strip()}


def build_tensor(
    evt: pd.DataFrame,
    label: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Mirrors BaselineConflictDataset.__getitem__ tensor logic, returns numpy."""
    t0 = float(label["t0"])
    is_conflict = int(label["is_conflict"])
    conflict_target_id = int(label["conflict_target_id"])

    # 1. carId → role mapping
    car_roles: dict[int, str] = {}
    for cid, role in zip(evt["carId"], evt["role"]):
        cid_i = int(cid)
        if cid_i not in car_roles:
            car_roles[cid_i] = role

    # Identify ego
    ego_id = None
    for cid, role in car_roles.items():
        if role == "ego":
            ego_id = cid
            break
    if ego_id is None:
        raise ValueError("no ego vehicle")

    # 2. Slot assignment
    slot_cars: dict[int, int] = {}
    frame_counts = evt.groupby("carId").size()
    for cid, role in car_roles.items():
        slot_idx = SLOT_TO_IDX.get(role)
        if slot_idx is None:
            continue
        if slot_idx not in slot_cars:
            slot_cars[slot_idx] = cid
        else:
            existing = slot_cars[slot_idx]
            if cid == conflict_target_id:
                slot_cars[slot_idx] = cid
            elif existing != conflict_target_id:
                if frame_counts.get(cid, 0) > frame_counts.get(existing, 0):
                    slot_cars[slot_idx] = cid

    # 3. Ego reference state at t0
    ego_rows = evt[evt["carId"] == ego_id]
    ego_at_t0 = ego_rows[ego_rows["frameNum"] <= t0]
    if ego_at_t0.empty:
        ego_rows_sorted = ego_rows.copy()
        ego_rows_sorted["_dist"] = np.abs(ego_rows_sorted["frameNum"] - t0)
        ego_at_t0 = ego_rows_sorted.loc[[ego_rows_sorted["_dist"].idxmin()]]
    ego_ref = ego_at_t0.iloc[-1]
    ego_x0 = float(ego_ref["carCenterXm"])
    ego_y0 = float(ego_ref["carCenterYm"])
    ego_h0 = float(ego_ref["heading"])

    # 4. Frame axis
    all_frames = sorted(evt["frameNum"].unique())
    history_frames = [f for f in all_frames if f <= t0]
    if len(history_frames) >= NUM_FRAMES:
        frame_window = history_frames[-NUM_FRAMES:]
    else:
        frame_window = history_frames
    T_actual = len(frame_window)

    # 5. Build tensors
    x = np.zeros((NUM_AGENTS, NUM_FRAMES, NUM_FEATURES), dtype=np.float32)
    agent_mask = np.zeros(NUM_AGENTS, dtype=bool)
    valid = np.zeros((NUM_AGENTS, NUM_FRAMES), dtype=bool)

    for slot_idx in range(NUM_AGENTS):
        car_id = slot_cars.get(slot_idx)
        if car_id is None:
            continue
        agent_mask[slot_idx] = True
        car_rows = evt[evt["carId"] == car_id]
        car_by_frame = {}
        for _, r in car_rows.iterrows():
            car_by_frame[int(r["frameNum"])] = r

        for t_idx, fn in enumerate(frame_window):
            out_t = NUM_FRAMES - T_actual + t_idx
            row = car_by_frame.get(int(fn))
            if row is None:
                continue
            x[slot_idx, out_t, 0] = float(row["carCenterXm"]) - ego_x0
            x[slot_idx, out_t, 1] = float(row["carCenterYm"]) - ego_y0
            x[slot_idx, out_t, 2] = _normalize_angle_diff(
                float(row["heading"]), ego_h0
            )
            x[slot_idx, out_t, 3] = float(row["speed"])
            valid[slot_idx, out_t] = True

    # 6. Target index
    target_idx = -1
    if is_conflict and conflict_target_id != -1:
        for slot_idx, cid in slot_cars.items():
            if cid == conflict_target_id and slot_idx in SLOT_TO_TARGET:
                target_idx = SLOT_TO_TARGET[slot_idx]
                break

    return x, agent_mask, valid, is_conflict, target_idx


def process_split(parquet_path: str, labels_path: str, valid_path: str,
                  out_prefix: str, name: str, valid_only: bool = False) -> None:
    """Iterate all valid events, build tensors, write memmap arrays.

    valid_only=True: only (re)write {out_prefix}_valid.npy, taking N and the
    exact row order from the EXISTING {out_prefix}_eid.npy.  This retrofits a
    validity mask onto a deployed cache without rebuilding x/mask/y/target and
    without needing the one-off .valid_events filter file.
    """
    labels_df = pd.read_csv(labels_path)

    if valid_only:
        eid_path = f"{out_prefix}_eid.npy"
        if not Path(eid_path).exists():
            raise FileNotFoundError(
                f"{eid_path} not found — --valid-only needs an existing cache "
                f"(run the full build first)."
            )
        event_ids = [int(e) for e in np.load(eid_path)]
        print(f"  [{name}] --valid-only: row order from {eid_path}")
    else:
        valid_set = _load_valid_set(valid_path)
        if valid_set is None:
            print(f"  [{name}] NOTE: {valid_path} not found — keeping all labelled events")
        else:
            labels_df = labels_df[labels_df["Event_id"].isin(valid_set)]
        event_ids = sorted(labels_df["Event_id"].tolist())

    N = len(event_ids)
    print(f"  [{name}] {N} events")

    # Allocate memmap files
    valid_mem = np.lib.format.open_memmap(
        f"{out_prefix}_valid.npy", mode="w+", dtype=bool,
        shape=(N, NUM_AGENTS, NUM_FRAMES),
    )
    if not valid_only:
        x_shape = (N, NUM_AGENTS, NUM_FRAMES, NUM_FEATURES)
        x_path = f"{out_prefix}_x.npy"
        mask_path = f"{out_prefix}_mask.npy"
        y_path = f"{out_prefix}_y.npy"
        target_path = f"{out_prefix}_target.npy"

        x_mem = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=x_shape)
        mask_mem = np.lib.format.open_memmap(mask_path, mode="w+", dtype=bool, shape=(N, NUM_AGENTS))
        y_mem = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=(N,))
        target_mem = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.int64, shape=(N,))
        eid_mem = np.lib.format.open_memmap(f"{out_prefix}_eid.npy", mode="w+", dtype=np.int64, shape=(N,))
        eid_mem[:] = np.array(event_ids, dtype=np.int64)
        eid_mem.flush()

    # Build labels lookup
    label_map: dict[int, pd.Series] = {}
    for _, row in labels_df.iterrows():
        label_map[int(row["Event_id"])] = row

    n_no_label = sum(1 for eid in event_ids if eid not in label_map)
    if n_no_label:
        print(f"  [{name}] WARNING: {n_no_label} cached events missing from "
              f"{labels_path} — their valid rows stay all-False")

    pf = pq.ParquetFile(parquet_path)
    num_rg = pf.metadata.num_row_groups

    # Build event → output index mapping
    eid_to_idx = {eid: i for i, eid in enumerate(event_ids)}

    for rg_idx in range(num_rg):
        if (rg_idx + 1) % 20 == 0:
            valid_mem.flush()
            if not valid_only:
                x_mem.flush()
            print(f"  [{name}] row group {rg_idx+1}/{num_rg} ...")

        rg_table = pf.read_row_group(rg_idx)
        rg_df = rg_table.to_pandas()

        # Group by Event_id within this row group
        for eid, evt in rg_df.groupby("Event_id"):
            idx = eid_to_idx.get(eid)
            if idx is None:
                continue  # event not in valid set

            label = label_map.get(eid)
            if label is None:
                continue

            try:
                x_arr, mask_arr, valid_arr, is_conflict, target_idx = build_tensor(evt, label)
            except ValueError:
                continue  # no ego in this row group's portion

            valid_mem[idx] = valid_arr
            if not valid_only:
                x_mem[idx] = x_arr
                mask_mem[idx] = mask_arr
                y_mem[idx] = float(is_conflict)
                target_mem[idx] = target_idx

    valid_mem.flush()
    if not valid_only:
        x_mem.flush()
        mask_mem.flush()
        y_mem.flush()
        target_mem.flush()

    # Sanity report: events that never got a single real frame written
    # (label present but event absent from parquet, or no ego portion found).
    dead_rows = int((~valid_mem.reshape(N, -1).any(axis=1)).sum())
    if dead_rows:
        print(f"  [{name}] WARNING: {dead_rows}/{N} events have an all-False "
              f"valid mask (all-zero tensors) — they carry no observation")

    if valid_only:
        size_mb = Path(f"{out_prefix}_valid.npy").stat().st_size / (1024**2)
        print(f"  [{name}] done: {N} events, valid.npy {size_mb:.1f} MB")
    else:
        size_mb = sum(
            Path(p).stat().st_size
            for p in [x_path, mask_path, y_path, target_path]
        ) / (1024**2)
        print(f"  [{name}] done: {N} events, {size_mb:.0f} MB total")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--valid-only", action="store_true",
        help="Only write {prefix}_valid.npy for an existing cache, keeping "
             "the row order of {prefix}_eid.npy (no x/mask/y/target rewrite).",
    )
    args = ap.parse_args(argv)
    for task in TASKS:
        print(f"\nProcessing {task['name']} ...")
        process_split(
            task["parquet"], task["labels"], task["valid"],
            task["out_prefix"], task["name"], valid_only=args.valid_only,
        )


if __name__ == "__main__":
    main()
