"""Merge highway and intersection event CSVs for combined model training.

Concatenates highway (_highway_{split}.csv) and intersection (_{split}.csv)
event CSVs, renumbering Event_ids across the full dataset to avoid collisions.

Usage (from repo root):
  python data_processing/scripts/merge_highway_intersection.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── config ──────────────────────────────────────────────────────────────────

OUT_DIR = Path("data/processed")

FILE_TYPES = [
    ("data", "events_data"),
    ("labels", "events_labels"),
    ("future_traj", "events_future_traj"),
]

SPLITS = ["train", "val"]


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    for split in SPLITS:
        # ── Load labels first to determine Event_id mapping ────────────────
        hw_labels_path = OUT_DIR / "labels" / f"events_labels_highway_{split}.csv"
        ix_labels_path = OUT_DIR / "labels" / f"events_labels_{split}.csv"

        if not hw_labels_path.exists():
            print(f"SKIP {split}: {hw_labels_path} not found (highway pipeline not run yet?)")
            continue
        if not ix_labels_path.exists():
            print(f"SKIP {split}: {ix_labels_path} not found")
            continue

        hw_labels = _load(str(hw_labels_path))
        ix_labels = _load(str(ix_labels_path))

        n_hw = len(hw_labels)
        n_ix = len(ix_labels)
        print(f"\n--- {split} ---")
        print(f"  highway:     {n_hw} events")
        print(f"  intersection: {n_ix} events")

        # ── Build Event_id mapping to avoid collisions ─────────────────────
        # Shift highway Event_ids by adding an offset: highway ids → offset + 1..N
        # Use a large offset to guarantee no collision with intersection ids.
        max_ix_id = int(ix_labels["Event_id"].max()) if n_ix > 0 else 0
        offset = max(0, max_ix_id) + 1
        hw_id_map = {old: old + offset for old in hw_labels["Event_id"]}

        print(f"  intersection max Event_id: {max_ix_id}")
        print(f"  highway Event_ids remapped to: [{offset + int(hw_labels['Event_id'].min())}, "
              f"{offset + int(hw_labels['Event_id'].max())}]")

        # ── Process each file type ─────────────────────────────────────────
        for dir_name, prefix in FILE_TYPES:
            hw_path = OUT_DIR / dir_name / f"{prefix}_highway_{split}.csv"
            ix_path = OUT_DIR / dir_name / f"{prefix}_{split}.csv"
            out_path = OUT_DIR / dir_name / f"{prefix}_combined_{split}.csv"

            hw_df = _load(str(hw_path))
            ix_df = _load(str(ix_path))

            # Remap highway Event_ids
            hw_df["Event_id"] = hw_df["Event_id"].map(hw_id_map)

            merged = pd.concat([hw_df, ix_df], ignore_index=True)
            merged.to_csv(out_path, index=False)

            print(f"  {prefix}_combined_{split}.csv: {len(merged)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
