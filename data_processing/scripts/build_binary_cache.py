"""Build flat binary cache + index CSV from existing memmap tensor cache.

Reads the pre-computed memmap .npy arrays and writes:
  - {prefix}_data.bin   — float32 tensor data, one event after another
  - {prefix}_index.csv  — per-event metadata (offset, labels, mask, etc.)
  - {prefix}_valid.bin  — bit-packed per-frame validity, if {prefix}_valid.npy
                          exists (see build_tensor_cache.py)

The binary format is simply concatenated float32 [7, 80, 12] tensors.
Each event occupies exactly 7 × 80 × 12 × 4 = 26880 bytes.
The byte offset of event N is N × 26880.
valid.bin packs the [7, 80] bool mask of event N into 70 bytes at N × 70.

Usage (from repo root):
  python data_processing/scripts/build_binary_cache.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────

TASKS = [
    {
        "prefix": "data/processed/train",
        "name": "train",
    },
    {
        "prefix": "data/processed/val",
        "name": "val",
    },
]

NUM_AGENTS = 7
NUM_FRAMES = 80
NUM_FEATURES = 12
RECORD_BYTES = NUM_AGENTS * NUM_FRAMES * NUM_FEATURES * 4  # 26880
VALID_RECORD_BYTES = (NUM_AGENTS * NUM_FRAMES + 7) // 8  # 70


def _load_labels(prefix: str) -> pd.DataFrame:
    """Try to locate the labels CSV from the prefix path."""
    # prefix is e.g. "data/processed/train"
    # labels are in "data/processed/labels/events_labels_train.csv"
    import re

    base = Path(prefix)
    # Try the standard naming convention
    candidates = [
        base.parent / "labels" / f"events_labels_{base.name}.csv",
        Path(f"data/processed/labels/events_labels_{base.name}.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    print(f"  WARNING: no labels CSV found for {prefix} — "
          f"is_conflict/target_idx will be from memmap only")
    return pd.DataFrame()


def process_split(prefix: str, name: str) -> None:
    """Read memmap arrays and write flat binary + index CSV."""

    # ── Load memmap arrays ─────────────────────────────────────────────
    x = np.load(f"{prefix}_x.npy", mmap_mode="r")
    mask = np.load(f"{prefix}_mask.npy", mmap_mode="r")
    y = np.load(f"{prefix}_y.npy", mmap_mode="r")
    target = np.load(f"{prefix}_target.npy", mmap_mode="r")
    eid = np.load(f"{prefix}_eid.npy", mmap_mode="r")

    N = len(x)
    assert x.shape == (N, NUM_AGENTS, NUM_FRAMES, NUM_FEATURES), \
        f"Unexpected x shape: {x.shape}"

    valid_npy = Path(f"{prefix}_valid.npy")
    valid = np.load(str(valid_npy), mmap_mode="r") if valid_npy.exists() else None
    if valid is None:
        print(f"  WARNING: {valid_npy} not found — skipping {prefix}_valid.bin "
              f"(run build_tensor_cache.py [--valid-only] first)")
    else:
        assert valid.shape == (N, NUM_AGENTS, NUM_FRAMES), \
            f"valid.npy shape {valid.shape} does not match cache N={N}"

    # ── Try to load labels CSV for scene_id / conflict_target_role ──────
    labels_df = _load_labels(prefix)
    has_labels = not labels_df.empty
    if has_labels:
        label_map: dict[int, dict] = {}
        for _, row in labels_df.iterrows():
            eid_val = int(row["Event_id"])
            label_map[eid_val] = {
                "scene_id": str(row.get("scene_id", "")),
                "conflict_target_role": str(row.get("conflict_target_role", "")),
            }

    # ── Write binary data file ─────────────────────────────────────────
    bin_path = f"{prefix}_data.bin"
    index_rows: list[dict] = []

    with open(bin_path, "wb") as f:
        for i in range(N):
            offset = i * RECORD_BYTES
            tensor = np.asarray(x[i], dtype=np.float32)
            f.write(tensor.tobytes())

            # Pack agent_mask as a single uint8 (7 bits)
            mask_bits = 0
            for j in range(NUM_AGENTS):
                if bool(mask[i, j]):
                    mask_bits |= (1 << j)

            meta = label_map.get(int(eid[i]), {}) if has_labels else {}

            index_rows.append({
                "event_id": int(eid[i]),
                "byte_offset": offset,
                "byte_length": RECORD_BYTES,
                "is_conflict": int(y[i]),
                "target_idx": int(target[i]),
                "agent_mask": mask_bits,
                "scene_id": meta.get("scene_id", ""),
                "conflict_target_role": meta.get("conflict_target_role", ""),
            })

        f.flush()

    # ── Write bit-packed validity file (index-aligned, 70 B/event) ──────
    if valid is not None:
        vbin_path = f"{prefix}_valid.bin"
        with open(vbin_path, "wb") as vf:
            for i in range(N):
                bits = np.packbits(
                    np.asarray(valid[i], dtype=bool).reshape(-1)
                )
                assert bits.nbytes == VALID_RECORD_BYTES
                vf.write(bits.tobytes())
        print(f"  [{name}] {vbin_path} "
              f"({Path(vbin_path).stat().st_size / 1024:.0f} KB)")

    # ── Write index CSV ─────────────────────────────────────────────────
    index_path = f"{prefix}_index.csv"
    index_df = pd.DataFrame(index_rows, columns=[
        "event_id", "byte_offset", "byte_length", "is_conflict",
        "target_idx", "agent_mask", "scene_id", "conflict_target_role",
    ])
    index_df.to_csv(index_path, index=False)

    bin_size_mb = Path(bin_path).stat().st_size / (1024 ** 2)
    idx_size_kb = Path(index_path).stat().st_size / 1024
    print(f"  [{name}] {N} events: {bin_path} ({bin_size_mb:.1f} MB), "
          f"{index_path} ({idx_size_kb:.0f} KB)")


def main() -> None:
    for task in TASKS:
        prefix = task["prefix"]
        name = task["name"]
        x_path = f"{prefix}_x.npy"
        if not Path(x_path).exists():
            print(f"  [{name}] SKIP: {x_path} not found")
            continue
        print(f"\nProcessing {name} ({prefix}) ...")
        process_split(prefix, name)


if __name__ == "__main__":
    main()
