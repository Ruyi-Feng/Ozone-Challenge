"""Post-hoc CSV rebalancer — filters exported CSVs to achieve target conflict:non-conflict ratio.

Reads the label CSV + data CSV produced by the pipeline, groups events by
scene_id, and downsamples the over-represented class (usually conflicts) to
meet the configured target ratio.

Optionally also balances conflict types (conflict_target_role) within each scene.

Usage (from repo root):
  python data_processing/scripts/rebalance_csv.py \\
      --labels data/processed/labels/events_labels_train.csv \\
      --data   data/processed/data/events_data_train.csv \\
      --config configs/data_processing.yaml

Produces:
  {labels}_rebalanced.csv
  {data}_rebalanced.csv
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from data_processing.io.readers import load_config
from data_processing.io.schema import DATA_COLUMNS, LABEL_COLUMNS, ProcessingConfig


def _load_labels(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _suffix_path(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def rebalance_labels(
    labels: pd.DataFrame,
    cfg: ProcessingConfig,
) -> pd.DataFrame:
    """Return a filtered label DataFrame with balanced conflict:non-conflict ratio.

    Groups by scene_id (when ``cfg.rebalance_per_scene`` is True), then
    downsamples the majority class within each scene.
    """
    target_ratio = cfg.rebalance_target_ratio
    seed = cfg.rebalance_seed
    per_scene = cfg.rebalance_per_scene

    if per_scene and "scene_id" in labels.columns:
        groups = dict(tuple(labels.groupby("scene_id")))
    else:
        groups = {"_all": labels}

    kept_ids: Set[str] = set()
    rng = random.Random(seed)

    for scene_id, group in groups.items():
        conflict_mask = group["is_conflict"] == 1
        conflict_rows = group[conflict_mask]
        non_rows = group[~conflict_mask]

        n_c = len(conflict_rows)
        n_n = len(non_rows)

        if n_n == 0:
            kept_ids.update(conflict_rows["Event_id"].astype(str))
            continue
        if n_c == 0:
            kept_ids.update(non_rows["Event_id"].astype(str))
            continue

        # Target: neither class should be more than target_ratio × the other
        max_c = max(n_n, int(n_n * target_ratio))
        max_n = max(n_c, int(n_c / target_ratio))

        c_ids = list(conflict_rows["Event_id"].astype(str))
        n_ids = list(non_rows["Event_id"].astype(str))

        if len(c_ids) > max_c:
            print(f"  [{scene_id}] downsampling conflicts: {len(c_ids)} → {max_c} "
                  f"(non-conflicts: {len(n_ids)})")
            c_ids = rng.sample(c_ids, max_c)

        if len(n_ids) > max_n:
            print(f"  [{scene_id}] downsampling non-conflicts: {len(n_ids)} → {max_n} "
                  f"(conflicts: {len(c_ids)})")
            n_ids = rng.sample(n_ids, max_n)

        kept_ids.update(c_ids)
        kept_ids.update(n_ids)

    return labels[labels["Event_id"].astype(str).isin(kept_ids)]


def rebalance_conflict_types(
    labels: pd.DataFrame,
    cfg: ProcessingConfig,
) -> pd.DataFrame:
    """Downsample over-represented conflict-target roles within each scene.

    Only affects rows where ``is_conflict == 1``.  Non-conflict rows pass
    through unchanged.
    """
    if not cfg.rebalance_balance_conflict_types:
        return labels

    if "conflict_target_role" not in labels.columns:
        print("  WARNING: conflict_target_role column not found — skipping type balance")
        return labels

    seed = cfg.rebalance_seed
    per_scene = cfg.rebalance_per_scene

    conflict_mask = labels["is_conflict"] == 1
    conflict_rows = labels[conflict_mask]
    non_rows = labels[~conflict_mask]

    if per_scene and "scene_id" in labels.columns:
        groups = dict(tuple(conflict_rows.groupby("scene_id")))
    else:
        groups = {"_all": conflict_rows}

    rng = random.Random(seed)
    kept_ids: List[str] = list(non_rows["Event_id"].astype(str))

    for scene_id, group in groups.items():
        role_groups: Dict[str, List[str]] = defaultdict(list)
        for _, row in group.iterrows():
            role = str(row.get("conflict_target_role", "")) or "unknown"
            role_groups[role].append(str(row["Event_id"]))

        if len(role_groups) <= 1:
            for ids in role_groups.values():
                kept_ids.extend(ids)
            continue

        min_count = min(len(v) for v in role_groups.values())
        for role, ids in role_groups.items():
            if len(ids) > min_count:
                kept = rng.sample(ids, min_count)
                kept_ids.extend(kept)
                print(f"  [{scene_id}] conflict type '{role}': "
                      f"{len(ids)} → {min_count}")
            else:
                kept_ids.extend(ids)

    kept_set = set(kept_ids)
    return labels[labels["Event_id"].astype(str).isin(kept_set)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebalance exported CSVs for conflict:non-conflict ratio"
    )
    parser.add_argument(
        "--labels", required=True,
        help="Path to events_labels CSV",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to events_data CSV",
    )
    parser.add_argument(
        "--config", default="configs/data_processing.yaml",
        help="Path to data_processing YAML config (for rebalance settings)",
    )
    parser.add_argument(
        "--out-labels", default=None,
        help="Output labels path (default: {labels}_rebalanced.csv)",
    )
    parser.add_argument(
        "--out-data", default=None,
        help="Output data path (default: {data}_rebalanced.csv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading labels: {args.labels}")
    labels = _load_labels(args.labels)
    print(f"  {len(labels)} events total")

    print(f"Loading data: {args.data}")
    data = _load_data(args.data)
    print(f"  {len(data)} rows")

    # 1. Balance conflict:non-conflict ratio
    labels_bal = rebalance_labels(labels, cfg)

    # 2. Optionally balance conflict types
    labels_bal = rebalance_conflict_types(labels_bal, cfg)

    # 3. Filter data to matching Event_ids
    kept_ids = set(labels_bal["Event_id"].astype(str))
    data_bal = data[data["Event_id"].astype(str).isin(kept_ids)]

    # 4. Report
    n_conflict = int((labels_bal["is_conflict"] == 1).sum())
    n_non = int((labels_bal["is_conflict"] == 0).sum())
    print(f"\nRebalanced: {len(labels_bal)} events "
          f"({n_conflict} conflict, {n_non} non-conflict, "
          f"ratio={n_conflict / max(n_non, 1):.1f}:1)")
    if "scene_id" in labels_bal.columns:
        for scene_id, grp in labels_bal.groupby("scene_id"):
            nc = int((grp["is_conflict"] == 1).sum())
            nn = int((grp["is_conflict"] == 0).sum())
            print(f"  {scene_id}: {nc}c / {nn}nc  ({nc / max(nn, 1):.1f}:1)")

    out_labels = args.out_labels or _suffix_path(args.labels, "_rebalanced")
    out_data = args.out_data or _suffix_path(args.data, "_rebalanced")

    labels_bal.to_csv(out_labels, index=False)
    data_bal.to_csv(out_data, index=False)
    print(f"\nSaved: {out_labels}")
    print(f"Saved: {out_data}")


if __name__ == "__main__":
    main()
