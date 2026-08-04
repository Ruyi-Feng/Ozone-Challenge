"""Evaluation entry — load checkpoint, run metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore

from model_baseline.config import BaselineRuntimeConfig, load_config
from model_baseline.factories import build_dataset, build_model

# Neighbor slot → model target index mapping (consistent with baseline.py)
TARGET_IDX_TO_ROLE: dict[int, str] = {
    0: "front",
    1: "rear",
    2: "left_front",
    3: "left_rear",
    4: "right_front",
    5: "right_rear",
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for evaluation.")


def _compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> dict[str, float]:
    """Binary classification metrics from accumulated predictions."""
    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def run_eval(
    cfg: BaselineRuntimeConfig,
    checkpoint: str | None = None,
) -> dict[str, float]:
    """Load model, run validation set, compute metrics."""
    _require_torch()

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = build_model(cfg.model.name, cfg=cfg.model)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint} (epoch {ckpt.get('epoch', '?')})")
    model.to(device)
    model.eval()

    # --- Data ---
    ds = build_dataset(
        cfg.model.name,
        data_path=cfg.data.data_val,
        label_path=cfg.data.label_val,
        split="val",
        future_path=cfg.data.future_val,
        num_agents=cfg.model.num_agents,
        num_features=cfg.model.num_features,
        num_frames=cfg.model.num_frames,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    print(f"Eval samples: {len(ds)}")

    # --- Accumulate predictions ---
    all_preds: list[int] = []
    all_labels: list[int] = []
    target_preds: list[int] = []   # predicted neighbor index
    target_labels: list[int] = []  # ground-truth neighbor index
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        agent_mask = batch["agent_mask"].to(device)
        is_conflict = batch["is_conflict"].to(device)
        target_idx = batch["target_idx"].to(device)

        outputs = model(x, agent_mask=agent_mask)
        loss_dict = model.compute_loss(outputs, is_conflict, target_idx)

        bs = x.size(0)
        total_loss += float(loss_dict["loss"]) * bs
        n += bs

        # Conflict prediction
        pred = (torch.sigmoid(outputs["conflict_logit"]) >= 0.5).long()
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(is_conflict.long().cpu().tolist())

        # Target prediction (conflict samples only)
        conf_mask = (
            is_conflict.to(dtype=torch.bool)
            & (target_idx >= 0).to(dtype=torch.bool)
        )
        if conf_mask.any():
            t_pred = outputs["target_logits"][conf_mask].argmax(dim=-1)
            target_preds.extend(t_pred.cpu().tolist())
            target_labels.extend(target_idx[conf_mask].long().cpu().tolist())

    # --- Conflict metrics ---
    metrics = _compute_metrics(np.array(all_preds), np.array(all_labels))
    metrics["loss"] = total_loss / max(n, 1)

    # --- Target metrics ---
    target_acc = 0.0
    per_role: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    if target_labels:
        t_preds_arr = np.array(target_preds)
        t_labels_arr = np.array(target_labels)
        target_acc = float((t_preds_arr == t_labels_arr).mean())

        for pred, label in zip(target_preds, target_labels):
            role = TARGET_IDX_TO_ROLE.get(label, f"unknown({label})")
            per_role[role]["total"] += 1
            if pred == label:
                per_role[role]["correct"] += 1

    metrics["target_acc"] = target_acc
    metrics["target_samples"] = len(target_labels)

    # --- Report ---
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"  Samples:         {n}")
    print(f"  Loss:            {metrics['loss']:.4f}")
    print()
    print("Conflict Prediction:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1:              {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")
    print()
    print("Target Prediction (conflict partner identification):")
    print(f"  Samples:         {metrics['target_samples']}")
    print(f"  Accuracy:        {metrics['target_acc']:.4f}")
    if per_role:
        print("  Per-role:")
        for role in ["front", "rear", "left_front", "left_rear",
                     "right_front", "right_rear"]:
            if role in per_role:
                r = per_role[role]
                acc = r["correct"] / max(r["total"], 1)
                print(f"    {role:>12s}: {acc:.3f} ({r['correct']}/{r['total']})")

    return metrics


def main(
    config_path: str | Path = "configs/model_baseline.yaml",
    checkpoint: str | None = None,
) -> None:
    cfg = load_config(config_path)
    if checkpoint is None:
        default_ckpt = Path("checkpoints/best_model.pt")
        if default_ckpt.exists():
            checkpoint = str(default_ckpt)
    run_eval(cfg, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
