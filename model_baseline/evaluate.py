"""Evaluation entry — load checkpoint, run metrics."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore

from model_baseline.config import (
    BaselineRuntimeConfig,
    checkpoint_filename,
    load_config,
)
from model_baseline.factories import build_dataset, build_model
from model_baseline.train import _disable_nested_tensor

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
    all_probs: np.ndarray | None = None,
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

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if len(np.unique(all_labels)) >= 2:
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(all_labels, all_preds)
        )
        if all_probs is not None:
            metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
            metrics["pr_auc"] = float(
                average_precision_score(all_labels, all_probs)
            )
    else:
        metrics["balanced_accuracy"] = float("nan")
        if all_probs is not None:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    if all_probs is not None:
        metrics["brier"] = float(brier_score_loss(all_labels, all_probs))
    return metrics


def _split_paths(cfg: BaselineRuntimeConfig, split: str) -> tuple[str, str, str | None]:
    """Return (data_prefix, label_csv, future_csv) for a named split."""
    data = cfg.data
    mapping = {
        "val": (data.data_val, data.label_val, data.future_val),
        "test": (data.data_test, data.label_test, data.future_test),
        "transfer": (data.data_transfer, data.label_transfer, data.future_transfer),
        "train": (data.data_train, data.label_train, data.future_train),
    }
    if split not in mapping:
        raise ValueError(f"Unknown split {split!r}. Use val/test/transfer/train.")
    data_path, label_path, future_path = mapping[split]
    if not data_path or not label_path:
        raise FileNotFoundError(
            f"Config has no data/label paths for split={split!r}"
        )
    return data_path, label_path, future_path


@torch.no_grad()
def run_eval(
    cfg: BaselineRuntimeConfig,
    checkpoint: str | None = None,
    output_dir: str | Path | None = None,
    split: str = "val",
) -> dict[str, Any]:
    """Load model, run a named split, compute metrics."""
    _require_torch()

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    data_path, label_path, future_path = _split_paths(cfg, split)

    # --- Model ---
    model = build_model(cfg.model.name, cfg=cfg.model)
    masked_regime = False
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        masked_regime = bool(ckpt.get("train_regime", {}).get("masked", False))
        print(f"Loaded checkpoint: {checkpoint} (epoch {ckpt.get('epoch', '?')})")
    if masked_regime:
        _disable_nested_tensor(model)
    model.to(device)
    model.eval()

    # --- Data ---
    ds = build_dataset(
        cfg.model.name,
        data_path=data_path,
        label_path=label_path,
        split=split,
        future_path=future_path,
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
    all_probs: list[float] = []
    all_event_ids: list[int] = []
    all_target_preds: list[int] = []
    all_target_labels: list[int] = []
    target_preds: list[int] = []   # predicted neighbor index
    target_labels: list[int] = []  # ground-truth neighbor index
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        agent_mask = batch["agent_mask"].to(device)
        is_conflict = batch["is_conflict"].to(device)
        target_idx = batch["target_idx"].to(device)

        if masked_regime:
            outputs = model(
                x,
                agent_mask=agent_mask,
                time_mask=~batch["valid_mask"].to(device),
            )
        else:
            outputs = model(x, agent_mask=agent_mask)
        loss_dict = model.compute_loss(outputs, is_conflict, target_idx)

        bs = x.size(0)
        total_loss += float(loss_dict["loss"]) * bs
        n += bs

        # Conflict prediction
        probs = torch.sigmoid(outputs["conflict_logit"])
        pred = (probs >= 0.5).long()
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(is_conflict.long().cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        event_batch = batch["event_id"]
        if hasattr(event_batch, "cpu"):
            event_batch = event_batch.cpu().tolist()
        all_event_ids.extend(int(v) for v in event_batch)

        has_neighbor = agent_mask[:, 1:].any(dim=1)
        batch_target_pred = outputs["target_logits"].argmax(dim=-1)
        batch_target_pred = torch.where(
            has_neighbor,
            batch_target_pred,
            torch.full_like(batch_target_pred, -1),
        )
        all_target_preds.extend(batch_target_pred.cpu().tolist())
        all_target_labels.extend(target_idx.long().cpu().tolist())

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
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    metrics = _compute_metrics(preds_arr, labels_arr, probs_arr)
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
    if target_labels:
        metrics["target_macro_f1"] = float(
            f1_score(target_labels, target_preds, average="macro")
        )
    else:
        metrics["target_macro_f1"] = 0.0

    # --- Persist event-level predictions and per-scene metrics ---
    labels_df = pd.read_csv(label_path)
    metadata = labels_df.set_index("Event_id")
    prediction_df = pd.DataFrame(
        {
            "event_id": all_event_ids,
            "is_conflict": all_labels,
            "conflict_probability": all_probs,
            "conflict_prediction": all_preds,
            "target_idx": all_target_labels,
            "target_prediction": all_target_preds,
        }
    )
    prediction_df["scene_id"] = prediction_df["event_id"].map(
        metadata["scene_id"].to_dict()
    )
    if "conflict_target_role" in metadata.columns:
        prediction_df["conflict_target_role"] = prediction_df["event_id"].map(
            metadata["conflict_target_role"].to_dict()
        )

    scene_ids = prediction_df["scene_id"].astype(str)
    prediction_df["is_transfer"] = scene_ids.str.startswith("UTE_xam")
    if split == "transfer":
        prediction_df["is_transfer"] = True

    scene_metrics: dict[str, dict[str, float]] = {}
    for scene_id, group in prediction_df.groupby("scene_id", dropna=False):
        scene_metrics[str(scene_id)] = _compute_metrics(
            group["conflict_prediction"].to_numpy(),
            group["is_conflict"].to_numpy(),
            group["conflict_probability"].to_numpy(),
        )
    metrics["split"] = split
    metrics["per_scene"] = scene_metrics

    transfer_group = prediction_df[prediction_df["is_transfer"]]
    if len(transfer_group) > 0:
        metrics["transfer_subset"] = _compute_metrics(
            transfer_group["conflict_prediction"].to_numpy(),
            transfer_group["is_conflict"].to_numpy(),
            transfer_group["conflict_probability"].to_numpy(),
        )
        metrics["transfer_subset"]["n"] = int(len(transfer_group))
        metrics["transfer_subset"]["scenes"] = sorted(
            {str(s) for s in transfer_group["scene_id"].unique()}
        )

    in_dist = prediction_df[~prediction_df["is_transfer"]]
    if len(in_dist) > 0 and len(transfer_group) > 0:
        metrics["in_distribution_subset"] = _compute_metrics(
            in_dist["conflict_prediction"].to_numpy(),
            in_dist["is_conflict"].to_numpy(),
            in_dist["conflict_probability"].to_numpy(),
        )
        metrics["in_distribution_subset"]["n"] = int(len(in_dist))

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = str(getattr(cfg.model, "tag", "") or cfg.model.name)
        prediction_df.to_csv(out / f"{tag}_{split}_predictions.csv", index=False)
        (out / f"{tag}_{split}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )

    # --- Report ---
    title = {
        "val": "Validation",
        "test": "In-distribution test",
        "transfer": "UTE transfer test",
        "train": "Train",
    }.get(split, split)
    print(f"\n{'='*50}")
    print(f"Evaluation Results — {title}")
    print(f"{'='*50}")
    print(f"  Samples:         {n}")
    print(f"  Loss:            {metrics['loss']:.4f}")
    print()
    print("Conflict Prediction:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1:              {metrics['f1']:.4f}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:          {metrics['pr_auc']:.4f}")
    print(f"  Brier:           {metrics['brier']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")
    if "transfer_subset" in metrics and split != "transfer":
        ts = metrics["transfer_subset"]
        print()
        print("UTE transfer subset (scene_id startswith UTE_xam):")
        print(f"  Samples:         {ts.get('n', 0)}")
        print(f"  Accuracy:        {ts['accuracy']:.4f}")
        print(f"  F1:              {ts['f1']:.4f}")
        print(f"  ROC-AUC:         {ts.get('roc_auc', float('nan')):.4f}")
    print()
    print("Target Prediction (conflict partner identification):")
    print(f"  Samples:         {metrics['target_samples']}")
    print(f"  Accuracy:        {metrics['target_acc']:.4f}")
    print(f"  Macro-F1:        {metrics['target_macro_f1']:.4f}")
    if per_role:
        print("  Per-role:")
        for role in ["front", "rear", "left_front", "left_rear",
                     "right_front", "right_rear"]:
            if role in per_role:
                r = per_role[role]
                acc = r["correct"] / max(r["total"], 1)
                print(f"    {role:>12s}: {acc:.3f} ({r['correct']}/{r['total']})")
    if scene_metrics:
        print()
        print("Per-scene conflict F1:")
        for scene_id, sm in sorted(scene_metrics.items()):
            print(f"    {scene_id}: f1={sm['f1']:.3f} n={int(sm['tp']+sm['tn']+sm['fp']+sm['fn'])}")

    return metrics


def main(
    config_path: str | Path = "configs/model_baseline.yaml",
    checkpoint: str | None = None,
    output_dir: str | Path | None = None,
    split: str = "val",
) -> None:
    cfg = load_config(config_path)
    if checkpoint is None:
        # resolves to checkpoints/{tag}_best_model.pt when model.tag is set
        default_ckpt = (
            Path(cfg.train.checkpoint_dir) / checkpoint_filename(cfg.model)
        )
        if default_ckpt.exists():
            checkpoint = str(default_ckpt)
    run_eval(cfg, checkpoint=checkpoint, output_dir=output_dir, split=split)


if __name__ == "__main__":
    main()
