"""Training entry — dataset, DataLoader, training loop.

Flow:
  config.model.name  →  build_dataset(...) + build_model(...)
  → DataLoader → train / val loop → checkpoint
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore

from model_baseline.config import (
    BaselineRuntimeConfig,
    MaskingConfig,
    checkpoint_filename,
    load_config,
)
from model_baseline.factories import build_dataset, build_model
from model_baseline.masking import MaskSamplerConfig, TrainingMaskSampler


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for training.")


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------


def _dataset_kwargs(cfg: BaselineRuntimeConfig) -> dict[str, Any]:
    max_agents = getattr(cfg.model, "max_agents", None) or getattr(
        cfg.data, "max_agents", cfg.model.num_agents
    )
    return {
        "num_agents": cfg.model.num_agents,
        "max_agents": max_agents,
        "num_features": cfg.model.num_features,
        "num_frames": cfg.model.num_frames,
    }


def _collate_fn_for(dataset: Any) -> Any:
    return getattr(dataset, "collate_fn", None)


def build_dataloader(
    dataset: Any, cfg: BaselineRuntimeConfig, *, shuffle: bool
) -> Any:
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        drop_last=False,
        collate_fn=_collate_fn_for(dataset),
    )


def build_dataloaders(
    cfg: BaselineRuntimeConfig,
) -> tuple[Any, Any]:
    """Construct train and validation DataLoaders from config."""
    _require_torch()
    ds_kwargs = _dataset_kwargs(cfg)

    train_ds = build_dataset(
        cfg.model.name,
        data_path=cfg.data.data_train,
        label_path=cfg.data.label_train,
        split="train",
        future_path=cfg.data.future_train,
        **ds_kwargs,
    )
    val_ds = build_dataset(
        cfg.model.name,
        data_path=cfg.data.data_val,
        label_path=cfg.data.label_val,
        split="val",
        future_path=cfg.data.future_val,
        **ds_kwargs,
    )
    return (
        build_dataloader(train_ds, cfg, shuffle=True),
        build_dataloader(val_ds, cfg, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Masked-training helpers (no-ops unless train.masking.enabled)
# ---------------------------------------------------------------------------


def _sampler_cfg(mcfg: MaskingConfig) -> MaskSamplerConfig:
    return MaskSamplerConfig(
        p_full=mcfg.p_full,
        seg_len_frames=tuple(mcfg.seg_len_frames),
        hierarchies=tuple(mcfg.hierarchies),
        order_modes=tuple(mcfg.order_modes),
    )


def _disable_nested_tensor(model: Any) -> None:
    """Force the vanilla (non-nested-tensor) encoder path.

    The SDPA/nested-tensor fast path activates in eval() with a key padding
    mask and produces small fp differences vs. the training path — bad for
    reproducible masked values.  Only called in the masked regime; the
    original training path is left untouched.
    """
    for name in ("encoder", "agent_encoder"):
        enc = getattr(model, name, None)
        if enc is None:
            continue
        for attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(enc, attr):
                setattr(enc, attr, False)


def _forward_extras(batch: dict[str, Any], device: "torch.device") -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if "role_ids" in batch:
        extras["role_ids"] = batch["role_ids"].to(device)
    return extras


def _compute_loss(
    model: Any,
    outputs: dict[str, "torch.Tensor"],
    is_conflict: "torch.Tensor",
    target_idx: "torch.Tensor",
    agent_mask: Optional["torch.Tensor"] = None,
) -> dict[str, "torch.Tensor"]:
    return model.compute_loss(
        outputs, is_conflict, target_idx, agent_mask=agent_mask
    )


def _masked_forward(
    model: Any,
    batch: dict[str, Any],
    x: "torch.Tensor",
    agent_mask: "torch.Tensor",
    device: torch.device,
    mask_sampler: Optional[TrainingMaskSampler],
    use_valid_mask: bool,
) -> dict[str, "torch.Tensor"]:
    """Forward with the regime selected by (mask_sampler, use_valid_mask).

    Neither set → the ORIGINAL call signature, byte-identical behaviour.
    """
    extras = _forward_extras(batch, device)
    if mask_sampler is not None:
        tm_np, cm_np = mask_sampler.sample_batch(
            batch["valid_mask"].numpy(), batch["agent_mask"].numpy()
        )
        return model(
            x,
            agent_mask=agent_mask,
            time_mask=torch.from_numpy(tm_np).to(device),
            channel_mask=torch.from_numpy(cm_np).to(device),
            **extras,
        )
    if use_valid_mask:
        return model(
            x,
            agent_mask=agent_mask,
            time_mask=~batch["valid_mask"].to(device),
            **extras,
        )
    if extras:
        return model(x, agent_mask=agent_mask, **extras)
    return model(x, agent_mask=agent_mask)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: torch.device,
    mask_sampler: Optional[TrainingMaskSampler] = None,
) -> dict[str, float]:
    """Run one training epoch.  Returns average losses.

    mask_sampler=None reproduces the original full-input training exactly.
    """
    model.train()
    total_loss = 0.0
    total_conflict = 0.0
    total_target = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        agent_mask = batch["agent_mask"].to(device)
        is_conflict = batch["is_conflict"].to(device)
        target_idx = batch["target_idx"].to(device)

        outputs = _masked_forward(
            model, batch, x, agent_mask, device, mask_sampler, False
        )
        loss_dict = _compute_loss(
            model, outputs, is_conflict, target_idx, agent_mask=agent_mask
        )

        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss_dict["loss"]) * bs
        total_conflict += float(loss_dict["conflict_loss"]) * bs
        total_target += float(loss_dict["target_loss"]) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "conflict_loss": total_conflict / max(n, 1),
        "target_loss": total_target / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: Any,
    loader: Any,
    device: torch.device,
    use_valid_mask: bool = False,
    mask_sampler: Optional[TrainingMaskSampler] = None,
) -> dict[str, float]:
    """Validation pass — losses + conflict accuracy + target accuracy.

    Defaults reproduce the original full-input validation exactly.
    use_valid_mask=True blocks padding frames (masked regime's "full input");
    mask_sampler draws random coalitions (masked validation; pass a sampler
    freshly seeded with masking.val_seed so epochs are comparable).
    """
    model.eval()
    total_loss = 0.0
    total_conflict = 0.0
    total_target = 0.0
    correct_conflict = 0
    correct_target = 0
    n_conflict = 0  # samples where is_conflict == 1 AND target in neighbor slots
    n_conflict_all = 0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        agent_mask = batch["agent_mask"].to(device)
        is_conflict = batch["is_conflict"].to(device)
        target_idx = batch["target_idx"].to(device)

        outputs = _masked_forward(
            model, batch, x, agent_mask, device, mask_sampler, use_valid_mask
        )
        loss_dict = _compute_loss(
            model, outputs, is_conflict, target_idx, agent_mask=agent_mask
        )

        bs = x.size(0)
        total_loss += float(loss_dict["loss"]) * bs
        total_conflict += float(loss_dict["conflict_loss"]) * bs
        total_target += float(loss_dict["target_loss"]) * bs

        # Conflict accuracy
        pred = (torch.sigmoid(outputs["conflict_logit"]) >= 0.5).long()
        correct_conflict += int((pred == is_conflict.long()).sum())
        n += bs

        # Target accuracy: only for conflict samples whose target is in
        # a neighbour slot (target_idx >= 0), same mask as compute_loss
        conf_mask = (
            is_conflict.to(dtype=torch.bool)
            & (target_idx >= 0).to(dtype=torch.bool)
        )
        n_conflict_all += int(is_conflict.to(dtype=torch.bool).sum().item())
        if conf_mask.any():
            target_pred = outputs["target_logits"][conf_mask].argmax(dim=-1)
            correct_target += int(
                (target_pred == target_idx[conf_mask].long()).sum()
            )
            n_conflict += int(conf_mask.sum().item())

    return {
        "loss": total_loss / max(n, 1),
        "conflict_loss": total_conflict / max(n, 1),
        "target_loss": total_target / max(n, 1),
        "conflict_acc": correct_conflict / max(n, 1),
        "target_acc": correct_target / max(n_conflict, 1),
        "target_coverage": n_conflict / max(n_conflict_all, 1),
    }


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------


def run_train(cfg: BaselineRuntimeConfig) -> None:
    """Wire factories, run training loop, save checkpoint."""
    _require_torch()

    # --- Reproducibility ---
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # --- Build model ---
    model = build_model(cfg.model.name, cfg=cfg.model)
    model.to(device)

    # --- Build data ---
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    # --- Masked surrogate regime (opt-in; default = original behaviour) ---
    masking_cfg = getattr(cfg.train, "masking", None)
    masking_on = bool(masking_cfg and masking_cfg.enabled)
    mask_sampler: Optional[TrainingMaskSampler] = None
    if masking_on:
        for ds, tag in ((train_loader.dataset, "train"),
                        (val_loader.dataset, "val")):
            if masking_cfg.require_valid and not getattr(ds, "has_valid_mask", False):
                raise RuntimeError(
                    f"train.masking.enabled=true but the {tag} cache has no "
                    f"per-frame validity mask — padded frames would enter the "
                    f"coalition game. Run: python data_processing/scripts/"
                    f"build_tensor_cache.py --valid-only  (then "
                    f"build_binary_cache.py), or set masking.require_valid: false."
                )
        _disable_nested_tensor(model)
        mask_sampler = TrainingMaskSampler(
            _sampler_cfg(masking_cfg),
            num_features=cfg.model.num_features,
            seed=cfg.train.seed,
        )
        print(f"Masked surrogate training ON: p_full={masking_cfg.p_full}, "
              f"seg_len={masking_cfg.seg_len_frames}, "
              f"orders={masking_cfg.order_modes}, "
              f"val_selection={masking_cfg.val_selection}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # --- Loop ---
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    ckpt_path = Path(cfg.train.checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.train.max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, mask_sampler=mask_sampler
        )

        if masking_on:
            # "full input" in the masked regime = padding frames blocked
            val_metrics = validate(model, val_loader, device, use_valid_mask=True)
            # masked validation with a fixed seed → comparable across epochs
            masked_val = validate(
                model, val_loader, device,
                mask_sampler=TrainingMaskSampler(
                    _sampler_cfg(masking_cfg),
                    num_features=cfg.model.num_features,
                    seed=masking_cfg.val_seed,
                ),
            )
            if masking_cfg.val_selection == "mixture":
                select_loss = (
                    masking_cfg.p_full * val_metrics["loss"]
                    + (1.0 - masking_cfg.p_full) * masked_val["loss"]
                )
            else:
                select_loss = val_metrics["loss"]
        else:
            val_metrics = validate(model, val_loader, device)
            masked_val = None
            select_loss = val_metrics["loss"]

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "masked_val": masked_val,
                "select_loss": select_loss,
            }
        )

        line = (
            f"Epoch {epoch:3d}/{cfg.train.max_epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"(c={train_metrics['conflict_loss']:.4f} "
            f"t={train_metrics['target_loss']:.4f}) | "
            f"val loss={val_metrics['loss']:.4f} "
            f"c_acc={val_metrics['conflict_acc']:.4f} "
            f"t_acc={val_metrics['target_acc']:.4f}"
        )
        if masked_val is not None:
            line += (f" | masked val loss={masked_val['loss']:.4f} "
                     f"c_acc={masked_val['conflict_acc']:.4f} "
                     f"(select={select_loss:.4f})")
        print(line)

        # Save best
        if select_loss < best_val_loss:
            best_val_loss = select_loss
            ckpt_file = ckpt_path / checkpoint_filename(cfg.model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "select_loss": select_loss,
                    "config": cfg,
                    "tag": getattr(cfg.model, "tag", ""),
                    "history": history,
                    # attribution driver asserts on this tag: only checkpoints
                    # trained under the masked regime yield a well-defined v(S)
                    "train_regime": {
                        "masked": masking_on,
                        "uses_valid_mask": masking_on,
                        "masking": asdict(masking_cfg) if masking_on else None,
                    },
                },
                ckpt_file,
            )
            print(f"  → saved checkpoint {ckpt_file} "
                  f"(select_loss={best_val_loss:.4f})")

    history_name = (
        f"{getattr(cfg.model, 'tag', '') or cfg.model.name}_training_history.json"
    )
    (ckpt_path / history_name).write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(f"Training finished. Best selection loss: {best_val_loss:.4f}")


def main(config_path: str | Path = "configs/model_baseline.yaml") -> None:
    cfg = load_config(config_path)
    run_train(cfg)


if __name__ == "__main__":
    main()
