"""Training entry — dataset, DataLoader, training loop.

Flow:
  config.model.name  →  build_dataset(...) + build_model(...)
  → DataLoader → train / val loop → checkpoint
"""

from __future__ import annotations

import random
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


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for training.")


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: BaselineRuntimeConfig,
) -> tuple[Any, Any]:
    """Construct train and validation DataLoaders from config."""
    _require_torch()

    train_ds = build_dataset(
        cfg.model.name,
        data_path=cfg.data.data_train,
        label_path=cfg.data.label_train,
        split="train",
        future_path=cfg.data.future_train,
        num_agents=cfg.model.num_agents,
        num_features=cfg.model.num_features,
        num_frames=cfg.model.num_frames,
    )
    val_ds = build_dataset(
        cfg.model.name,
        data_path=cfg.data.data_val,
        label_path=cfg.data.label_val,
        split="val",
        future_path=cfg.data.future_val,
        num_agents=cfg.model.num_agents,
        num_features=cfg.model.num_features,
        num_frames=cfg.model.num_frames,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch.  Returns average losses."""
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

        outputs = model(x, agent_mask=agent_mask)
        loss_dict = model.compute_loss(outputs, is_conflict, target_idx)

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
) -> dict[str, float]:
    """Validation pass — losses + conflict accuracy."""
    model.eval()
    total_loss = 0.0
    total_conflict = 0.0
    total_target = 0.0
    correct_conflict = 0
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
        total_conflict += float(loss_dict["conflict_loss"]) * bs
        total_target += float(loss_dict["target_loss"]) * bs

        # Conflict accuracy
        pred = (torch.sigmoid(outputs["conflict_logit"]) >= 0.5).long()
        correct_conflict += int((pred == is_conflict.long()).sum())
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "conflict_loss": total_conflict / max(n, 1),
        "target_loss": total_target / max(n, 1),
        "conflict_acc": correct_conflict / max(n, 1),
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

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # --- Build model ---
    model = build_model(cfg.model.name, cfg=cfg.model)
    model.to(device)

    # --- Build data ---
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # --- Loop ---
    best_val_loss = float("inf")
    for epoch in range(1, cfg.train.max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{cfg.train.max_epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"(c={train_metrics['conflict_loss']:.4f} "
            f"t={train_metrics['target_loss']:.4f}) | "
            f"val loss={val_metrics['loss']:.4f} "
            f"c_acc={val_metrics['conflict_acc']:.4f}"
        )

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = Path("checkpoints")
            ckpt_path.mkdir(exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": cfg,
                },
                ckpt_path / "best_model.pt",
            )
            print(f"  → saved checkpoint (val_loss={best_val_loss:.4f})")

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


def main(config_path: str | Path = "configs/model_baseline.yaml") -> None:
    cfg = load_config(config_path)
    run_train(cfg)


if __name__ == "__main__":
    main()
