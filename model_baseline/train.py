"""Training entry — wiring only; loop left empty.

Flow:
  config.model.name  →  build_dataset(...) + build_model(...)
  → DataLoader (TODO) → train / val loop (TODO)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_baseline.config import BaselineRuntimeConfig, load_config
from model_baseline.factories import build_model


def build_dataloaders(cfg: BaselineRuntimeConfig) -> tuple[Any, Any]:
    """
    TODO: construct train/val DataLoaders.

    Intended sketch (not implemented):
      train_ds = build_dataset(cfg.model.name, data_path=..., label_path=..., split=\"train\")
      val_ds   = build_dataset(cfg.model.name, data_path=..., label_path=..., split=\"val\")
      return DataLoader(train_ds, ...), DataLoader(val_ds, ...)
    """
    raise NotImplementedError("DataLoader construction left empty for now")


def train_one_epoch(*args: Any, **kwargs: Any) -> dict[str, float]:
    """TODO: one training epoch."""
    raise NotImplementedError


def validate(*args: Any, **kwargs: Any) -> dict[str, float]:
    """TODO: validation pass (metrics)."""
    raise NotImplementedError


def run_train(cfg: BaselineRuntimeConfig) -> None:
    """
    Wire factories; train/val body intentionally empty.
    """
    model_name = cfg.model.name

    # --- factory wiring (kept as the architectural anchor) ---
    _model = build_model(model_name, cfg=cfg.model)
    # datasets / loaders left empty until Dataset is implemented:
    # train_loader, val_loader = build_dataloaders(cfg)

    # TODO: optimizer, loop over epochs, checkpointing
    _ = _model
    raise NotImplementedError(
        f"Training loop not implemented yet (model_name={model_name!r}). "
        "Factories are wired; fill Dataset / DataLoader / train+val next."
    )


def main(config_path: str | Path = "configs/model_baseline.yaml") -> None:
    cfg = load_config(config_path)
    run_train(cfg)


if __name__ == "__main__":
    main()
