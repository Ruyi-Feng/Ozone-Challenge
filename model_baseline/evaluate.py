"""Evaluation entry — wiring only; metrics left empty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_baseline.config import BaselineRuntimeConfig, load_config
from model_baseline.factories import build_model


def run_eval(cfg: BaselineRuntimeConfig, checkpoint: str | None = None) -> dict[str, float]:
    """
    TODO: load checkpoint, run val/test loader, compute metrics.

    Wiring sketch:
      model = build_model(cfg.model.name, cfg=cfg.model)
      # load state_dict from checkpoint
      ds = build_dataset(cfg.model.name, data_path=cfg.data.data_val, ...)
    """
    model_name = cfg.model.name
    _model = build_model(model_name, cfg=cfg.model)
    _ = (_model, checkpoint)
    raise NotImplementedError(
        f"Evaluation not implemented yet (model_name={model_name!r})."
    )


def main(
    config_path: str | Path = "configs/model_baseline.yaml",
    checkpoint: str | None = None,
) -> None:
    cfg = load_config(config_path)
    run_eval(cfg, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
