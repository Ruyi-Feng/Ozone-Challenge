"""Masked surrogate training tests.

Verifies:
- mask parameters do NOT change the original no-mask path (trivial masks ≡
  no masks);
- the §4 rule "time-segment mask ≡ all channels masked" holds in the model;
- everything-masked input yields finite outputs (CLS-only prior);
- run_train end-to-end: masked regime trains, tags the checkpoint with
  train_regime.masked, refuses caches without valid_mask, and the default
  (masking disabled) path still runs unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from model_baseline.config import (  # noqa: E402
    BaselineRuntimeConfig,
    DataPaths,
    MaskingConfig,
    ModelConfig,
    TrainConfig,
)
from model_baseline.models.transformer import TransformerConflictModel  # noqa: E402
from model_baseline.train import run_train  # noqa: E402

A, T, F = 7, 80, 4


def _tiny_model() -> TransformerConflictModel:
    cfg = ModelConfig(
        name="transformer", num_agents=A, num_features=F,
        hidden_dim=8, num_frames=T, n_layers=1, n_heads=2, dropout=0.0,
    )
    torch.manual_seed(0)
    model = TransformerConflictModel(cfg)
    model.eval()
    return model


def _inputs(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, A, T, F, generator=g)
    agent_mask = torch.ones(B, A, dtype=torch.bool)
    return x, agent_mask


# ---------------------------------------------------------------------------
# Model-side mask semantics
# ---------------------------------------------------------------------------


def test_trivial_masks_equal_no_masks():
    """All-False time/channel masks must reproduce the original path exactly."""
    model = _tiny_model()
    x, agent_mask = _inputs()
    with torch.no_grad():
        out_plain = model(x, agent_mask=agent_mask)
        out_trivial = model(
            x, agent_mask=agent_mask,
            time_mask=torch.zeros(2, A, T, dtype=torch.bool),
            channel_mask=torch.zeros(2, A, T, F, dtype=torch.bool),
        )
    assert torch.allclose(
        out_plain["conflict_logit"], out_trivial["conflict_logit"], atol=1e-6
    )
    assert torch.allclose(
        out_plain["target_logits"], out_trivial["target_logits"], atol=1e-6
    )


def test_time_mask_equals_all_channels_masked():
    """§4 rule: masking a frame's every channel ≡ masking the frame."""
    model = _tiny_model()
    x, agent_mask = _inputs()

    tm = torch.zeros(2, A, T, dtype=torch.bool)
    tm[:, :, 10:20] = True
    cm = torch.zeros(2, A, T, F, dtype=torch.bool)
    cm[:, :, 10:20, :] = True

    with torch.no_grad():
        out_t = model(x, agent_mask=agent_mask, time_mask=tm)
        out_c = model(x, agent_mask=agent_mask, channel_mask=cm)
    assert torch.allclose(
        out_t["conflict_logit"], out_c["conflict_logit"], atol=1e-6
    )


def test_everything_masked_is_finite():
    """v(∅): all frames blocked → CLS-only readout, finite outputs."""
    model = _tiny_model()
    x, agent_mask = _inputs()
    tm = torch.ones(2, A, T, dtype=torch.bool)
    with torch.no_grad():
        out = model(x, agent_mask=agent_mask, time_mask=tm)
    assert torch.isfinite(out["conflict_logit"]).all()
    assert torch.isfinite(out["target_logits"]).all()

    # and v(∅) must not depend on the (unobserved) trajectory content
    x2 = x + 123.0
    with torch.no_grad():
        out2 = model(x2, agent_mask=agent_mask, time_mask=tm)
    assert torch.allclose(
        out["conflict_logit"], out2["conflict_logit"], atol=1e-5
    )


# ---------------------------------------------------------------------------
# run_train end-to-end (tiny synthetic cache, CPU)
# ---------------------------------------------------------------------------


def _write_cache(prefix: str, with_valid: bool = True, N: int = 8) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, A, T, F)).astype(np.float32)
    mask = np.zeros((N, A), dtype=bool)
    mask[:, 0] = True
    mask[:, 1] = rng.random(N) < 0.8
    valid = np.zeros((N, A, T), dtype=bool)
    for i in range(N):
        for a in range(A):
            if mask[i, a]:
                valid[i, a, int(rng.integers(0, 40)):] = True
    x[~np.repeat(valid[..., None], F, axis=-1)] = 0.0
    y = (rng.random(N) < 0.5).astype(np.float32)
    target = np.where((y > 0) & mask[:, 1], 0, -1).astype(np.int64)

    np.save(f"{prefix}_x.npy", x)
    np.save(f"{prefix}_mask.npy", mask)
    np.save(f"{prefix}_y.npy", y)
    np.save(f"{prefix}_target.npy", target)
    np.save(f"{prefix}_eid.npy", np.arange(N, dtype=np.int64))
    if with_valid:
        np.save(f"{prefix}_valid.npy", valid)


def _tiny_cfg(prefix: str, masking: MaskingConfig) -> BaselineRuntimeConfig:
    return BaselineRuntimeConfig(
        model=ModelConfig(
            name="transformer", num_agents=A, num_features=F,
            hidden_dim=8, num_frames=T, n_layers=1, n_heads=2, dropout=0.0,
        ),
        data=DataPaths(
            data_train=prefix, label_train="", data_val=prefix, label_val="",
        ),
        train=TrainConfig(
            batch_size=4, num_workers=0, lr=1e-3, max_epochs=1,
            seed=7, device="cpu", masking=masking,
        ),
    )


def test_run_train_masked_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = str(tmp_path / "train")
    _write_cache(prefix, with_valid=True)

    cfg = _tiny_cfg(prefix, MaskingConfig(enabled=True, p_full=0.5))
    run_train(cfg)

    ckpt = torch.load(
        tmp_path / "checkpoints" / "best_model.pt", weights_only=False
    )
    assert ckpt["train_regime"]["masked"] is True
    assert ckpt["train_regime"]["masking"]["p_full"] == 0.5
    assert np.isfinite(ckpt["val_loss"])

    # weights load back into a fresh model
    _tiny_model().load_state_dict(ckpt["model_state_dict"])


def test_run_train_masked_requires_valid_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = str(tmp_path / "train")
    _write_cache(prefix, with_valid=False)

    cfg = _tiny_cfg(prefix, MaskingConfig(enabled=True))
    with pytest.raises(RuntimeError, match="valid"):
        run_train(cfg)


def test_run_train_default_path_unchanged(tmp_path, monkeypatch):
    """masking disabled: trains fine even without valid files (original path)."""
    monkeypatch.chdir(tmp_path)
    prefix = str(tmp_path / "train")
    _write_cache(prefix, with_valid=False)

    cfg = _tiny_cfg(prefix, MaskingConfig(enabled=False))
    run_train(cfg)

    ckpt = torch.load(
        tmp_path / "checkpoints" / "best_model.pt", weights_only=False
    )
    assert ckpt["train_regime"]["masked"] is False
