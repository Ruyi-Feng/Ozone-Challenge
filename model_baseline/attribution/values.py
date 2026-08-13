"""Value function v(S) for one event: conflict logit under coalition masks."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from model_baseline.train import _disable_nested_tensor


class ConflictValueFn:
    """Batch evaluator of v(S) for a single event.

    The coalition state is carried entirely by ``channel_mask``: masked cells
    have their channels masked (value zeroed + indicator bit), and the model
    itself attention-blocks any frame whose channels are all masked
    (TransformerConflictModel._apply_masks).  Invalid frames (padding) are
    permanently blocked via ``time_mask = ~valid`` — they are context, never
    players.

    Note: masking every cell of an agent does NOT flip agent_mask.  The
    "present but unobserved" neighbor CLS still enters the masked-mean
    readout — the mask pattern itself is observable, which is part of the
    game definition (design doc §2).
    """

    def __init__(
        self,
        model: Any,
        x: "torch.Tensor | np.ndarray",
        agent_mask: "torch.Tensor | np.ndarray",
        valid_mask: "torch.Tensor | np.ndarray",
        device: "torch.device | str" = "cpu",
        target: str = "conflict_logit",
        target_idx: Optional[int] = None,
    ) -> None:
        if target not in ("conflict_logit", "target_logit"):
            raise ValueError(f"unknown attribution target {target!r}")
        if target == "target_logit" and target_idx is None:
            raise ValueError("target='target_logit' requires target_idx")

        self.model = model.to(device)
        self.model.eval()
        _disable_nested_tensor(self.model)  # reproducible masked attention
        self.device = torch.device(device)
        self.target = target
        self.target_idx = target_idx

        self.x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.agent_mask = torch.as_tensor(
            agent_mask, dtype=torch.bool, device=self.device
        )
        self.valid = torch.as_tensor(
            valid_mask, dtype=torch.bool, device=self.device
        )
        if self.x.dim() != 3:
            raise ValueError(f"x must be [A, T, F], got {tuple(self.x.shape)}")
        self.n_forwards = 0

    def __call__(self, channel_masks: np.ndarray) -> np.ndarray:
        """channel_masks [B, A, T, F] bool → v values [B] float."""
        cm = torch.from_numpy(np.ascontiguousarray(channel_masks)).to(
            self.device
        )
        B = cm.shape[0]
        # _apply_masks is out-of-place, so expanded views are safe (no clone)
        x = self.x.unsqueeze(0).expand(B, -1, -1, -1)
        am = self.agent_mask.unsqueeze(0).expand(B, -1)
        tm = (~self.valid).unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            out = self.model(x, agent_mask=am, time_mask=tm, channel_mask=cm)
        self.n_forwards += B
        if self.target == "conflict_logit":
            vals = out["conflict_logit"]
        else:
            vals = out["target_logits"][:, self.target_idx]
        return vals.detach().cpu().numpy().astype(np.float64)
