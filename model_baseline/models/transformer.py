"""Transformer conflict model: shared temporal encoder + dual heads.

Replaces the LSTM encoder of ``BaselineConflictModel`` while keeping the
same I/O contract and dual-head readout.  Designed so that time / channel
masks can later drive attention blocking (Shapley), but masks default to
None (= fully visible) and are not required for ordinary train / eval.

I/O contract
------------
Input:
  x            : [B, A, T, F]
  agent_mask   : [B, A]  bool, True = slot occupied (ego should be True)
  time_mask    : [B, A, T] bool, True = frame blocked (optional)
  channel_mask : [B, A, T, F] bool, True = channel masked (optional)

Output dict:
  conflict_logit : [B]
  target_logits  : [B, A-1]   # neighbor slots only; empty → -inf
"""

from __future__ import annotations

from typing import Any, Optional

from model_baseline.config import ModelConfig

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # torch may still be optional in requirements
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for TransformerConflictModel. "
            "Uncomment / install torch in requirements.txt."
        )


if nn is not None:

    class TransformerConflictModel(nn.Module):
        """Per-agent Transformer encoder + CLS readout + dual heads."""

        def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
            super().__init__()
            self.cfg = cfg
            self.num_agents = cfg.num_agents
            self.num_features = cfg.num_features
            self.hidden_dim = cfg.hidden_dim  # d_model
            self.num_neighbors = self.num_agents - 1
            self.n_layers = cfg.n_layers
            self.n_heads = cfg.n_heads
            self.dropout = cfg.dropout
            # PE length; fall back to a generous default if unset
            self.max_frames = int(cfg.num_frames) if cfg.num_frames else 128

            if self.hidden_dim % self.n_heads != 0:
                raise ValueError(
                    f"hidden_dim={self.hidden_dim} must be divisible by "
                    f"n_heads={self.n_heads}"
                )

            # F features + F mask-indicator bits → d_model
            self.input_proj = nn.Linear(self.num_features * 2, self.hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            # Learnable absolute PE over original frame indices (position-preserving)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.max_frames, self.hidden_dim)
            )

            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                batch_first=True,
                # LayerNorm is per-position (safe w.r.t. mask leakage)
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=self.n_layers
            )

            # Same dual heads as LSTM baseline
            self.conflict_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.target_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )

            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)

        def _apply_masks(
            self,
            x: "torch.Tensor",
            time_mask: Optional["torch.Tensor"],
            channel_mask: Optional["torch.Tensor"],
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Zero masked channels, set indicator bits, derive frame blocking.

            Returns
            -------
            x_in : [B, A, T, F]  values with masked channels zeroed
            indicators : [B, A, T, F]  1 where channel was masked
            frame_blocked : [B, A, T] bool  True → attention key/value blocked
            """
            b, a, t, f = x.shape
            indicators = torch.zeros_like(x)
            x_in = x

            if channel_mask is not None:
                channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)
                if channel_mask.shape != x.shape:
                    raise ValueError(
                        f"channel_mask shape {tuple(channel_mask.shape)} "
                        f"!= x shape {tuple(x.shape)}"
                    )
                indicators = channel_mask.to(dtype=x.dtype)
                x_in = x * (1.0 - indicators)

            frame_blocked = torch.zeros(b, a, t, dtype=torch.bool, device=x.device)

            if time_mask is not None:
                time_mask = time_mask.to(device=x.device, dtype=torch.bool)
                if time_mask.shape != (b, a, t):
                    raise ValueError(
                        f"time_mask shape {tuple(time_mask.shape)} "
                        f"!= {(b, a, t)}"
                    )
                # Time-segment mask ≡ all channels masked on those frames
                tm = time_mask.unsqueeze(-1)  # [B, A, T, 1]
                x_in = x_in.masked_fill(tm, 0.0)
                indicators = indicators.masked_fill(tm, 1.0)
                frame_blocked = frame_blocked | time_mask

            # Fully channel-masked frame → attention block
            if channel_mask is not None:
                frame_blocked = frame_blocked | channel_mask.all(dim=-1)

            return x_in, indicators, frame_blocked

        def encode_agents(
            self,
            x: "torch.Tensor",
            time_mask: Optional["torch.Tensor"] = None,
            channel_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Encode each agent track independently with a shared Transformer.

            Parameters
            ----------
            x : Tensor [B, A, T, F]
            time_mask, channel_mask : optional masks (see class docstring)

            Returns
            -------
            h : Tensor [B, A, H]  CLS representation per agent
            """
            b, a, t, f = x.shape
            if t > self.max_frames:
                raise ValueError(
                    f"T={t} exceeds max_frames={self.max_frames} "
                    f"(set model.num_frames in config)"
                )

            x_in, indicators, frame_blocked = self._apply_masks(
                x, time_mask, channel_mask
            )
            # [B, A, T, 2F] → project → [B, A, T, H]
            emb = self.input_proj(torch.cat([x_in, indicators], dim=-1))
            emb = emb + self.pos_embed[:, :t, :].unsqueeze(1)

            # Flatten agent axis: [B*A, T, H]
            flat = emb.reshape(b * a, t, self.hidden_dim)
            cls = self.cls_token.expand(b * a, -1, -1)  # [B*A, 1, H]
            tokens = torch.cat([cls, flat], dim=1)  # [B*A, 1+T, H]

            # key_padding_mask: True = ignore. CLS (index 0) always visible.
            pad = torch.zeros(b * a, 1 + t, dtype=torch.bool, device=x.device)
            pad[:, 1:] = frame_blocked.reshape(b * a, t)

            encoded = self.encoder(tokens, src_key_padding_mask=pad)
            h_cls = encoded[:, 0, :]  # [B*A, H]
            return h_cls.view(b, a, self.hidden_dim)

        @staticmethod
        def _masked_mean(
            h_nbr: "torch.Tensor",
            nbr_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """Mean-pool neighbor embeddings over valid slots only."""
            weights = nbr_mask.to(dtype=h_nbr.dtype).unsqueeze(-1)  # [B, A-1, 1]
            denom = weights.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            return (h_nbr * weights).sum(dim=1) / denom

        def forward(
            self,
            x: "torch.Tensor",
            agent_mask: Optional["torch.Tensor"] = None,
            time_mask: Optional["torch.Tensor"] = None,
            channel_mask: Optional["torch.Tensor"] = None,
        ) -> dict[str, "torch.Tensor"]:
            """
            Parameters
            ----------
            x : Tensor [B, A, T, F]
            agent_mask : Tensor [B, A] bool/float, or None (= all present)
            time_mask : Tensor [B, A, T] bool, True = blocked frame (optional)
            channel_mask : Tensor [B, A, T, F] bool, True = masked channel (optional)

            Returns
            -------
            dict with:
              conflict_logit : [B]
              target_logits  : [B, A-1]  (empty slots filled with -inf)
            """
            if x.dim() != 4:
                raise ValueError(f"Expected x [B,A,T,F], got shape {tuple(x.shape)}")
            b, a, _, _ = x.shape
            if a != self.num_agents:
                raise ValueError(
                    f"num_agents mismatch: cfg={self.num_agents}, got A={a}"
                )

            if agent_mask is None:
                agent_mask = torch.ones(b, a, dtype=torch.bool, device=x.device)
            else:
                agent_mask = agent_mask.to(device=x.device, dtype=torch.bool)

            h = self.encode_agents(x, time_mask=time_mask, channel_mask=channel_mask)
            h_ego = h[:, 0]  # [B, H]
            h_nbr = h[:, 1:]  # [B, A-1, H]
            nbr_mask = agent_mask[:, 1:]  # [B, A-1]

            h_nbr_pool = self._masked_mean(h_nbr, nbr_mask)
            conflict_logit = self.conflict_head(
                torch.cat([h_ego, h_nbr_pool], dim=-1)
            ).squeeze(-1)

            ego_exp = h_ego.unsqueeze(1).expand_as(h_nbr)
            target_logits = self.target_head(
                torch.cat([ego_exp, h_nbr], dim=-1)
            ).squeeze(-1)  # [B, A-1]
            target_logits = target_logits.masked_fill(~nbr_mask, float("-inf"))

            return {
                "conflict_logit": conflict_logit,
                "target_logits": target_logits,
            }

        def compute_loss(
            self,
            outputs: dict[str, "torch.Tensor"],
            is_conflict: "torch.Tensor",
            target_idx: "torch.Tensor",
            *,
            lambda_target: float = 1.0,
        ) -> dict[str, "torch.Tensor"]:
            """Same loss as LSTM baseline: conflict BCE + target CE."""
            conflict_logit = outputs["conflict_logit"]
            target_logits = outputs["target_logits"]

            conflict_loss = F.binary_cross_entropy_with_logits(
                conflict_logit, is_conflict.to(dtype=conflict_logit.dtype)
            )

            conf_mask = (
                is_conflict.to(dtype=torch.bool)
                & (target_idx >= 0).to(dtype=torch.bool)
            )
            if conf_mask.any():
                target_loss = F.cross_entropy(
                    target_logits[conf_mask],
                    target_idx[conf_mask].long(),
                )
            else:
                target_loss = conflict_logit.new_zeros(())

            total = conflict_loss + lambda_target * target_loss
            return {
                "loss": total,
                "conflict_loss": conflict_loss.detach(),
                "target_loss": target_loss.detach(),
            }

else:

    class TransformerConflictModel:  # type: ignore[no-redef]
        """Stub when torch is not installed."""

        def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
            _require_torch()
