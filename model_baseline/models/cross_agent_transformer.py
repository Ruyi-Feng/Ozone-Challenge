"""Causal temporal Transformer followed by cross-agent self-attention.

At prediction time T, each agent is first encoded from observations at times
<= T only.  Cross-agent attention then operates on those history summaries;
no post-T trajectory is part of the model input.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from model_baseline.config import ModelConfig
from model_baseline.models.transformer import TransformerConflictModel


class CausalTemporalTransformerConflictModel(TransformerConflictModel):
    """Per-agent temporal Transformer with strict causal frame attention."""

    def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)

        # Masked and causal paths are not compatible with PyTorch's prototype
        # nested-tensor fast path for every padding pattern.
        self.encoder.enable_nested_tensor = False
        self.encoder.use_nested_tensor = False

    def _encode_temporal_tokens(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-agent causal frame tokens plus a final summary token.

        Output shape is [B, A, T+1, H]. Frame token t can attend only to
        frames <= t. The summary token is placed last, so it can attend to the
        complete history available at prediction time.
        """
        b, a, t, _ = x.shape
        if t > self.max_frames:
            raise ValueError(
                f"T={t} exceeds max_frames={self.max_frames} "
                f"(set model.num_frames in config)"
            )

        x_in, indicators, frame_blocked = self._apply_masks(
            x, time_mask, channel_mask
        )
        emb = self.input_proj(torch.cat([x_in, indicators], dim=-1))
        emb = emb + self.pos_embed[:, :t, :].unsqueeze(1)

        flat = emb.reshape(b * a, t, self.hidden_dim)
        summary = self.cls_token.expand(b * a, -1, -1)
        # Frames precede summary: causal attention makes every frame
        # independent of later frames while summary sees all history.
        tokens = torch.cat([flat, summary], dim=1)

        causal = torch.triu(
            torch.ones(t + 1, t + 1, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        padding = torch.zeros(
            b * a, t + 1, dtype=torch.bool, device=x.device
        )
        padding[:, :t] = frame_blocked.reshape(b * a, t)

        # Merge causality and blocked-key rules into one per-sample attention
        # mask. A blocked query may attend only to itself; it therefore stays
        # finite but remains unavailable as a key to every other query.
        # This avoids all--inf softmax rows without exposing masked history.
        attention_mask = causal.unsqueeze(0).expand(
            b * a, -1, -1
        ).clone()
        attention_mask |= padding.unsqueeze(1)
        blocked_rows, blocked_queries = padding.nonzero(as_tuple=True)
        attention_mask[
            blocked_rows, blocked_queries, blocked_queries
        ] = False
        attention_mask = attention_mask.repeat_interleave(
            self.n_heads, dim=0
        )

        encoded = tokens
        for layer in self.encoder.layers:
            encoded = layer(
                encoded,
                src_mask=attention_mask,
                src_key_padding_mask=None,
                is_causal=True,
            )
        if self.encoder.norm is not None:
            encoded = self.encoder.norm(encoded)
        return encoded.view(b, a, t + 1, self.hidden_dim)

    def encode_agents(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return each agent's history-only representation at prediction T."""
        temporal = self._encode_temporal_tokens(
            x, time_mask=time_mask, channel_mask=channel_mask
        )
        return temporal[:, :, -1, :]


class CrossAgentTransformerConflictModel(CausalTemporalTransformerConflictModel):
    """Causal temporal summaries followed by cross-agent self-attention."""

    def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.cross_layers = cfg.cross_layers
        if self.cross_layers <= 0:
            raise ValueError("cross_layers must be positive")

        self.agent_embed = nn.Parameter(
            torch.zeros(1, self.num_agents, self.hidden_dim)
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.agent_encoder = nn.TransformerEncoder(
            cross_layer,
            num_layers=self.cross_layers,
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.agent_embed, std=0.02)

    def encode_agents(
        self,
        x: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode history causally, then exchange information across agents."""
        b, a, _, _ = x.shape
        if agent_mask is None:
            agent_mask = torch.ones(b, a, dtype=torch.bool, device=x.device)
        else:
            agent_mask = agent_mask.to(device=x.device, dtype=torch.bool)
        summaries = super().encode_agents(
            x, time_mask=time_mask, channel_mask=channel_mask
        )
        summaries = summaries + self.agent_embed[:, :a, :]
        return self.agent_encoder(
            summaries,
            src_key_padding_mask=~agent_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
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

        h = self.encode_agents(
            x,
            agent_mask=agent_mask,
            time_mask=time_mask,
            channel_mask=channel_mask,
        )
        h_ego = h[:, 0]
        h_nbr = h[:, 1:]
        nbr_mask = agent_mask[:, 1:]

        h_nbr_pool = self._masked_mean(h_nbr, nbr_mask)
        conflict_logit = self.conflict_head(
            torch.cat([h_ego, h_nbr_pool], dim=-1)
        ).squeeze(-1)

        ego_expanded = h_ego.unsqueeze(1).expand_as(h_nbr)
        target_logits = self.target_head(
            torch.cat([ego_expanded, h_nbr], dim=-1)
        ).squeeze(-1)
        target_logits = target_logits.masked_fill(~nbr_mask, float("-inf"))
        return {
            "conflict_logit": conflict_logit,
            "target_logits": target_logits,
        }
