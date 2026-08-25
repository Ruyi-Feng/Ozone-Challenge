"""Causal temporal Transformer + role-conditioned cross-agent attention.

Unlike CrossAgentTransformerConflictModel this network accepts a variable
agent count 1 <= A <= max_agents and does not use a fixed slot embedding.
Agent identity is carried only by a role embedding (plus the trajectory).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from data_processing.core.variable_agents import MAX_AGENTS, NUM_ROLES, PAD_ROLE_ID
from model_baseline.config import ModelConfig
from model_baseline.models.cross_agent_transformer import (
    CausalTemporalTransformerConflictModel,
)


class VariableCrossAgentTransformerConflictModel(
    CausalTemporalTransformerConflictModel
):
    """Variable-agent Cross-Agent Transformer (no absolute slot embedding)."""

    def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.max_agents = int(getattr(cfg, "max_agents", None) or MAX_AGENTS)
        self.num_roles = int(getattr(cfg, "num_roles", None) or NUM_ROLES)
        self.pad_role_id = int(getattr(cfg, "pad_role_id", None) or PAD_ROLE_ID)
        self.cross_layers = int(cfg.cross_layers)
        if self.cross_layers <= 0:
            raise ValueError("cross_layers must be positive")

        self.role_embedding = nn.Embedding(
            self.num_roles + 1,
            self.hidden_dim,
            padding_idx=self.pad_role_id,
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
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        with torch.no_grad():
            self.role_embedding.weight[self.pad_role_id].zero_()

    def _check_agent_count(self, a: int) -> None:
        if not 1 <= a <= self.max_agents:
            raise ValueError(
                f"agent count A={a} outside [1, max_agents={self.max_agents}]"
            )

    def encode_agents(
        self,
        x: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, a, _, _ = x.shape
        self._check_agent_count(a)
        if agent_mask is None:
            agent_mask = torch.ones(b, a, dtype=torch.bool, device=x.device)
        else:
            agent_mask = agent_mask.to(device=x.device, dtype=torch.bool)
        if role_ids is None:
            raise ValueError("role_ids is required for the variable-agent model")
        role_ids = role_ids.to(device=x.device, dtype=torch.long)
        if role_ids.shape != (b, a):
            raise ValueError(
                f"role_ids shape {tuple(role_ids.shape)} != {(b, a)}"
            )

        summaries = CausalTemporalTransformerConflictModel.encode_agents(
            self, x, time_mask=time_mask, channel_mask=channel_mask
        )
        summaries = summaries + self.role_embedding(role_ids)
        return self.agent_encoder(
            summaries,
            src_key_padding_mask=~agent_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected x [B,A,T,F], got shape {tuple(x.shape)}")
        b, a, _, _ = x.shape
        self._check_agent_count(a)
        if agent_mask is None:
            agent_mask = torch.ones(b, a, dtype=torch.bool, device=x.device)
        else:
            agent_mask = agent_mask.to(device=x.device, dtype=torch.bool)
        if not bool(agent_mask[:, 0].all()):
            raise ValueError("ego agent_mask must be True for every sample")

        h = self.encode_agents(
            x,
            agent_mask=agent_mask,
            role_ids=role_ids,
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

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        is_conflict: torch.Tensor,
        target_idx: torch.Tensor,
        *,
        lambda_target: float = 1.0,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        conflict_logit = outputs["conflict_logit"]
        target_logits = outputs["target_logits"]
        conflict_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            conflict_logit, is_conflict.to(dtype=conflict_logit.dtype)
        )

        n_nbr = target_logits.size(-1)
        conf_mask = (
            is_conflict.to(dtype=torch.bool)
            & (target_idx >= 0)
            & (target_idx < n_nbr)
        )
        if agent_mask is not None and n_nbr > 0:
            nbr_mask = agent_mask.to(dtype=torch.bool)[:, 1:]
            safe_idx = target_idx.clamp(min=0, max=max(n_nbr - 1, 0)).long()
            in_set = nbr_mask.gather(1, safe_idx.unsqueeze(1)).squeeze(1)
            conf_mask = conf_mask & in_set

        if conf_mask.any():
            selected = target_logits[conf_mask]
            labels = target_idx[conf_mask].long()
            finite = torch.isfinite(
                selected.gather(1, labels.unsqueeze(1)).squeeze(1)
            )
            if finite.any():
                target_loss = torch.nn.functional.cross_entropy(
                    selected[finite], labels[finite]
                )
            else:
                target_loss = conflict_logit.new_zeros(())
        else:
            target_loss = conflict_logit.new_zeros(())

        total = conflict_loss + lambda_target * target_loss
        return {
            "loss": total,
            "conflict_loss": conflict_loss.detach(),
            "target_loss": target_loss.detach(),
        }
