"""Baseline conflict model: shared LSTM + dual heads.

Task
----
Given multi-agent previous tracks (history), predict:
  1) whether a conflict will occur in the future window;
  2) which neighbor vehicle is the conflict partner.

Variable neighbor count
-----------------------
Each sample is padded to a fixed agent axis A (ego + 6 slots).
Missing slots are zero-filled and marked False in ``agent_mask``.
  - conflict head: mean-pools only valid neighbors
  - target head: empty-slot logits set to -inf (ignored by softmax / CE)

I/O contract
------------
Input:
  x          : [B, A, T, F]
  agent_mask : [B, A]  bool, True = slot occupied (ego should be True)

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
            "PyTorch is required for BaselineConflictModel. "
            "Uncomment / install torch in requirements.txt."
        )


if nn is not None:

    class BaselineConflictModel(nn.Module):
        """Shared LSTM encoder + conflict / target dual heads."""

        def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
            super().__init__()
            self.cfg = cfg
            self.num_agents = cfg.num_agents
            self.num_features = cfg.num_features
            self.hidden_dim = cfg.hidden_dim
            self.num_neighbors = self.num_agents - 1  # exclude ego

            self.encoder = nn.LSTM(
                input_size=self.num_features,
                hidden_size=self.hidden_dim,
                batch_first=True,
            )
            # [h_ego ; h_nbr_pool] → conflict logit
            self.conflict_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            # [h_ego ; h_a] → scalar score per neighbor
            self.target_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )

        def encode_agents(self, x: "torch.Tensor") -> "torch.Tensor":
            """Encode each agent track independently with a shared LSTM.

            Parameters
            ----------
            x : Tensor [B, A, T, F]

            Returns
            -------
            h : Tensor [B, A, H]  last hidden state per agent
            """
            b, a, t, f = x.shape
            flat = x.reshape(b * a, t, f)
            _, (h_n, _) = self.encoder(flat)
            # h_n: [num_layers, B*A, H] → take last layer
            return h_n[-1].view(b, a, self.hidden_dim)

        @staticmethod
        def _masked_mean(
            h_nbr: "torch.Tensor",
            nbr_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """Mean-pool neighbor embeddings over valid slots only.

            If a sample has zero neighbors, returns a zero vector.
            """
            weights = nbr_mask.to(dtype=h_nbr.dtype).unsqueeze(-1)  # [B, A-1, 1]
            denom = weights.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            return (h_nbr * weights).sum(dim=1) / denom

        def forward(
            self,
            x: "torch.Tensor",
            agent_mask: Optional["torch.Tensor"] = None,
            time_mask: Optional["torch.Tensor"] = None,
            channel_mask: Optional["torch.Tensor"] = None,
            role_ids: Optional["torch.Tensor"] = None,
        ) -> dict[str, "torch.Tensor"]:
            """
            Parameters
            ----------
            x : Tensor [B, A, T, F]
            agent_mask : Tensor [B, A] bool/float, or None (= all present)

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

            h = self.encode_agents(x)  # [B, A, H]
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
            # empty slots must not win softmax / CE
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
            agent_mask: Optional["torch.Tensor"] = None,
        ) -> dict[str, "torch.Tensor"]:
            """
            Parameters
            ----------
            outputs : forward() result
            is_conflict : [B]  0/1
            target_idx : [B]   neighbor index in {0..A-2}, or -1 if no conflict
            lambda_target : weight for target CE (only on conflict samples)

            Notes
            -----
            Empty-slot logits are already -inf, so CE ignores them for present
            neighbors. Target loss is computed only where ``is_conflict == 1``.
            """
            conflict_logit = outputs["conflict_logit"]
            target_logits = outputs["target_logits"]

            conflict_loss = F.binary_cross_entropy_with_logits(
                conflict_logit, is_conflict.to(dtype=conflict_logit.dtype)
            )

            # Target loss only on conflict samples whose target is in a
            # neighbour slot (target_idx >= 0).  When the conflict partner
            # is outside the neighbourhood the sample still contributes to
            # the conflict BCE but the target CE is skipped.
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

    class BaselineConflictModel:  # type: ignore[no-redef]
        """Stub when torch is not installed."""

        def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
            _require_torch()
