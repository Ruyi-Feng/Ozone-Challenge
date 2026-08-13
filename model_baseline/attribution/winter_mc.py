"""Winter value via nested-permutation Monte Carlo + exact Shapley for small games.

Ported from the text-version engine (token_winter_shapley_analysis.py):
same chain-batched incremental unmasking, coalition value cache, and P4
telescoping — with the tokenizer masker replaced by (agent, segment, channel)
cell masks and the value function replaced by the conflict logit.

Differences from the text version:
- coalition cache key = packed bitmask bytes of the masked-cell vector
  (canonical, 56 B for 448 cells) instead of frozenset of token indices;
- the coalition state is rebuilt as a channel_mask per step instead of
  in-place unmasking of input ids (x is never mutated);
- per-chain marginals are kept to report Monte-Carlo standard errors.

Estimator note: for ANY fixed distribution over nested orders — uniform
(symmetric Winter) or order-restricted (chrono/reverse, asymmetric Shapley
sensu Frye et al. 2020) — each chain telescopes from v(∅) to v(N), so
per-level efficiency (P4) and aggregation consistency (P1) hold by
construction; only the symmetry axiom differs between readouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Callable, Optional

import numpy as np

from model_baseline.masking import (
    CellPartition,
    LevelOrderConfig,
    coalition_masks,
    sample_nested_order,
)

ValueFn = Callable[[np.ndarray], np.ndarray]  # [B,A,T,F] bool → [B] float


def _key(masked_row: np.ndarray) -> bytes:
    return np.packbits(masked_row).tobytes()


@dataclass
class WinterResult:
    cell_psi: np.ndarray          # [n_cells] Winter value per cell
    cell_se: np.ndarray           # [n_cells] MC standard error (0 for n=1)
    v_full: float
    v_empty: float
    n_samples: int
    n_forwards: int               # value evaluations spent by THIS call
    n_cache_hits: int
    n_marginals: int

    @property
    def p4_residual(self) -> float:
        """v(N) − v(∅) − Σψ; ~float-sum noise by construction."""
        return float(self.v_full - self.v_empty - self.cell_psi.sum())


def _boundary_values(
    value_fn: ValueFn,
    p: CellPartition,
    valid_mask: np.ndarray,
    cache: dict[bytes, float],
) -> tuple[float, float, int]:
    """Ensure v(N) (nothing masked) and v(∅) (all masked) are in the cache."""
    n_forwards = 0
    all_visible = np.zeros(p.n_cells, dtype=bool)
    all_masked = np.ones(p.n_cells, dtype=bool)
    todo = []
    for state in (all_visible, all_masked):
        if _key(state) not in cache:
            todo.append(state)
    if todo:
        cms = np.stack(
            [coalition_masks(p, s, valid_mask)[1] for s in todo], axis=0
        )
        vals = value_fn(cms)
        n_forwards += len(todo)
        for s, v in zip(todo, vals):
            cache[_key(s)] = float(v)
    return cache[_key(all_visible)], cache[_key(all_masked)], n_forwards


def winter_value_nested_mc(
    value_fn: ValueFn,
    p: CellPartition,
    valid_mask: np.ndarray,
    orders: LevelOrderConfig,
    *,
    n_samples: int = 25,
    batch_size: int = 32,
    rng: np.random.Generator,
    cache: Optional[dict[bytes, float]] = None,
    verbose: bool = False,
) -> WinterResult:
    """Nested-permutation MC estimate of the Winter value for one event.

    ``cache`` may be shared across readouts of the same event (v(S) is the
    same function; only the order distribution differs).  Never share it
    across events.
    """
    if cache is None:
        cache = {}
    n_cells = p.n_cells
    valid_mask = np.asarray(valid_mask, dtype=bool)

    v_full, v_empty, n_forwards = _boundary_values(
        value_fn, p, valid_mask, cache
    )
    n_hits = 0
    n_marginals = 0

    if n_cells == 0:
        return WinterResult(
            cell_psi=np.zeros(0), cell_se=np.zeros(0),
            v_full=v_full, v_empty=v_empty, n_samples=0,
            n_forwards=n_forwards, n_cache_hits=0, n_marginals=0,
        )

    # per-chain marginals → mean = psi, std/sqrt(n) = MC standard error
    marginals = np.zeros((n_samples, n_cells), dtype=np.float64)

    done = 0
    while done < n_samples:
        B = min(batch_size, n_samples - done)
        chain_orders = [sample_nested_order(p, orders, rng) for _ in range(B)]
        masked = np.ones((B, n_cells), dtype=bool)
        prev_vals = np.full(B, v_empty, dtype=np.float64)

        for step in range(n_cells):
            step_cells = np.array(
                [chain_orders[b][step] for b in range(B)], dtype=np.int64
            )
            for b in range(B):
                masked[b, step_cells[b]] = False

            keys = [_key(masked[b]) for b in range(B)]
            curr_vals = np.empty(B, dtype=np.float64)
            uncached: list[int] = []
            for b, k in enumerate(keys):
                v = cache.get(k)
                if v is None:
                    uncached.append(b)
                else:
                    curr_vals[b] = v
                    n_hits += 1

            if uncached:
                cms = np.stack(
                    [
                        coalition_masks(p, masked[b], valid_mask)[1]
                        for b in uncached
                    ],
                    axis=0,
                )
                vals = value_fn(cms)
                n_forwards += len(uncached)
                for i, b in enumerate(uncached):
                    v = float(vals[i])
                    cache[keys[b]] = v
                    curr_vals[b] = v

            for b in range(B):
                marginals[done + b, step_cells[b]] = curr_vals[b] - prev_vals[b]
            prev_vals = curr_vals
            n_marginals += B

        done += B
        if verbose:
            print(f"  MC chains {done}/{n_samples}, cache={len(cache)}, "
                  f"forwards={n_forwards}, hits={n_hits}")

    cell_psi = marginals.mean(axis=0)
    if n_samples > 1:
        cell_se = marginals.std(axis=0, ddof=1) / np.sqrt(n_samples)
    else:
        cell_se = np.zeros(n_cells)

    return WinterResult(
        cell_psi=cell_psi, cell_se=cell_se,
        v_full=v_full, v_empty=v_empty, n_samples=n_samples,
        n_forwards=n_forwards, n_cache_hits=n_hits, n_marginals=n_marginals,
    )


# ---------------------------------------------------------------------------
# Exact Shapley for small single-level games (case studies)
# ---------------------------------------------------------------------------


@dataclass
class ExactResult:
    phi: np.ndarray               # [n_players]
    v_full: float                 # all players visible (rest of scene visible too)
    v_empty: float                # all players masked (context stays visible)
    n_forwards: int


def exact_shapley(
    value_fn: ValueFn,
    p: CellPartition,
    valid_mask: np.ndarray,
    player_cells: list[np.ndarray],
    *,
    batch_size: int = 256,
    max_players: int = 16,
    cache: Optional[dict[bytes, float]] = None,
) -> ExactResult:
    """Exact Shapley over 2^n coalitions of cell GROUPS (players).

    Cells not covered by any player stay visible throughout (context).
    n > max_players is refused: 2^n forwards grow out of case-study range
    (n=16 → 65k evaluations; the joint agent×segment game is impossible).
    """
    n = len(player_cells)
    if n > max_players:
        raise ValueError(
            f"{n} players exceeds max_players={max_players} "
            f"(2^{n} = {1 << n} coalitions) — use the MC estimator instead"
        )
    if cache is None:
        cache = {}
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n_sub = 1 << n

    # coalition bitmask m: bit i set = player i VISIBLE
    values = np.empty(n_sub, dtype=np.float64)
    n_forwards = 0
    pend_ids: list[int] = []
    pend_masks: list[np.ndarray] = []

    def _flush() -> None:
        nonlocal n_forwards
        if not pend_ids:
            return
        vals = value_fn(np.stack(pend_masks, axis=0))
        n_forwards += len(pend_ids)
        for m, v, state_key in zip(pend_ids, vals, pend_keys):
            values[m] = float(v)
            cache[state_key] = float(v)
        pend_ids.clear()
        pend_masks.clear()
        pend_keys.clear()

    pend_keys: list[bytes] = []
    for m in range(n_sub):
        masked = np.zeros(p.n_cells, dtype=bool)
        for i in range(n):
            if not (m >> i) & 1:
                masked[player_cells[i]] = True
        k = _key(masked)
        cached = cache.get(k)
        if cached is not None:
            values[m] = cached
            continue
        pend_ids.append(m)
        pend_keys.append(k)
        pend_masks.append(coalition_masks(p, masked, valid_mask)[1])
        if len(pend_ids) >= batch_size:
            _flush()
    _flush()

    # phi_i = Σ_{S ⊆ N\{i}} |S|!(n-1-|S|)!/n! · (v(S∪{i}) − v(S))
    weights = np.array(
        [1.0 / (n * comb(n - 1, s)) for s in range(n)], dtype=np.float64
    )
    popcount = np.array(
        [bin(m).count("1") for m in range(n_sub)], dtype=np.int64
    )
    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bit = 1 << i
        for m in range(n_sub):
            if m & bit:
                continue
            phi[i] += weights[popcount[m]] * (values[m | bit] - values[m])

    return ExactResult(
        phi=phi,
        v_full=float(values[n_sub - 1]),
        v_empty=float(values[0]),
        n_forwards=n_forwards,
    )
