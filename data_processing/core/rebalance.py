"""Post-Stage-3 rebalancing: downsample conflicts by conflict_target_role.

Operates on ``TrackedNeighborhoodEvent`` objects (which carry
``conflict_target_role`` assigned during Stage 3 neighbour extraction).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from data_processing.io.schema import ProcessingConfig, TrackedNeighborhoodEvent


def balance_conflict_types(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> List[TrackedNeighborhoodEvent]:
    """Downsample over-represented conflict-target roles within each scene.

    For each scene, conflict events are grouped by ``conflict_target_role``.
    Every role whose count exceeds the minimum role count is randomly
    downsampled (never upsampled).

    Non-conflict events pass through unchanged.

    Parameters
    ----------
    events : list of TrackedNeighborhoodEvent
    cfg : ProcessingConfig
        Uses ``rebalance_per_scene`` and ``rebalance_seed``.

    Returns
    -------
    list of TrackedNeighborhoodEvent
        Rebalanced events (same order within each group; only subsets removed).
    """
    if not cfg.rebalance_balance_conflict_types:
        return events

    seed = cfg.rebalance_seed
    per_scene = cfg.rebalance_per_scene

    conflicts: List[TrackedNeighborhoodEvent] = []
    non_conflicts: List[TrackedNeighborhoodEvent] = []
    for e in events:
        if e.window.is_conflict:
            conflicts.append(e)
        else:
            non_conflicts.append(e)

    if not conflicts:
        return events

    # ── Group conflicts ──────────────────────────────────────────────────
    if per_scene:
        groups: Dict[str, List[TrackedNeighborhoodEvent]] = defaultdict(list)
        for e in conflicts:
            groups[e.window.scene_id].append(e)
    else:
        groups = {"_all": conflicts}

    balanced: List[TrackedNeighborhoodEvent] = []

    for scene_id, group in groups.items():
        # Group by conflict_target_role within this scene
        role_groups: Dict[str, List[TrackedNeighborhoodEvent]] = defaultdict(list)
        for e in group:
            role = e.conflict_target_role or "unknown"
            role_groups[role].append(e)

        if len(role_groups) <= 1:
            balanced.extend(group)
            continue

        # Find the smallest role count — that's our cap per role
        min_count = min(len(v) for v in role_groups.values())
        if min_count == 0:
            balanced.extend(group)
            continue

        rng = random.Random(seed)
        for role, role_events in role_groups.items():
            if len(role_events) > min_count:
                kept = rng.sample(role_events, min_count)
                balanced.extend(kept)
                print(f"  [{scene_id}] conflict type '{role}': "
                      f"{len(role_events)} → {min_count}")
            else:
                balanced.extend(role_events)

    return balanced + non_conflicts
