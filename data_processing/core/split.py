"""Train/validation split to prevent data leakage across ego trajectories.

The core principle: all events sharing the same ego_id are assigned to the
same split.  This guarantees that no trajectory segment appears as *future*
in one sample and *history* in another across the train/val boundary.
"""

from __future__ import annotations

import random
from typing import Any, List, Set, Tuple

from data_processing.io.schema import TrackedNeighborhoodEvent


def split_by_ego(
    events: List[TrackedNeighborhoodEvent],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[TrackedNeighborhoodEvent], List[TrackedNeighborhoodEvent]]:
    """Split events into train/val sets, keeping each ego_id in one split only.

    Parameters
    ----------
    events : List[TrackedNeighborhoodEvent]
        All events from the pipeline (conflict + non-conflict).
    train_ratio : float
        Fraction of unique ego_ids assigned to the training set.
    seed : int
        Fixed seed for reproducible splits.

    Returns
    -------
    (train_events, val_events) : Tuple of two event lists.
    """
    if not events:
        return [], []

    # Collect unique ego ids in deterministic order, then shuffle
    ego_ids = sorted({e.window.ego_id for e in events})
    rng = random.Random(seed)
    rng.shuffle(ego_ids)

    n_train = max(1, int(len(ego_ids) * train_ratio))
    train_egos: Set[Any] = set(ego_ids[:n_train])

    train = [e for e in events if e.window.ego_id in train_egos]
    val = [e for e in events if e.window.ego_id not in train_egos]
    return train, val
