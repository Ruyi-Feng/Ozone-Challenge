"""Train/validation split to prevent data leakage across ego trajectories.

The core principle: all events sharing the same ego_id are assigned to the
same split.  This guarantees that no trajectory segment appears as *future*
in one sample and *history* in another across the train/val boundary.
"""

from __future__ import annotations

import random
from typing import Any, List, Set, Tuple

from data_processing.io.schema import TrackedNeighborhoodEvent, WindowedEventCandidate


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


def label_windows_with_split(
    windows: List[WindowedEventCandidate],
    *,
    transfer_prefixes: tuple[str, ...] = (),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Write ``window.meta["split"]`` in-place: train / val / test / transfer.

    Any scene whose ``scene_id`` starts with a transfer prefix is assigned
    entirely to ``transfer`` (never train/val). Remaining egos are split by
    ``(scene_id, ego_id)`` so the same vehicle never crosses the
    train/val/test boundary.
    """
    rest: list[WindowedEventCandidate] = []
    prefixes = tuple(p for p in transfer_prefixes if p)
    for window in windows:
        scene = str(window.scene_id)
        if any(scene.startswith(prefix) for prefix in prefixes):
            window.meta["split"] = "transfer"
        else:
            rest.append(window)

    keys = sorted({(w.scene_id, w.ego_id) for w in rest})
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    if n == 0:
        return

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(1, n_val - 1)
            n_test = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
    else:
        n_train = max(1, min(n, n_train or 1))
        n_val = min(n - n_train, max(0, n_val))
        n_test = n - n_train - n_val

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    for window in rest:
        key = (window.scene_id, window.ego_id)
        if key in train_keys:
            window.meta["split"] = "train"
        elif key in val_keys:
            window.meta["split"] = "val"
        else:
            window.meta["split"] = "test"
