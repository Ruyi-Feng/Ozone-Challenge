"""Geometry helpers: distance, relative pose, neighbor slot classification."""

from __future__ import annotations

from typing import Optional, Tuple


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    raise NotImplementedError


def compute_relative_pose(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    other_x: float,
    other_y: float,
) -> Tuple[float, float]:
    """
    Transform other vehicle into ego frame.

    Returns
    -------
    (dx_forward, dy_left) or project-specific (longitudinal, lateral).
    """
    raise NotImplementedError


def classify_neighbor_slot(
    dx: float,
    dy: float,
    *,
    slots: Tuple[str, ...] = (
        "front",
        "rear",
        "left_front",
        "left_rear",
        "right_front",
        "right_rear",
    ),
) -> Optional[str]:
    """
    Map relative pose to one of the six neighbor slots.

    Returns None if the vehicle does not fall into any configured slot.
    """
    raise NotImplementedError


def is_within_max_distance(distance_m: float, max_distance_m: float = 200.0) -> bool:
    """Return True if distance is within the inclusion threshold."""
    return distance_m <= max_distance_m
