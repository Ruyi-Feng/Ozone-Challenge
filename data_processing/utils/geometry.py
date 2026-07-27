"""Geometry helpers: distance, relative pose, neighbor slot classification."""

from __future__ import annotations

import math
from typing import Optional, Tuple

# Default lateral threshold (meters) for separating pure front/rear from
# left/right variants.  When |dy_left| <= this value the vehicle is
# considered to be directly ahead / behind.
DEFAULT_LATERAL_THRESHOLD_M = 2.0


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


def compute_relative_pose(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    other_x: float,
    other_y: float,
) -> Tuple[float, float]:
    """
    Transform other vehicle into ego vehicle's local coordinate frame.

    Coordinate convention (matching NBDT / Ozone standard):
      - heading 0   = East  (+X axis)
      - heading 90  = North (+Y axis)

    Returns
    -------
    (longitudinal, lateral)
        longitudinal > 0  →  other is ahead of ego
        longitudinal < 0  →  other is behind ego
        lateral     > 0  →  other is to the left of ego
        lateral     < 0  →  other is to the right of ego
    """
    theta = math.radians(ego_heading)
    dx = other_x - ego_x
    dy = other_y - ego_y
    # Project (dx, dy) onto ego forward (cosθ, sinθ) and left (-sinθ, cosθ)
    longitudinal = dx * math.cos(theta) + dy * math.sin(theta)
    lateral = -dx * math.sin(theta) + dy * math.cos(theta)
    return longitudinal, lateral


def classify_neighbor_slot(
    dx: float,
    dy: float,
    *,
    lateral_threshold: float = DEFAULT_LATERAL_THRESHOLD_M,
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
    Map relative pose (dx=longitudinal, dy=lateral) to one of the six neighbor slots.

    Classification rules
    --------------------
    - |dy| <= lateral_threshold  →  pure longitudinal: "front" or "rear"
    - |dy| >  lateral_threshold  →  lateral quadrant (e.g. "left_front", "right_rear")

    Returns None if no matching slot is found among the configured *slots*.
    """
    is_front = dx >= 0
    is_rear = dx < 0
    is_left = dy > lateral_threshold
    is_right = dy < -lateral_threshold
    is_center = not is_left and not is_right

    candidate: Optional[str] = None
    if is_front and is_center:
        candidate = "front"
    elif is_rear and is_center:
        candidate = "rear"
    elif is_front and is_left:
        candidate = "left_front"
    elif is_rear and is_left:
        candidate = "left_rear"
    elif is_front and is_right:
        candidate = "right_front"
    elif is_rear and is_right:
        candidate = "right_rear"

    if candidate is not None and candidate in slots:
        return candidate
    return None


def is_within_max_distance(distance_m: float, max_distance_m: float = 200.0) -> bool:
    """Return True if distance is within the inclusion threshold."""
    return distance_m <= max_distance_m
