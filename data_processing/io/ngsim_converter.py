"""NGSIM raw (10fps) → Ozone standardized format converter.

Adapted from NBDT ``NGSIMTransfer`` in
``standardized_data_toolkit/dataloader/dataloader.py``.

Key differences from the NBDT original:
- Standalone functions (no BasicTransfer base class).
- Column-name casing fixed (the real CSV uses ``v_Length``, etc.).
- Keeps 10 fps — no frame-rate upsampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEET_TO_METER = 0.3048

# NGSIM v_Class → NBDT objClass
#   1 = motorcycle → 4
#   2 = auto       → 0  (car)
#   3 = truck      → 3
NGSIM_CLASS_MAP: dict = {
    1: 4,
    2: 0,
    3: 3,
}

# Standardized output columns (matching NBDT / Ozone schema)
STANDARD_COLUMNS: list[str] = [
    "frameNum",
    "carId",
    "laneId",
    "carCenterX",
    "carCenterY",
    "boundingBox1X", "boundingBox1Y",
    "boundingBox2X", "boundingBox2Y",
    "boundingBox3X", "boundingBox3Y",
    "boundingBox4X", "boundingBox4Y",
    "carCenterXm", "carCenterYm",
    "boundingBox1Xm", "boundingBox1Ym",
    "boundingBox2Xm", "boundingBox2Ym",
    "boundingBox3Xm", "boundingBox3Ym",
    "boundingBox4Xm", "boundingBox4Ym",
    "heading",
    "course",
    "speed",
    "objClass",
    "carCenterLon",
    "carCenterLat",
]


# ---------------------------------------------------------------------------
# Coordinate helpers (from NBDT NGSIMTransfer)
# ---------------------------------------------------------------------------

def _build_local_frame(
    gx_ft: np.ndarray,
    gy_ft: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Build a local Cartesian frame from NGSIM Global_X / Global_Y (feet).

    Origin:  x = min(Global_X_m),  y = max(Global_Y_m).
    Local X =  (Global_X_m - x_origin)
    Local Y = -(Global_Y_m - y_origin)   ← Y-axis flipped to image convention.

    Returns
    -------
    (local_x_m, local_y_m, (x_origin_m, y_origin_m))
    """
    gx_m = gx_ft * FEET_TO_METER
    gy_m = gy_ft * FEET_TO_METER

    x_origin = float(gx_m.min())
    y_origin = float(gy_m.max())

    local_x_m = gx_m - x_origin
    local_y_m = -(gy_m - y_origin)

    return local_x_m, local_y_m, (x_origin, y_origin)


def _compute_heading_and_course(
    vehicle_ids: np.ndarray,
    frame_nums: np.ndarray,
    gx_ft: np.ndarray,
    gy_ft: np.ndarray,
    local_x_m: np.ndarray,
    local_y_m: np.ndarray,
    max_frame_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute heading (image degrees) and course (geographic degrees).

    Heading is derived from the image-coordinate displacement between
    consecutive frames of the same vehicle.  Course is derived from the
    geographic displacement.

    Parameters
    ----------
    max_frame_gap : int
        Maximum allowed Frame_ID gap between consecutive observations of
        the same vehicle.  Larger gaps are treated as trajectory breaks
        and heading/course is left as -1 for those rows.

    Returns
    -------
    (heading, course) — each a float ndarray of length *n*.
        Uncomputable entries are set to -1.
    """
    gx_m = gx_ft * FEET_TO_METER
    gy_m = gy_ft * FEET_TO_METER

    n = len(vehicle_ids)
    heading = np.full(n, -1.0)
    course = np.full(n, -1.0)

    for i in range(n):
        # Prefer next frame; fall back to previous frame
        has_next = (
            i + 1 < n
            and vehicle_ids[i + 1] == vehicle_ids[i]
            and abs(frame_nums[i + 1] - frame_nums[i]) <= max_frame_gap
        )
        has_prev = (
            i - 1 >= 0
            and vehicle_ids[i - 1] == vehicle_ids[i]
            and abs(frame_nums[i] - frame_nums[i - 1]) <= max_frame_gap
        )

        if has_next:
            curr, ref = i, i + 1
        elif has_prev:
            curr, ref = i - 1, i
        else:
            continue  # isolated frame — leave as -1

        # Image-coordinate heading (what the pipeline actually uses)
        dx_img = local_x_m[ref] - local_x_m[curr]
        dy_img = local_y_m[ref] - local_y_m[curr]
        heading[i] = float(np.degrees(np.arctan2(dy_img, dx_img)) % 360)

        # Geographic course (global East-North)
        dE = gx_m[ref] - gx_m[curr]
        dN = gy_m[ref] - gy_m[curr]
        course[i] = float(np.degrees(np.arctan2(dE, dN)) % 360)

    return heading, course


def _oriented_bbox(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    length_m: np.ndarray,
    width_m: np.ndarray,
    heading_deg: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Compute OBB corner coordinates (metres) from centre + heading + extent.

    Corner ordering follows the NBDT / CitySim convention:
        0  front-right   (boundingBox1)
        1  front-left    (boundingBox2)
        2  rear-left     (boundingBox3)
        3  rear-right    (boundingBox4)
    """
    l = length_m / 2.0
    w = width_m / 2.0
    theta = np.radians(heading_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    lc, ls = l * cos_t, l * sin_t
    wc, ws = w * cos_t, w * sin_t

    bb1Xm = cx_m + lc + ws
    bb1Ym = cy_m + ls - wc
    bb2Xm = cx_m + lc - ws
    bb2Ym = cy_m + ls + wc
    bb3Xm = cx_m - lc - ws
    bb3Ym = cy_m - ls + wc
    bb4Xm = cx_m - lc + ws
    bb4Ym = cy_m - ls - wc

    return bb1Xm, bb1Ym, bb2Xm, bb2Ym, bb3Xm, bb3Ym, bb4Xm, bb4Ym


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def _normalise_ngsim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names so casing mismatches are harmless."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def convert_ngsim_to_standard(
    raw_path: Union[str, Path],
    *,
    scene_id: str = "peachtree",
) -> pd.DataFrame:
    """Convert a single NGSIM raw CSV into the Ozone / NBDT standard format.

    Parameters
    ----------
    raw_path : str or Path
        Path to the NGSIM raw CSV (e.g. ``peachtree.csv``).
    scene_id : str
        Identifier written into the ``scene_id`` column of the output.

    Returns
    -------
    pd.DataFrame
        Standardized trajectories at **10 fps** (no frame-rate upsampling).
    """
    # ------------------------------------------------------------------
    # 1. Load & normalise
    # ------------------------------------------------------------------
    raw = pd.read_csv(raw_path)
    raw = _normalise_ngsim_columns(raw)

    # ------------------------------------------------------------------
    # 2. Sort by vehicle then frame (NGSIM raw is unsorted)
    # ------------------------------------------------------------------
    raw = raw.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Extract arrays
    # ------------------------------------------------------------------
    vid = raw["Vehicle_ID"].values
    fnum = raw["Frame_ID"].values
    gx_ft = raw["Global_X"].values.astype(float)
    gy_ft = raw["Global_Y"].values.astype(float)

    # ------------------------------------------------------------------
    # 4. Build local frame (metres)
    # ------------------------------------------------------------------
    local_x_m, local_y_m, _origin = _build_local_frame(gx_ft, gy_ft)

    # ------------------------------------------------------------------
    # 5. Heading & course
    # ------------------------------------------------------------------
    heading, course = _compute_heading_and_course(
        vid, fnum, gx_ft, gy_ft, local_x_m, local_y_m,
    )

    # ------------------------------------------------------------------
    # 6. Vehicle dimensions & speed (feet → metres)
    # ------------------------------------------------------------------
    length_m = raw["v_Length"].values.astype(float) * FEET_TO_METER
    width_m = raw["v_Width"].values.astype(float) * FEET_TO_METER
    speed_ms = raw["v_Vel"].values.astype(float) * FEET_TO_METER

    # ------------------------------------------------------------------
    # 7. OBB corners (metres)
    # ------------------------------------------------------------------
    (bb1Xm, bb1Ym, bb2Xm, bb2Ym,
     bb3Xm, bb3Ym, bb4Xm, bb4Ym) = _oriented_bbox(
        local_x_m, local_y_m, length_m, width_m, heading,
    )

    # ------------------------------------------------------------------
    # 8. Pixel coordinates (1 px = 1 m placeholder — no background image)
    # ------------------------------------------------------------------
    PIX2METER = 1.0
    carCenterX = local_x_m / PIX2METER
    carCenterY = local_y_m / PIX2METER
    bb1X = bb1Xm / PIX2METER
    bb1Y = bb1Ym / PIX2METER
    bb2X = bb2Xm / PIX2METER
    bb2Y = bb2Ym / PIX2METER
    bb3X = bb3Xm / PIX2METER
    bb3Y = bb3Ym / PIX2METER
    bb4X = bb4Xm / PIX2METER
    bb4Y = bb4Ym / PIX2METER

    # ------------------------------------------------------------------
    # 9. Vehicle class mapping
    # ------------------------------------------------------------------
    objClass = raw["v_Class"].map(NGSIM_CLASS_MAP).fillna(-1).astype(int)

    # ------------------------------------------------------------------
    # 10. Assemble output
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        "frameNum":       fnum,
        "carId":          vid,
        "laneId":         raw["Lane_ID"].values,
        # pixel
        "carCenterX":     carCenterX,
        "carCenterY":     carCenterY,
        "boundingBox1X":  bb1X,
        "boundingBox1Y":  bb1Y,
        "boundingBox2X":  bb2X,
        "boundingBox2Y":  bb2Y,
        "boundingBox3X":  bb3X,
        "boundingBox3Y":  bb3Y,
        "boundingBox4X":  bb4X,
        "boundingBox4Y":  bb4Y,
        # metre
        "carCenterXm":    local_x_m,
        "carCenterYm":    local_y_m,
        "boundingBox1Xm": bb1Xm,
        "boundingBox1Ym": bb1Ym,
        "boundingBox2Xm": bb2Xm,
        "boundingBox2Ym": bb2Ym,
        "boundingBox3Xm": bb3Xm,
        "boundingBox3Ym": bb3Ym,
        "boundingBox4Xm": bb4Xm,
        "boundingBox4Ym": bb4Ym,
        # motion
        "heading":        heading,
        "course":         course,
        "speed":          speed_ms,
        "objClass":       objClass,
        # geo (unavailable)
        "carCenterLon":   -1,
        "carCenterLat":   -1,
    })

    # scene_id column (used by pipeline when present)
    result["scene_id"] = scene_id

    return result


def convert_ngsim_and_save(
    raw_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    scene_id: str = "peachtree",
) -> Path:
    """Convert NGSIM raw → standard CSV and write to *out_path*."""
    df = convert_ngsim_to_standard(raw_path, scene_id=scene_id)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
