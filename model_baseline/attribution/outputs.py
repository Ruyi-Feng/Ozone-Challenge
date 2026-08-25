"""Aggregation + serialization of Winter-Shapley results.

Aggregations are plain sums of cell values (Winter P1: lower-level values sum
to the enclosing group's value).  P1 residuals are identically zero by
construction here; the p4_residual of each WinterResult is reported as a
wiring check instead.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from model_baseline.attribution.winter_mc import WinterResult
from model_baseline.masking import CellPartition

SLOT_NAMES = (
    "ego", "front", "rear",
    "left_front", "left_rear", "right_front", "right_rear",
)
CHANNEL_NAMES = ("dx", "dy", "heading", "speed")


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _agent_name(a: int, labels: list[dict[str, Any]] | None) -> str:
    if labels and 0 <= a < len(labels):
        item = labels[a]
        role = item.get("role", "")
        car_id = item.get("car_id", item.get("carId", -1))
        try:
            car_id_i = int(car_id)
        except (TypeError, ValueError):
            car_id_i = -1
        if car_id_i >= 0 and role:
            local = item.get("local_index", a)
            return f"{local}:{role}:{car_id_i}"
        if role:
            return str(role)
    return SLOT_NAMES[a] if a < len(SLOT_NAMES) else f"agent{a}"


def aggregate(
    p: CellPartition,
    cell_psi: np.ndarray,
    agent_labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sum cell values up the levels: (agent, segment), agent, segment, channel."""
    agent_seg: dict[str, float] = {}
    per_agent: dict[str, float] = {}
    per_segment: dict[str, float] = {}
    per_channel: dict[str, float] = {}
    per_agent_rows: list[dict[str, Any]] = []
    agent_psi: dict[int, float] = {}

    for cell_id, (a, s, c) in enumerate(p.cells):
        psi = float(cell_psi[cell_id])
        a_name = _agent_name(a, agent_labels)
        c_name = CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"
        agent_seg[f"{a_name}/seg{s}"] = agent_seg.get(f"{a_name}/seg{s}", 0.0) + psi
        per_agent[a_name] = per_agent.get(a_name, 0.0) + psi
        agent_psi[a] = agent_psi.get(a, 0.0) + psi
        per_segment[f"seg{s}"] = per_segment.get(f"seg{s}", 0.0) + psi
        per_channel[c_name] = per_channel.get(c_name, 0.0) + psi

    for a, psi in sorted(agent_psi.items()):
        label = (agent_labels[a] if agent_labels and a < len(agent_labels) else {})
        per_agent_rows.append({
            "local_agent_index": a,
            "carId": label.get("car_id", label.get("carId", -1)),
            "role": label.get("role", _agent_name(a, None)),
            "agent_shapley": float(psi),
        })

    return {
        "agent_segment": agent_seg,
        "agent": per_agent,
        "segment": per_segment,
        "channel": per_channel,
        "agent_rows": per_agent_rows,
    }


def cell_table(
    p: CellPartition,
    res: WinterResult,
    agent_labels: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for cell_id, (a, s, c) in enumerate(p.cells):
        start, end = p.seg_bounds[s]
        label = (agent_labels[a] if agent_labels and a < len(agent_labels) else {})
        rows.append({
            "agent": _agent_name(a, agent_labels),
            "agent_idx": a,
            "local_agent_index": a,
            "carId": label.get("car_id", label.get("carId", -1)),
            "role": label.get("role", SLOT_NAMES[a] if a < len(SLOT_NAMES) else f"agent{a}"),
            "seg": s,
            "frame_start": int(start),
            "frame_end": int(end),
            "n_valid_frames": int(len(p.cell_frames[cell_id])),
            "channel": CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}",
            "psi": float(res.cell_psi[cell_id]),
            "se": float(res.cell_se[cell_id]),
        })
    return rows


def readout_result(
    p: CellPartition,
    res: WinterResult,
    agent_labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """JSON-serializable result of one readout on one event."""
    aggs = aggregate(p, res.cell_psi, agent_labels=agent_labels)
    return {
        "v_full": res.v_full,
        "v_empty": res.v_empty,
        "p_full": _sigmoid(res.v_full),
        "p_empty": _sigmoid(res.v_empty),
        "delta_v": res.v_full - res.v_empty,
        "sum_psi": float(res.cell_psi.sum()),
        "p4_residual": res.p4_residual,
        "n_samples": res.n_samples,
        "n_forwards": res.n_forwards,
        "n_cache_hits": res.n_cache_hits,
        "aggregations": aggs,
        "agent_shapley": aggs.get("agent_rows", []),
        "cells": cell_table(p, res, agent_labels=agent_labels),
    }


def event_result(
    meta: dict[str, Any],
    game_config: dict[str, Any],
    p: CellPartition,
    readouts: dict[str, WinterResult],
    extra: dict[str, Any] | None = None,
    agent_labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event": meta,
        "game": game_config,
        "seg_bounds": [[int(s), int(e)] for s, e in p.seg_bounds],
        "n_cells": p.n_cells,
        "n_players": len(p.agents),
        "agents": agent_labels or [],
        "readouts": {
            name: readout_result(p, res, agent_labels=agent_labels)
            for name, res in readouts.items()
        },
    }
    if extra:
        out.update(extra)
    return out


def save_event_json(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One summary.csv row per (event, readout)."""
    rows = []
    meta = result["event"]
    for name, r in result["readouts"].items():
        agg = r["aggregations"]
        top_agent = max(agg["agent"], key=lambda k: abs(agg["agent"][k])) \
            if agg["agent"] else ""
        top_seg = max(agg["segment"], key=lambda k: abs(agg["segment"][k])) \
            if agg["segment"] else ""
        rows.append({
            "event_id": meta.get("event_id"),
            "scene_id": meta.get("scene_id", ""),
            "is_conflict": meta.get("is_conflict", ""),
            "conflict_target_role": meta.get("conflict_target_role", ""),
            "readout": name,
            "v_full": r["v_full"],
            "v_empty": r["v_empty"],
            "p_full": r["p_full"],
            "p_empty": r["p_empty"],
            "sum_psi": r["sum_psi"],
            "p4_residual": r["p4_residual"],
            "top_agent": top_agent,
            "top_agent_psi": agg["agent"].get(top_agent, 0.0),
            "top_segment": top_seg,
            "top_segment_psi": agg["segment"].get(top_seg, 0.0),
            "n_forwards": r["n_forwards"],
        })
    return rows


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
