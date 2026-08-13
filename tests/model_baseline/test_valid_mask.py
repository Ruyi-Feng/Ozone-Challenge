"""valid_mask retrofit tests: builders + both dataset classes.

Covers:
- build_tensor_cache full build writes {prefix}_valid.npy marking front
  zero-padding, interior missing frames and empty slots as invalid;
- --valid-only rebuild reproduces the same mask against an existing cache
  (row order taken from {prefix}_eid.npy);
- build_binary_cache writes an index-aligned bit-packed {prefix}_valid.bin;
- BaselineConflictDataset / BinaryBaselineDataset return identical
  valid_mask, and fall back to agent-mask broadcast when files are absent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from model_baseline.datasets.baseline import BaselineConflictDataset  # noqa: E402
from model_baseline.datasets.binary_baseline import BinaryBaselineDataset  # noqa: E402


def _load_script(name: str):
    path = project_root / "data_processing" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


btc = _load_script("build_tensor_cache")
bbc = _load_script("build_binary_cache")


# ---------------------------------------------------------------------------
# Synthetic scenario
#
# Event 101 (row 0): ego car 1 frames 0..99 (full 80-frame window 20..99),
#   front car 2 frames 60..99 with frame 80 missing (interior hole).
# Event 202 (row 1): ego car 3 frames 50..99 only → 30 frames front padding.
# ---------------------------------------------------------------------------


def _rows(event_id, car_id, role, frames):
    return [
        {
            "Event_id": event_id,
            "carId": car_id,
            "role": role,
            "frameNum": fn,
            "carCenterXm": 100.0 + car_id + 0.1 * fn,
            "carCenterYm": 200.0 + car_id + 0.05 * fn,
            "heading": 15.0 + car_id,
            "speed": 10.0 + car_id,
        }
        for fn in frames
    ]


def _build_cache(tmp: Path) -> str:
    """Write synthetic parquet + labels, run the full tensor build.

    Returns the cache prefix (str).
    """
    front_frames = [f for f in range(60, 100) if f != 80]
    data = (
        _rows(101, 1, "ego", range(100))
        + _rows(101, 2, "front", front_frames)
        + _rows(202, 3, "ego", range(50, 100))
    )
    parquet_path = tmp / "train_events.parquet"
    pd.DataFrame(data).to_parquet(parquet_path)

    labels_dir = tmp / "labels"
    labels_dir.mkdir(exist_ok=True)
    labels_path = labels_dir / "events_labels_train.csv"
    pd.DataFrame(
        [
            {
                "Event_id": 101, "t0": 99, "is_conflict": 1,
                "conflict_target_id": 2, "scene_id": "sceneA",
                "conflict_target_role": "front",
            },
            {
                "Event_id": 202, "t0": 99, "is_conflict": 0,
                "conflict_target_id": -1, "scene_id": "sceneB",
                "conflict_target_role": "",
            },
        ]
    ).to_csv(labels_path, index=False)

    prefix = str(tmp / "train")
    btc.process_split(
        str(parquet_path), str(labels_path),
        str(tmp / "train_events.valid_events"),  # does not exist → keep all
        prefix, "train",
    )
    return prefix


def _expected_valid() -> np.ndarray:
    v = np.zeros((2, 7, 80), dtype=bool)
    v[0, 0, :] = True                # ego 101: full window
    v[0, 1, 40:] = True              # front 101: frames 60..99 → out_t 40..79
    v[0, 1, 60] = False              # ... minus the frame-80 hole
    v[1, 0, 30:] = True              # ego 202: 50 frames right-aligned
    return v


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------


def test_full_build_valid_mask(tmp_path):
    prefix = _build_cache(tmp_path)

    eid = np.load(f"{prefix}_eid.npy")
    assert eid.tolist() == [101, 202]

    valid = np.load(f"{prefix}_valid.npy")
    assert valid.shape == (2, 7, 80)
    np.testing.assert_array_equal(valid, _expected_valid())

    # agent_mask consistency: slots without any valid frame are absent slots
    mask = np.load(f"{prefix}_mask.npy")
    assert mask[0].tolist() == [True, True] + [False] * 5
    assert mask[1].tolist() == [True] + [False] * 6


def test_valid_only_rebuild_matches_full(tmp_path):
    prefix = _build_cache(tmp_path)
    reference = np.load(f"{prefix}_valid.npy").copy()

    Path(f"{prefix}_valid.npy").unlink()
    btc.process_split(
        str(tmp_path / "train_events.parquet"),
        str(tmp_path / "labels" / "events_labels_train.csv"),
        str(tmp_path / "train_events.valid_events"),
        prefix, "train", valid_only=True,
    )

    rebuilt = np.load(f"{prefix}_valid.npy")
    np.testing.assert_array_equal(rebuilt, reference)
    # x cache untouched by the valid-only pass
    assert Path(f"{prefix}_x.npy").exists()


def test_valid_only_requires_existing_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        btc.process_split(
            str(tmp_path / "missing.parquet"),
            str(tmp_path / "missing_labels.csv"),
            str(tmp_path / "missing.valid_events"),
            str(tmp_path / "nocache"), "train", valid_only=True,
        )


# ---------------------------------------------------------------------------
# Binary cache + dataset round-trip
# ---------------------------------------------------------------------------


def test_binary_cache_and_dataset_roundtrip(tmp_path):
    prefix = _build_cache(tmp_path)
    bbc.process_split(prefix, "train")

    vbin = Path(f"{prefix}_valid.bin")
    assert vbin.exists()
    assert vbin.stat().st_size == 2 * bbc.VALID_RECORD_BYTES

    ds_npy = BaselineConflictDataset(prefix)
    ds_bin = BinaryBaselineDataset(prefix)
    assert ds_npy.has_valid_mask and ds_bin.has_valid_mask

    expected = torch.from_numpy(_expected_valid())
    for i in range(2):
        a, b = ds_npy[i], ds_bin[i]
        assert torch.equal(a["valid_mask"], expected[i])
        assert torch.equal(a["valid_mask"], b["valid_mask"])
        assert torch.equal(a["x"], b["x"])
        assert torch.equal(a["agent_mask"], b["agent_mask"])
        assert a["event_id"] == b["event_id"]

    meta = ds_bin.get_metadata(0)
    assert meta["event_id"] == 101
    assert meta["scene_id"] == "sceneA"
    assert meta["conflict_target_role"] == "front"
    assert meta["is_conflict"] == 1


def test_fallback_without_valid_files(tmp_path):
    prefix = _build_cache(tmp_path)
    bbc.process_split(prefix, "train")

    Path(f"{prefix}_valid.npy").unlink()
    Path(f"{prefix}_valid.bin").unlink()

    ds_npy = BaselineConflictDataset(prefix)
    ds_bin = BinaryBaselineDataset(prefix)
    assert not ds_npy.has_valid_mask and not ds_bin.has_valid_mask

    for ds in (ds_npy, ds_bin):
        sample = ds[0]
        expected = sample["agent_mask"][:, None].expand(-1, 80)
        assert torch.equal(sample["valid_mask"], expected)
