"""
Main processing pipeline: wire stages without embedding algorithm details.

Flow
----
raw CSV
  -> detect conflict / non-conflict candidates
  -> validate 8s history + 3s future windows
  -> filter & persistently collect neighbors
  -> export data CSV + label CSV (+ future traj CSV)
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd

from data_processing.core.conflict_detect import detect_all_candidates
from data_processing.core.export import (
    export_chunk,
    export_events,
    suffix_path,
    write_csv_headers,
    write_csv_headers_split,
)
from data_processing.core.neighbor_filter import (
    _precompute_frame_neighbors,
    extract_neighbors_for_events,
)
from data_processing.core.trajectory_window import build_windowed_events
from data_processing.io.readers import list_raw_csv_files, load_config, load_raw_csv
from data_processing.io.schema import (
    ConflictCandidate,
    ProcessingConfig,
    TrackedNeighborhoodEvent,
    WindowedEventCandidate,
)
from data_processing.io.writers import write_interim_candidates


def run_conflict_stage(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
) -> List[ConflictCandidate]:
    """Stage 1 wrapper."""
    return detect_all_candidates(raw_df, cfg)


def run_window_stage(
    raw_df: pd.DataFrame,
    candidates: List[ConflictCandidate],
    cfg: ProcessingConfig,
    *,
    car_index: dict | None = None,
) -> List[WindowedEventCandidate]:
    """Stage 2 wrapper."""
    return build_windowed_events(raw_df, candidates, cfg, car_index=car_index)


def run_neighbor_stage(
    raw_df: pd.DataFrame,
    windows: List[WindowedEventCandidate],
    cfg: ProcessingConfig,
    *,
    frame_index: dict | None = None,
    car_index: dict | None = None,
    neighbor_cache: dict | None = None,
) -> List[TrackedNeighborhoodEvent]:
    """Stage 3 wrapper."""
    return extract_neighbors_for_events(
        raw_df, windows, cfg,
        frame_index=frame_index, car_index=car_index,
        neighbor_cache=neighbor_cache,
    )


def run_export_stage(
    events: List[TrackedNeighborhoodEvent],
    cfg: ProcessingConfig,
) -> Tuple[str, str, str]:
    """Stage 4 wrapper."""
    return export_events(events, cfg)


def _determine_split_egos(
    windows: List[WindowedEventCandidate],
    ratio: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[Set, Set, Set]:
    """Assign ego_ids to train/val/test, stratified per scene.

    Within each scene (location) the ego_ids are shuffled and split by
    *ratio* so every location contributes proportionally to each split.
    This avoids a location-biased split where one scene's traffic pattern
    (different density / speed) dominates a single split.

    Returns sets of ``(scene_id, ego_id)`` tuples — carId is only unique
    within a scene, so the pair is the correct ego key.
    """
    train_ratio, val_ratio, _ = ratio
    scene_egos: dict = {}
    for w in windows:
        scene_egos.setdefault(w.scene_id, set()).add(w.ego_id)

    rng = random.Random(seed)
    train_keys: Set = set()
    val_keys: Set = set()
    test_keys: Set = set()
    for scene_id, ego_ids in scene_egos.items():
        ego_list = sorted(ego_ids)
        rng.shuffle(ego_list)
        n = len(ego_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n > 0 and n_train == 0:
            n_train = 1
        n_val = min(n_val, n - n_train)
        train_keys.update((scene_id, e) for e in ego_list[:n_train])
        val_keys.update((scene_id, e) for e in ego_list[n_train:n_train + n_val])
        test_keys.update((scene_id, e) for e in ego_list[n_train + n_val:])
    return train_keys, val_keys, test_keys


def process_raw_dataframe(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
    *,
    dump_interim: bool = False,
    interim_name: str = "candidates.csv",
) -> Tuple[str, str, str]:
    """
    Run full pipeline on an in-memory raw dataframe.

    Returns (data_path, label_path, future_traj_path).
    """
    print(f"Pipeline start: {len(raw_df)} rows, "
          f"history={cfg.history_sec}s, future={cfg.future_sec}s")

    candidates = run_conflict_stage(raw_df, cfg)
    n_conf = sum(1 for c in candidates if c.is_conflict)
    n_non = sum(1 for c in candidates if not c.is_conflict)
    print(f"  → {len(candidates)} candidates ({n_conf} conflict, {n_non} non-conflict)")

    # Pre-build lookup indices so Stages 2–3 don't re-scan the full table.
    # frame_index: frameNum → DataFrame slice   (used by Stage 3)
    # car_index:   carId    → DataFrame slice   (used by Stage 2 & 3)
    print("  building lookup indices …")
    frame_index = dict(tuple(raw_df.groupby("frameNum")))
    car_index = dict(tuple(raw_df.groupby("carId")))
    print(f"  → {len(frame_index)} frame groups, {len(car_index)} vehicle groups")

    if dump_interim:
        interim_path = Path(cfg.interim_dir) / interim_name
        rows = [
            {
                "scene_id": c.scene_id,
                "ego_id": c.ego_id,
                "is_conflict": c.is_conflict,
                "t_conflict": c.t_conflict,
                "conflict_target_id": c.conflict_target_id,
            }
            for c in candidates
        ]
        write_interim_candidates(pd.DataFrame(rows), interim_path)
        print(f"  → interim candidates saved to {interim_path}")

    print("  Stage 2/4: window validation …")
    windows = run_window_stage(raw_df, candidates, cfg, car_index=car_index)
    print(f"  → {len(windows)} windowed events")

    # Pre-compute neighbor cache once for all chunks
    print("  precomputing frame-neighbor cache …")
    neighbor_cache = _precompute_frame_neighbors(frame_index, cfg)

    CHUNK_SIZE = 5000
    test_only = cfg.test_only
    ratio = cfg.train_val_test_ratio
    split_mode = ratio is not None and not test_only
    next_id = cfg.event_id_offset

    if test_only:
        write_csv_headers_split(cfg, ("test",))
        n_chunks = (len(windows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Stage 3+4: {len(windows)} events in "
              f"{n_chunks} chunks (test-only) …")
    elif split_mode:
        train_egos, val_egos, test_egos = _determine_split_egos(windows, ratio)
        write_csv_headers_split(cfg, ("train", "val", "test"))
        n_chunks = (len(windows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Stage 3+4: {len(windows)} events in "
              f"{n_chunks} chunks (split mode) …")
    else:
        write_csv_headers(cfg)
        n_chunks = (len(windows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Stage 3+4: {len(windows)} events in "
              f"{n_chunks} chunks …")

    total_tracked = 0
    for i in range(0, len(windows), CHUNK_SIZE):
        chunk = windows[i:i + CHUNK_SIZE]
        tracked = run_neighbor_stage(
            raw_df, chunk, cfg,
            frame_index=frame_index, car_index=car_index,
            neighbor_cache=neighbor_cache,
        )
        total_tracked += len(tracked)

        if test_only:
            next_id = export_chunk(
                tracked, cfg,
                suffix_path(cfg.data_out, "test"),
                suffix_path(cfg.label_out, "test"),
                suffix_path(cfg.future_traj_out, "test"),
                start_id=next_id,
            )
        elif split_mode:
            train_evts = [e for e in tracked
                          if (e.window.scene_id, e.window.ego_id) in train_egos]
            val_evts = [e for e in tracked
                        if (e.window.scene_id, e.window.ego_id) in val_egos]
            test_evts = [e for e in tracked
                         if (e.window.scene_id, e.window.ego_id) in test_egos]
            next_id = export_chunk(
                train_evts, cfg,
                suffix_path(cfg.data_out, "train"),
                suffix_path(cfg.label_out, "train"),
                suffix_path(cfg.future_traj_out, "train"),
                start_id=next_id,
            )
            next_id = export_chunk(
                val_evts, cfg,
                suffix_path(cfg.data_out, "val"),
                suffix_path(cfg.label_out, "val"),
                suffix_path(cfg.future_traj_out, "val"),
                start_id=next_id,
            )
            next_id = export_chunk(
                test_evts, cfg,
                suffix_path(cfg.data_out, "test"),
                suffix_path(cfg.label_out, "test"),
                suffix_path(cfg.future_traj_out, "test"),
                start_id=next_id,
            )
        else:
            next_id = export_chunk(
                tracked, cfg,
                cfg.data_out, cfg.label_out, cfg.future_traj_out,
                start_id=next_id,
            )

        del tracked, chunk
        gc.collect()

    print(f"  → {total_tracked} tracked events, done")

    if test_only:
        return (
            suffix_path(cfg.data_out, "test"),
            suffix_path(cfg.label_out, "test"),
            suffix_path(cfg.future_traj_out, "test"),
        )
    if split_mode:
        return (
            suffix_path(cfg.data_out, "train"),
            suffix_path(cfg.label_out, "train"),
            suffix_path(cfg.future_traj_out, "train"),
        )
    return cfg.data_out, cfg.label_out, cfg.future_traj_out


def process_raw_file(
    raw_path: Union[str, Path],
    cfg: ProcessingConfig,
    *,
    dump_interim: bool = False,
) -> Tuple[str, str, str]:
    """Load one raw CSV and run the full pipeline."""
    raw_df = load_raw_csv(raw_path)
    interim_name = f"{Path(raw_path).stem}_candidates.csv"
    return process_raw_dataframe(
        raw_df,
        cfg,
        dump_interim=dump_interim,
        interim_name=interim_name,
    )


def run_pipeline(
    cfg: ProcessingConfig,
    *,
    raw_path: Optional[Union[str, Path]] = None,
    dump_interim: bool = False,
) -> Tuple[str, str, str]:
    """
    Entry used by scripts.

    If raw_path is given, process that file; otherwise process all CSVs in cfg.raw_dir
    (current stub: first file only — multi-file merge TBD when implementing).
    """
    if raw_path is not None:
        return process_raw_file(raw_path, cfg, dump_interim=dump_interim)

    files = list_raw_csv_files(cfg.raw_dir)
    if not files:
        raise FileNotFoundError(f"No CSV found under {cfg.raw_dir}")

    if len(files) == 1:
        return process_raw_file(files[0], cfg, dump_interim=dump_interim)

    # ------------------------------------------------------------------
    # Multi-file: load, tag with scene_id, concat, then process together.
    # This is needed for CitySim (e.g. IntersectionA-01.csv … A-12.csv)
    # and inD datasets where one recording is split across many CSVs.
    # ------------------------------------------------------------------
    import pandas as pd

    from data_processing.io.readers import load_raw_csv

    parts: list[pd.DataFrame] = []
    for fp in files:
        scene_name = Path(fp).stem  # e.g. "IntersectionA-01"
        df = load_raw_csv(fp)
        df["scene_id"] = scene_name
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True)
    print(f"Multi-file: merged {len(files)} files → {len(merged)} rows, "
          f"{merged['carId'].nunique()} unique vehicles")

    return process_raw_dataframe(
        merged, cfg, dump_interim=dump_interim,
        interim_name="merged_candidates.csv",
    )


def load_pipeline_config(config_path: Union[str, Path]) -> ProcessingConfig:
    """Thin alias for script convenience."""
    return load_config(config_path)
