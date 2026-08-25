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
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

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
from data_processing.utils.progress import tqdm_bar


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


def _determine_train_egos(
    windows: List[WindowedEventCandidate],
    train_ratio: float,
    seed: int = 42,
) -> Set:
    """Pre-compute which scene-local ego identities belong to training.

    Mirrors ``split_by_ego`` logic but operates on lightweight window
    objects so the split can be determined before Stage 3 materializes
    any DataFrame-heavy ``TrackedNeighborhoodEvent``.

    Vehicle ids are only guaranteed to be unique inside one recording.
    Therefore the split key must include ``scene_id`` when multiple files
    are processed together.
    """
    ego_ids = sorted({(w.scene_id, w.ego_id) for w in windows})
    rng = random.Random(seed)
    rng.shuffle(ego_ids)
    n_train = max(1, int(len(ego_ids) * train_ratio))
    return set(ego_ids[:n_train])


def _thin_windows_by_ego(
    windows: List[WindowedEventCandidate],
    *,
    min_t0_gap_sec: float,
    max_per_ego_per_class: int,
    fps: float,
    seed: int,
) -> List[WindowedEventCandidate]:
    """Drop overlapping same-ego windows; do not unique-ify neighbor cars.

    A vehicle appearing as someone else's neighbor is a different training
    sample (different ego, t0, and relative features). The waste is the
    same ego emitting many near-identical 8s windows.
    """
    if min_t0_gap_sec <= 0 and max_per_ego_per_class <= 0:
        return windows

    min_gap_frames = max(0.0, float(min_t0_gap_sec) * float(fps))
    groups: dict[tuple, list[WindowedEventCandidate]] = {}
    for window in windows:
        groups.setdefault(
            (window.scene_id, window.ego_id, window.is_conflict), []
        ).append(window)

    selected: list[WindowedEventCandidate] = []
    for key, group in sorted(groups.items()):
        rng = random.Random(f"{seed}:thin:{key[0]}:{key[1]}:{int(key[2])}")
        order = list(group)
        rng.shuffle(order)
        kept: list[WindowedEventCandidate] = []
        for window in order:
            if min_gap_frames > 0 and any(
                abs(window.t0 - other.t0) < min_gap_frames for other in kept
            ):
                continue
            kept.append(window)
            if max_per_ego_per_class > 0 and len(kept) >= max_per_ego_per_class:
                break
        selected.extend(kept)

    selected.sort(key=lambda w: (w.scene_id, w.t0, w.ego_id))
    print(
        f"  ego-window sample: {len(windows)} → {len(selected)} "
        f"(min_t0_gap={min_t0_gap_sec}s, "
        f"max_per_ego_class={max_per_ego_per_class or 'off'})"
    )
    return selected


def _cap_windows_for_pilot(
    windows: List[WindowedEventCandidate],
    max_per_class_per_scene: int,
    seed: int,
) -> List[WindowedEventCandidate]:
    """Deterministically cap validated events for a lightweight pilot run."""
    if max_per_class_per_scene <= 0:
        return windows

    groups: dict[tuple[str, bool], list[WindowedEventCandidate]] = {}
    for window in windows:
        groups.setdefault((window.scene_id, window.is_conflict), []).append(window)

    selected: list[WindowedEventCandidate] = []
    for (scene_id, is_conflict), group in sorted(groups.items()):
        if len(group) <= max_per_class_per_scene:
            selected.extend(group)
            continue
        rng = random.Random(f"{seed}:{scene_id}:{int(is_conflict)}")
        selected.extend(rng.sample(group, max_per_class_per_scene))

    selected.sort(key=lambda w: (w.scene_id, w.t0, w.ego_id))
    print(
        f"  pilot cap: {len(windows)} → {len(selected)} windows "
        f"(max {max_per_class_per_scene}/class/scene)"
    )
    return selected


def process_raw_dataframe(
    raw_df: pd.DataFrame,
    cfg: ProcessingConfig,
    *,
    dump_interim: bool = False,
    interim_name: str = "candidates.csv",
    split_assigner: Callable[[List[WindowedEventCandidate]], None] | None = None,
    split_counters: Dict[str, int] | None = None,
    init_outputs: bool = True,
) -> Tuple[str, str, str] | Dict[str, int]:
    """
    Run full pipeline on an in-memory raw dataframe.

    Returns (data_path, label_path, future_traj_path) for the legacy train/val
    path, or the mutated *split_counters* when *split_assigner* is provided.
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
    windows = _thin_windows_by_ego(
        windows,
        min_t0_gap_sec=cfg.min_t0_gap_sec,
        max_per_ego_per_class=cfg.max_events_per_ego_per_class,
        fps=cfg.fps,
        seed=cfg.rebalance_seed,
    )
    windows = _cap_windows_for_pilot(
        windows,
        cfg.max_events_per_class_per_scene,
        cfg.rebalance_seed,
    )

    # Pre-compute neighbor cache once for all chunks
    print("  precomputing frame-neighbor cache …")
    neighbor_cache = _precompute_frame_neighbors(frame_index, cfg)

    CHUNK_SIZE = 5000

    if split_assigner is not None:
        split_assigner(windows)
        if split_counters is None:
            split_counters = {}
        n_chunks = (len(windows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Stage 3+4: {len(windows)} events in "
              f"{n_chunks} chunks (named-split mode) …")
        total_tracked = 0
        from tqdm import tqdm
        chunk_starts = range(0, len(windows), CHUNK_SIZE)
        if n_chunks > 1:
            chunk_starts = tqdm(
                chunk_starts,
                total=n_chunks,
                desc="  export chunks",
                unit="chk",
                **tqdm_bar(leave=False, position=1),
            )
        for i in chunk_starts:
            chunk = windows[i:i + CHUNK_SIZE]
            tracked = run_neighbor_stage(
                raw_df, chunk, cfg,
                frame_index=frame_index, car_index=car_index,
                neighbor_cache=neighbor_cache,
            )
            total_tracked += len(tracked)
            grouped: dict[str, list] = defaultdict(list)
            for event in tracked:
                split = str(event.window.meta.get("split", "train"))
                grouped[split].append(event)
            for split, events in grouped.items():
                split_counters[split] = export_chunk(
                    events, cfg,
                    suffix_path(cfg.data_out, split),
                    suffix_path(cfg.label_out, split),
                    suffix_path(cfg.future_traj_out, split),
                    start_id=split_counters.get(split, 0),
                )
            del tracked, chunk
            gc.collect()
        print(f"  → {total_tracked} tracked events, done")
        return split_counters

    ratio = cfg.train_val_split_ratio
    split_mode = ratio is not None and 0.0 < ratio < 1.0

    if split_mode:
        train_egos = _determine_train_egos(windows, ratio)
        if init_outputs:
            write_csv_headers_split(cfg)
        train_id = 0
        val_id = 0
        n_chunks = (len(windows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Stage 3+4: {len(windows)} events in "
              f"{n_chunks} chunks (split mode) …")
    else:
        if init_outputs:
            write_csv_headers(cfg)
        next_id = 0
        n_chunks = (len(windows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Stage 3+4: {len(windows)} events in "
              f"{n_chunks} chunks …")

    total_tracked = 0
    from tqdm import tqdm
    chunk_starts = range(0, len(windows), CHUNK_SIZE)
    if n_chunks > 1:
        chunk_starts = tqdm(
            chunk_starts,
            total=n_chunks,
            desc="  export chunks",
            unit="chk",
            **tqdm_bar(leave=False, position=1),
        )
    for i in chunk_starts:
        chunk = windows[i:i + CHUNK_SIZE]
        tracked = run_neighbor_stage(
            raw_df, chunk, cfg,
            frame_index=frame_index, car_index=car_index,
            neighbor_cache=neighbor_cache,
        )
        total_tracked += len(tracked)

        if split_mode:
            train_evts = [
                e for e in tracked
                if (e.window.scene_id, e.window.ego_id) in train_egos
            ]
            val_evts = [
                e for e in tracked
                if (e.window.scene_id, e.window.ego_id) not in train_egos
            ]
            train_id = export_chunk(
                train_evts, cfg,
                suffix_path(cfg.data_out, "train"),
                suffix_path(cfg.label_out, "train"),
                suffix_path(cfg.future_traj_out, "train"),
                start_id=train_id,
            )
            val_id = export_chunk(
                val_evts, cfg,
                suffix_path(cfg.data_out, "val"),
                suffix_path(cfg.label_out, "val"),
                suffix_path(cfg.future_traj_out, "val"),
                start_id=val_id,
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

    if split_mode:
        return (
            suffix_path(cfg.data_out, "train"),
            suffix_path(cfg.label_out, "train"),
            suffix_path(cfg.future_traj_out, "train"),
        )
    else:
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
