"""Attribution driver: sample selection → per-event Winter MC → outputs.

Usage (from repo root):
  python -m model_baseline.attribution.driver --config configs/attribution.yaml
  python -m model_baseline.attribution.driver --config configs/attribution.yaml \
      --checkpoint checkpoints/transformer_masked_best_model.pt --max-n 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_baseline.attribution.config import (
    READOUT_SEED_TAG,
    READOUTS,
    AttributionConfig,
    SelectionConfig,
    load_attribution_config,
)
from model_baseline.attribution.outputs import (
    SLOT_NAMES,
    event_result,
    save_event_json,
    summary_rows,
    write_summary_csv,
)
from model_baseline.attribution.values import ConflictValueFn
from model_baseline.attribution.winter_mc import exact_shapley, winter_value_nested_mc
from model_baseline.datasets.binary_baseline import BinaryBaselineDataset
from model_baseline.factories import build_model
from model_baseline.masking import SegmentSpec, build_partition


def load_frozen_model(
    checkpoint: str,
    device: "torch.device",
    allow_unmasked: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load model architecture + weights from a training checkpoint.

    The architecture is rebuilt from the ModelConfig pickled INTO the
    checkpoint (guarantees arch/weights consistency).  Refuses checkpoints
    not trained under the masked surrogate regime: their v(S) is OOD.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    regime = ckpt.get("train_regime", {}) or {}
    if not regime.get("masked", False):
        msg = (
            f"Checkpoint {checkpoint} was NOT trained under the masked "
            f"surrogate regime (train.masking.enabled) — masked inputs are "
            f"out-of-distribution and v(S) is not a well-defined value "
            f"function. Retrain with configs/model_transformer_masked.yaml."
        )
        if not allow_unmasked:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}\nProceeding because "
              f"allow_unmasked_checkpoint=true (case studies only).")

    model_cfg = ckpt["config"].model
    model = build_model(model_cfg.name, cfg=model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"masked={regime.get('masked', False)})")
    return model, ckpt


def select_samples(
    ds: BinaryBaselineDataset, sel: SelectionConfig
) -> list[int]:
    """Filter dataset indices by the selection config (index-CSV metadata)."""
    picked: list[int] = []
    for i in range(len(ds)):
        meta = ds.get_metadata(i)
        if sel.event_ids and meta["event_id"] not in sel.event_ids:
            continue
        if sel.conflict_only and not meta["is_conflict"]:
            continue
        if sel.roles and meta["conflict_target_role"] not in sel.roles:
            continue
        if sel.scenes and not any(s in meta["scene_id"] for s in sel.scenes):
            continue
        picked.append(i)
        if sel.max_n and len(picked) >= sel.max_n:
            break
    return picked


def _event_rng(base_seed: int, event_id: int, readout: str) -> np.random.Generator:
    """Reproducible per-(event, readout) RNG, independent of subset/order."""
    return np.random.default_rng(
        np.random.SeedSequence([base_seed, event_id, READOUT_SEED_TAG[readout]])
    )


def run_attribution(cfg: AttributionConfig) -> list[dict[str, Any]]:
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    model, _ckpt = load_frozen_model(
        cfg.checkpoint, device, allow_unmasked=cfg.allow_unmasked_checkpoint
    )

    ds = BinaryBaselineDataset(cfg.data_prefix)
    if not ds.has_valid_mask:
        raise RuntimeError(
            f"{cfg.data_prefix}_valid.bin not found — the attribution game "
            f"requires per-frame validity. Run build_tensor_cache.py "
            f"--valid-only, then build_binary_cache.py."
        )

    idxs = select_samples(ds, cfg.selection)
    print(f"Selected {len(idxs)}/{len(ds)} events "
          f"(conflict_only={cfg.selection.conflict_only}, "
          f"max_n={cfg.selection.max_n})")

    spec = (
        SegmentSpec(boundaries=cfg.game.boundaries)
        if cfg.game.boundaries
        else SegmentSpec(seg_len_frames=cfg.game.seg_len_frames)
    )
    game_config = {
        "segment": {"seg_len_frames": cfg.game.seg_len_frames,
                    "boundaries": list(cfg.game.boundaries or ())} ,
        "readouts": {
            name: {"hierarchy": READOUTS[name].hierarchy,
                   "segments": READOUTS[name].segments}
            for name in cfg.game.readouts
        },
        "target": cfg.game.target,
        "mc": asdict(cfg.mc),
        "checkpoint": cfg.checkpoint,
    }

    out_dir = Path(cfg.output.dir)
    all_rows: list[dict[str, Any]] = []

    for count, i in enumerate(idxs, start=1):
        sample = ds[i]
        meta = ds.get_metadata(i)
        eid = meta["event_id"]
        valid = sample["valid_mask"].numpy()
        agent = sample["agent_mask"].numpy()

        partition = build_partition(
            valid, agent, spec, num_features=model.num_features
        )
        if partition.n_cells == 0:
            print(f"  [{count}/{len(idxs)}] event {eid}: no valid cells — skipped")
            continue

        target_idx = None
        if cfg.game.target == "target_logit":
            target_idx = int(sample["target_idx"])
            if target_idx < 0:
                print(f"  [{count}/{len(idxs)}] event {eid}: no target slot "
                      f"for target_logit attribution — skipped")
                continue

        value_fn = ConflictValueFn(
            model, sample["x"], agent, valid, device,
            target=cfg.game.target, target_idx=target_idx,
        )

        t0 = time.time()
        cache: dict[bytes, float] = {}  # shared across readouts, per event
        readout_results = {}
        for name in cfg.game.readouts:
            readout_results[name] = winter_value_nested_mc(
                value_fn, partition, valid, READOUTS[name],
                n_samples=cfg.mc.n_samples,
                batch_size=cfg.mc.batch_size,
                rng=_event_rng(cfg.mc.seed, eid, name),
                cache=cache,
            )

        extra: dict[str, Any] = {}
        if cfg.exact.enabled:
            exact_out: dict[str, Any] = {}
            for a in cfg.exact.agents:
                if a not in partition.by_agent_seg:
                    continue
                segs = partition.segments_of(a)
                player_cells = [
                    np.asarray(partition.by_agent_seg[a][s]) for s in segs
                ]
                er = exact_shapley(
                    value_fn, partition, valid, player_cells,
                    max_players=cfg.exact.max_players, cache=cache,
                )
                a_name = SLOT_NAMES[a] if a < len(SLOT_NAMES) else f"agent{a}"
                exact_out[a_name] = {
                    "players": [f"seg{s}" for s in segs],
                    "phi": [float(v) for v in er.phi],
                    "v_full": er.v_full,
                    "v_empty": er.v_empty,
                    "n_forwards": er.n_forwards,
                }
            extra["exact"] = exact_out

        result = event_result(meta, game_config, partition, readout_results, extra)
        result["elapsed_sec"] = round(time.time() - t0, 3)
        if cfg.output.save_per_event:
            save_event_json(out_dir / f"event_{eid}.json", result)
        all_rows.extend(summary_rows(result))

        p4 = max(abs(r.p4_residual) for r in readout_results.values())
        print(f"  [{count}/{len(idxs)}] event {eid}: cells={partition.n_cells}, "
              f"forwards={value_fn.n_forwards}, max|P4|={p4:.2e}, "
              f"{result['elapsed_sec']:.1f}s")

    write_summary_csv(out_dir / "summary.csv", all_rows)
    print(f"Done: {len(all_rows)} readout rows → {out_dir / 'summary.csv'}")
    return all_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/attribution.yaml")
    ap.add_argument("--checkpoint", default=None, help="override cfg.checkpoint")
    ap.add_argument("--data-prefix", default=None, help="override cfg.data_prefix")
    ap.add_argument("--max-n", type=int, default=None, help="override selection.max_n")
    args = ap.parse_args()

    cfg = load_attribution_config(args.config)
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    if args.data_prefix:
        cfg.data_prefix = args.data_prefix
    if args.max_n is not None:
        cfg.selection.max_n = args.max_n
    run_attribution(cfg)


if __name__ == "__main__":
    main()
