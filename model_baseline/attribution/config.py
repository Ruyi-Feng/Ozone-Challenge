"""Attribution run configuration (YAML → dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from model_baseline.masking import LevelOrderConfig

# Named readouts — the two complementary games (+ optional third):
#   symmetric : agent-major uniform orders → "which evidence does the model
#               rely on"; agent-level Shapley falls out via P1 aggregation.
#   chrono    : time-major, segments ascending → chain-respecting asymmetric
#               value; coalitions are scene prefixes ("risk revelation").
#   reverse   : time-major, segments descending → proximate sufficiency.
READOUTS: dict[str, LevelOrderConfig] = {
    "symmetric": LevelOrderConfig(hierarchy="agent_major", segments="uniform"),
    "chrono": LevelOrderConfig(hierarchy="time_major", segments="chrono"),
    "reverse": LevelOrderConfig(hierarchy="time_major", segments="reverse"),
}
# fixed per-readout seed tags → per-event RNG independent of readout list order
READOUT_SEED_TAG: dict[str, int] = {"symmetric": 0, "chrono": 1, "reverse": 2}


@dataclass
class SelectionConfig:
    conflict_only: bool = True
    roles: tuple[str, ...] = ()          # e.g. ("front", "left_front")
    scenes: tuple[str, ...] = ()         # substring match on scene_id
    event_ids: tuple[int, ...] = ()      # explicit allowlist (overrides nothing else)
    max_n: int = 100                     # 0 = no cap


@dataclass
class GameConfig:
    seg_len_frames: int = 10             # 1 s at 10 fps → 8 segments
    boundaries: Optional[tuple[int, ...]] = None  # semantic cut, overrides seg_len
    readouts: tuple[str, ...] = ("symmetric", "chrono")
    target: str = "conflict_logit"       # or "target_logit"


@dataclass
class MCConfig:
    n_samples: int = 25
    batch_size: int = 32
    seed: int = 20260810


@dataclass
class ExactConfig:
    enabled: bool = False
    agents: tuple[int, ...] = (0,)       # slot indices for per-agent segment games
    max_players: int = 16


@dataclass
class OutputConfig:
    dir: str = "outputs/attribution/run_001"
    save_per_event: bool = True


@dataclass
class AttributionConfig:
    checkpoint: str = "checkpoints/best_model.pt"
    data_prefix: str = "data/processed/val"
    device: str = "cuda"
    # v(S) is only well-defined on a checkpoint trained under the masked
    # regime; overriding this yields OOD masked inputs — case studies only.
    allow_unmasked_checkpoint: bool = False
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    game: GameConfig = field(default_factory=GameConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    exact: ExactConfig = field(default_factory=ExactConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        for r in self.game.readouts:
            if r not in READOUTS:
                raise ValueError(
                    f"unknown readout {r!r}; available: {sorted(READOUTS)}"
                )


def load_attribution_config(path: str | Path) -> AttributionConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    sel = raw.get("selection", {}) or {}
    game = raw.get("game", {}) or {}
    mc = raw.get("mc", {}) or {}
    exact = raw.get("exact", {}) or {}
    out = raw.get("output", {}) or {}

    boundaries = game.get("boundaries")
    return AttributionConfig(
        checkpoint=str(raw.get("checkpoint", "checkpoints/best_model.pt")),
        data_prefix=str(raw.get("data_prefix", "data/processed/val")),
        device=str(raw.get("device", "cuda")),
        allow_unmasked_checkpoint=bool(raw.get("allow_unmasked_checkpoint", False)),
        selection=SelectionConfig(
            conflict_only=bool(sel.get("conflict_only", True)),
            roles=tuple(sel.get("roles", ()) or ()),
            scenes=tuple(sel.get("scenes", ()) or ()),
            event_ids=tuple(int(e) for e in (sel.get("event_ids", ()) or ())),
            max_n=int(sel.get("max_n", 100)),
        ),
        game=GameConfig(
            seg_len_frames=int(game.get("seg_len_frames", 10)),
            boundaries=tuple(int(b) for b in boundaries) if boundaries else None,
            readouts=tuple(game.get("readouts", ("symmetric", "chrono"))),
            target=str(game.get("target", "conflict_logit")),
        ),
        mc=MCConfig(
            n_samples=int(mc.get("n_samples", 25)),
            batch_size=int(mc.get("batch_size", 32)),
            seed=int(mc.get("seed", 20260810)),
        ),
        exact=ExactConfig(
            enabled=bool(exact.get("enabled", False)),
            agents=tuple(int(a) for a in (exact.get("agents", (0,)) or ())),
            max_players=int(exact.get("max_players", 16)),
        ),
        output=OutputConfig(
            dir=str(out.get("dir", "outputs/attribution/run_001")),
            save_per_event=bool(out.get("save_per_event", True)),
        ),
    )
