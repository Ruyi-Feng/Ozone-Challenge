"""Attribution engine + driver tests.

Engine correctness on analytic games:
- linear value function → MC psi equals the weights EXACTLY (any readout,
  even n_samples=1: every chain marginal is the constant weight);
- redundant-pair game → chrono readout credits the EARLIER cell (innovation),
  symmetric readout splits 50/50 — the semantic core of the dual readout;
- exact Shapley: AND game → 0.5/0.5.

Integration:
- real tiny transformer: P4 telescoping, cross-readout cache reuse,
  determinism;
- driver end-to-end on a synthetic binary cache + masked-regime checkpoint;
- driver refuses checkpoints not trained under the masked regime;
- visualize renders PNGs from the produced JSON.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from model_baseline.attribution.config import (  # noqa: E402
    READOUTS,
    AttributionConfig,
    ExactConfig,
    GameConfig,
    MCConfig,
    OutputConfig,
    SelectionConfig,
)
from model_baseline.attribution.driver import (  # noqa: E402
    load_frozen_model,
    run_attribution,
)
from model_baseline.attribution.values import ConflictValueFn  # noqa: E402
from model_baseline.attribution.winter_mc import (  # noqa: E402
    exact_shapley,
    winter_value_nested_mc,
)
from model_baseline.config import (  # noqa: E402
    BaselineRuntimeConfig,
    ModelConfig,
)
from model_baseline.masking import (  # noqa: E402
    SegmentSpec,
    build_partition,
)
from model_baseline.models.transformer import TransformerConflictModel  # noqa: E402


def _load_script(name: str):
    path = project_root / "data_processing" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Analytic value functions over a partition
# ---------------------------------------------------------------------------


def _visible_cells(p, cm: np.ndarray) -> np.ndarray:
    vis = np.zeros(p.n_cells, dtype=bool)
    for cid, (a, _s, c) in enumerate(p.cells):
        vis[cid] = not cm[a, p.cell_frames[cid], c].any()
    return vis


class LinearValue:
    """v(S) = c0 + Σ_{i∈S} w_i — psi must equal w exactly for ANY readout."""

    def __init__(self, p, w, c0=0.25):
        self.p, self.w, self.c0 = p, np.asarray(w, dtype=float), c0

    def __call__(self, cms: np.ndarray) -> np.ndarray:
        return np.array(
            [self.c0 + self.w[_visible_cells(self.p, cm)].sum() for cm in cms]
        )


class AnyOfPair:
    """v(S) = 1 if cell_a ∈ S or cell_b ∈ S else 0 (pure redundancy)."""

    def __init__(self, p, cell_a, cell_b):
        self.p, self.a, self.b = p, cell_a, cell_b

    def __call__(self, cms: np.ndarray) -> np.ndarray:
        out = []
        for cm in cms:
            vis = _visible_cells(self.p, cm)
            out.append(1.0 if (vis[self.a] or vis[self.b]) else 0.0)
        return np.array(out)


def _pair_partition():
    """1 agent, 8 frames, 1 channel, 4-frame segments → 2 cells."""
    valid = np.ones((1, 8), dtype=bool)
    agent = np.array([True])
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=4), 1)
    assert p.n_cells == 2
    early = p.by_agent_seg[0][0][0]  # segment 0 (frames 0-3)
    late = p.by_agent_seg[0][1][0]   # segment 1 (frames 4-7)
    return p, valid, early, late


def test_linear_game_recovers_weights_exactly():
    valid = np.ones((2, 8), dtype=bool)
    valid[1, :4] = False  # agent 1 has front padding
    agent = np.array([True, True])
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=4), 2)
    rng_w = np.random.default_rng(0)
    w = rng_w.normal(size=p.n_cells)

    for name, orders in READOUTS.items():
        res = winter_value_nested_mc(
            LinearValue(p, w), p, valid, orders,
            n_samples=1, batch_size=4, rng=np.random.default_rng(1),
        )
        np.testing.assert_allclose(res.cell_psi, w, atol=1e-12, err_msg=name)
        assert abs(res.p4_residual) < 1e-12
        assert (res.cell_se == 0).all()


def test_chrono_credits_innovation_symmetric_splits():
    p, valid, early, late = _pair_partition()
    vf = AnyOfPair(p, early, late)

    chrono = winter_value_nested_mc(
        vf, p, valid, READOUTS["chrono"],
        n_samples=5, batch_size=5, rng=np.random.default_rng(2),
    )
    # chain-respecting: all credit lands where the information FIRST appears
    assert chrono.cell_psi[early] == pytest.approx(1.0)
    assert chrono.cell_psi[late] == pytest.approx(0.0)

    reverse = winter_value_nested_mc(
        vf, p, valid, READOUTS["reverse"],
        n_samples=5, batch_size=5, rng=np.random.default_rng(3),
    )
    assert reverse.cell_psi[late] == pytest.approx(1.0)
    assert reverse.cell_psi[early] == pytest.approx(0.0)

    sym = winter_value_nested_mc(
        vf, p, valid, READOUTS["symmetric"],
        n_samples=400, batch_size=64, rng=np.random.default_rng(4),
    )
    # uniform segment order → 50/50 split in expectation
    assert sym.cell_psi[early] == pytest.approx(0.5, abs=0.08)
    assert sym.cell_psi[late] == pytest.approx(0.5, abs=0.08)


def test_exact_shapley_and_game():
    p, valid, early, late = _pair_partition()

    class BothNeeded:
        def __call__(self, cms):
            out = []
            for cm in cms:
                vis = _visible_cells(p, cm)
                out.append(1.0 if (vis[early] and vis[late]) else 0.0)
            return np.array(out)

    res = exact_shapley(
        BothNeeded(), p, valid,
        [np.array([early]), np.array([late])],
    )
    np.testing.assert_allclose(res.phi, [0.5, 0.5], atol=1e-12)
    assert res.v_full == 1.0 and res.v_empty == 0.0

    with pytest.raises(ValueError, match="max_players"):
        exact_shapley(
            BothNeeded(), p, valid,
            [np.array([early]), np.array([late])], max_players=1,
        )


# ---------------------------------------------------------------------------
# Real model integration
# ---------------------------------------------------------------------------

A, T, F = 7, 80, 4


def _tiny_model() -> TransformerConflictModel:
    cfg = ModelConfig(
        name="transformer", num_agents=A, num_features=F,
        hidden_dim=8, num_frames=T, n_layers=1, n_heads=2, dropout=0.0,
    )
    torch.manual_seed(0)
    m = TransformerConflictModel(cfg)
    m.eval()
    return m


def _tiny_event(seed=5):
    rng = np.random.default_rng(seed)
    agent = np.array([True, True, True] + [False] * 4)
    valid = np.zeros((A, T), dtype=bool)
    valid[0, :] = True
    valid[1, 20:] = True
    valid[2, 5:70] = True
    x = rng.normal(size=(A, T, F)).astype(np.float32)
    x[~np.repeat(valid[..., None], F, axis=-1)] = 0.0
    return x, agent, valid


def test_real_model_p4_cache_and_determinism():
    model = _tiny_model()
    x, agent, valid = _tiny_event()
    p = build_partition(valid, agent, SegmentSpec(seg_len_frames=20), F)
    assert p.n_cells > 0

    def _run():
        vf = ConflictValueFn(model, x, agent, valid, "cpu")
        cache: dict[bytes, float] = {}
        sym = winter_value_nested_mc(
            vf, p, valid, READOUTS["symmetric"],
            n_samples=3, batch_size=3, rng=np.random.default_rng(10),
            cache=cache,
        )
        chrono = winter_value_nested_mc(
            vf, p, valid, READOUTS["chrono"],
            n_samples=3, batch_size=3, rng=np.random.default_rng(11),
            cache=cache,
        )
        return sym, chrono

    sym, chrono = _run()
    # P4 telescoping (float-sum noise only)
    assert abs(sym.p4_residual) < 1e-4
    assert abs(chrono.p4_residual) < 1e-4
    assert sym.v_full == chrono.v_full and sym.v_empty == chrono.v_empty
    # second readout reuses the shared per-event cache
    assert chrono.n_cache_hits > 0

    sym2, chrono2 = _run()
    np.testing.assert_array_equal(sym.cell_psi, sym2.cell_psi)
    np.testing.assert_array_equal(chrono.cell_psi, chrono2.cell_psi)


# ---------------------------------------------------------------------------
# Driver end-to-end
# ---------------------------------------------------------------------------


def _write_npy_cache(prefix: str, N=4):
    rng = np.random.default_rng(0)
    x = np.zeros((N, A, T, F), dtype=np.float32)
    mask = np.zeros((N, A), dtype=bool)
    valid = np.zeros((N, A, T), dtype=bool)
    for i in range(N):
        xi, agent_i, valid_i = _tiny_event(seed=100 + i)
        x[i], mask[i], valid[i] = xi, agent_i, valid_i
    y = np.array([1, 1, 0, 1], dtype=np.float32)[:N]
    target = np.array([0, 0, -1, 0], dtype=np.int64)[:N]
    np.save(f"{prefix}_x.npy", x)
    np.save(f"{prefix}_mask.npy", mask)
    np.save(f"{prefix}_y.npy", y)
    np.save(f"{prefix}_target.npy", target)
    np.save(f"{prefix}_eid.npy", np.arange(1000, 1000 + N, dtype=np.int64))
    np.save(f"{prefix}_valid.npy", valid)


def _save_ckpt(path: Path, masked: bool = True):
    model = _tiny_model()
    cfg = BaselineRuntimeConfig(
        model=ModelConfig(
            name="transformer", num_agents=A, num_features=F,
            hidden_dim=8, num_frames=T, n_layers=1, n_heads=2, dropout=0.0,
        )
    )
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.5,
            "config": cfg,
            "train_regime": {"masked": masked, "uses_valid_mask": masked},
        },
        path,
    )


@pytest.fixture()
def driver_setup(tmp_path):
    prefix = str(tmp_path / "val")
    _write_npy_cache(prefix)
    bbc = _load_script("build_binary_cache")
    bbc.process_split(prefix, "val")
    ckpt = tmp_path / "ckpt.pt"
    _save_ckpt(ckpt, masked=True)
    return tmp_path, prefix, ckpt


def _attr_cfg(tmp_path, prefix, ckpt, **over) -> AttributionConfig:
    kw = dict(
        checkpoint=str(ckpt),
        data_prefix=prefix,
        device="cpu",
        selection=SelectionConfig(conflict_only=True, max_n=2),
        game=GameConfig(seg_len_frames=20, readouts=("symmetric", "chrono")),
        mc=MCConfig(n_samples=2, batch_size=4, seed=99),
        exact=ExactConfig(enabled=True, agents=(0,), max_players=8),
        output=OutputConfig(dir=str(tmp_path / "out"), save_per_event=True),
    )
    kw.update(over)
    return AttributionConfig(**kw)


def test_driver_end_to_end(driver_setup):
    tmp_path, prefix, ckpt = driver_setup
    rows = run_attribution(_attr_cfg(tmp_path, prefix, ckpt))

    # 2 conflict events picked (max_n=2) × 2 readouts
    assert len(rows) == 4
    assert (tmp_path / "out" / "summary.csv").exists()

    ev_json = tmp_path / "out" / "event_1000.json"
    assert ev_json.exists()
    with open(ev_json, encoding="utf-8") as f:
        result = json.load(f)

    for name in ("symmetric", "chrono"):
        r = result["readouts"][name]
        assert abs(r["p4_residual"]) < 1e-4
        assert r["sum_psi"] == pytest.approx(r["delta_v"], abs=1e-4)
        assert r["cells"] and "psi" in r["cells"][0]
    # exact ego segment game present: 4 segments (seg_len 20)
    assert result["exact"]["ego"]["players"] == ["seg0", "seg1", "seg2", "seg3"]
    assert len(result["exact"]["ego"]["phi"]) == 4

    # determinism at the driver level
    rows2 = run_attribution(_attr_cfg(tmp_path, prefix, ckpt))
    for r1, r2 in zip(rows, rows2):
        assert r1["v_full"] == r2["v_full"]
        assert r1["top_segment_psi"] == r2["top_segment_psi"]


def test_driver_refuses_unmasked_checkpoint(driver_setup):
    tmp_path, prefix, ckpt = driver_setup
    bad_ckpt = tmp_path / "bad.pt"
    _save_ckpt(bad_ckpt, masked=False)

    with pytest.raises(RuntimeError, match="masked"):
        load_frozen_model(str(bad_ckpt), torch.device("cpu"))
    # but the escape hatch works (case studies)
    model, _ = load_frozen_model(
        str(bad_ckpt), torch.device("cpu"), allow_unmasked=True
    )
    assert model is not None


def test_visualize_renders_pngs(driver_setup):
    from model_baseline.attribution.visualize import render_event

    tmp_path, prefix, ckpt = driver_setup
    run_attribution(_attr_cfg(tmp_path, prefix, ckpt))
    written = render_event(tmp_path / "out" / "event_1000.json")
    assert len(written) == 6  # 2 readouts × 3 plots
    for w in written:
        assert w.exists() and w.stat().st_size > 0
