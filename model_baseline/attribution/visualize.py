"""Render per-event attribution JSONs (agent×segment heatmap, risk timeline,
channel bars).  Bipolar convention everywhere: red = pushes toward conflict
(ψ > 0), blue = pushes away (ψ < 0).

Usage:
  python -m model_baseline.attribution.visualize outputs/attribution/run_001/event_123.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from model_baseline.attribution.outputs import SLOT_NAMES


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for attribution.visualize "
            "(pip install matplotlib)"
        ) from e


def _seg_labels(result: dict[str, Any], fps: float) -> list[str]:
    T = max(e for _s, e in result["seg_bounds"])
    return [
        f"{(s - T) / fps:.1f}~{(e - T) / fps:.1f}s"
        for s, e in result["seg_bounds"]
    ]


def plot_agent_segment_heatmap(
    result: dict[str, Any], readout: str, save: Optional[Path] = None,
    fps: float = 10.0,
):
    """Agents × segments ψ heatmap (diverging, centred at 0)."""
    plt = _mpl()
    r = result["readouts"][readout]
    agg = r["aggregations"]["agent_segment"]
    n_seg = len(result["seg_bounds"])
    agents = [a for a in SLOT_NAMES if any(k.startswith(f"{a}/") for k in agg)]

    m = np.zeros((len(agents), n_seg))
    for k, v in agg.items():
        a_name, seg = k.split("/")
        m[agents.index(a_name), int(seg[3:])] = v

    vmax = max(1e-12, np.abs(m).max())
    fig, ax = plt.subplots(figsize=(max(6, n_seg * 0.9), 1 + 0.6 * len(agents)))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_seg), _seg_labels(result, fps), rotation=45, ha="right")
    ax.set_yticks(range(len(agents)), agents)
    ax.set_title(
        f"event {result['event']['event_id']} — {readout} readout "
        f"(ψ>0 → conflict)"
    )
    fig.colorbar(im, ax=ax, label="ψ (logit)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        plt.close(fig)
    return fig


def plot_risk_timeline(
    result: dict[str, Any], readout: str = "chrono",
    save: Optional[Path] = None, fps: float = 10.0,
):
    """Per-segment ψ bars + cumulative risk curve from v(∅) to v(N).

    For the chrono readout the cumulative curve reads as "the model's risk
    estimate as the scene history is revealed time-slice by time-slice".
    """
    plt = _mpl()
    r = result["readouts"][readout]
    seg_agg = r["aggregations"]["segment"]
    n_seg = len(result["seg_bounds"])
    psi = np.array([seg_agg.get(f"seg{s}", 0.0) for s in range(n_seg)])
    cum = r["v_empty"] + np.concatenate([[0.0], np.cumsum(psi)])

    fig, ax = plt.subplots(figsize=(max(6, n_seg * 0.9), 4))
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in psi]
    ax.bar(range(n_seg), psi, color=colors, alpha=0.85, label="segment ψ")
    ax.step(
        np.arange(-1, n_seg) + 0.5, cum, where="post",
        color="black", lw=1.5, label="cumulative v",
    )
    ax.axhline(r["v_empty"], color="gray", ls=":", lw=1, label="v(∅) prior")
    ax.axhline(r["v_full"], color="gray", ls="--", lw=1, label="v(N) full")
    ax.set_xticks(range(n_seg), _seg_labels(result, fps), rotation=45, ha="right")
    ax.set_ylabel("logit")
    ax.set_title(
        f"event {result['event']['event_id']} — {readout} risk timeline"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        plt.close(fig)
    return fig


def plot_channel_bars(
    result: dict[str, Any], readout: str, save: Optional[Path] = None,
):
    plt = _mpl()
    ch = result["readouts"][readout]["aggregations"]["channel"]
    names = list(ch.keys())
    vals = [ch[k] for k in names]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, vals,
           color=["#c0392b" if v > 0 else "#2471a3" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ψ (logit)")
    ax.set_title(f"event {result['event']['event_id']} — {readout} channels")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        plt.close(fig)
    return fig


def render_event(json_path: str | Path, out_dir: Optional[Path] = None,
                 fps: float = 10.0) -> list[Path]:
    """All plots for one event JSON; PNGs land next to the JSON by default."""
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        result = json.load(f)
    out_dir = Path(out_dir) if out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem

    written: list[Path] = []
    for readout in result["readouts"]:
        p1 = out_dir / f"{stem}_{readout}_heatmap.png"
        plot_agent_segment_heatmap(result, readout, save=p1, fps=fps)
        p2 = out_dir / f"{stem}_{readout}_timeline.png"
        plot_risk_timeline(result, readout, save=p2, fps=fps)
        p3 = out_dir / f"{stem}_{readout}_channels.png"
        plot_channel_bars(result, readout, save=p3)
        written += [p1, p2, p3]
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+", help="event_*.json paths")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    for jp in args.json:
        written = render_event(jp, args.out_dir and Path(args.out_dir), args.fps)
        for w in written:
            print(w)


if __name__ == "__main__":
    main()
