"""Baseline-anchored delta radar for color-jitter / grayscale augmentation conditions.

Every model's no-augmentation level collapses onto one shared reference circle
(radius = offset). Radius at other spokes = offset + (drop(condition) -
drop(none)); outside the circle = the augmentation increased skin-tone
sensitivity, inside = it decreased it.

Uses ONLY the bug-fixed color-jitter runs (--color_jitter_consistent; see
data/rgb.py and llm_reports/color_jitter_strength_experiment.md for why the
uncorrected runs are excluded) plus the grayscale condition, which was never
affected by the per-frame-flicker bug.

Each model gets a distinct marker shape (not just a color) for colorblind
accessibility. No variance/significance overlay on the plot itself -- see
the console output and llm_reports/color_jitter_strength_experiment.md for
per-model and pooled significance figures.

Aggregation identical to the other augmentation figures:
  unit drop      = matched_unseen F1 - shifted_unseen F1   per (pair, seed)
  per-seed value = mean over pairs within that seed
  point          = mean over seeds
All conditions are restricted to the (pair, seed) units shared across every
condition supplied via --roots (no on-figure footnote; see console output).

Usage:
  python scripts/plot_augmentation_radar.py \
    --roots none=out/skin_tone_probe_rgb_torchvision_v6_cj0p0 \
            "jitter 40%=out/skin_tone_probe_rgb_torchvision_v6_cj0p4_consistent" \
            "jitter 80%=out/skin_tone_probe_rgb_torchvision_v6_cj0p8_consistent" \
            "jitter 80% strong=out/skin_tone_probe_rgb_torchvision_v6_cj0p8_strong_consistent" \
            "grayscale 50%=out/skin_tone_probe_rgb_torchvision_v6_grayscale0p5" \
    --baseline none --out_dir out/.../augmentation_conditions
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font sizes scaled to this figure's own native width, matching the pt-per-
# inch ratio used in benchmarks/skin_tone/summarize_skin_tone_significance.py
# (reference: 10.4in wide, title=12.5, label=10.5, tick=10.0, annotation=8.7)
# so text reads at the same apparent size across figures once each is scaled
# to a shared column width in the paper.
_FONT_RATIO_TITLE = 12.5 / 10.4
_FONT_RATIO_LABEL = 10.5 / 10.4
_FONT_RATIO_TICK = 10.0 / 10.4
_FONT_RATIO_ANNOTATION = 8.7 / 10.4


def font_sizes(width_in: float) -> dict[str, float]:
    return {
        "title": _FONT_RATIO_TITLE * width_in,
        "label": _FONT_RATIO_LABEL * width_in,
        "tick": _FONT_RATIO_TICK * width_in,
        "annotation": _FONT_RATIO_ANNOTATION * width_in,
    }

MODELS = ["mc3_18", "mvit_v2_s", "r2plus1d_18", "r3d_18", "s3d", "swin3d_s"]

# Validated categorical palette (dataviz reference instance), fixed per model
# so colors match every other augmentation figure in this series.
MODEL_COLOR = {
    "mc3_18":      "#2a78d6",
    "mvit_v2_s":   "#1baf7a",
    "r2plus1d_18": "#eda100",
    "r3d_18":      "#008300",
    "s3d":         "#4a3aa7",
    "swin3d_s":    "#e34948",
}
# Distinct marker per model -- identity is never carried by color alone.
MODEL_MARKER = {
    "mc3_18":      "o",
    "mvit_v2_s":   "s",
    "r2plus1d_18": "^",
    "r3d_18":      "D",
    "s3d":         "v",
    "swin3d_s":    "P",
}

INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE_C = "#e1e0d9", "#c3c2b7"

MATCHED, SHIFTED = "eval_matched_unseen_ids", "eval_shifted_unseen_ids"


SEED_RE = re.compile(r"^seed_(\d+)")


def unit_drops(root: Path, model: str) -> dict[tuple[str, str], float]:
    """(pair_tag, seed_N) -> matched-shifted F1 drop.

    CV runs name their directories seed_{N}fold{M}; the fold suffix is
    stripped and folds are averaged into the seed, so CV and non-CV roots
    key on the same (pair_tag, seed_N) units and can share a unit-intersection
    with plain-seed (leaky-split) roots.
    """
    accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    base = root / "rgb_torchvision" / model
    for summary in sorted(base.glob(f"*/seed_*/summary_rgb_{model}_model.json")):
        pair_tag, seed_dir = summary.parent.parent.name, summary.parent.name
        match = SEED_RE.match(seed_dir)
        if not match:
            continue
        seed_key = f"seed_{match.group(1)}"
        d = json.loads(summary.read_text())
        splits = d.get("splits", {})
        try:
            value = float(splits[MATCHED]["f1_macro"]) - float(splits[SHIFTED]["f1_macro"])
        except KeyError:
            continue
        accum[(pair_tag, seed_key)].append(value)
    return {key: float(np.mean(values)) for key, values in accum.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_name", default="augmentation_radar_delta")
    args = ap.parse_args()

    conditions: list[tuple[str, Path]] = []
    for spec in args.roots:
        label, _, path = spec.partition("=")
        conditions.append((label, Path(path)))
    labels = [l for l, _ in conditions]
    baseline_label = args.baseline or labels[0]

    # ── collect, restricted to units shared across every supplied condition ─
    raw: dict[tuple[str, str], dict] = {}
    for label, root in conditions:
        for model in MODELS:
            raw[(model, label)] = unit_drops(root, model)

    seed_means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    unit_cache: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        shared = set.intersection(*(set(raw[(model, l)]) for l in labels))
        for label in labels:
            units = {k: v for k, v in raw[(model, label)].items() if k in shared}
            unit_cache[(model, label)] = units
            per_seed = defaultdict(list)
            for (pair, seed), v in units.items():
                per_seed[seed].append(v)
            seed_means[model][label] = {s: float(np.mean(v)) for s, v in per_seed.items()}
        print(f"{model}: {len(shared)} shared (pair, seed) units across all conditions")

    mean_of = lambda m, l: float(np.mean(list(seed_means[m][l].values())))

    n = len(labels)
    theta = np.arange(n) * 2 * np.pi / n
    close = lambda a: np.concatenate([a, a[:1]])

    deltas = {(m, l): mean_of(m, l) - mean_of(m, baseline_label)
              for m in MODELS for l in labels}
    dmin, dmax = min(deltas.values()), max(deltas.values())
    offset = max(0.03, -dmin * 1.3)
    r_max = offset + dmax * 1.25

    # Font sizes are pinned to the 8.6in ratio (matching the other figures)
    # but the canvas itself is drawn smaller at the same aspect ratio -- that
    # makes the fixed-point-size text and markers occupy more of the figure.
    fig_w = 8.6
    fonts = font_sizes(fig_w)
    fig = plt.figure(figsize=(6.3, 5.3))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color(BASELINE_C)
    ax.spines["polar"].set_linewidth(0.7)
    ax.set_ylim(0, r_max)

    d_ticks = [d for d in (-0.02, 0.0, 0.02, 0.04, 0.06) if 0 < offset + d < r_max]
    ax.set_yticks([offset + d for d in d_ticks])
    ax.set_yticklabels([("0" if d == 0 else f"{d:+.2f}") for d in d_ticks],
                       fontsize=fonts["tick"], color=INK_MUTED)
    ax.set_rlabel_position(90 / n)
    ax.set_xticks(theta)
    xtick_labels = [("none\n(reference)" if l == baseline_label else l) for l in labels]
    ax.set_xticklabels(xtick_labels, fontsize=fonts["tick"], color=INK_2)

    # shared reference circle
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(tt, np.full_like(tt, offset), color=INK_2, lw=2.4, zorder=2)

    for model in MODELS:
        c = MODEL_COLOR[model]
        vals = np.array([offset + deltas[(model, l)] for l in labels])

        ax.plot(close(theta), close(vals), "-", color=c, lw=2.0, alpha=0.95, zorder=3)
        ax.plot(theta, vals, MODEL_MARKER[model], color=c, ms=8, mec="white",
                mew=1.0, zorder=4)

    handles = [plt.Line2D([], [], color=MODEL_COLOR[m], lw=2, marker=MODEL_MARKER[m],
                          mec="white", ms=8, label=m) for m in MODELS]
    handles.append(plt.Line2D([], [], color=INK_2, lw=2.4,
                              label="no augmentation (Δ = 0)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.05),
              fontsize=fonts["tick"], frameon=False)
    ax.set_title("Change in skin-tone swap drop relative to no augmentation\n(lower is better)",
                 fontsize=fonts["title"], color=INK, pad=24)

    plt.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{args.out_name}.{ext}",
                    dpi=200, facecolor="white", bbox_inches="tight")
    print(f"\nsaved to {args.out_dir}/{args.out_name}.{{pdf,png}}")


if __name__ == "__main__":
    main()
