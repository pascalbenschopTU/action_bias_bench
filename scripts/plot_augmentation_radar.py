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
from matplotlib.transforms import ScaledTranslation

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


# Poster variant keeps the paper figure's data and layout but pins every font
# larger relative to the canvas, so text stays legible once the figure is
# scaled into a poster column and read from a distance.
POSTER_FONT_SCALE = 1.45

TITLE = "Change in skin-tone swap drop relative to no augmentation\n(lower is better)"

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
    ap.add_argument("--from_cache", action="store_true",
                    help="Reuse the cached per-(model, condition) deltas written by a "
                         "previous run instead of rescanning every condition root. "
                         "Layout-only changes need nothing else; drop the flag after "
                         "any run whose underlying results changed.")
    args = ap.parse_args()

    conditions: list[tuple[str, Path]] = []
    for spec in args.roots:
        label, _, path = spec.partition("=")
        conditions.append((label, Path(path)))
    labels = [l for l, _ in conditions]
    baseline_label = args.baseline or labels[0]

    # The scan below opens thousands of small per-seed JSONs across every
    # condition root, which is slow on a network mount, while all the figure
    # needs from it is one delta per (model, condition). Cache those so that
    # re-running purely to adjust the layout costs nothing.
    cache_path = args.out_dir / f"{args.out_name}_deltas.json"
    if args.from_cache:
        cached = json.loads(cache_path.read_text())
        deltas = {tuple(k.split("|", 1)): v for k, v in cached.items()}
        print(f"loaded {len(deltas)} cached deltas from {cache_path}")
    else:
        # ── collect, restricted to units shared across every supplied condition ─
        raw: dict[tuple[str, str], dict] = {}
        for label, root in conditions:
            for model in MODELS:
                raw[(model, label)] = unit_drops(root, model)

        seed_means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for model in MODELS:
            shared = set.intersection(*(set(raw[(model, l)]) for l in labels))
            for label in labels:
                units = {k: v for k, v in raw[(model, label)].items() if k in shared}
                per_seed = defaultdict(list)
                for (pair, seed), v in units.items():
                    per_seed[seed].append(v)
                seed_means[model][label] = {s: float(np.mean(v)) for s, v in per_seed.items()}
            print(f"{model}: {len(shared)} shared (pair, seed) units across all conditions")

        mean_of = lambda m, l: float(np.mean(list(seed_means[m][l].values())))
        deltas = {(m, l): mean_of(m, l) - mean_of(m, baseline_label)
                  for m in MODELS for l in labels}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({f"{m}|{l}": v for (m, l), v in deltas.items()}, indent=2))
        print(f"cached deltas to {cache_path}")

    n = len(labels)
    theta = np.arange(n) * 2 * np.pi / n
    close = lambda a: np.concatenate([a, a[:1]])
    dmin, dmax = min(deltas.values()), max(deltas.values())
    offset = max(0.03, -dmin * 1.3)
    r_max = offset + dmax * 1.25

    def draw(fig_w: float, fig_h: float, fonts: dict[str, float], marker_size: float,
             out_name: str, xtick_pad: float | None = None, legend_below: bool = False,
             axes_rect: tuple | None = None, legend_x: float | None = None,
             title: str | None = TITLE, title_pad: float = 24,
             compact_labels: bool = False, legend_y: float = 0.5,
             bottom_label_drop: float = 0.0, suptitle: str | None = None,
             suptitle_y: float = 0.95) -> None:
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")
        # An explicit rect (poster variants) fixes the polar axes' size directly,
        # so the saved figure keeps the requested aspect ratio and the circle --
        # whose diameter is the shorter side of the rect -- can be sized on
        # purpose instead of being whatever tight_layout leaves over.
        if axes_rect is None:
            ax = fig.add_subplot(111, projection="polar")
        else:
            ax = fig.add_axes(axes_rect, projection="polar")
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
        # One line rather than stacked for the poster variants: the stacked form
        # costs vertical space, which is exactly what limits the circle's size.
        baseline_spoke = "none (reference)" if compact_labels else "none\n(reference)"
        xtick_labels = [(baseline_spoke if l == baseline_label else l) for l in labels]
        ax.set_xticklabels(xtick_labels, fontsize=fonts["tick"], color=INK_2)
        if xtick_pad is not None:
            # Spoke labels are centred on their spoke, so at larger font sizes the
            # inner half runs over the plotted lines; push them clear radially.
            ax.tick_params(axis="x", pad=xtick_pad)
        if bottom_label_drop:
            # The radial pad is uniform, but the lower spokes sit at a shallow
            # angle where it buys little vertical clearance, so those two labels
            # still touch the circle. Drop just them straight down instead;
            # raising the pad for everyone would push the side labels into the
            # legend. cos(theta) is the vertical component with zero at north.
            drop = ScaledTranslation(0, -bottom_label_drop / 72.0, fig.dpi_scale_trans)
            for label, angle in zip(ax.get_xticklabels(), theta):
                if np.cos(angle) < -0.3:
                    label.set_transform(label.get_transform() + drop)

        # shared reference circle
        tt = np.linspace(0, 2 * np.pi, 200)
        ax.plot(tt, np.full_like(tt, offset), color=INK_2, lw=2.4, zorder=2)

        for model in MODELS:
            c = MODEL_COLOR[model]
            vals = np.array([offset + deltas[(model, l)] for l in labels])

            ax.plot(close(theta), close(vals), "-", color=c, lw=2.0, alpha=0.95, zorder=3)
            ax.plot(theta, vals, MODEL_MARKER[model], color=c, ms=marker_size, mec="white",
                    mew=1.0, zorder=4)

        handles = [plt.Line2D([], [], color=MODEL_COLOR[m], lw=2, marker=MODEL_MARKER[m],
                              mec="white", ms=marker_size, label=m) for m in MODELS]
        handles.append(plt.Line2D([], [], color=INK_2, lw=2.4,
                                  label="reference (Δ = 0)" if compact_labels
                                  else "no augmentation (Δ = 0)"))
        if legend_x is not None:
            # Figure coordinates, so the legend sits beside the plot without
            # stealing width from the axes rect the way an axes-anchored one does.
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(legend_x, legend_y),
                       fontsize=fonts["tick"], frameon=False)
        elif legend_below:
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                      ncol=3, fontsize=fonts["tick"], frameon=False)
        else:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.05),
                      fontsize=fonts["tick"], frameon=False)
        if title:
            ax.set_title(title, fontsize=fonts["title"], color=INK, pad=title_pad)
        if suptitle:
            # Figure-level so it centres over the whole canvas and, unlike an axes
            # title, takes no space from the axes rect -- the radar and legend keep
            # the exact geometry they had before the title was added.
            fig.suptitle(suptitle, fontsize=fonts["title"], color=INK, y=suptitle_y)

        # An explicit rect already sets deliberate margins, so tight_layout would
        # fight it -- but always save with a tight bbox, which only trims surplus
        # whitespace and so can never clip a label the way a fixed canvas can.
        if axes_rect is None:
            plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(args.out_dir / f"{out_name}.{ext}",
                        dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    # Font sizes are pinned to the 8.6in ratio (matching the other figures)
    # but the canvas itself is drawn smaller at the same aspect ratio -- that
    # makes the fixed-point-size text and markers occupy more of the figure.
    fig_w = 8.6
    fonts = font_sizes(fig_w)
    poster_fonts = {k: v * POSTER_FONT_SCALE for k, v in fonts.items()}
    big_fonts = {k: v * 1.62 for k, v in fonts.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    draw(6.3, 5.3, fonts, 8, args.out_name)

    # Landscape poster variant, legend beside the plot, compact labels. The
    # circle's diameter is the axes rect's shorter side -- its height -- so the
    # rect spans most of the canvas height.
    #
    # The canvas is 6.3in rather than 5.6in only to make room for the title; the
    # rect and legend are re-expressed as fractions of the taller canvas so both
    # keep the exact same size and position in inches as the untitled version
    # (rect 5.94x4.648in at y=0.476in, legend centred on the radar at y=2.8in).
    draw(9.0, 6.3, big_fonts, 12, f"{args.out_name}_poster_2",
         xtick_pad=16, axes_rect=(0.005, 0.07556, 0.66, 0.73778),
         legend_x=0.70, legend_y=0.4444, title=None, compact_labels=True,
         bottom_label_drop=18,
         suptitle="Change in skin-tone swap drop vs. no augmentation (lower is better)",
         suptitle_y=0.965)
    print(f"\nsaved to {args.out_dir}/{args.out_name}.{{pdf,png}} (+ _poster_2/_3/_4)")


if __name__ == "__main__":
    main()
