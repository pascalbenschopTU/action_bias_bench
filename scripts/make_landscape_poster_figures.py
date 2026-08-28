"""Figure variants for the 140 x 100 cm landscape ECCV poster.

Kept separate from the paper figure scripts on purpose: nothing written here
feeds the paper, so the paper pipeline (`summarize_skin_tone_robustness.py`,
`plot_probe_ssm.py`, `plot_augmentation_radar.py`) stays untouched and its
figures keep their published geometry. See llm_reports/landscape_poster.md for
the layout budget these sizes come from.

Every figure is drawn at a native canvas whose width in inches equals its slot
width on the printed poster, so `\\includegraphics[width=\\linewidth]` scales it
by exactly 1.0 and a point size set here is the point size that gets printed.
`_fonts()` converts the shared on-poster targets in `POSTER_PT` for figures
that cannot be drawn at their slot width (the hero strip, which is a 100 cm
wide raster).

Usage (run from the ActionBiasBench directory, env with cv2 + matplotlib):
    python scripts/make_landscape_poster_figures.py --all
    python scripts/make_landscape_poster_figures.py hero heatmap
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_actionbiasbench"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent

# ── poster geometry ──────────────────────────────────────────────────────────
# 140 cm wide board, 2 cm text margins, four columns separated by 0.02\linewidth.
TEXT_W_CM = 136.0
IN = 2.54
# Body columns are not equal: section 2's heatmap needs the width and section
# 1's schematic does not, so each figure is drawn at its own slot width. These
# fractions must match the \begin{column} widths in landscape_poster.tex; the
# four plus three 0.02 \hfill gaps come to 1.0.
COL_FRAC = {"schematic": 0.195, "heatmap": 0.265, "probe": 0.245, "radar": 0.235}
SLOT_CM = {name: frac * TEXT_W_CM for name, frac in COL_FRAC.items()}
HERO_CM = 0.71 * TEXT_W_CM        # hero row; the rest of that row is the EU AI Act box

# Point sizes as they should measure on the printed poster.
POSTER_PT = {"title": 40.0, "label": 34.0, "tick": 30.0, "cell": 27.0,
             "annot": 28.0, "legend": 30.0}

# bbox_inches="tight" trims a different amount off each canvas, and it is the
# *delivered* width, not the native one, that LaTeX scales to the column. So a
# figure cropped to 73% of its canvas has its text magnified by 1/0.73 on the
# board. These are the measured delivered/native width ratios; _save() prints
# the current value on every run, so they can be re-derived whenever a figure's
# layout changes. Without this the radar printed a third larger than the rest.
CROP = {"pair_heatmap_paired_flip_rate_landscape": 0.981,
        "probe_drop_by_model_landscape": 0.992,
        "augmentation_radar_landscape": 0.702,
        "swap_design_schematic": 0.988}

DATASET_ROOT = Path("/Volumes/MoDDL/Pascal/motion_only_AR/datasets/skin_tone_actions/camera_far")
OUT_DIR = ROOT / "out" / "poster_landscape"

PAIR_ORDER = [
    "squat_vs_tie", "tie_vs_squat",
    "clap_vs_celebrate", "celebrate_vs_clap",
    "dribble_vs_golf", "golf_vs_dribble",
    "lunge_vs_cartwheel", "cartwheel_vs_lunge",
    "yawn_vs_fish", "fish_vs_yawn",
]

# Frozen-feature probe taxonomy, mirrored from scripts/plot_probe_ssm.py (that
# module runs its whole pipeline at import time, so it cannot be imported).
FAMILY = {
    "clip": "language", "siglip": "language", "tc_clip": "language+K400",
    "dinov2": "img-ssl", "dinov3": "img-ssl", "hiera": "img-ssl",
    "hiera_large": "img-ssl", "eva02": "img-ssl+ImgNet", "vjepa2": "video-ssl",
    "r3d_18": "K400", "mc3_18": "K400", "r2plus1d_18": "K400",
    "mvit_v2_s": "K400", "s3d": "K400", "swin3d_s": "K400",
}
FAM_COLOR = {
    "language": "#d62728", "language+K400": "#8c564b", "img-ssl": "#1f77b4",
    "img-ssl+ImgNet": "#9467bd", "video-ssl": "#ff7f0e", "K400": "#2ca02c",
}

# Augmentation radar palette, shared with the paper figure so a model reads the
# same colour everywhere.
sys.path.insert(0, str(ROOT / "scripts"))
from plot_augmentation_radar import (  # noqa: E402
    BASELINE_C, GRID, INK, INK_2, INK_MUTED, MODEL_COLOR, MODEL_MARKER, MODELS,
)

# TU Delft house colours (hex from poster/_style/tudelft-colors.sty), so the
# schematic matches the poster's headings and boxes rather than introducing a
# second palette.
TUD = {"primary": "#00A6D6", "navy": "#0C2340", "orange": "#EC6842",
       "primary_wash": "#E6F6FB", "orange_wash": "#FDEEE9", "ink": "#111111"}

RADAR_LABELS = ["none", "weak jitter", "strong jitter",
                "strong jitter + grayscale", "planckian"]


def _fonts(native_w_in: float, slot_cm: float, stem: str | None = None) -> dict[str, float]:
    """POSTER_PT re-expressed in the figure's own points, given that LaTeX will
    scale this canvas to `slot_cm` on the board. `stem` applies that figure's
    measured tight-bbox crop, so every figure's text prints at the same size."""
    k = native_w_in * CROP.get(stem, 1.0) * IN / slot_cm
    return {name: value * k for name, value in POSTER_PT.items()}


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    ratio = fig.get_tightbbox().width / fig.get_size_inches()[0]
    plt.close(fig)
    print(f"[OK] {stem}.pdf/.png  delivered/native = {ratio:.3f} "
          f"(CROP says {CROP.get(stem, 1.0)})", flush=True)


def _pretty_pair(pair_tag: str) -> str:
    left, _, right = pair_tag.partition("_vs_")
    return f"{left} vs {right}" if right else pair_tag


DISPLAY_NAME = {
    "i3d_flow": "I3D-flow", "mc3_18": "MC3-18", "mvit_v2_s": "MViT-v2-S",
    "r2plus1d_18": "R(2+1)D-18", "r3d_18": "R3D-18", "s3d": "S3D",
    "swin3d_s": "Swin3D-S", "clip": "CLIP", "siglip": "SigLIP",
    "tc_clip": "TC-CLIP", "dinov2": "DINOv2", "dinov3": "DINOv3",
    "eva02": "EVA-02", "hiera": "Hiera", "hiera_large": "Hiera",
    "vjepa2": "V-JEPA 2",
}


def _model_label(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset frames (hero strip and design schematic)
# ══════════════════════════════════════════════════════════════════════════════
def _video(background: str, action: str, base_id: int, variant: str) -> Path:
    return (DATASET_ROOT / background / "__generated_synthetic_videos" / action
            / f"{action}_{base_id}_modified_{variant}.mp4")


def _read_frames(path: Path, idxs):
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = {}
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {idx} from {path}")
        out[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def _actor_bbox(frames_a, frames_b, thr: int = 25):
    """Union bbox of the pixels where the two tone renders disagree.

    The two clips are byte-identical except for the skin texture, so this marks
    exactly the actor's visible skin and nothing else. It is preferred over
    background subtraction here because the Konzerthaus pavement is wet and
    reflective: a background difference also catches the actor's reflection and
    pushes the box a hundred pixels wide of the body.
    """
    import cv2

    boxes = []
    for idx, frame_a in frames_a.items():
        diff = np.abs(frame_a.astype(int) - frames_b[idx].astype(int)).max(axis=2)
        mask = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(xs):
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    if not boxes:
        raise RuntimeError("no tone difference found; are both renders the same variant?")
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _crop_window(bbox, frame_shape, aspect: float, zoom: float):
    """Window `zoom` times the actor's height, at the requested width/height
    `aspect`, centred on the actor and clipped into the frame.

    Zooming in rather than keeping the full 16:9 render is what makes the skin
    tone itself readable from poster distance; at `zoom` around 1.5 the building
    and the ground line still frame the actor, so "same background" survives."""
    height, width = frame_shape
    win_h = min(height, zoom * (bbox[3] - bbox[1]))
    win_w = min(width, aspect * win_h)
    win_h = min(height, win_w / aspect)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    x0 = int(round(min(max(cx - win_w / 2, 0), width - win_w)))
    y0 = int(round(min(max(cy - win_h / 2, 0), height - win_h)))
    return x0, y0, x0 + int(round(win_w)), y0 + int(round(win_h))


def _hstack(images, gap_px: int, colour: int = 255):
    height = images[0].shape[0]
    gap = np.full((height, gap_px, 3), colour, dtype=np.uint8)
    out = images[0]
    for image in images[1:]:
        out = np.concatenate([out, gap, image], axis=1)
    return out


def make_hero(args) -> None:
    """Three timepoints through one cartwheel; each panel shows the White and
    African render of the same frame flush against each other.

    Deliberately uncaptioned: the section heading says only the skin tone
    changes, and per-panel tone labels turned out to read as clutter over an
    image whose whole point is that the two halves are otherwise identical.
    """
    tones = ["white", "african"]
    frame_idxs = [int(v) for v in args.hero_frames.split(",")]

    paths = {t: _video(args.background, "cartwheel", args.hero_base_id, t) for t in tones}
    frames = {t: _read_frames(paths[t], frame_idxs) for t in tones}
    bbox = _actor_bbox(frames["white"], frames["african"])
    x0, y0, x1, y1 = _crop_window(bbox, frames["white"][frame_idxs[0]].shape[:2],
                                  args.hero_aspect, args.hero_zoom)

    panels = [_hstack([frames[t][idx][y0:y1, x0:x1] for t in tones], gap_px=0)
              for idx in frame_idxs]
    strip = _hstack(panels, gap_px=args.hero_panel_gap)

    fig = plt.figure(figsize=(12.0, 12.0 * strip.shape[0] / strip.shape[1]), dpi=320)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(strip, interpolation="none")
    ax.axis("off")
    _save(fig, "hero_tone_pairs_landscape")


def make_schematic(args) -> None:
    """Section 1's design figure: in training every action is only ever seen
    with one skin tone; at test the tone-action mapping is swapped and nothing
    else changes.

    The action is labelled once per column because it is the constant -- what
    differs between the two rows is only the tone, which the reader is meant to
    see rather than read.
    """
    actions = ["cartwheel", "lunge"]
    # Frames picked by eye from a contact sheet: the cartwheel at peak inverted
    # extension and the lunge at its deepest, so each panel reads as its action
    # from across the room.
    frame_idx = {"cartwheel": args.schematic_cartwheel_frame,
                 "lunge": args.schematic_lunge_frame}
    rows = [("TRAIN", {"cartwheel": "white", "lunge": "african"},
             TUD["primary"], TUD["primary_wash"]),
            ("TEST", {"cartwheel": "african", "lunge": "white"},
             TUD["orange"], TUD["orange_wash"])]

    by_action = {}
    for action in actions:
        idx = frame_idx[action]
        by_action[action] = {
            tone: _read_frames(_video(args.background, action, args.base_id, tone), [idx])
            for tone in ("white", "african")
        }
    # One window shared by both actions, so the two panels in a row differ only
    # in what the actor is doing -- a per-action crop would move the building
    # and read as a background change.
    boxes = [_actor_bbox(v["white"], v["african"]) for v in by_action.values()]
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes))
    shape = next(iter(by_action[actions[0]]["white"].values())).shape[:2]
    x0, y0, x1, y1 = _crop_window(union, shape, args.schematic_aspect, args.schematic_zoom)

    crops = {(action, tone): frames[frame_idx[action]][y0:y1, x0:x1]
             for action, by_tone in by_action.items() for tone, frames in by_tone.items()}

    slot = SLOT_CM["schematic"]
    fig_w = slot / IN
    fonts = _fonts(fig_w, slot, "swap_design_schematic")

    pad, tag_w, gap = 0.006, 0.085, 0.013
    img_w = (1.0 - 2 * pad - tag_w - 2 * gap) / 2
    img_h_in = img_w * fig_w / args.schematic_aspect
    header_in = fonts["label"] / 72 * 2.1
    row_gap_in = 0.16
    fig_h = header_in + 2 * img_h_in + row_gap_in + 0.10

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    img_hf = img_h_in / fig_h
    lefts = [pad + tag_w + gap + i * (img_w + gap) for i in range(len(actions))]

    for left, action in zip(lefts, actions):
        fig.text(left + img_w / 2, 1.0 - header_in / fig_h * 0.55, action,
                 ha="center", va="center", fontsize=fonts["label"], weight="bold",
                 color=TUD["navy"])

    top = 1.0 - header_in / fig_h
    for title, tone_of, accent, wash in rows:
        bottom = top - img_hf
        fig.patches.append(FancyBboxPatch(
            (pad, bottom), tag_w, img_hf,
            boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
            facecolor=accent, edgecolor="none", zorder=1))
        fig.text(pad + tag_w / 2, bottom + img_hf / 2, title, ha="center", va="center",
                 rotation=90, fontsize=fonts["label"], weight="bold", color="white",
                 zorder=3)
        for left, action in zip(lefts, actions):
            ax = fig.add_axes([left, bottom, img_w, img_hf], zorder=2)
            ax.imshow(crops[(action, tone_of[action])], interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(accent)
                spine.set_linewidth(2.4)
        top = bottom - row_gap_in / fig_h

    _save(fig, "swap_design_schematic")


def make_titlebar(args) -> None:
    """Split the four-figure SMPL strip into two halves for the title band.

    The cut is placed on a column of pure background so no figure and no label
    is clipped -- x=264 of 511 is the only fully-clear column near the middle,
    since the "celebrating" and "jumping" captions very nearly touch.

    The white background is keyed out so the halves sit on the blue band rather
    than in white boxes. Keying is done by flood-filling the background from the
    image border rather than by thresholding every light pixel, so the cream
    robe of the praying figure is not eaten along with it.
    """
    from scipy import ndimage

    src = Path(args.smpl_source)
    from PIL import Image

    rgba = np.array(Image.open(src).convert("RGBA"))
    near_white = rgba[..., :3].min(axis=2) > args.smpl_white_threshold
    labels, _ = ndimage.label(near_white)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    background = np.isin(labels, [v for v in border if v])
    rgba[..., 3] = np.where(background, 0, rgba[..., 3])

    cut = args.smpl_cut
    column = rgba[:, cut, 3]
    if column.max() != 0:
        raise SystemExit(f"column x={cut} is not empty; pick a clear cut column")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, half in (("smpl_title_left", rgba[:, :cut]), ("smpl_title_right", rgba[:, cut:])):
        # Trim the now-transparent margins so both halves butt up against their
        # own content and can be placed by height alone.
        cols = np.nonzero(half[..., 3].any(axis=0))[0]
        rows = np.nonzero(half[..., 3].any(axis=1))[0]
        crop = half[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        path = OUT_DIR / f"{name}.png"
        Image.fromarray(crop).save(path)
        print(f"[OK] {name}.png  {crop.shape[1]}x{crop.shape[0]}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — paired accuracy drop, transposed for a poster column
# ══════════════════════════════════════════════════════════════════════════════
def _paired_drop_matrix(csv_path: Path):
    """(b-c)/n per (action pair, model) from the raw per-direction accuracy
    counts. b-c equals correct_matched - correct_shifted exactly: a clip right
    under both tones cancels out of both, so the marginal counts carry the same
    difference as the paired ones."""
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = totals[(row["pair_tag"], row["model"])]
            item[0] += int(row["correct_matched"])
            item[1] += int(row["correct_shifted"])
            item[2] += int(row["n"])

    models = sorted({m for _pair, m in totals},
                    key=lambda m: (0, m) if m == "i3d_flow" else (1, m))
    matrix = np.full((len(PAIR_ORDER), len(models)), np.nan)
    for i, pair in enumerate(PAIR_ORDER):
        for j, model in enumerate(models):
            item = totals.get((pair, model))
            if item and item[2]:
                matrix[i, j] = (item[0] - item[1]) / item[2]
    return matrix, models


def make_heatmap(args) -> None:
    csv_path = ROOT / args.swap_root / "skin_tone_raw_accuracy_by_direction_testonly.csv"
    matrix, models = _paired_drop_matrix(csv_path)

    max_abs = max(0.05, float(np.nanmax(np.abs(matrix))))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    slot = SLOT_CM["heatmap"]
    fig_w = slot / IN
    fonts = _fonts(fig_w, slot, "pair_heatmap_paired_flip_rate_landscape")
    fig = plt.figure(figsize=(fig_w, fig_w * args.heatmap_aspect), dpi=200)
    grid = fig.add_gridspec(2, 1, height_ratios=[40, 1.6], hspace=0.045,
                            left=0.30, right=0.90, top=0.855, bottom=0.075)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[1, 0])

    im = ax.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_model_label(m) for m in models], fontsize=fonts["tick"],
                       rotation=40, ha="left", va="bottom", rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(PAIR_ORDER)))
    ax.set_yticklabels([_pretty_pair(p) for p in PAIR_ORDER], fontsize=fonts["tick"])
    ax.set_ylabel("Action pair", fontsize=fonts["label"], fontweight="bold")

    # Thin white gridlines between cells: with no separator the annotated
    # numbers ran into their neighbours, which is what made the grid feel busy.
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PAIR_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value != value:
                continue
            ax.text(j, i, f"{value:.02f}", ha="center", va="center",
                    fontsize=fonts["cell"],
                    color="#111111" if abs(value) < max_abs * 0.45 else "white")

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Paired accuracy drop $(b-c)/n$",
                   fontsize=fonts["cell"])
    cbar.ax.tick_params(labelsize=fonts["cell"])
    _save(fig, "pair_heatmap_paired_flip_rate_landscape")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — linear-probe drop per frozen backbone
# ══════════════════════════════════════════════════════════════════════════════
def make_probe(args) -> None:
    probe = json.loads((ROOT / args.probe_summary).read_text())

    rows = []
    for model, record in probe.items():
        if model == "hiera":
            continue  # superseded by hiera_large, shown below relabelled as "hiera"
        matched = record["eval_matched_unseen_ids"]
        shifted = record["eval_shifted_unseen_ids"]
        rows.append(dict(
            model=_model_label("hiera" if model == "hiera_large" else model),
            family=FAMILY[model], matched=matched, drop=matched - shifted,
            ci_lo=record.get("drop_unseen_ci_low", np.nan),
            ci_hi=record.get("drop_unseen_ci_high", np.nan),
        ))
    rows.sort(key=lambda r: r["drop"])

    slot = SLOT_CM["probe"]
    fig_w = slot / IN
    fonts = _fonts(fig_w, slot, "probe_drop_by_model_landscape")
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * args.probe_aspect), dpi=200)

    for i, row in enumerate(rows):
        ax.barh(i, -row["drop"], height=args.probe_bar_height,
                color=FAM_COLOR[row["family"]], edgecolor="white", linewidth=0.8)
        lo, hi = row["ci_lo"], row["ci_hi"]
        if lo == lo and hi == hi:
            ax.errorbar(-row["drop"], i,
                        xerr=[[hi - row["drop"]], [row["drop"] - lo]],
                        fmt="none", ecolor="#2b2b2b", elinewidth=1.6, capsize=0,
                        alpha=0.55, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=fonts["tick"])
    ax.tick_params(axis="x", labelsize=fonts["tick"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlim(-max(r["drop"] for r in rows) - args.probe_left_margin, 0.06)
    ax.set_xlabel("shifted $-$ matched  ($\\Delta$F1)", fontsize=fonts["label"])

    for i, row in enumerate(rows):
        hi = row["ci_hi"]
        outer = -(hi if hi == hi else row["drop"])
        ax.text(outer - 0.012, i, f"F1$_\\mathrm{{matched}}$={row['matched']:.2f}",
                va="center", ha="right", fontsize=fonts["annot"], color="#555555")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in FAM_COLOR.values()]
    fig.legend(handles, list(FAM_COLOR), fontsize=fonts["legend"], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, columnspacing=1.4,
               handlelength=1.4)
    fig.tight_layout(pad=0.4, rect=(0, 0.135, 1, 1))
    _save(fig, "probe_drop_by_model_landscape")


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — augmentation radar with the legend beneath the plot
# ══════════════════════════════════════════════════════════════════════════════
def make_radar(args) -> None:
    cached = json.loads((ROOT / args.radar_deltas).read_text())
    deltas = {tuple(key.split("|", 1)): value for key, value in cached.items()}

    n = len(RADAR_LABELS)
    theta = np.arange(n) * 2 * np.pi / n
    close = lambda a: np.concatenate([a, a[:1]])
    dmin = min(deltas.values())
    dmax = max(deltas.values())
    offset = max(0.03, -dmin * 1.3)
    r_max = offset + dmax * 1.25

    slot = SLOT_CM["radar"]
    fig_w = slot / IN
    fonts = _fonts(fig_w, slot, "augmentation_radar_landscape")
    fig = plt.figure(figsize=(fig_w, fig_w * args.radar_aspect), dpi=200)
    fig.patch.set_facecolor("white")
    # Explicit rect rather than add_subplot: the circle's diameter is the rect's
    # shorter side, so pinning the rect is what guarantees the radar fills the
    # column width instead of shrinking to whatever the legend leaves over.
    ax = fig.add_axes(args.radar_rect, projection="polar")
    ax.set_facecolor("white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(BASELINE_C)
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_ylim(0, r_max)

    d_ticks = [d for d in (-0.02, 0.0, 0.02, 0.04, 0.06) if 0 < offset + d < r_max]
    ax.set_yticks([offset + d for d in d_ticks])
    ax.set_yticklabels([("0" if d == 0 else f"{d:+.2f}") for d in d_ticks],
                       fontsize=fonts["cell"], color=INK_MUTED)
    ax.set_rlabel_position(90 / n)
    ax.set_xticks(theta)
    ax.set_xticklabels([("none (reference)" if l == "none" else l) for l in RADAR_LABELS],
                       fontsize=fonts["tick"], color=INK_2)
    ax.tick_params(axis="x", pad=args.radar_tick_pad)

    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(tt, np.full_like(tt, offset), color=INK_2, lw=3.0, zorder=2)
    for model in MODELS:
        vals = np.array([offset + deltas[(model, label)] for label in RADAR_LABELS])
        ax.plot(close(theta), close(vals), "-", color=MODEL_COLOR[model], lw=2.6,
                alpha=0.95, zorder=3)
        ax.plot(theta, vals, MODEL_MARKER[model], color=MODEL_COLOR[model],
                ms=args.radar_marker, mec="white", mew=1.2, zorder=4)

    handles = [plt.Line2D([], [], color=MODEL_COLOR[m], lw=2.6, marker=MODEL_MARKER[m],
                          mec="white", ms=args.radar_marker,
                          label=_model_label(m)) for m in MODELS]
    handles.append(plt.Line2D([], [], color=INK_2, lw=3.0, label="reference ($\\Delta$ = 0)"))
    # No in-figure title: the poster's section heading and caption already carry
    # it, and a title here collides with the "none (reference)" spoke label.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, fontsize=fonts["legend"], frameon=False,
               columnspacing=1.4, handlelength=1.8)
    _save(fig, "augmentation_radar_landscape")


BUILDERS = {
    "hero": make_hero,
    "schematic": make_schematic,
    "heatmap": make_heatmap,
    "probe": make_probe,
    "radar": make_radar,
    "titlebar": make_titlebar,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # No argparse `choices` here: with nargs="*" argparse validates its own empty
    # default against them, so `--all` alone would be rejected.
    ap.add_argument("figures", nargs="*",
                    help="Any of: " + ", ".join(sorted(BUILDERS)) + " (default: all).")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--background", default="konzerthaus")
    ap.add_argument("--base_id", type=int, default=0)
    ap.add_argument("--hero_base_id", type=int, default=7,
                    help="Motion instance for the hero strip; 7 is the teal-coat "
                         "actor used on the earlier poster.")
    ap.add_argument("--hero_frames", default="20,52,64")
    ap.add_argument("--hero_aspect", type=float, default=1.05,
                    help="Per-tone panel width/height. Below 1.0 the 16:9 render "
                         "is trimmed in x only, so the panels stack more densely.")
    ap.add_argument("--hero_zoom", type=float, default=1.62,
                    help="Crop height as a multiple of the actor's height.")
    ap.add_argument("--hero_panel_gap", type=int, default=44)
    ap.add_argument("--schematic_aspect", type=float, default=0.915)
    ap.add_argument("--schematic_zoom", type=float, default=1.45)
    ap.add_argument("--schematic_cartwheel_frame", type=int, default=22)
    ap.add_argument("--schematic_lunge_frame", type=int, default=88)
    ap.add_argument("--heatmap_aspect", type=float, default=1.05,
                    help="Figure height as a multiple of the column width.")
    ap.add_argument("--probe_aspect", type=float, default=1.05,
                    help="Taller than 14 bars strictly need: column 3's content "
                         "ends well above section 5's block. Deliberately short "
                         "of closing that gap -- some whitespace is wanted.")
    ap.add_argument("--probe_bar_height", type=float, default=0.64,
                    help="Lowered in step with probe_aspect: the added height "
                         "should become space between bars, not thicker bars.")
    ap.add_argument("--probe_left_margin", type=float, default=0.42)
    ap.add_argument("--radar_aspect", type=float, default=0.70)
    ap.add_argument("--radar_rect", type=float, nargs=4,
                    default=[0.02, 0.20, 0.96, 0.70])
    ap.add_argument("--radar_tick_pad", type=float, default=48.0)
    ap.add_argument("--radar_marker", type=float, default=13.0)
    ap.add_argument("--swap_root", default="out/skin_tone_probe_v7_cv")
    ap.add_argument("--probe_summary", default="out/linear_probes/_probe_summary_cv.json")
    ap.add_argument("--radar_deltas", default="out/poster_landscape/radar_scan_deltas.json")
    ap.add_argument("--smpl_source", default="poster/smpl.png")
    ap.add_argument("--smpl_cut", type=int, default=264,
                    help="Column to split the SMPL strip on; must be fully background.")
    ap.add_argument("--smpl_white_threshold", type=int, default=238)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    wanted = sorted(BUILDERS) if args.all or not args.figures else args.figures
    unknown = [name for name in wanted if name not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {sorted(BUILDERS)}")
    unknown = [name for name in wanted if name not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {sorted(BUILDERS)}")
    for name in wanted:
        BUILDERS[name](args)


if __name__ == "__main__":
    main()
