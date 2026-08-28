"""Hero figure: 3 timepoints through one cartwheel repetition, each panel showing
the White and African renders of that same frame index side by side (identical
pose/camera/lighting -- only skin tone differs).

Motivation: the paper's intro hook is that an action-recognition model can misread
the same motion differently depending on skin tone. This figure makes that
directly visible: same choreography, same instant, only the texture changes.

The clip contains several cartwheel repetitions; frame indices are chosen by hand
within the *first* repetition (found via background-subtraction bbox scan --
see out/tmp_scripts/bbox_scan.py) so the three panels show progress through one
cartwheel, not three different reps.

Crops are computed directly from full-resolution decoded frames (no manifest /
cache / resize in the pipeline) using a shared bounding box per timepoint (union
of both tones' actor masks, since the two renders share identical geometry) so
both tones in a panel are pixel-aligned and neither is resampled to match the
other.

Usage (run from the ActionBiasBench directory, env with cv2+matplotlib):
    python scripts/make_cartwheel_tone_pairs.py \
        --dataset_root /Volumes/MoDDL/Pascal/motion_only_AR/datasets/skin_tone_actions/camera_far \
        --background konzerthaus --action cartwheel --base_id 0 \
        --frames 4,16,26 --out_dir out/frame_grids
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TONES = ["white", "african"]
TONE_LABEL = {"white": "White", "african": "African"}


def video_path(dataset_root: Path, background: str, action: str, base_id: int, variant: str) -> Path:
    return (dataset_root / background / "__generated_synthetic_videos" / action
            / f"{action}_{base_id}_modified_{variant}.mp4")


def read_frame(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--background", default="konzerthaus")
    ap.add_argument("--action", default="cartwheel")
    ap.add_argument("--base_id", type=int, default=0)
    ap.add_argument("--frames", default="4,16,26",
                     help="Comma-separated absolute frame indices, one per panel, "
                          "chosen within a single repetition of the action.")
    ap.add_argument("--margin_frac", type=float, default=0.18,
                     help="Padding added around the detected actor bbox, as a "
                          "fraction of the bbox size.")
    ap.add_argument("--width_crop_left_frac", type=float, default=0.0)
    ap.add_argument("--width_crop_right_frac", type=float, default=0.0)
    ap.add_argument("--height_crop_top_frac", type=float, default=0.0)
    ap.add_argument("--height_crop_bottom_frac", type=float, default=0.0)
    ap.add_argument("--gap_px", type=int, default=14, help="Gap between the two tones within a panel.")
    ap.add_argument("--panel_gap_px", type=int, default=36, help="Gap between panels.")
    ap.add_argument("--out_dir", type=Path, default=Path("out/frame_grids"))
    ap.add_argument("--out_name", default=None)
    args = ap.parse_args()

    frame_idxs = [int(f.strip()) for f in args.frames.split(",") if f.strip()]
    out_name = args.out_name or f"{args.action}_{args.background}_tone_pairs"

    paths = {t: video_path(args.dataset_root, args.background, args.action, args.base_id, t) for t in TONES}
    for t, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing render: {p}")

    all_frames = {idx: {t: read_frame(paths[t], idx) for t in TONES} for idx in frame_idxs}

    h, w = all_frames[frame_idxs[0]][TONES[0]].shape[:2]
    crop_windows = {idx: (0, 0, w, h) for idx in frame_idxs}

    trimmed = {}
    for idx, (x0, y0, x1, y1) in crop_windows.items():
        width = x1 - x0
        height = y1 - y0
        trimmed[idx] = (x0 + width * args.width_crop_left_frac, y0 + height * args.height_crop_top_frac,
                            x1 - width * args.width_crop_right_frac, y1 - height * args.height_crop_bottom_frac)
    crop_windows = trimmed

    panels = []
    for idx in frame_idxs:
        x0, y0, x1, y1 = crop_windows[idx]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        crops = [all_frames[idx][t][y0:y1, x0:x1] for t in TONES]
        h = min(c.shape[0] for c in crops)
        w = min(c.shape[1] for c in crops)
        crops = [c[:h, :w] for c in crops]

        gap = np.full((h, args.gap_px, 3), 255, dtype=np.uint8)
        panel = np.concatenate([crops[0], gap, crops[1]], axis=1)
        panels.append(panel)
        print(f"frame {idx}: crop {w}x{h} per tone (bbox x[{x0},{x1}] y[{y0},{y1}])")

    # Pad shorter panels with white space at the top (not crop the taller
    # ones) so every panel keeps its full pose and all panels share a common
    # ground line at the bottom. A no-op when --shared_crop keeps every
    # panel identically sized already.
    panel_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < panel_h:
            pad = np.full((panel_h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
            p = np.concatenate([pad, p], axis=0)
        padded.append(p)
    panels = padded
    panel_gap = np.full((panel_h, args.panel_gap_px, 3), 255, dtype=np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, panel_gap, p], axis=1)

    dpi = 300
    fig_w, fig_h = strip.shape[1] / dpi, strip.shape[0] / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(strip, interpolation="none")
    ax.axis("off")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{out_name}.png"
    pdf_path = args.out_dir / f"{out_name}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)
    print(f"panel size: {strip.shape[1]}x{strip.shape[0]} px")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
