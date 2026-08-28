"""Build a frame grid: same action/background/motion instance, several
timepoints (rows) x several skin tones (columns), for a qualitative "does
this look like it could be misread" figure.

Reads raw renders directly from the dataset (no cached embeddings/manifests
needed): {dataset_root}/{background}/__generated_synthetic_videos/{action}/
{action}_{base_id}_modified_{variant}.mp4

Usage (run from the ActionBiasBench directory, in an env with cv2+matplotlib,
e.g. `conda run -n demo python scripts/make_skin_tone_frame_grid.py`):
    python scripts/make_skin_tone_frame_grid.py \
        --dataset_root /Volumes/MoDDL/Pascal/motion_only_AR/datasets/skin_tone_actions/camera_far \
        --background konzerthaus --action cartwheel --base_id 0 \
        --out_dir out/frame_grids
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paper table order (Fig. 1 / Supp. Table 1): light-to-dark, White/Asian
# then Indian/African, not the schema.py VARIANT_ORDER -- kept consistent
# with the rest of the paper's figures/tables.
DEFAULT_VARIANTS = ["white", "asian", "indian", "african"]
VARIANT_LABEL = {
    "white": "White", "asian": "Asian", "indian": "Indian", "african": "African",
}


def video_path(dataset_root: Path, background: str, action: str, base_id: int, variant: str) -> Path:
    return (dataset_root / background / "__generated_synthetic_videos" / action
            / f"{action}_{base_id}_modified_{variant}.mp4")


def read_frame_at_fraction(path: Path, fraction: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(max(int(round(fraction * (n_frames - 1))), 0), n_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx}/{n_frames} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def center_crop(frame: np.ndarray, keep_frac: float) -> np.ndarray:
    h, w = frame.shape[:2]
    ch, cw = int(h * keep_frac), int(w * keep_frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return frame[y0:y0 + ch, x0:x0 + cw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--background", default="konzerthaus")
    ap.add_argument("--action", default="cartwheel")
    ap.add_argument("--base_id", type=int, default=0)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--fractions", default="0.2,0.5,0.8",
                     help="Comma-separated points through the clip (0=first frame, 1=last).")
    ap.add_argument("--center_crop", type=float, default=None,
                     help="Optional center-crop fraction, e.g. 0.5 keeps the middle 50%%.")
    ap.add_argument("--out_dir", type=Path, default=Path("out/frame_grids"))
    ap.add_argument("--out_name", default=None,
                     help="Defaults to {action}_{background}_skin_tone_grid.")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    fractions = [float(f.strip()) for f in args.fractions.split(",") if f.strip()]
    out_name = args.out_name or f"{args.action}_{args.background}_skin_tone_grid"

    n_rows, n_cols = len(fractions), len(variants)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.6 * n_rows * 0.75 + 0.4),
        dpi=220,
        squeeze=False,
    )

    for j, variant in enumerate(variants):
        path = video_path(args.dataset_root, args.background, args.action, args.base_id, variant)
        if not path.exists():
            raise FileNotFoundError(f"Missing render: {path}")
        for i, frac in enumerate(fractions):
            frame = read_frame_at_fraction(path, frac)
            if args.center_crop:
                frame = center_crop(frame, args.center_crop)
            ax = axes[i][j]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(VARIANT_LABEL.get(variant, variant), fontsize=15, fontweight="bold")

    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.90, bottom=0.02)
    fig.suptitle(
        f"Same {args.action} motion, {args.background} background -- only skin tone changes",
        fontsize=16, fontweight="bold", y=0.985,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{out_name}.pdf"
    png_path = args.out_dir / f"{out_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
