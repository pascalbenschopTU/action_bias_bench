"""Visual debug tool for the RGB color-augmentation pipeline.

Loads one frame from a video and renders a grid comparing the original
against 3 random draws each of weak ColorJitter, strong ColorJitter,
grayscale, and Planckian jitter -- the same parameter values used in the
skin-tone augmentation-mitigation sweep
(jobs/bias/run_skin_tone_augmentation_sweep.sbatch).

Usage:
  python data/color_jitter_analysis.py /path/to/video.mp4 --frame_index 0 --out debug.png
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu

from augment import planckian_gains, sample_log_uniform_temperature

WEAK_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
STRONG_JITTER = dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
PLANCKIAN_MIN_K, PLANCKIAN_MAX_K, PLANCKIAN_REFERENCE_K = 3000.0, 12000.0, 6504.0


def load_frame(video_path: str, frame_index: int) -> torch.Tensor:
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_index = min(max(frame_index, 0), len(vr) - 1)
    frame = vr[frame_index].asnumpy()  # (H, W, 3) uint8, RGB
    return torch.from_numpy(frame).permute(2, 0, 1).contiguous()  # (3, H, W) uint8


def apply_color_jitter(frame: torch.Tensor, params: dict) -> torch.Tensor:
    """Sample ColorJitter params once and apply the same draw to the frame --
    mirrors RGBVideoClipDataset._apply_color_jitter_consistent in data/rgb.py."""
    jitter = T.ColorJitter(**params)
    fn_idx, b, c, s, h = T.ColorJitter.get_params(
        jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
    )
    out = frame
    for fn_id in fn_idx:
        if fn_id == 0 and b is not None:
            out = TF.adjust_brightness(out, b)
        elif fn_id == 1 and c is not None:
            out = TF.adjust_contrast(out, c)
        elif fn_id == 2 and s is not None:
            out = TF.adjust_saturation(out, s)
        elif fn_id == 3 and h is not None:
            out = TF.adjust_hue(out, h)
    return out


def apply_grayscale(frame: torch.Tensor) -> torch.Tensor:
    return TF.rgb_to_grayscale(frame, num_output_channels=3)


def apply_planckian(frame: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Mirrors RGBVideoClipDataset._apply_planckian_jitter in data/rgb.py."""
    temperature_k = sample_log_uniform_temperature(rng, PLANCKIAN_MIN_K, PLANCKIAN_MAX_K)
    gains = planckian_gains(temperature_k, PLANCKIAN_REFERENCE_K)
    gain_tensor = torch.tensor(gains, dtype=torch.float32).view(3, 1, 1)
    return (frame.to(torch.float32) * gain_tensor).clamp_(0, 255).to(torch.uint8)


def to_numpy_img(frame: torch.Tensor) -> np.ndarray:
    return frame.permute(1, 2, 0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_path", type=str)
    ap.add_argument("--frame_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="color_jitter_analysis.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = load_frame(args.video_path, args.frame_index)
    rng = np.random.default_rng(args.seed)

    rows = [
        ("weak jitter", lambda: apply_color_jitter(frame, WEAK_JITTER)),
        ("strong jitter", lambda: apply_color_jitter(frame, STRONG_JITTER)),
        ("grayscale", lambda: apply_grayscale(frame)),
        ("planckian jitter", lambda: apply_planckian(frame, rng)),
    ]

    fig, axes = plt.subplots(
        len(rows), 4, figsize=(14, 2.6 * len(rows)),
        gridspec_kw={"hspace": -0.05, "wspace": 0.03},
    )
    for row_idx, (name, transform_fn) in enumerate(rows):
        axes[row_idx, 0].imshow(to_numpy_img(frame))
        axes[row_idx, 0].set_ylabel(name, fontsize=11, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        for col_idx in range(1, 4):
            axes[row_idx, col_idx].imshow(to_numpy_img(transform_fn()))
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
