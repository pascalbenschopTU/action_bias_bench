"""Debug script: decode video(s), run one foundation model, print output shape.
With --video2 and --out_dir, also plots the SSM of both videos side by side.

Usage:
    python scripts/debug_hf_model.py --video path/to/video.mp4 --model clip
    python scripts/debug_hf_model.py --video v1.mp4 --video2 v2.mp4 --model dinov2 --out_dir out/ssm
    python scripts/debug_hf_model.py --video v1.mp4 --video2 v2.mp4 --model vjepa2 --frames 64 --out_dir out/ssm
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.huggingface_models as hf

MODELS = ["clip", "dinov2", "dinov3", "siglip", "eva02", "hiera", "vjepa2"]

ap = argparse.ArgumentParser()
ap.add_argument("--video",    required=True)
ap.add_argument("--video2",   default=None)
ap.add_argument("--model",    required=True, choices=MODELS)
ap.add_argument("--frames",   type=int, default=16, help="frames to sample (use 64 for vjepa2, -1 or 0 for all frames)")
ap.add_argument("--out_dir",  default=None)
args = ap.parse_args()

# ── device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device: {device}")
hf.DEVICE = device


# ── helpers ───────────────────────────────────────────────────────────────────
def load_frames(path: str, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = total
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return np.stack(frames)   # (T, H, W, 3)


def ssm(emb: np.ndarray) -> np.ndarray:
    """(T, D) L2-normalised -> (T, T) cosine similarity matrix."""
    return emb @ emb.T


# ── load frames ───────────────────────────────────────────────────────────────
videos = [args.video] + ([args.video2] if args.video2 else [])
embeddings = []

load_fn   = getattr(hf, f"load_{args.model}")
encode_fn = getattr(hf, f"encode_{args.model}")
print(f"loading {args.model} ...")
model, processor = load_fn()

for path in videos:
    frames_bgr = load_frames(path, args.frames)
    print(f"video:  {path}  frames={frames_bgr.shape}")
    emb = encode_fn(frames_bgr, model, processor, device=device)
    print(f"output: {emb.shape}  dtype={emb.dtype}")
    embeddings.append(emb)

# ── SSM plot ──────────────────────────────────────────────────────────────────
if args.video2 and args.out_dir:
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [Path(args.video).stem, Path(args.video2).stem]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, emb, label in zip(axes, embeddings, labels):
        m = ssm(emb)
        im = ax.imshow(m, cmap="RdBu_r")
        ax.set_title(f"{label}\n[{m.min():.3f}, {m.max():.3f}]", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("frame")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"SSM — {args.model}", fontsize=11)
    fig.tight_layout()

    out_path = out_dir / f"ssm_{args.model}_{labels[0]}_vs_{labels[1]}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")
