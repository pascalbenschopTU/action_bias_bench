"""
Quantify the synthetic-to-real domain gap for the BEDLAM-rendered skin-tone
actions, addressing the reviewer request for quantitative evidence of how
close the synthetic clips are to a real HAR distribution (Kinetics-400).

Reuses the exact DINOv2 extractor already used for the SSM/skin-tone-bias
analysis (models.huggingface_models.encode_dinov2, 16 uniformly sampled
frames, CLS token, L2-normalised, mean-pooled over frames -- see
scripts/skin_tone_bias_analysis.py). Synthetic-side embeddings are read from
the existing cache (out/bias_analysis/embeddings/dinov2); real-side clips are
sampled from the local Kinetics-400 val split for the closest matching class
and embedded/cached the same way.

For each action we report a distance ratio, analogous to the paper's own
d_skin/d_action separation ratio:

    gap_ratio = d_cross(real, synthetic) / d_real_intra(real, real)

gap_ratio ~ 1   synthetic clips sit inside the natural spread of real clips
gap_ratio >> 1  synthetic clips are further from real clips than real clips
                are from each other (a real domain gap)

Usage (run from the ActionBiasBench directory):
    python scripts/measure_synthetic_real_gap.py
    python scripts/measure_synthetic_real_gap.py --n_real 50 --frames 16
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.huggingface_models as hf  # noqa: E402
from scripts.skin_tone_bias_analysis import load_frames, l2_norm  # noqa: E402

# action tag (as used in the synthetic dataset / cache filenames) -> closest
# Kinetics-400 val class folder. golf/fish do not have an exact 1:1 class in
# K400 (K400 splits golf into driving/chipping/putting, and "fish" into
# catching/feeding/ice_fishing); we pick the closest full-body-motion analogue
# and flag this explicitly rather than pretending it's an exact match.
ACTION_TO_K400 = {
    "squat": "squat",
    "lunge": "lunge",
    "cartwheel": "cartwheeling",
    "clap": "clapping",
    "celebrate": "celebrating",
    "dribble": "dribbling_basketball",
    "yawn": "yawning",
    "tie": "tying_tie",
    "golf": "golf_driving",
    "fish": "catching_fish",
}
APPROXIMATE_MATCHES = {"golf", "fish"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k400_root", default="../../datasets/Kinetics/k400/val")
    ap.add_argument("--synthetic_cache", default="out/bias_analysis/embeddings/dinov2")
    ap.add_argument("--real_cache", default="out/bias_analysis/embeddings_real_k400/dinov2")
    ap.add_argument("--out_dir", default="out/bias_analysis")
    ap.add_argument("--n_real", type=int, default=50, help="Max real clips per class.")
    ap.add_argument("--frames", type=int, default=16, help="Frames per clip (matches synthetic-side default).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return ap.parse_args()


def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embed_real_clips(k400_root: Path, k400_class: str, n_real: int, n_frames: int,
                      cache_dir: Path, model, processor, device) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    class_dir = k400_root / k400_class
    paths = sorted(class_dir.glob("*.mp4"))[:n_real]

    embs = []
    for path in paths:
        cache_path = cache_dir / f"{k400_class}_{path.stem}.npz"
        if cache_path.exists():
            with np.load(cache_path) as d:
                embs.append(d["mean"])
            continue
        frames = load_frames(path, n_frames)
        if frames is None:
            print(f"  [skip] could not decode {path.name}")
            continue
        seq = hf.encode_dinov2(frames, model, processor, device=device)
        mean_emb = l2_norm(seq.mean(axis=0))
        np.savez(cache_path, mean=mean_emb)
        embs.append(mean_emb)
        print(f"  embedded real {k400_class}/{path.name}: {seq.shape}")
    return np.stack(embs) if embs else np.zeros((0, 1024), dtype=np.float32)


def load_synthetic_clip_means(synthetic_cache: Path, action: str) -> np.ndarray:
    embs = []
    for npz_path in sorted(synthetic_cache.glob(f"dinov2_{action}_*.npz")):
        with np.load(npz_path) as d:
            embs.append(d["mean"])
    return np.stack(embs) if embs else np.zeros((0, 1024), dtype=np.float32)


def mean_pairwise_cosine_dist(a: np.ndarray, b: np.ndarray | None = None) -> float:
    """Mean cosine distance over all pairs. If b is None, all distinct pairs within a."""
    if b is None:
        sims = a @ a.T
        n = a.shape[0]
        if n < 2:
            return float("nan")
        iu = np.triu_indices(n, k=1)
        return float(np.mean(1.0 - sims[iu]))
    sims = a @ b.T
    return float(np.mean(1.0 - sims))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device={device}")

    k400_root = (ROOT / args.k400_root).resolve() if not Path(args.k400_root).is_absolute() else Path(args.k400_root)
    synthetic_cache = ROOT / args.synthetic_cache
    real_cache = ROOT / args.real_cache

    model, processor = hf.load_dinov2()
    model = model.to(device)

    rows = []
    for action, k400_class in ACTION_TO_K400.items():
        print(f"\n=== {action} <-> {k400_class} ===")
        real = embed_real_clips(k400_root, k400_class, args.n_real, args.frames,
                                 real_cache, model, processor, device)
        synth = load_synthetic_clip_means(synthetic_cache, action)
        if real.shape[0] < 2 or synth.shape[0] < 2:
            print(f"  [skip] insufficient clips: real={real.shape[0]} synth={synth.shape[0]}")
            continue

        d_real_intra = mean_pairwise_cosine_dist(real)
        d_synth_intra = mean_pairwise_cosine_dist(synth)
        d_cross = mean_pairwise_cosine_dist(real, synth)
        gap_ratio = d_cross / d_real_intra if d_real_intra > 0 else float("nan")

        rows.append(dict(
            action=action, k400_class=k400_class,
            approximate_match=action in APPROXIMATE_MATCHES,
            n_real=real.shape[0], n_synth=synth.shape[0],
            d_real_intra=d_real_intra, d_synth_intra=d_synth_intra,
            d_cross=d_cross, gap_ratio=gap_ratio,
        ))
        print(f"  n_real={real.shape[0]} n_synth={synth.shape[0]}  "
              f"d_real_intra={d_real_intra:.4f}  d_synth_intra={d_synth_intra:.4f}  "
              f"d_cross={d_cross:.4f}  gap_ratio={gap_ratio:.3f}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = out_dir / "synthetic_real_domain_gap.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    ratios = [r["gap_ratio"] for r in rows]
    print(f"\nmean gap_ratio over {len(ratios)} actions: {np.mean(ratios):.3f}  "
          f"(median {np.median(ratios):.3f}, min {np.min(ratios):.3f}, max {np.max(ratios):.3f})")


if __name__ == "__main__":
    main()
