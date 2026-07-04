"""
Skin-tone bias analysis via cross-video embedding distances.

For each clip, computes a separation ratio:
    r = d_skin / d_action

    d_skin   = mean cosine distance to same action + same performer, opposite skin group
               (e.g. lunge_7_african vs lunge_7_white/asian)
    d_action = mean cosine distance to the paired action, all performers + all skins
               (e.g. lunge_7_african vs all cartwheel clips)

Interpretation:
    r << 1  skin swap is a tiny perturbation relative to the action boundary → safe
    r ~  1  skin swap rivals the action boundary → potentially problematic
    r >  1  skin swap pushes the clip past the action boundary → bias

Also computes cross-video temporal alignment:
    align(A, B) = mean diagonal of cosine cross-similarity matrix (T_A, T_B),
                  after resampling both to the same number of steps.
    Higher = more similar temporal motion structure.

Usage (run from the ActionBiasBench directory):
    python scripts/skin_tone_bias_analysis.py --model dinov2
    python scripts/skin_tone_bias_analysis.py --model vjepa2 --frames 64
    python scripts/skin_tone_bias_analysis.py --model clip --frames -1
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.huggingface_models as hf

# ── experiment constants ──────────────────────────────────────────────────────

ACTION_PAIRS = [
    ("lunge", "cartwheel"),
    ("squat", "tie"),
    ("clap", "celebrate"),
    ("dribble", "golf"),
    ("yawn", "fish"),
]
ALL_ACTIONS    = sorted({a for pair in ACTION_PAIRS for a in pair})
PAIR_LOOKUP    = {a: b for a, b in ACTION_PAIRS} | {b: a for a, b in ACTION_PAIRS}

LIGHT_VARIANTS = ["white", "asian"]
DARK_VARIANTS  = ["african", "indian"]
ALL_VARIANTS   = LIGHT_VARIANTS + DARK_VARIANTS

BACKGROUNDS = ["autumn_hockey", "konzerthaus", "stadium_01"]
ALL_IDS     = list(range(10))

MODELS = ["clip", "dinov2", "dinov3", "siglip", "eva02", "hiera", "vjepa2", "tc_clip"]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_frames(path: Path, n: int) -> np.ndarray | None:
    """Uniformly sample n frames from video. n <= 0 means all frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    if n <= 0:
        n = total
    n = min(n, total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else None


def video_path(dataset_root: Path, action: str, base_id: int, variant: str, bg: str) -> Path:
    fname = f"{action}_{base_id}_modified_{variant}.mp4"
    return dataset_root / bg / "__generated_synthetic_videos" / action / fname


def l2_norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1-D vectors (already L2-normalised)."""
    return float(1.0 - np.dot(a, b))


def cross_align(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Mean diagonal of cross-similarity matrix between two (T, D) sequences.
    Both expected to be L2-normalised per row.
    Resample to the shorter length so dimensions match.
    """
    T = min(seq_a.shape[0], seq_b.shape[0])
    a = seq_a[np.linspace(0, seq_a.shape[0] - 1, T, dtype=int)]
    b = seq_b[np.linspace(0, seq_b.shape[0] - 1, T, dtype=int)]
    return float(np.diag(a @ b.T).mean())


def skin_group(variant: str) -> str:
    return "light" if variant in LIGHT_VARIANTS else "dark"


def opposite_variants(variant: str) -> list[str]:
    return DARK_VARIANTS if variant in LIGHT_VARIANTS else LIGHT_VARIANTS


class ClipEmbeddingDataset(Dataset):
    """Video decode/cache-read dataset; model inference stays in the main process."""

    def __init__(self, dataset_root: Path, n_frames: int, cache_dir: Path, model_name: str):
        self.n_frames = n_frames
        self.items = []

        for action in ALL_ACTIONS:
            for base_id in ALL_IDS:
                for variant in ALL_VARIANTS:
                    for bg in BACKGROUNDS:
                        path = video_path(dataset_root, action, base_id, variant, bg)
                        if not path.exists():
                            continue

                        cache_key = f"{model_name}_{action}_{base_id}_{variant}_{bg}.npz"
                        cache_path = cache_dir / cache_key
                        self.items.append((action, base_id, variant, bg, path, cache_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        action, base_id, variant, bg, path, cache_path = self.items[idx]
        key = (action, base_id, variant, bg)

        if cache_path.exists():
            with np.load(cache_path) as d:
                return {
                    "key": key,
                    "cached": True,
                    "mean": d["mean"],
                    "seq": d["seq"],
                }

        frames = load_frames(path, self.n_frames)
        return {
            "key": key,
            "cached": False,
            "frames": frames,
            "cache_path": cache_path,
        }


def collate_clip_items(batch: list[dict]) -> list[dict]:
    return batch


def init_loader_worker(_worker_id: int) -> None:
    cv2.setNumThreads(0)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Skin-tone bias analysis via embedding distances.")
    ap.add_argument("--model",        required=True)
    ap.add_argument("--frames",       type=int, default=16,
                    help="Frames to sample per video (-1 = all). For vjepa2 use 64.")
    ap.add_argument("--dataset_root", default="../../datasets/skin_tone_actions/camera_far",
                    help="Path to the camera_far video root.")
    ap.add_argument("--out_dir",      default="out/bias_analysis")
    ap.add_argument("--cache_dir",    default="out/bias_analysis/embeddings",
                    help="Directory to cache per-video embeddings (NPZ). Reused on re-runs.")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="CPU DataLoader workers for video decode/cache reads.")
    ap.add_argument("--batch_clips", type=int, default=1,
                    help="Number of uncached clips to combine per GPU encode step.")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="DataLoader prefetch factor when num_workers > 0.")
    ap.add_argument("--pin_memory", action="store_true",
                    help="Enable DataLoader pinned-memory handling.")
    return ap.parse_args()


def embed_all_clips(
    dataset_root: Path,
    encode_fn,
    model,
    processor,
    device: torch.device,
    n_frames: int,
    cache_dir: Path,
    model_name: str,
    batch_clips: int = 1,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> dict:
    """
    Returns dict: (action, base_id, variant, bg) -> {"mean": (D,), "seq": (T, D)}
    Caches each embedding as an NPZ so the model doesn't need to re-run.
    """
    embeddings = {}
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = ClipEmbeddingDataset(
        dataset_root=dataset_root,
        n_frames=n_frames,
        cache_dir=cache_dir,
        model_name=model_name,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": max(1, batch_clips),
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "collate_fn": collate_clip_items,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = init_loader_worker

    loader = DataLoader(**loader_kwargs)

    print(
        f"Embedding dataset clips: {len(dataset)}  batch_clips={max(1, batch_clips)}  "
        f"num_workers={max(0, num_workers)}",
        flush=True,
    )

    for batch in loader:
        uncached = []

        for item in batch:
            if item["cached"]:
                embeddings[item["key"]] = {
                    "mean": item["mean"],
                    "seq": item["seq"],
                }
            elif item["frames"] is not None:
                uncached.append(item)

        if not uncached:
            continue

        if model_name in {"vjepa2", "tc_clip"}:
            for item in uncached:
                seq = encode_fn(item["frames"], model, processor, device=device)
                mean_emb = l2_norm(seq.mean(axis=0))
                np.savez(item["cache_path"], mean=mean_emb, seq=seq)
                embeddings[item["key"]] = {"mean": mean_emb, "seq": seq}
                action, base_id, variant, bg = item["key"]
                print(f"  embedded {action} id={base_id} {variant} {bg}: {seq.shape}", flush=True)
            continue

        frame_counts = [item["frames"].shape[0] for item in uncached]
        all_frames = np.concatenate([item["frames"] for item in uncached], axis=0)
        all_seq = encode_fn(all_frames, model, processor, device=device)

        offset = 0
        for item, count in zip(uncached, frame_counts):
            seq = all_seq[offset:offset + count]
            offset += count

            mean_emb = l2_norm(seq.mean(axis=0))
            np.savez(item["cache_path"], mean=mean_emb, seq=seq)
            embeddings[item["key"]] = {"mean": mean_emb, "seq": seq}
            action, base_id, variant, bg = item["key"]
            print(f"  embedded {action} id={base_id} {variant} {bg}: {seq.shape}", flush=True)

    return embeddings


def compute_separation_ratios(embeddings: dict) -> list[dict]:
    rows = []

    for (action, base_id, variant, bg), clip in embeddings.items():
        partner = PAIR_LOOKUP.get(action)
        if partner is None:
            continue

        emb_mean = clip["mean"]
        emb_seq  = clip["seq"]

        # d_skin: same action, same performer, opposite skin group
        skin_dists, skin_aligns = [], []
        for ov in opposite_variants(variant):
            for bg2 in BACKGROUNDS:
                key = (action, base_id, ov, bg2)
                if key not in embeddings:
                    continue
                other = embeddings[key]
                skin_dists.append(cosine_dist(emb_mean, other["mean"]))
                skin_aligns.append(cross_align(emb_seq, other["seq"]))

        # d_action: partner action, all performers, all skins
        action_dists, action_aligns = [], []
        for pid in ALL_IDS:
            for pv in ALL_VARIANTS:
                for bg2 in BACKGROUNDS:
                    key = (partner, pid, pv, bg2)
                    if key not in embeddings:
                        continue
                    other = embeddings[key]
                    action_dists.append(cosine_dist(emb_mean, other["mean"]))
                    action_aligns.append(cross_align(emb_seq, other["seq"]))

        if not skin_dists or not action_dists:
            continue

        d_skin   = float(np.mean(skin_dists))
        d_action = float(np.mean(action_dists))
        r        = d_skin / (d_action + 1e-8)

        a_skin   = float(np.mean(skin_aligns))
        a_action = float(np.mean(action_aligns))

        rows.append({
            "action":      action,
            "partner":     partner,
            "base_id":     base_id,
            "variant":     variant,
            "skin_group":  skin_group(variant),
            "background":  bg,
            "d_skin":      round(d_skin, 6),
            "d_action":    round(d_action, 6),
            "r":           round(r, 6),
            "align_skin":  round(a_skin, 6),
            "align_action": round(a_action, 6),
        })

    return rows


def print_summary(rows: list[dict], model_name: str) -> None:
    print(f"\n=== Separation ratio r = d_skin / d_action  [{model_name}] ===")
    print("r < 1 → skin swap smaller than action gap (safe)")
    print("r > 1 → skin swap rivals action gap (problematic)\n")

    by_action = defaultdict(list)
    for row in rows:
        by_action[row["action"]].append(row["r"])

    print(f"{'Action':<14} {'mean_r':>8}  {'max_r':>8}  {'n':>6}")
    print("-" * 42)
    for action in sorted(by_action, key=lambda a: -np.mean(by_action[a])):
        rs = np.array(by_action[action])
        print(f"{action:<14} {rs.mean():>8.4f}  {rs.max():>8.4f}  {len(rs):>6}")

    flagged = sorted([r for r in rows if r["r"] > 0.5], key=lambda x: -x["r"])
    if flagged:
        print(f"\n=== High-r clips (r > 0.5) — skin swap ≥ 50% of action distance ===")
        print(f"{'action':<12} {'id':>4} {'variant':<10} {'background':<18} {'r':>8}  {'d_skin':>8}  {'d_action':>8}")
        for row in flagged[:30]:
            print(f"{row['action']:<12} {row['base_id']:>4} {row['variant']:<10} "
                  f"{row['background']:<18} {row['r']:>8.4f}  {row['d_skin']:>8.4f}  {row['d_action']:>8.4f}")

    print(f"\n=== By skin group ===")
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["skin_group"]].append(row["r"])
    for group, rs in by_group.items():
        arr = np.array(rs)
        print(f"  {group}: mean_r={arr.mean():.4f}  max_r={arr.max():.4f}  n={len(arr)}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir  = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(
        f"device: {device}  model: {args.model}  frames: {args.frames}  "
        f"batch_clips: {args.batch_clips}  num_workers: {args.num_workers}"
    )
    hf.DEVICE = device

    import models.torchvision_models as tv
    load_fn   = getattr(hf, f"load_{args.model}", None) or getattr(tv, f"load_{args.model}")
    encode_fn = getattr(hf, f"encode_{args.model}", None) or getattr(tv, f"encode_{args.model}")
    print(f"Loading {args.model} weights...")
    model_obj, processor = load_fn()

    # embed
    print("Embedding clips (cached after first run)...")
    embeddings = embed_all_clips(
        dataset_root=dataset_root,
        encode_fn=encode_fn,
        model=model_obj,
        processor=processor,
        device=device,
        n_frames=args.frames,
        cache_dir=cache_dir / args.model,
        model_name=args.model,
        batch_clips=args.batch_clips,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )
    print(f"Total clips: {len(embeddings)}")

    # compute separation ratios
    print("Computing separation ratios...")
    rows = compute_separation_ratios(embeddings)

    # save
    out_csv = out_dir / f"bias_{args.model}.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {out_csv}")

    print_summary(rows, args.model)


if __name__ == "__main__":
    main()
