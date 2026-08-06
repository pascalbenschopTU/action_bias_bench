"""
Diagnostic for the TC-CLIP anomaly discussed in Section 3.2 (frozen-feature
results): TC-CLIP's SSM ratio (computed on the untouched per-frame sequence)
is the 2nd-highest of all models, but its linear-probe drop (computed on the
*mean-pooled-over-time* embedding) is near the bottom, close to V-JEPA2. The
paper currently *proposes* that Kinetics-400 fine-tuning reduces sensitivity
specifically in the temporally-pooled representation without removing it from
the frame-level features -- this script tests that directly rather than
leaving it as an unverified hypothesis.

If the hypothesis is right, replacing mean-pooling with max-pooling over the
frame sequence (same cached per-frame embeddings, no new feature extraction)
should recover more of TC-CLIP's drop, since max-pooling does not let a
tone-correlated signal that varies in sign/magnitude across frames cancel
out. CLIP is included as a control: it has no joint temporal modelling
(each frame's CLIP embedding is computed independently), so its drop should
be comparatively insensitive to how frames are pooled.

Reuses run_probe_for_pair from train_embedding_linear_probe.py unchanged --
only the frame-pooling function is swapped (monkeypatched), so everything
else (train/eval manifests, logistic regression, metrics, output format)
stays identical to the model's regular mean-pooled probe.

Usage (run from the ActionBiasBench directory):
    python benchmarks/skin_tone/pooling_ablation_probe.py
    python benchmarks/skin_tone/pooling_ablation_probe.py --models clip,tc_clip,dinov2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_embedding_linear_probe as tep  # noqa: E402


def load_embeddings_maxpool(entries, cache_dir, model, subsample_frames: int = 0):
    """Same contract as tep.load_embeddings_for_manifest, but always pools the
    cached per-frame sequence with max instead of mean (ignores the cached
    'mean' field entirely, so it also requires 'seq' to be present)."""
    X, y, meta = [], [], []
    for rel_path, label in entries:
        identity = tep.extract_clip_identity(rel_path)
        key = tep.cache_key(model, identity["action"], identity["base_id"],
                             identity["variant"], identity["bg"])
        npz_path = cache_dir / key
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        if "seq" not in d:
            continue
        seq = d["seq"]
        if subsample_frames > 0:
            T = seq.shape[0]
            n = min(subsample_frames, T)
            idx = np.linspace(0, T - 1, n, dtype=int)
            seq = seq[idx]
        emb = seq.max(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        X.append(emb)
        y.append(label)
        meta.append(identity)
    if not X:
        return np.empty((0, 0)), np.empty(0, dtype=int), []
    return np.stack(X), np.array(y, dtype=int), meta


def unseen_drop(out_root: Path, model: str) -> float | None:
    """matched_unseen f1_macro - shifted_unseen f1_macro, averaged over all
    pair/seed/fold summaries under out_root (mirrors build_probe_summary.py's
    eval_matched_unseen_ids / eval_shifted_unseen_ids fields)."""
    mode_name = f"rgb_{model}_linear_model"
    drops = []
    for summary_path in out_root.glob(f"rgb_torchvision/{model}_linear/*/*/summary_{mode_name}.json"):
        d = json.loads(summary_path.read_text())
        splits = d.get("splits", {})
        mm = splits.get("eval_matched_unseen_ids", {}).get("f1_macro")
        ms = splits.get("eval_shifted_unseen_ids", {}).get("f1_macro")
        if mm is not None and ms is not None:
            drops.append(mm - ms)
    return float(np.mean(drops)) if drops else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="clip,tc_clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default="out/bias_analysis/embeddings")
    ap.add_argument("--out_root_prefix", default="out/skin_tone_probe")
    args = ap.parse_args()

    # Swap in max-pooling for the duration of this run.
    tep.load_embeddings_for_manifest = load_embeddings_maxpool

    results = {}
    for model in args.models.split(","):
        model = model.strip()
        out_root = Path(f"{args.out_root_prefix}_{model}_linear_maxpool")
        cache_dir = Path(args.cache_dir) / model
        if not cache_dir.exists():
            # try alternate location (e.g. clip_embeddings/clip), matching
            # train_embedding_linear_probe.py's main()
            alt = Path(args.cache_dir).parent / f"{model}_embeddings" / model
            if alt.exists():
                cache_dir = alt
            else:
                print(f"Cache dir not found: {cache_dir}", file=sys.stderr)
                sys.exit(1)
        manifests_root = tep.manifests_root_for_fold(None)
        pair_tags = sorted(
            p.name for p in manifests_root.iterdir()
            if p.is_dir() and "_vs_" in p.name and not p.name.endswith("_smoke")
        )
        mode_name = f"rgb_{model}_linear_model"
        for pair_tag in pair_tags:
            out_dir = out_root / "rgb_torchvision" / f"{model}_linear" / pair_tag / f"seed_{args.seed}"
            if (out_dir / f"summary_{mode_name}.json").exists():
                continue
            tep.run_probe_for_pair(
                pair_tag=pair_tag, seed=args.seed, cache_dir=cache_dir, model=model,
                out_dir=out_dir, mode_name=mode_name, C=1.0, max_iter=2000,
                manifests_root=manifests_root, subsample_frames=0,
            )
        results[model] = unseen_drop(out_root, model)

    print("\n=== max-pool probe: matched-unseen minus shifted-unseen F1 (seed=%d, original split) ===" % args.seed)
    for model, drop in results.items():
        print(f"  {model:12s} max-pool drop = {drop}")
    print("\n(compare against the standard mean-pool CV drop reported in the paper: "
          "clip=0.427, tc_clip=0.075 -- see out/linear_probes/_probe_summary_cv.json)")


if __name__ == "__main__":
    main()
