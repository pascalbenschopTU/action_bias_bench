"""
Direction-alignment test for the CLIP/TC-CLIP Fig.3-vs-Fig.4 disagreement
(Section 3.2, reviewer 2): TC-CLIP's frame-level SSM ratio is high (second
highest of all models) but its pooled linear-probe drop is low (near
DINOv3). The pooling ablation (pooling_ablation_probe.py) already ruled out
simple mean-pooling sign-cancellation as the explanation -- max-pooling made
TC-CLIP's drop smaller, not larger.

This tests a more specific mechanism. SSM measures how far a clip's
embedding moves under a skin-tone swap, in *any* direction. The linear
probe only cares about movement *along its own decision boundary* --
movement parallel to the boundary never flips a prediction. So a model can
show large SSM (large movement) and a small probe drop (little of that
movement crosses the boundary) if fine-tuning rotated the classifier's
discriminative direction away from the tone-sensitive dimensions, without
shrinking the tone signal itself.

For each (model, pair_tag) we train the same probe used for the paper's
numbers (train_embedding_linear_probe.train_logistic, same manifests, same
cached mean-pooled embeddings), extract its decision direction
w = weight[class_1] - weight[class_0], then for every same-clip
opposite-skin-tone-group counterfactual pair (same action/base_id/
background, one dark-group variant vs one light-group variant -- the same
pairing skin_tone_bias_analysis.py uses for its own d_skin metric) compute
the raw embedding delta and measure what fraction of its squared magnitude
lies along w vs orthogonal to it.

If TC-CLIP's tone-swap deltas are disproportionately orthogonal to its own
decision boundary compared to CLIP's, that supports "fine-tuning rotated the
discriminative direction away from the tone-sensitive subspace" as the
mechanism, rather than "pooling destroyed the signal".

Usage (run from the ActionBiasBench directory):
    python benchmarks/skin_tone/probe_direction_alignment.py
    python benchmarks/skin_tone/probe_direction_alignment.py --models clip,tc_clip
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_embedding_linear_probe as tep  # noqa: E402
from scripts.skin_tone_bias_analysis import ALL_IDS, DARK_VARIANTS, opposite_variants  # noqa: E402


def resolve_cache_dir(cache_dir_prefix: Path, model: str) -> Path:
    cache_dir = cache_dir_prefix / model
    if cache_dir.exists():
        return cache_dir
    alt = cache_dir_prefix.parent / f"{model}_embeddings" / model
    if alt.exists():
        return alt
    print(f"Cache dir not found: {cache_dir}", file=sys.stderr)
    sys.exit(1)


def load_all_embeddings(cache_dir: Path, model: str) -> dict:
    """(action, base_id, variant, bg) -> cached mean-pooled embedding, read from
    the exact same cache the probe itself trains on.

    Lists the directory once (single bulk network call) instead of doing one
    Path.exists() stat per candidate key -- the latter is thousands of
    individual round-trips on the network-mounted filesystem and was the
    actual bottleneck (each stat pays network latency; a single readdir does
    not)."""
    # filename -> (action, base_id, variant, bg), inverse of tep.cache_key
    wanted = {
        tep.cache_key(model, action, base_id, variant, bg): (action, base_id, variant, bg)
        for action in tep.ALL_ACTIONS
        for base_id in ALL_IDS
        for variant in tep.ALL_VARIANTS
        for bg in tep.BACKGROUNDS
    }
    out = {}
    for entry in cache_dir.iterdir():
        identity = wanted.get(entry.name)
        if identity is None:
            continue
        with np.load(entry) as d:
            out[identity] = d["mean"]
    return out


def direction_for_pair(pair_tag: str, seed: int, cache_dir: Path, model: str,
                        manifests_root: Path, C: float, max_iter: int):
    train_manifest = manifests_root / pair_tag / "train_in_domain.txt"
    if not train_manifest.exists():
        return None
    X_train, y_train, _ = tep.load_embeddings_for_manifest(
        tep.parse_manifest(train_manifest), cache_dir, model)
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return None
    weight, _bias = tep.train_logistic(X_train, y_train, n_classes=2, C=C, max_iter=max_iter, seed=seed)
    w = (weight[1] - weight[0]).detach().numpy()
    norm = np.linalg.norm(w)
    return w / norm if norm > 0 else None


def alignment_for_pair(pair_tag: str, w_hat: np.ndarray, embeddings: dict) -> list[dict]:
    action_a, action_b = pair_tag.split("_vs_")
    rows = []
    for action in (action_a, action_b):
        for base_id in ALL_IDS:
            for bg in tep.BACKGROUNDS:
                for v1 in DARK_VARIANTS:
                    key1 = (action, base_id, v1, bg)
                    if key1 not in embeddings:
                        continue
                    for v2 in opposite_variants(v1):
                        key2 = (action, base_id, v2, bg)
                        if key2 not in embeddings:
                            continue
                        delta = embeddings[key2] - embeddings[key1]
                        total = float(np.linalg.norm(delta))
                        if total < 1e-8:
                            continue
                        proj = float(np.dot(delta, w_hat))
                        rows.append(dict(
                            action=action, base_id=base_id, bg=bg, v1=v1, v2=v2,
                            total=total, proj_abs=abs(proj), frac=(proj * proj) / (total * total),
                        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="clip,tc_clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default="out/bias_analysis/embeddings")
    ap.add_argument("--out_csv", default="out/skin_tone_probe_v7_cv_analysis/probe_direction_alignment.csv")
    args = ap.parse_args()

    manifests_root = tep.manifests_root_for_fold(None)
    pair_tags = sorted(
        p.name for p in manifests_root.iterdir()
        if p.is_dir() and "_vs_" in p.name and not p.name.endswith("_smoke")
    )

    all_rows = []
    for model in args.models.split(","):
        model = model.strip()
        cache_dir = resolve_cache_dir(Path(args.cache_dir), model)
        embeddings = load_all_embeddings(cache_dir, model)
        print(f"{model}: loaded {len(embeddings)} cached clip embeddings from {cache_dir}")

        for pair_tag in pair_tags:
            w_hat = direction_for_pair(pair_tag, args.seed, cache_dir, model,
                                        manifests_root, C=1.0, max_iter=2000)
            if w_hat is None:
                print(f"  [SKIP] {pair_tag}: could not train probe")
                continue
            rows = alignment_for_pair(pair_tag, w_hat, embeddings)
            for r in rows:
                r["model"] = model
                r["pair_tag"] = pair_tag
            all_rows.extend(rows)
            if rows:
                fracs = [r["frac"] for r in rows]
                totals = [r["total"] for r in rows]
                print(f"  [{pair_tag}] n_pairs={len(rows)}  "
                      f"mean_frac_along_w={np.mean(fracs):.4f}  mean_total_delta={np.mean(totals):.4f}")

    if not all_rows:
        print("No alignment rows computed -- check cache_dir / manifests.", file=sys.stderr)
        sys.exit(1)

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {out_csv}")

    print("\n=== summary: mean squared-alignment fraction along the probe's own decision direction ===")
    for model in args.models.split(","):
        model = model.strip()
        fracs = [r["frac"] for r in all_rows if r["model"] == model]
        totals = [r["total"] for r in all_rows if r["model"] == model]
        if fracs:
            print(f"  {model:12s} n={len(fracs):4d}  mean_frac_along_w={np.mean(fracs):.4f}  "
                  f"median_frac_along_w={np.median(fracs):.4f}  mean_total_delta={np.mean(totals):.4f}")


if __name__ == "__main__":
    main()
