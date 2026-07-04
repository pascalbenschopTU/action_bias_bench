"""
Linear probe on cached foundation-model embeddings for the skin-tone shortcut benchmark.

Trains a logistic regression classifier on the mean embedding of each video clip
(pre-computed by skin_tone_bias_analysis.py and stored as NPZ files), using the
same train/eval manifests as the torchvision RGB probe.

Writes summary JSONs in exactly the same format and directory structure as
train_torchvision_rgb_probe.py, so all downstream analysis scripts
(aggregate_skin_tone_probe.py, compute_skin_tone_probe_stats.py,
summarize_skin_tone_robustness.py) work without modification.

Output root structure (compatible with aggregate_skin_tone_probe.py):
    {out_root}/rgb_torchvision/{model}_linear/{pair_tag}/seed_{seed}/
        summary_rgb_{model}_linear_model.json
        eval_matched_seen_ids/summary_rgb_{model}_linear_model.json
        eval_shifted_seen_ids/summary_rgb_{model}_linear_model.json
        eval_matched_unseen_ids/summary_rgb_{model}_linear_model.json
        eval_shifted_unseen_ids/summary_rgb_{model}_linear_model.json

No GPU required — runs entirely from cached NPZ embeddings.

Usage (from ActionBiasBench directory):
    python scripts/train_embedding_linear_probe.py --model dinov2
    python scripts/train_embedding_linear_probe.py --model clip --seeds 0,1,2
    python scripts/train_embedding_linear_probe.py --model dinov2 --run_analysis
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── constants (mirror run_action_bias_bench.sh defaults) ─────────────────────

ACTION_PAIRS = [
    ("lunge",   "cartwheel"),
    ("squat",   "tie"),
    ("clap",    "celebrate"),
    ("dribble", "golf"),
    ("yawn",    "fish"),
]
ALL_ACTIONS = sorted({a for pair in ACTION_PAIRS for a in pair})

BACKGROUNDS     = ["autumn_hockey", "konzerthaus", "stadium_01"]
DARK_VARIANTS   = ["african", "indian"]
LIGHT_VARIANTS  = ["white", "asian"]
ALL_VARIANTS    = DARK_VARIANTS + LIGHT_VARIANTS

EVAL_SPLITS = [
    "eval_matched_unseen_ids",
    "eval_matched_seen_ids",
    "eval_shifted_seen_ids",
    "eval_shifted_unseen_ids",
]

MANIFESTS_ROOT = ROOT / "benchmarks/skin_tone/generated/manifests/skin_tone_camera_far_binary"
LABELS_ROOT    = ROOT / "benchmarks/skin_tone/generated/labels/skin_tone_camera_far_binary"

_VARIANT_RE = re.compile(r"_modified_([^/_]+?)(?:\.mp4|\.avi|$)", re.IGNORECASE)
_BASE_ID_RE = re.compile(
    r"^(?P<action>.+)_(?P<base_id>\d+)_(?:modified_(?P<variant>[^.]+)|(?P<initial>initial))(?:\..+)?$",
    re.IGNORECASE,
)

# ── model code ────────────────────────────────────────────────────────────────

def softmax(X):
    max = torch.max(X, 1, keepdim=True)[0]
    # without safety you get overflow
    safe_exp = torch.exp(X - max)
    safe_exp_sum = torch.sum(safe_exp, 1, keepdim=True)
    return safe_exp / (safe_exp_sum + 1e-10)

def cross_entropy(X, y):
    # y is (T,) integer class indices — pick log-prob of the true class per sample
    T, D = X.shape
    X_norm = softmax(X)
    loss = 0

    for i in range(T):
        loss = loss + (-torch.log(X_norm[i, y[i]] + 1e-10))

    return loss / T


def train_logistic(
    X: np.ndarray, y: np.ndarray, *, n_classes: int,
    C: float = 1.0, max_iter: int = 2000, seed: int = 0,
) -> tuple:
    torch.manual_seed(seed)
    T, D = X.shape
    weight = torch.zeros(n_classes, D, requires_grad=True)
    bias   = torch.zeros(n_classes,    requires_grad=True)
    inp    = torch.tensor(X, dtype=torch.float32)
    gt     = torch.tensor(y, dtype=torch.long)

    opt = torch.optim.LBFGS([weight, bias], max_iter=max_iter, line_search_fn="strong_wolfe")
    wd  = 1.0 / (C * T)   # L2 penalty matching sklearn's C convention

    def closure():
        opt.zero_grad()
        logits = inp @ weight.T + bias              # (T, n_classes)
        loss   = cross_entropy(logits, gt) + wd * weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return weight, bias


def predict(clf: tuple, X: np.ndarray) -> np.ndarray:
    weight, bias = clf
    inp = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = inp @ weight.T + bias          # (N, n_classes)
    return logits.argmax(dim=1).numpy()

    

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_manifest(path: Path) -> list[tuple[str, int]]:
    """Return list of (rel_path, label) from a manifest .txt file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                entries.append((parts[0], int(parts[1])))
    return entries


def extract_clip_identity(rel_path: str) -> dict:
    """Extract action, base_id, variant, background from a manifest rel_path."""
    from pathlib import PurePosixPath
    pure  = PurePosixPath(rel_path)
    parts = list(pure.parts)
    bg    = parts[0] if parts else ""
    action = parts[-2] if len(parts) >= 2 else ""
    stem   = pure.name
    for suf in (".mp4", ".avi", ".mov", ".zst"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    m = _BASE_ID_RE.match(stem)
    if m:
        base_id = int(m.group("base_id"))
        if not action:
            action  = str(m.group("action") or "")
        variant = str(m.group("variant") or ("initial" if m.group("initial") else "unknown")).lower()
    else:
        base_id = -1
        mv = _VARIANT_RE.search(rel_path)
        variant = mv.group(1).lower() if mv else "unknown"
    return {"bg": bg, "action": action, "base_id": base_id, "variant": variant}


def cache_key(model: str, action: str, base_id: int, variant: str, bg: str) -> str:
    return f"{model}_{action}_{base_id}_{variant}_{bg}.npz"


def load_embeddings_for_manifest(
    entries: list[tuple[str, int]],
    cache_dir: Path,
    model: str,
    subsample_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    For each manifest entry, load the cached mean embedding.
    If subsample_frames > 0, uniformly subsample that many frames from the
    stored sequence and recompute the mean — useful for comparing models
    that were encoded at different frame counts (e.g. DINOv2 all-frames vs
    CLIP 64-frames).
    Returns (X, y, meta) where X is (N, D), y is (N,), meta is list of identity dicts.
    Entries whose embedding is not found are silently skipped.
    """
    X, y, meta = [], [], []
    for rel_path, label in entries:
        identity = extract_clip_identity(rel_path)
        key      = cache_key(model, identity["action"], identity["base_id"],
                              identity["variant"], identity["bg"])
        npz_path = cache_dir / key
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        if subsample_frames > 0 and "seq" in d:
            seq = d["seq"]                          # (T, D)
            T   = seq.shape[0]
            n   = min(subsample_frames, T)
            idx = np.linspace(0, T - 1, n, dtype=int)
            emb = seq[idx].mean(axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        else:
            emb = d["mean"]
        X.append(emb)
        y.append(label)
        meta.append(identity)
    if not X:
        return np.empty((0, 0)), np.empty(0, dtype=int), []
    return np.stack(X), np.array(y, dtype=int), meta


# ── metrics ───────────────────────────────────────────────────────────────────

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_from_cm(cm: np.ndarray, eps: float = 1e-12):
    tp      = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_s  = cm.sum(axis=0).astype(np.float64)
    prec    = tp / (pred_s  + eps)
    rec     = tp / (support + eps)
    f1      = 2 * prec * rec / (prec + rec + eps)
    return prec, rec, f1, support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n  = max(int(y_true.max()), int(y_pred.max())) + 1
    cm = confusion_matrix(y_true, y_pred, n)
    prec, rec, f1, support = prf_from_cm(cm)
    top1     = float((y_true == y_pred).sum()) / max(1, len(y_true))
    f1_mac   = float(np.nanmean(f1))
    f1_wt    = float(np.nansum(f1 * support) / (support.sum() + 1e-12))
    p_mac    = float(np.nanmean(prec))
    p_wt     = float(np.nansum(prec * support) / (support.sum() + 1e-12))
    r_mac    = float(np.nanmean(rec))
    r_wt     = float(np.nansum(rec * support) / (support.sum() + 1e-12))
    mca      = float(np.nanmean(rec))
    return {
        "top1":               top1,
        "top5":               top1,   # binary — top5 == top1
        "mean_class_acc":     mca,
        "precision_macro":    p_mac,
        "recall_macro":       r_mac,
        "f1_macro":           f1_mac,
        "precision_weighted": p_wt,
        "recall_weighted":    r_wt,
        "f1_weighted":        f1_wt,
    }


def compute_per_variant(y_true, y_pred, meta, n_classes) -> dict:
    by_variant: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for yt, yp, m in zip(y_true, y_pred, meta):
        v = m["variant"]
        by_variant[v][0].append(yt)
        by_variant[v][1].append(yp)

    per_variant = {}
    for variant, (yt_list, yp_list) in by_variant.items():
        if not yt_list:
            continue
        yt = np.array(yt_list, dtype=int)
        yp = np.array(yp_list, dtype=int)
        cm = confusion_matrix(yt, yp, n_classes)
        _, _, f1, support = prf_from_cm(cm)
        f1_mac = float(np.nanmean(f1))
        top1   = float((yt == yp).sum()) / len(yt)
        per_variant[variant] = {"count": len(yt), "top1": top1, "f1_macro": f1_mac}

    dark_counts  = {v: per_variant[v]["count"] for v in DARK_VARIANTS  if v in per_variant}
    light_counts = {v: per_variant[v]["count"] for v in LIGHT_VARIANTS if v in per_variant}
    dark_top1  = (sum(per_variant[v]["top1"] * per_variant[v]["count"] for v in dark_counts)
                  / max(1, sum(dark_counts.values()))) if dark_counts else float("nan")
    light_top1 = (sum(per_variant[v]["top1"] * per_variant[v]["count"] for v in light_counts)
                  / max(1, sum(light_counts.values()))) if light_counts else float("nan")
    gap = (light_top1 - dark_top1) if (dark_top1 == dark_top1 and light_top1 == light_top1) else float("nan")

    return {
        "per_variant": per_variant,
        "per_tone_group": {
            "dark":  {"count": sum(dark_counts.values()),  "top1": dark_top1},
            "light": {"count": sum(light_counts.values()), "top1": light_top1},
            "gap_light_minus_dark": gap,
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Linear probe on cached embeddings.")
    ap.add_argument("--model",       required=True,
                    help="Model name matching cache files (e.g. dinov2, clip).")
    ap.add_argument("--cache_dir",   default="out/bias_analysis/embeddings",
                    help="Parent of the per-model cache directory.")
    ap.add_argument("--out_root",    default=None,
                    help="Output root. Defaults to out/skin_tone_probe_{model}_linear.")
    ap.add_argument("--seeds",       default="0,1,2",
                    help="Comma-separated training seeds.")
    ap.add_argument("--C",           type=float, default=1.0,
                    help="LogisticRegression regularisation strength.")
    ap.add_argument("--max_iter",    type=int, default=2000)
    ap.add_argument("--subsample_frames", type=int, default=0,
                    help="Uniformly subsample this many frames from the cached sequence "
                         "before computing the mean embedding (0 = use cached mean as-is). "
                         "Use 64 to match CLIP's frame count when comparing with DINOv2.")
    ap.add_argument("--run_analysis", action="store_true",
                    help="Run aggregate + stats + heatmap scripts after probe.")
    return ap.parse_args()


def run_probe_for_pair(
    pair_tag: str,
    seed: int,
    cache_dir: Path,
    model: str,
    out_dir: Path,
    mode_name: str,
    C: float,
    max_iter: int,
    subsample_frames: int = 0,
) -> None:
    train_manifest = MANIFESTS_ROOT / pair_tag / "train_in_domain.txt"
    if not train_manifest.exists():
        print(f"  [SKIP] manifest not found: {train_manifest}", flush=True)
        return

    X_train, y_train, _ = load_embeddings_for_manifest(
        parse_manifest(train_manifest), cache_dir, model, subsample_frames)
    if len(X_train) == 0:
        print(f"  [SKIP] no embeddings found for {pair_tag}", flush=True)
        return
    if len(np.unique(y_train)) < 2:
        print(f"  [SKIP] only one class in training data for {pair_tag} "
              f"(missing embeddings for one action?)", flush=True)
        return

    n_classes = len(np.unique(y_train))
    clf = train_logistic(X_train, y_train, n_classes=n_classes,
                          C=C, max_iter=max_iter, seed=seed)

    split_metrics: dict[str, dict] = {}
    per_variant_by_split: dict[str, dict] = {}

    for split in EVAL_SPLITS:
        eval_manifest = MANIFESTS_ROOT / pair_tag / f"{split}.txt"
        if not eval_manifest.exists():
            continue
        X_eval, y_eval, meta_eval = load_embeddings_for_manifest(
            parse_manifest(eval_manifest), cache_dir, model, subsample_frames)
        if len(X_eval) == 0:
            continue
        y_pred = predict(clf, X_eval)
        metrics = compute_metrics(y_eval, y_pred)
        split_metrics[split] = metrics
        per_variant_by_split[split] = compute_per_variant(y_eval, y_pred, meta_eval, n_classes)

        # write per-split summary
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_summary = {
            "mode":     mode_name,
            "num_splits": 1,
            "splits":   {split: metrics},
            "aggregate": {k: {"mean": v, "std": 0.0} for k, v in metrics.items()},
            "per_variant_splits": {split: per_variant_by_split[split]},
        }
        (split_dir / f"summary_{mode_name}.json").write_text(
            json.dumps(split_summary, indent=2), encoding="utf-8")

    if not split_metrics:
        return

    # aggregate summary across all eval splits
    metric_names = list(next(iter(split_metrics.values())).keys())
    aggregate = {}
    for mn in metric_names:
        vals = [split_metrics[s][mn] for s in split_metrics]
        aggregate[mn] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    summary = {
        "mode":         mode_name,
        "num_splits":   len(split_metrics),
        "splits":       split_metrics,
        "aggregate":    aggregate,
        "per_variant_splits": per_variant_by_split,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary_{mode_name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  [{pair_tag} seed={seed}] train={len(X_train)}  "
          + "  ".join(f"{s.replace('eval_','')[:8]}={split_metrics[s]['f1_macro']:.3f}"
                      for s in EVAL_SPLITS if s in split_metrics),
          flush=True)


def main() -> None:
    args = parse_args()
    seeds    = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    model    = args.model
    # model name for directory/mode: "dinov2_linear", "clip_linear", etc.
    model_dir_name = f"{model}_linear"
    mode_name      = f"rgb_{model_dir_name}_model"
    out_root = Path(args.out_root) if args.out_root else Path(f"out/skin_tone_probe_{model}_linear")
    cache_dir = Path(args.cache_dir) / model

    if not cache_dir.exists():
        # try alternate location (e.g. clip_embeddings/clip)
        alt = Path(args.cache_dir).parent / f"{model}_embeddings" / model
        if alt.exists():
            cache_dir = alt
        else:
            print(f"Cache dir not found: {cache_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"model={model}  mode={mode_name}  cache={cache_dir}  out={out_root}", flush=True)

    pair_tags = sorted(
        d.name for d in MANIFESTS_ROOT.iterdir()
        if d.is_dir() and "_vs_" in d.name and not d.name.endswith("_smoke")
    )

    for pair_tag in pair_tags:
        for seed in seeds:
            out_dir = out_root / "rgb_torchvision" / model_dir_name / pair_tag / f"seed_{seed}"
            if (out_dir / f"summary_{mode_name}.json").exists():
                print(f"  [SKIP cached] {pair_tag} seed={seed}", flush=True)
                continue
            run_probe_for_pair(
                pair_tag=pair_tag,
                seed=seed,
                cache_dir=cache_dir,
                model=model,
                out_dir=out_dir,
                mode_name=mode_name,
                C=args.C,
                max_iter=args.max_iter,
                subsample_frames=args.subsample_frames,
            )

    if args.run_analysis:
        import subprocess
        python = sys.executable
        for script, extra in [
            ("benchmarks/skin_tone/aggregate_skin_tone_probe.py",      ["--root", str(out_root)]),
            ("benchmarks/skin_tone/compute_skin_tone_probe_stats.py",  ["--root", str(out_root), "--metric", "f1_macro"]),
            ("benchmarks/skin_tone/summarize_skin_tone_robustness.py", ["--root", str(out_root), "--metric", "f1_macro"]),
        ]:
            print(f"\nRunning {script} ...", flush=True)
            subprocess.run([python, script] + extra, check=True)

    print(f"\nDone. Results in {out_root}", flush=True)


if __name__ == "__main__":
    main()
