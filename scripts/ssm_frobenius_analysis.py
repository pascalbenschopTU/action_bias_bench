"""
SSM structural-distance analysis from cached embeddings.

For each (action, base_id, background) cell, computes:
    d_skin   = mean SSM distance across skin-tone swaps
               e.g. dist(SSM(lunge_7_african), SSM(lunge_7_white))
    d_action = mean SSM distance to a paired action
               e.g. dist(SSM(lunge_7_african), SSM(cartwheel_7_african))

Bias signal: d_skin / d_action
    << 1  skin swap barely changes the SSM → structure invariant to skin tone
    ~  1  skin swap changes SSM as much as switching action → problematic

Two independent toggles:
    --metric {frobenius,rsa}  distance between two SSMs (see frob_dist/rsa_dist)
    --pairs  {matching,all}   matching = 5 curated action pairs (default, original
                               behaviour); all = all C(10,2)=45 pairs among the
                               same 10 actions

No model or GPU needed — runs entirely on cached NPZ files.

Usage (from ActionBiasBench directory):
    python scripts/ssm_frobenius_analysis.py --model dinov2
    python scripts/ssm_frobenius_analysis.py --model dinov2 --metric rsa
    python scripts/ssm_frobenius_analysis.py --model dinov2 --pairs all
"""

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

ACTION_PAIRS = [
    ("lunge", "cartwheel"),
    ("squat", "tie"),
    ("clap", "celebrate"),
    ("dribble", "golf"),
    ("yawn", "fish"),
]
ALL_ACTIONS = sorted({a for pair in ACTION_PAIRS for a in pair})

LIGHT_VARIANTS = ["white", "asian"]
DARK_VARIANTS  = ["african", "indian"]
ALL_VARIANTS   = LIGHT_VARIANTS + DARK_VARIANTS

BACKGROUNDS = ["autumn_hockey", "konzerthaus", "stadium_01"]
ALL_IDS     = list(range(10))


def load_cache(cache_dir: Path, model: str) -> dict:
    """Load all cached (seq,) arrays. Returns dict keyed by (action, id, variant, bg)."""
    embeddings = {}
    for npz_path in sorted(cache_dir.glob(f"{model}_*.npz")):
        # filename: {model}_{action}_{id}_{variant}_{bg}.npz
        stem = npz_path.stem[len(model) + 1:]   # strip "dinov2_"
        parts = stem.split("_")
        # action may be multi-part (none here), bg may be multi-part (autumn_hockey)
        # format: action _ id _ variant _ bg (bg can be 1 or 2 tokens)
        # find id (first numeric token)
        id_idx = next(i for i, p in enumerate(parts) if p.isdigit())
        action  = "_".join(parts[:id_idx])
        base_id = int(parts[id_idx])
        rest    = parts[id_idx + 1:]
        # variant is one of the known values
        known_variants = set(ALL_VARIANTS)
        var_idx = next(i for i, p in enumerate(rest) if p in known_variants)
        variant = rest[var_idx]
        bg      = "_".join(rest[var_idx + 1:])
        if action not in ALL_ACTIONS:
            continue
        d = np.load(npz_path)
        embeddings[(action, base_id, variant, bg)] = d["seq"]   # (T, D)
    return embeddings


def compute_ssm(seq: np.ndarray, T: int) -> np.ndarray:
    """Resample seq to T steps, return (T, T) cosine self-similarity matrix."""
    src_T = seq.shape[0]
    if src_T != T:
        idx = np.linspace(0, src_T - 1, T, dtype=int)
        seq = seq[idx]
    return seq @ seq.T   # already L2-normalised per row


def frob_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius distance between two same-shape matrices. Sensitive to the
    absolute similarity level of each SSM, not just its temporal pattern."""
    return float(np.linalg.norm(A - B, "fro"))


def rsa_dist(A: np.ndarray, B: np.ndarray) -> float:
    """1 - Pearson correlation between the two SSMs' off-diagonal entries.
    Mean-centred and scale-invariant, so it compares temporal *structure*
    only, ignoring differences in overall similarity level (RSA-style)."""
    iu = np.triu_indices_from(A, k=1)
    corr = np.corrcoef(A[iu], B[iu])[0, 1]
    return float(1.0 - corr)


def opposite_variants(variant: str) -> list:
    return DARK_VARIANTS if variant in LIGHT_VARIANTS else LIGHT_VARIANTS


def skin_group(variant: str) -> str:
    return "light" if variant in LIGHT_VARIANTS else "dark"


def build_partner_map(pairs: list) -> dict:
    """action -> set of partner actions. `pairs` is a list of (a, b) tuples."""
    partners = defaultdict(set)
    for a, b in pairs:
        partners[a].add(b)
        partners[b].add(a)
    return partners

DIST_FUNCS = {"frobenius": frob_dist, "rsa": rsa_dist}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     default="dinov2")
    ap.add_argument("--cache_dir", default="out/bias_analysis/embeddings")
    ap.add_argument("--out_dir",   default="out/bias_analysis")
    ap.add_argument("--T",         type=int, default=64,
                    help="Common number of steps to resample all SSMs to before comparing.")
    ap.add_argument("--metric", choices=list(DIST_FUNCS), default="frobenius",
                    help="Distance between two SSMs (default: frobenius, original behaviour).")
    ap.add_argument("--pairs", choices=["matching", "all"], default="matching",
                    help="matching = 5 curated pairs (default, original behaviour); "
                         "all = all C(10,2)=45 pairs among the same 10 actions.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) / args.model
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_fn = DIST_FUNCS[args.metric]
    pairs = ACTION_PAIRS if args.pairs == "matching" else list(itertools.combinations(ALL_ACTIONS, 2))
    partner_map = build_partner_map(pairs)

    print(f"Loading cached embeddings from {cache_dir} ...", flush=True)
    embeddings = load_cache(cache_dir, args.model)
    print(f"  {len(embeddings)} clips loaded", flush=True)
    print(f"  metric={args.metric}  pairs={args.pairs} ({len(pairs)} action pairs)", flush=True)

    T = args.T
    rows = []

    for (action, base_id, variant, bg), seq in embeddings.items():
        partners = partner_map.get(action)
        if not partners:
            continue

        ssm_ref = compute_ssm(seq, T)

        # --- d_skin: same action, same performer, same bg, opposite skin variants ---
        # (independent of partner action, so computed once per clip)
        skin_dists = []
        for ov in opposite_variants(variant):
            key = (action, base_id, ov, bg)
            if key not in embeddings:
                continue
            ssm_other = compute_ssm(embeddings[key], T)
            skin_dists.append(dist_fn(ssm_ref, ssm_other))
        if not skin_dists:
            continue
        d_skin = float(np.mean(skin_dists))

        # --- d_action: each partner action, same performer, same bg, all skin variants ---
        for partner in sorted(partners):
            action_dists = []
            for pv in ALL_VARIANTS:
                key = (partner, base_id, pv, bg)
                if key not in embeddings:
                    continue
                ssm_other = compute_ssm(embeddings[key], T)
                action_dists.append(dist_fn(ssm_ref, ssm_other))
            if not action_dists:
                continue

            d_action = float(np.mean(action_dists))
            r        = d_skin / (d_action + 1e-8)

            rows.append({
                "action":     action,
                "partner":    partner,
                "base_id":    base_id,
                "variant":    variant,
                "skin_group": skin_group(variant),
                "background": bg,
                "d_skin":     round(d_skin, 5),
                "d_action":   round(d_action, 5),
                "r":          round(r, 5),
            })

    # save
    out_csv = out_dir / f"ssm_{args.metric}_{args.model}.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {out_csv}")

    # summary
    print(f"\n=== SSM r = d_skin / d_action  [{args.model}, T={T}, metric={args.metric}] ===")
    print("r << 1  skin swap barely changes SSM structure (safe)")
    print("r ~  1  skin swap rivals action change (bias)\n")

    by_action = defaultdict(list)
    by_id     = defaultdict(list)
    for row in rows:
        by_action[row["action"]].append(row["r"])
        if row["action"] == "lunge":
            by_id[row["base_id"]].append(row["r"])

    print(f"{'Action':<14} {'mean_r':>8}  {'max_r':>8}  {'n':>5}")
    print("-" * 40)
    for action in sorted(by_action, key=lambda a: -np.mean(by_action[a])):
        rs = np.array(by_action[action])
        print(f"{action:<14} {rs.mean():>8.4f}  {rs.max():>8.4f}  {len(rs):>5}")

    if by_id:
        print(f"\n=== Lunge: per-performer breakdown ===")
        print(f"{'id':>4}  {'mean_r':>8}  {'max_r':>8}  {'n':>5}")
        for pid in sorted(by_id):
            rs = np.array(by_id[pid])
            print(f"{pid:>4}  {rs.mean():>8.4f}  {rs.max():>8.4f}  {len(rs):>5}")

    print(f"\n=== By skin group ===")
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["skin_group"]].append(row["r"])
    for group, rs in by_group.items():
        arr = np.array(rs)
        print(f"  {group}: mean_r={arr.mean():.4f}  max_r={arr.max():.4f}  n={len(arr)}")


if __name__ == "__main__":
    main()
