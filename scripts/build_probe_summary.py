"""Build the flat {model: {split: f1_macro}} summary that plot_probe_ssm.py reads.

For each model, averages f1_macro over every (pair_tag, seed/fold) run directory
found under out/linear_probes/skin_tone_probe_{model}{suffix}/rgb_torchvision/{model}_linear/*/*/,
for each of the 4 eval splits.

Usage:
    python scripts/build_probe_summary.py --suffix _linear_cv --out out/linear_probes/_probe_summary_cv.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = [
    "clip", "dinov2", "dinov3", "siglip", "eva02", "hiera", "hiera_large",
    "vjepa2", "tc_clip", "r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s", "s3d", "swin3d_s",
]
EVAL_SPLITS = [
    "eval_matched_unseen_ids", "eval_shifted_unseen_ids",
    "eval_matched_seen_ids", "eval_shifted_seen_ids",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/linear_probes")
    ap.add_argument("--suffix", default="_linear_cv",
                     help="Appended to skin_tone_probe_{model} to find each model's out_root.")
    ap.add_argument("--out", default="out/linear_probes/_probe_summary_cv.json")
    args = ap.parse_args()

    root = Path(args.root)
    summary: dict[str, dict[str, float]] = {}
    for model in MODELS:
        model_dir_name = f"{model}_linear"
        base = root / f"skin_tone_probe_{model}{args.suffix}" / "rgb_torchvision" / model_dir_name
        if not base.exists():
            continue
        per_split_vals: dict[str, list[float]] = {s: [] for s in EVAL_SPLITS}
        # Group the per-(pair) unseen drop by training run (seed[/fold] dir, the
        # parent's parent), so the CI reflects run-to-run variability, not the
        # much larger across-pair spread. Each run's drop is the mean over its
        # action pairs; the 95% CI is then taken over the runs.
        drop_by_run: dict[str, list[float]] = defaultdict(list)
        n_summaries = 0
        for summary_path in sorted(base.glob(f"*/*/summary_rgb_{model_dir_name}_model.json")):
            d = json.loads(summary_path.read_text())
            splits = d.get("splits", {})
            for s in EVAL_SPLITS:
                if s in splits and "f1_macro" in splits[s]:
                    per_split_vals[s].append(float(splits[s]["f1_macro"]))
            mu = splits.get("eval_matched_unseen_ids", {}).get("f1_macro")
            su = splits.get("eval_shifted_unseen_ids", {}).get("f1_macro")
            if mu is not None and su is not None:
                run_id = summary_path.parent.name          # e.g. seed_0fold2
                drop_by_run[run_id].append(float(mu) - float(su))
            n_summaries += 1
        if n_summaries == 0:
            continue
        entry = {
            s: (sum(vals) / len(vals) if vals else float("nan"))
            for s, vals in per_split_vals.items()
        }
        run_drops = np.array([float(np.mean(v)) for v in drop_by_run.values()])
        if run_drops.size >= 2:
            lo, hi = np.percentile(run_drops, [2.5, 97.5])
            entry["drop_unseen_ci_low"] = float(lo)
            entry["drop_unseen_ci_high"] = float(hi)
            entry["n_runs"] = int(run_drops.size)
        summary[model] = entry
        ci = (f"  drop_CI=[{entry.get('drop_unseen_ci_low', float('nan')):+.3f},"
              f"{entry.get('drop_unseen_ci_high', float('nan')):+.3f}]"
              if "drop_unseen_ci_low" in entry else "")
        print(f"{model:13s} n_runs={run_drops.size:2d} ({n_summaries:3d} summaries)  " +
              "  ".join(f"{s.replace('eval_', '')[:8]}={entry[s]:.3f}" for s in EVAL_SPLITS) + ci)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}  ({len(summary)} models)")


if __name__ == "__main__":
    main()
