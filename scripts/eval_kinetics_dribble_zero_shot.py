"""
Zero-shot Kinetics-400 skin-tone check on REAL Kinetics-400 "dribbling_basketball"
clips (not HMDB51/PA-HMDB51), stratified by a hand-assigned skin-tone label.

Why this exists: eval_dribble_scaffold.py runs the same check on the 145 PA-HMDB51
"dribble" clips, but those clips come from only 29 distinct source videos (many clips
are the same performer/court/camera cut into pieces), so most of that script's logic
is dedicated to correcting for that pseudo-replication. A duplication check on the
Kinetics-400 "dribbling_basketball" class (see llm_reports/kinetics_dribble_scale_plan.md)
found NO such problem: 806 clips (756 train + 50 val) resolve to 806 distinct YouTube
source-video IDs, zero duplicates, zero corrupted files. Every clip here is already an
independent unit - no source-video aggregation step is needed (unlike
eval_dribble_scaffold.py), so this script is the simpler direct clip-level test, just
at much larger n.

Skin-tone labels are NOT provided by Kinetics-400 (unlike PA-HMDB51) - they were
hand-assigned by visual review of one representative frame per clip (see
kinetics_dribble_skin_tone_labels.json and its accompanying notes on labeling
methodology/limitations - this is a single reviewer's visual judgment from one still
frame, not a validated annotation protocol).

Usage (run from the ActionBiasBench directory):
    python scripts/eval_kinetics_dribble_zero_shot.py --dry_run
    python scripts/eval_kinetics_dribble_zero_shot.py --models r3d_18
    python scripts/eval_kinetics_dribble_zero_shot.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval_pahmdb51_zero_shot import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    MODEL_NAMES,
    REPO_ROOT,
    WORKSPACE_ROOT,
    ClipRecord,
    _binary_contingency,
    _check_runtime_dependencies,
    _full_contingency,
    configure_caches,
    fisher_exact_2x2_fallback,
    resolve_device,
    run_model_with_device_fallback,
    write_csv,
    write_json,
)

DEFAULT_KINETICS_ROOT = WORKSPACE_ROOT / "datasets" / "Kinetics" / "k400"
DEFAULT_LABELS_JSON = REPO_ROOT / "benchmarks" / "skin_tone" / "generated" / "kinetics_dribble_skin_tone_labels.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "out" / "kinetics_dribble_zero_shot"

K400_CLASS_DIR = "dribbling_basketball"
K400_TARGET = "dribbling basketball"
BASKETBALL_FAMILY = ["dribbling basketball", "shooting basketball", "playing basketball", "dunking basketball"]
SKIN_COLOR_NAMES = ["unidentifiable", "white", "yellow", "black", "mixed_skin_color"]
SKIN_COLOR_ID = {name: i for i, name in enumerate(SKIN_COLOR_NAMES)}


def load_labeled_clips(
    labels_json: Path,
    kinetics_root: Path,
    max_clips: Optional[int],
    seed: int,
) -> Tuple[List[ClipRecord], List[Dict[str, str]]]:
    data = json.loads(labels_json.read_text())
    clips: List[ClipRecord] = []
    skipped: List[Dict[str, str]] = []
    for video_name, meta in sorted(data.items()):
        skin_name = meta["skin_color"]
        if skin_name not in SKIN_COLOR_ID:
            skipped.append({"rel_path": video_name, "reason": f"unknown_skin_color_label:{skin_name}"})
            continue
        split = meta["split"]
        rel_path = f"{split}/{K400_CLASS_DIR}/{video_name}"
        abs_path = kinetics_root / split / K400_CLASS_DIR / video_name
        if not abs_path.is_file():
            skipped.append({"rel_path": rel_path, "reason": "missing_video_file"})
            continue
        clips.append(
            ClipRecord(
                hmdb_class=K400_CLASS_DIR,
                k400_group=K400_CLASS_DIR,
                k400_target=K400_TARGET,
                video_name=video_name,
                rel_path=rel_path,
                abs_path=abs_path,
                skin_color=skin_name,
                skin_color_id=SKIN_COLOR_ID[skin_name],
                review=bool(meta.get("review", False)),
                # Each Kinetics clip is its own distinct source video (verified - see
                # module docstring) so, unlike eval_dribble_scaffold.py, there is no
                # aggregation group smaller than the clip itself.
                source_video_id=video_name,
            )
        )

    if max_clips is not None and len(clips) > max_clips:
        import random

        clips = random.Random(seed).sample(clips, max_clips)
    return clips, skipped


def add_basketball_family_column(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        row["correct_basketball_family"] = row["y_pred_top1_400"] in BASKETBALL_FAMILY


def print_label_distribution(clips: List[ClipRecord]) -> None:
    from collections import Counter

    by_color: Counter = Counter(c.skin_color for c in clips)
    by_split: Counter = Counter(c.rel_path.split("/")[0] for c in clips)
    n_review = sum(1 for c in clips if c.review)
    print(f"{len(clips)} labeled clips.")
    print("Skin-tone distribution:")
    for name in SKIN_COLOR_NAMES:
        print(f"  {name:16s} {by_color.get(name, 0):4d}")
    print(f"Split distribution: {dict(by_split)}")
    print(f"Flagged for review: {n_review}")


def binary_fisher_test(rows: List[Dict[str, object]], accuracy_field: str) -> Dict[str, object]:
    counts = _binary_contingency(rows, accuracy_field)
    if "white" not in counts or "non_white" not in counts:
        return {"error": "one or both groups (white/non_white) empty - cannot run test"}
    table = [
        [counts["white"]["correct"], counts["white"]["incorrect"]],
        [counts["non_white"]["correct"], counts["non_white"]["incorrect"]],
    ]
    try:
        from scipy import stats  # type: ignore

        statistic, p_value = stats.fisher_exact(table)
        method = "scipy_fisher_exact"
    except Exception:
        statistic, p_value, method = fisher_exact_2x2_fallback(table)
    groups_out = {
        name: {
            "n": counts[name]["correct"] + counts[name]["incorrect"],
            "n_correct": counts[name]["correct"],
            "accuracy": counts[name]["correct"] / (counts[name]["correct"] + counts[name]["incorrect"]),
        }
        for name in ("white", "non_white")
    }
    return {"groups": groups_out, "test": method, "statistic": statistic, "p_value": p_value}


def full_chi2_test(rows: List[Dict[str, object]], accuracy_field: str) -> Dict[str, object]:
    counts = _full_contingency(rows, accuracy_field)
    names = sorted(counts)
    if len(names) < 2:
        return {"error": "fewer than 2 non-'unidentifiable' skin-tone groups present - cannot run test"}
    table = [[counts[name]["correct"], counts[name]["incorrect"]] for name in names]
    try:
        from scipy import stats  # type: ignore

        statistic, p_value, _, _ = stats.chi2_contingency(table)
        method = "scipy_chi2_contingency"
    except Exception:
        statistic, p_value, method = float("nan"), float("nan"), "scipy_unavailable_no_manual_fallback"
    groups_out = {
        name: {
            "n": counts[name]["correct"] + counts[name]["incorrect"],
            "n_correct": counts[name]["correct"],
            "accuracy": counts[name]["correct"] / (counts[name]["correct"] + counts[name]["incorrect"]),
        }
        for name in names
    }
    return {"groups": groups_out, "test": method, "statistic": statistic, "p_value": p_value}


def print_console_summary(results: Dict[str, object]) -> None:
    print("\n=== Clip-level test (each clip is an independently-sourced video - see module docstring) ===")
    for field, r in results["binary_white_vs_non_white"].items():
        if "error" in r:
            print(f"  [{field}] {r['error']}")
            continue
        w, nw = r["groups"]["white"], r["groups"]["non_white"]
        print(
            f"  [{field}] white acc={w['accuracy']:.3f} (n={w['n']}) vs non_white acc={nw['accuracy']:.3f} "
            f"(n={nw['n']}) -> {r['test']} p={r['p_value']}"
        )
    print("\n=== Full skin-tone-group breakdown (chi-square) ===")
    for field, r in results["full_breakdown"].items():
        if "error" in r:
            print(f"  [{field}] {r['error']}")
            continue
        print(f"  [{field}] -> {r['test']} p={r['p_value']}")
        for name, g in r["groups"].items():
            print(f"      {name:16s} acc={g['accuracy']:.3f} (n={g['n']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--max_clips", type=int, default=None, help="Cap total clips for a quick smoke test.")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--kinetics_root", type=Path, default=DEFAULT_KINETICS_ROOT)
    parser.add_argument("--labels_json", type=Path, default=DEFAULT_LABELS_JSON)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run_tag", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument(
        "--dry_run", action="store_true", help="Resolve labeled clips + print skin-tone distribution, no inference."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_caches(args.cache_dir)

    clips, skipped = load_labeled_clips(args.labels_json, args.kinetics_root, args.max_clips, args.seed)
    if not clips:
        print("No clips resolved - check --labels_json / --kinetics_root paths.", file=sys.stderr)
        raise SystemExit(1)

    out_dir = args.out_dir / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print_label_distribution(clips)
        write_json(
            {"clips": [{**asdict(c), "abs_path": str(c.abs_path)} for c in clips], "skipped": skipped},
            out_dir / "dry_run_clips.json",
        )
        return

    _check_runtime_dependencies()
    device = resolve_device(args.device)
    print(f"Resolved device: {device}")
    print_label_distribution(clips)

    class_map = {K400_CLASS_DIR: K400_TARGET}
    all_rows: List[Dict[str, object]] = []
    all_failed: List[Dict[str, str]] = list(skipped)
    for model_name in args.models:
        print(f"=== {model_name} ===")
        rows, failed = run_model_with_device_fallback(model_name, clips, class_map, device, args.num_frames, args.log_every)
        add_basketball_family_column(rows)
        for row in rows:
            row.pop("y_pred_top1_restricted", None)
            row.pop("top1_prob_restricted", None)
            row.pop("correct_top1_restricted", None)
        all_rows.extend(rows)
        all_failed.extend(failed)

    write_csv(all_rows, out_dir / "predictions.csv")

    results = {
        "binary_white_vs_non_white": {
            field: binary_fisher_test(all_rows, field) for field in ("correct_top1_400", "correct_basketball_family")
        },
        "full_breakdown": {
            field: full_chi2_test(all_rows, field) for field in ("correct_top1_400", "correct_basketball_family")
        },
    }
    write_json(results, out_dir / "skin_tone_stats.json")
    write_json(all_failed, out_dir / "skipped_clips.json")
    write_json(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "n_clips": len(clips),
        },
        out_dir / "run_config.json",
    )

    print_console_summary(results)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
