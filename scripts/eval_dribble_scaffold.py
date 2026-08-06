"""
Focused zero-shot Kinetics-400 skin-tone check on the expanded PA-HMDB51 "dribble"
scaffold labels (145 clips vs. the original 6 in dribble.json), with a background/
performer confound control.

Why this exists: HMDB51 clips are often cut from the same source video (same
performer, same court/gym, same lighting/camera). Multiple clips from one source are
NOT independent evidence about skin tone - they're repeated measures of one
"background". Naively pooling all 145 clips (as eval_pahmdb51_zero_shot.py does for
the other classes) would treat e.g. 10 clips from one video as 10 independent data
points, wildly overstating statistical power. This script:

  1. Groups clips by source-video ID (parsed from the HMDB51 filename convention:
     everything before "_dribble_").
  2. Runs the same off-the-shelf Kinetics-400-pretrained torchvision models as
     eval_pahmdb51_zero_shot.py (no fine-tuning) on every clip.
  3. Reports TWO tests side by side:
       - "naive" clip-level test (n=145, what you'd get ignoring the background
         confound)
       - "background-controlled" test (one accuracy value per *source video*,
         n = number of distinct source videos - the actually-independent sample)
     so you can see how much the naive number was inflated by pseudo-replication.

Two accuracy definitions are reported: exact match to "dribbling basketball", and a
"basketball family" tolerant match (dribbling/shooting/playing/dunking basketball) -
the earlier full-dataset run showed models often pick a sibling basketball label.

Usage (run from the ActionBiasBench directory):
    python scripts/eval_dribble_scaffold.py --dry_run
    python scripts/eval_dribble_scaffold.py --models r3d_18
    python scripts/eval_dribble_scaffold.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval_pahmdb51_zero_shot import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_HMDB51_ROOT,
    MODEL_NAMES,
    REPO_ROOT,
    ClipRecord,
    _binary_contingency,
    _check_runtime_dependencies,
    configure_caches,
    fisher_exact_2x2_fallback,
    resolve_device,
    run_model_with_device_fallback,
    write_csv,
    write_json,
)

DEFAULT_SCAFFOLD_JSON = (
    REPO_ROOT.parent
    / "appearance_free_cross_domain_action_recognition"
    / "privacy"
    / "data"
    / "pa_hmdb51"
    / "PrivacyAttributes"
    / "dribble_scaffold.json"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "out" / "dribble_scaffold_zero_shot"

HMDB_CLASS = "dribble"
K400_TARGET = "dribbling basketball"
# Confirmed present in the real torchvision Kinetics-400 category list.
BASKETBALL_FAMILY = ["dribbling basketball", "shooting basketball", "playing basketball", "dunking basketball"]
SKIN_COLOR_NAMES = ["unidentifiable", "white", "yellow", "black", "mixed_skin_color"]


def _collapse_skin_color_label(raw_label) -> int:
    # Matches appearance_free_cross_domain_action_recognition/privacy/pa_hmdb51.py's
    # _collapse_label: any multi-label segment counts as mixed_skin_color (id 4).
    if isinstance(raw_label, list):
        return 4
    return int(raw_label)


def _majority_skin_color(segments) -> int:
    counts: Dict[int, int] = defaultdict(int)
    for start, end, raw_label in segments:
        duration = max(1, int(end) - int(start) + 1)
        counts[_collapse_skin_color_label(raw_label)] += duration
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def load_scaffold_clips(
    scaffold_json: Path,
    hmdb51_root: Path,
    max_clips: Optional[int],
    seed: int,
) -> Tuple[List[ClipRecord], List[Dict[str, str]]]:
    data = json.loads(scaffold_json.read_text())
    clips: List[ClipRecord] = []
    skipped: List[Dict[str, str]] = []
    for video_name, meta in sorted(data.items()):
        skin_id = _majority_skin_color(meta["skin_color"])
        rel_path = f"{HMDB_CLASS}/{video_name}"
        abs_path = hmdb51_root / HMDB_CLASS / video_name
        if not abs_path.is_file():
            skipped.append({"rel_path": rel_path, "reason": "missing_video_file"})
            continue
        source_id = video_name.split(f"_{HMDB_CLASS}_")[0]
        clips.append(
            ClipRecord(
                hmdb_class=HMDB_CLASS,
                k400_group=HMDB_CLASS,
                k400_target=K400_TARGET,
                video_name=video_name,
                rel_path=rel_path,
                abs_path=abs_path,
                skin_color=SKIN_COLOR_NAMES[skin_id],
                skin_color_id=skin_id,
                review=bool(meta.get("review", False)),
                source_video_id=source_id,
            )
        )

    if max_clips is not None and len(clips) > max_clips:
        import random

        clips = random.Random(seed).sample(clips, max_clips)
    return clips, skipped


def group_by_source(clips: List[ClipRecord]) -> Dict[str, List[ClipRecord]]:
    groups: Dict[str, List[ClipRecord]] = defaultdict(list)
    for c in clips:
        groups[c.source_video_id].append(c)
    return groups


def print_source_video_summary(clips: List[ClipRecord]) -> None:
    groups = group_by_source(clips)
    print(f"{len(clips)} clips across {len(groups)} distinct source videos.")
    label_counts: Dict[str, int] = defaultdict(int)
    inconsistent = 0
    for members in groups.values():
        labels = {m.skin_color for m in members}
        if len(labels) > 1:
            inconsistent += 1
            label_counts["inconsistent"] += 1
        else:
            label_counts[next(iter(labels))] += 1
    print(f"Sources with inconsistent skin_color across their own clips: {inconsistent}")
    print("Source-level skin-tone distribution:")
    for name, n in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:16s} {n:3d} sources")
    sizes: Dict[int, int] = defaultdict(int)
    for members in groups.values():
        sizes[len(members)] += 1
    print(f"Clips-per-source histogram (clips -> #sources with that many): {dict(sorted(sizes.items()))}")


def add_basketball_family_column(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        row["correct_basketball_family"] = row["y_pred_top1_400"] in BASKETBALL_FAMILY


def aggregate_by_source_per_model(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["source_video_id"]))
        g = groups.setdefault(
            key,
            {
                "model": row["model"],
                "source_video_id": row["source_video_id"],
                "skin_colors": set(),
                "n_clips": 0,
                "n_correct_exact": 0,
                "n_correct_family": 0,
            },
        )
        g["skin_colors"].add(row["skin_color"])
        g["n_clips"] += 1
        g["n_correct_exact"] += int(bool(row["correct_top1_400"]))
        g["n_correct_family"] += int(bool(row["correct_basketball_family"]))

    out: List[Dict[str, object]] = []
    for (model_name, source_id), g in sorted(groups.items()):
        n = int(g["n_clips"])
        out.append(
            {
                "model": model_name,
                "source_video_id": source_id,
                "skin_color": sorted(g["skin_colors"])[0] if len(g["skin_colors"]) == 1 else "inconsistent",
                "n_skin_color_labels": len(g["skin_colors"]),
                "n_clips": n,
                "accuracy_exact": g["n_correct_exact"] / n,
                "accuracy_family": g["n_correct_family"] / n,
            }
        )
    return out


def aggregate_by_source_pooled(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """One row per source video, pooling across all clips AND all models - the
    actually-independent unit for the background-controlled test."""
    groups: Dict[str, Dict[str, object]] = {}
    for row in rows:
        source_id = str(row["source_video_id"])
        g = groups.setdefault(
            source_id,
            {"source_video_id": source_id, "skin_colors": set(), "n": 0, "n_correct_exact": 0, "n_correct_family": 0},
        )
        g["skin_colors"].add(row["skin_color"])
        g["n"] += 1
        g["n_correct_exact"] += int(bool(row["correct_top1_400"]))
        g["n_correct_family"] += int(bool(row["correct_basketball_family"]))

    out: List[Dict[str, object]] = []
    for source_id, g in sorted(groups.items()):
        n = int(g["n"])
        out.append(
            {
                "source_video_id": source_id,
                "skin_color": sorted(g["skin_colors"])[0] if len(g["skin_colors"]) == 1 else "inconsistent",
                "n_skin_color_labels": len(g["skin_colors"]),
                "n_observations": n,  # clips x models for this source
                "accuracy_exact": g["n_correct_exact"] / n,
                "accuracy_family": g["n_correct_family"] / n,
            }
        )
    return out


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


def source_level_test(per_source_pooled: List[Dict[str, object]], accuracy_field: str) -> Dict[str, object]:
    white_acc = [r[accuracy_field] for r in per_source_pooled if r["skin_color"] == "white"]
    black_acc = [r[accuracy_field] for r in per_source_pooled if r["skin_color"] == "black"]
    if len(white_acc) < 2 or len(black_acc) < 2:
        return {"error": f"fewer than 2 source videos in white (n={len(white_acc)}) or black (n={len(black_acc)}) group"}
    try:
        from scipy import stats  # type: ignore

        statistic, p_value = stats.mannwhitneyu(white_acc, black_acc, alternative="two-sided")
        method = "scipy_mannwhitneyu"
    except Exception:
        statistic, p_value, method = float("nan"), float("nan"), "scipy_unavailable_no_manual_fallback"
    return {
        "white": {"n_sources": len(white_acc), "mean_accuracy": statistics.mean(white_acc)},
        "black": {"n_sources": len(black_acc), "mean_accuracy": statistics.mean(black_acc)},
        "test": method,
        "statistic": statistic,
        "p_value": p_value,
    }


def print_console_summary(results: Dict[str, object]) -> None:
    print("\n=== NAIVE clip-level test (pseudo-replicated - same source video repeated many times) ===")
    for field, r in results["clip_level_naive"].items():
        if "error" in r:
            print(f"  [{field}] {r['error']}")
            continue
        w, nw = r["groups"]["white"], r["groups"]["non_white"]
        print(
            f"  [{field}] white acc={w['accuracy']:.3f} (n={w['n']}) vs non_white acc={nw['accuracy']:.3f} "
            f"(n={nw['n']}) -> {r['test']} p={r['p_value']}"
        )

    print("\n=== BACKGROUND-CONTROLLED test (one row per source video - the independent sample) ===")
    for field, r in results["source_level_controlled"].items():
        if "error" in r:
            print(f"  [{field}] {r['error']}")
            continue
        w, b = r["white"], r["black"]
        print(
            f"  [{field}] white mean_acc={w['mean_accuracy']:.3f} (n_sources={w['n_sources']}) vs "
            f"black mean_acc={b['mean_accuracy']:.3f} (n_sources={b['n_sources']}) -> {r['test']} p={r['p_value']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--max_clips", type=int, default=None, help="Cap total clips for a quick smoke test.")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--hmdb51_root", type=Path, default=DEFAULT_HMDB51_ROOT)
    parser.add_argument("--scaffold_json", type=Path, default=DEFAULT_SCAFFOLD_JSON)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run_tag", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--dry_run", action="store_true", help="Resolve clips + print source-video distribution, no inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_caches(args.cache_dir)

    clips, skipped = load_scaffold_clips(args.scaffold_json, args.hmdb51_root, args.max_clips, args.seed)
    if not clips:
        print("No clips resolved - check --scaffold_json / --hmdb51_root paths.", file=sys.stderr)
        raise SystemExit(1)

    out_dir = args.out_dir / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print_source_video_summary(clips)
        write_json(
            {"clips": [{**asdict(c), "abs_path": str(c.abs_path)} for c in clips], "skipped": skipped},
            out_dir / "dry_run_clips.json",
        )
        return

    _check_runtime_dependencies()
    device = resolve_device(args.device)
    print(f"Resolved device: {device}")
    print_source_video_summary(clips)

    class_map = {HMDB_CLASS: K400_TARGET}
    all_rows: List[Dict[str, object]] = []
    all_failed: List[Dict[str, str]] = list(skipped)
    for model_name in args.models:
        print(f"=== {model_name} ===")
        rows, failed = run_model_with_device_fallback(model_name, clips, class_map, device, args.num_frames, args.log_every)
        add_basketball_family_column(rows)
        # These columns are degenerate here (class_map has one entry -> restricted
        # softmax always "predicts" the only candidate) - drop them to avoid
        # confusion; correct_basketball_family is the meaningful lenient metric.
        for row in rows:
            row.pop("y_pred_top1_restricted", None)
            row.pop("top1_prob_restricted", None)
            row.pop("correct_top1_restricted", None)
        all_rows.extend(rows)
        all_failed.extend(failed)

    write_csv(all_rows, out_dir / "predictions.csv")

    per_source_per_model = aggregate_by_source_per_model(all_rows)
    write_csv(per_source_per_model, out_dir / "per_source_per_model.csv")

    per_source_pooled = aggregate_by_source_pooled(all_rows)
    write_csv(per_source_pooled, out_dir / "per_source_pooled.csv")

    results = {
        "clip_level_naive": {
            field: binary_fisher_test(all_rows, field) for field in ("correct_top1_400", "correct_basketball_family")
        },
        "source_level_controlled": {
            field: source_level_test(per_source_pooled, field) for field in ("accuracy_exact", "accuracy_family")
        },
    }
    write_json(results, out_dir / "background_controlled_stats.json")
    write_json(all_failed, out_dir / "skipped_clips.json")
    write_json(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "n_clips": len(clips),
            "n_sources": len(per_source_pooled),
        },
        out_dir / "run_config.json",
    )

    print_console_summary(results)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
