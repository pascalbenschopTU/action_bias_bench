"""
Zero-shot Kinetics-400 evaluation of real PA-HMDB51 clips, stratified by the
PA-HMDB51 skin-tone annotation, to check whether an off-the-shelf K400-pretrained
video classifier's accuracy correlates with the annotated skin-tone group on real
video (not synthetic). No fine-tuning: each torchvision video model is used exactly
as pretrained, with its full 400-way Kinetics-400 classification head intact.

Restricted to a hand-verified subset of HMDB51 classes that have a clean Kinetics-400
counterpart (see HMDB_TO_K400 below), e.g. HMDB51 "dribble" -> K400 "dribbling
basketball". draw_sword/fencing/sword/sword_exercise are merged into one
"sword_family" group in the summary outputs since K400 only has a single generic
"sword fighting" class - there is no distinct K400 target for each.

Background: see llm_reports/pa_hmdb51_real_data_plan.md for why this experiment
exists and its caveats (skin tone here is observational/confounded by performer
identity, video source/era, and the action class itself - not causal like the
synthetic Ctrl-A-Bias counterfactual-recoloring work).

Usage (run from the ActionBiasBench directory):
    # 1. sanity-check the class mapping + label distribution, no model loading:
    python scripts/eval_pahmdb51_zero_shot.py --dry_run

    # 2. quick smoke test, one model, 2 clips/class:
    python scripts/eval_pahmdb51_zero_shot.py --models r3d_18 --max_clips_per_class 2

    # 3. full run, all 6 models:
    python scripts/eval_pahmdb51_zero_shot.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
DEFAULT_HMDB51_ROOT = WORKSPACE_ROOT / "datasets" / "hmdb51"
DEFAULT_PA_HMDB51_REPO = REPO_ROOT.parent / "appearance_free_cross_domain_action_recognition"
DEFAULT_OUT_ROOT = REPO_ROOT / "out" / "hmdb_pahmdb51_zero_shot"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache"

MODEL_NAMES = ["r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s", "s3d", "swin3d_s"]

# Hand-verified against the real torchvision Kinetics-400 category list (all 6
# architectures share the same 400 names/order). See llm_reports/pa_hmdb51_real_data_plan.md
# for classes considered and rejected as too ambiguous to map.
SWORD_FAMILY = {"draw_sword", "fencing", "sword", "sword_exercise"}

HMDB_TO_K400: Dict[str, str] = {
    "cartwheel": "cartwheeling",
    "clap": "clapping",
    "climb": "rock climbing",
    "dive": "springboard diving",
    "dribble": "dribbling basketball",
    "shoot_ball": "shooting basketball",
    "drink": "drinking",
    "hug": "hugging",
    "kiss": "kissing",
    "laugh": "laughing",
    "pullup": "pull ups",
    "pushup": "push up",
    "situp": "situp",
    "ride_bike": "riding a bike",
    "ride_horse": "riding or walking with horse",
    "somersault": "somersaulting",
    "smoke": "smoking",
    "brush_hair": "brushing hair",
    "shoot_bow": "archery",
    "punch": "punching person (boxing)",
    "golf": "golf driving",
    # sword family: no distinct K400 target exists for each - all merged onto the
    # one available K400 class and reported as a single "sword_family" group.
    "draw_sword": "sword fighting",
    "fencing": "sword fighting",
    "sword": "sword fighting",
    "sword_exercise": "sword fighting",
}

# unidentifiable is excluded from both groupings below (no visible skin tone, not a
# fair comparison group).
BINARY_GROUP_MAP: Dict[str, Optional[str]] = {
    "white": "white",
    "yellow": "non_white",
    "black": "non_white",
    "mixed_skin_color": "non_white",
    "unidentifiable": None,
}

CONFOUND_NOTE = (
    "Pooled across HMDB51 classes with very different skin-tone label distributions "
    "(e.g. 'dribble' is 0% white, several other classes are ~100% white) - a pooled "
    "accuracy gap conflates class difficulty with skin tone. Cross-check against "
    "per_class_summary before treating this as evidence of a skin-tone effect."
)


@dataclass
class ClipRecord:
    hmdb_class: str
    k400_group: str
    k400_target: str
    video_name: str
    rel_path: str
    abs_path: Path
    skin_color: str
    skin_color_id: int
    review: bool
    # Empty for the main multi-class pipeline. Populated by focused single-class
    # analyses (e.g. eval_dribble_scaffold.py) that have enough clips per class to
    # group by source video - clips cut from the same source video share the same
    # performer/background and are not independent evidence about skin tone.
    source_video_id: str = ""


def configure_caches(cache_root: Path) -> None:
    """Route all torch/HF/xdg caches under a project-local dir, never ~/.cache."""
    torch_home = cache_root / "torch"
    hf_home = cache_root / "huggingface"
    xdg_home = cache_root / "xdg"
    for d in (torch_home, hf_home, xdg_home):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["XDG_CACHE_HOME"] = str(xdg_home)


def _check_runtime_dependencies() -> None:
    missing = []
    for module_name in ("torch", "torchvision", "av"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing required package(s): "
            + ", ".join(missing)
            + ". Run this script inside a conda env that has torch/torchvision/av "
            "installed, e.g.:\n"
            "  conda run -n video_features python scripts/eval_pahmdb51_zero_shot.py ...\n"
            "(no environment is validated as ready by this script - check before running)."
        )


def _import_pa_hmdb51(pa_hmdb51_repo: Path):
    repo_str = str(pa_hmdb51_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from privacy.pa_hmdb51 import ATTRIBUTE_CLASS_NAMES, load_pa_hmdb51_records  # type: ignore

    return load_pa_hmdb51_records, ATTRIBUTE_CLASS_NAMES


def load_annotated_clips(
    pa_hmdb51_repo: Path,
    hmdb51_root: Path,
    class_map: Dict[str, str],
    classes_filter: Optional[Sequence[str]],
    max_clips_per_class: Optional[int],
    seed: int,
) -> Tuple[List[ClipRecord], List[Dict[str, str]], List[str]]:
    load_pa_hmdb51_records, attribute_class_names = _import_pa_hmdb51(pa_hmdb51_repo)
    skin_color_names = list(attribute_class_names["skin_color"])

    allowed = set(class_map)
    if classes_filter:
        requested = set(classes_filter)
        unknown = requested - allowed
        if unknown:
            print(f"[WARN] --classes not in mapping table (ignored): {sorted(unknown)}", file=sys.stderr)
        allowed = allowed & requested

    attr_dir = pa_hmdb51_repo / "privacy" / "data" / "pa_hmdb51" / "PrivacyAttributes"
    records = load_pa_hmdb51_records(attr_dir)

    by_class: Dict[str, List] = defaultdict(list)
    for rec in records:
        if rec.action_class in allowed:
            by_class[rec.action_class].append(rec)

    rng = random.Random(seed)
    clips: List[ClipRecord] = []
    skipped: List[Dict[str, str]] = []
    for hmdb_class, recs in sorted(by_class.items()):
        recs = sorted(recs, key=lambda r: r.video_name)
        if max_clips_per_class is not None and len(recs) > max_clips_per_class:
            recs = rng.sample(recs, max_clips_per_class)
        group = "sword_family" if hmdb_class in SWORD_FAMILY else hmdb_class
        k400_target = class_map[hmdb_class]
        for rec in recs:
            abs_path = hmdb51_root / rec.rel_path
            if not abs_path.is_file():
                skipped.append({"rel_path": rec.rel_path, "reason": "missing_video_file"})
                continue
            skin_id = rec.labels["skin_color"]
            clips.append(
                ClipRecord(
                    hmdb_class=hmdb_class,
                    k400_group=group,
                    k400_target=k400_target,
                    video_name=rec.video_name,
                    rel_path=rec.rel_path,
                    abs_path=abs_path,
                    skin_color=skin_color_names[skin_id],
                    skin_color_id=skin_id,
                    review=rec.review,
                )
            )
    return clips, skipped, skin_color_names


def decode_video_frames(path: Path, num_frames: int = 16):
    """Decode a video with PyAV and uniformly sample num_frames as (T,C,H,W) uint8."""
    import av
    import numpy as np
    import torch

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    container.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")

    total = len(frames)
    idxs = np.round(np.linspace(0, total - 1, num=min(num_frames, total))).astype(int)
    sampled = [frames[i] for i in idxs]
    while len(sampled) < num_frames:
        sampled.append(sampled[-1])
    clip = np.stack(sampled[:num_frames], axis=0)  # (T,H,W,C) uint8
    return torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()  # (T,C,H,W) uint8


def get_model_registry():
    from torchvision.models.video import (
        MC3_18_Weights,
        MViT_V2_S_Weights,
        R2Plus1D_18_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_S_Weights,
        mc3_18,
        mvit_v2_s,
        r2plus1d_18,
        r3d_18,
        s3d,
        swin3d_s,
    )

    return {
        "r3d_18": (r3d_18, R3D_18_Weights),
        "mc3_18": (mc3_18, MC3_18_Weights),
        "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights),
        "s3d": (s3d, S3D_Weights),
        "swin3d_s": (swin3d_s, Swin3D_S_Weights),
    }


def resolve_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name: str, device):
    registry = get_model_registry()
    builder, weights_cls = registry[model_name]
    weights = weights_cls.DEFAULT
    model = builder(weights=weights)
    model.eval()
    model.to(device)
    transform = weights.transforms()
    categories = list(weights.meta["categories"])
    return model, transform, categories


def predict_clip(
    model,
    transform,
    frames,
    device,
    category_index: Dict[str, int],
    categories: List[str],
    restricted_indices: List[int],
    restricted_targets: List[str],
    y_true_target: str,
) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F

    x = transform(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
    probs = F.softmax(logits.float(), dim=0)

    top2 = torch.topk(probs, k=2)
    top1_idx = int(top2.indices[0].item())
    clamped = probs.clamp_min(1e-12)
    entropy = float(-(clamped * clamped.log()).sum().item())

    restricted_probs = probs[restricted_indices]
    restricted_probs = restricted_probs / restricted_probs.sum()
    r_top1 = int(torch.argmax(restricted_probs).item())

    true_idx = category_index[y_true_target]

    return {
        "y_pred_top1_400": categories[top1_idx],
        "top1_prob_400": float(top2.values[0].item()),
        "top2_prob_400": float(top2.values[1].item()),
        "margin_400": float((top2.values[0] - top2.values[1]).item()),
        "entropy_400": entropy,
        "true_class_prob_400": float(probs[true_idx].item()),
        "y_pred_top1_restricted": restricted_targets[r_top1],
        "top1_prob_restricted": float(restricted_probs[r_top1].item()),
    }


def run_model_over_clips(
    model_name: str,
    clips: List[ClipRecord],
    class_map: Dict[str, str],
    device,
    num_frames: int,
    log_every: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    model, transform, categories = load_pretrained_model(model_name, device)
    category_index = {name: i for i, name in enumerate(categories)}
    restricted_targets = sorted(set(class_map.values()))
    missing = [t for t in restricted_targets if t not in category_index]
    if missing:
        raise RuntimeError(f"K400 target(s) not found in category list: {missing}")
    restricted_indices = [category_index[t] for t in restricted_targets]

    rows: List[Dict[str, object]] = []
    failed: List[Dict[str, str]] = []
    start = time.time()
    for i, clip in enumerate(clips, start=1):
        try:
            frames = decode_video_frames(clip.abs_path, num_frames=num_frames)
            pred = predict_clip(
                model,
                transform,
                frames,
                device,
                category_index,
                categories,
                restricted_indices,
                restricted_targets,
                clip.k400_target,
            )
        except Exception as exc:  # noqa: BLE001 - log and keep going, one bad clip shouldn't kill the run
            failed.append({"model": model_name, "rel_path": clip.rel_path, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        row: Dict[str, object] = {
            "model": model_name,
            "hmdb_class": clip.hmdb_class,
            "k400_group": clip.k400_group,
            "k400_target": clip.k400_target,
            "rel_path": clip.rel_path,
            "video_name": clip.video_name,
            "skin_color": clip.skin_color,
            "source_video_id": clip.source_video_id,
            "y_true": clip.k400_target,
            **pred,
        }
        row["correct_top1_400"] = row["y_pred_top1_400"] == clip.k400_target
        row["correct_top1_restricted"] = row["y_pred_top1_restricted"] == clip.k400_target
        rows.append(row)

        if log_every and i % log_every == 0:
            print(f"  [{model_name}] {i}/{len(clips)} clips ({time.time() - start:.1f}s)", flush=True)
    return rows, failed


def run_model_with_device_fallback(
    model_name: str,
    clips: List[ClipRecord],
    class_map: Dict[str, str],
    device,
    num_frames: int,
    log_every: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    try:
        return run_model_over_clips(model_name, clips, class_map, device, num_frames, log_every)
    except RuntimeError as exc:
        if device.type == "cpu":
            raise
        import torch

        print(f"[{model_name}] device={device} failed ({exc}); retrying on cpu", flush=True)
        return run_model_over_clips(model_name, clips, class_map, torch.device("cpu"), num_frames, log_every)


def summarize_per_class(rows: List[Dict[str, object]], skin_color_names: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["k400_group"]))
        record = groups.setdefault(
            key,
            {
                "model": row["model"],
                "k400_group": row["k400_group"],
                "k400_target": row["k400_target"],
                "hmdb_classes": set(),
                "n": 0,
                "skin_counts": Counter(),
                "n_correct_top1_400": 0,
                "n_correct_top1_restricted": 0,
            },
        )
        record["hmdb_classes"].add(row["hmdb_class"])
        record["n"] += 1
        record["skin_counts"][row["skin_color"]] += 1
        record["n_correct_top1_400"] += int(bool(row["correct_top1_400"]))
        record["n_correct_top1_restricted"] += int(bool(row["correct_top1_restricted"]))

    out: List[Dict[str, object]] = []
    for (model_name, k400_group), record in sorted(groups.items()):
        n = int(record["n"])
        row_out: Dict[str, object] = {
            "model": model_name,
            "k400_group": k400_group,
            "k400_target": record["k400_target"],
            "hmdb_classes": ",".join(sorted(record["hmdb_classes"])),
            "n": n,
        }
        for name in skin_color_names:
            row_out[f"n_{name}"] = record["skin_counts"].get(name, 0)
        row_out["n_correct_top1_400"] = record["n_correct_top1_400"]
        row_out["accuracy_top1_400"] = record["n_correct_top1_400"] / n if n else float("nan")
        row_out["n_correct_top1_restricted"] = record["n_correct_top1_restricted"]
        row_out["accuracy_top1_restricted"] = record["n_correct_top1_restricted"] / n if n else float("nan")
        out.append(row_out)
    return out


def _hypergeom_pmf(a: int, row1: int, col1: int, n: int) -> float:
    b, c, d = row1 - a, col1 - a, n - row1 - col1 + a
    if a < 0 or b < 0 or c < 0 or d < 0:
        return 0.0
    return (math.comb(row1, a) * math.comb(n - row1, col1 - a)) / math.comb(n, col1)


def fisher_exact_2x2_fallback(table: List[List[int]]) -> Tuple[float, float, str]:
    """Two-sided Fisher's exact test with no scipy dependency (small counts only)."""
    (a, b), (c, d) = table
    row1, row2 = a + b, c + d
    col1 = a + c
    n = row1 + row2
    observed_p = _hypergeom_pmf(a, row1, col1, n)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    p_value = sum(
        p for k in range(lo, hi + 1) if (p := _hypergeom_pmf(k, row1, col1, n)) <= observed_p * (1 + 1e-9)
    )
    odds_ratio = (a * d) / (b * c) if b * c != 0 else float("inf")
    return odds_ratio, p_value, "manual_fisher_exact"


def _binary_contingency(rows: List[Dict[str, object]], accuracy_field: str) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for row in rows:
        group = BINARY_GROUP_MAP.get(str(row["skin_color"]))
        if group is None:
            continue
        counts[group]["correct" if row[accuracy_field] else "incorrect"] += 1
    return counts


def _full_contingency(rows: List[Dict[str, object]], accuracy_field: str) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for row in rows:
        color = str(row["skin_color"])
        if color == "unidentifiable":
            continue
        counts[color]["correct" if row[accuracy_field] else "incorrect"] += 1
    return counts


def pooled_skin_tone_stats(
    rows: List[Dict[str, object]],
    accuracy_field: str,
    scope_model: Optional[str],
) -> List[Dict[str, object]]:
    if scope_model is not None:
        rows = [r for r in rows if r["model"] == scope_model]
    scope = scope_model or "pooled_all_models"
    results: List[Dict[str, object]] = []

    binary_counts = _binary_contingency(rows, accuracy_field)
    if "white" in binary_counts and "non_white" in binary_counts:
        table = [
            [binary_counts["white"]["correct"], binary_counts["white"]["incorrect"]],
            [binary_counts["non_white"]["correct"], binary_counts["non_white"]["incorrect"]],
        ]
        try:
            from scipy import stats  # type: ignore

            statistic, p_value = stats.fisher_exact(table)
            method = "scipy_fisher_exact"
        except Exception:
            statistic, p_value, method = fisher_exact_2x2_fallback(table)
        groups_out = {
            name: {
                "n": binary_counts[name]["correct"] + binary_counts[name]["incorrect"],
                "n_correct": binary_counts[name]["correct"],
                "accuracy": binary_counts[name]["correct"] / (binary_counts[name]["correct"] + binary_counts[name]["incorrect"])
                if (binary_counts[name]["correct"] + binary_counts[name]["incorrect"])
                else float("nan"),
            }
            for name in ("white", "non_white")
        }
        results.append(
            {
                "scope": scope,
                "grouping": "binary",
                "accuracy_field": accuracy_field,
                "groups": groups_out,
                "test": method,
                "statistic": statistic,
                "p_value": p_value,
                "n_total": sum(g["n"] for g in groups_out.values()),
                "confound_note": CONFOUND_NOTE,
            }
        )
    else:
        results.append(
            {
                "scope": scope,
                "grouping": "binary",
                "accuracy_field": accuracy_field,
                "error": "one or both groups (white/non_white) empty after excluding 'unidentifiable' - cannot run test",
            }
        )

    full_counts = _full_contingency(rows, accuracy_field)
    full_names = sorted(full_counts)
    if len(full_names) >= 2:
        table = [[full_counts[name]["correct"], full_counts[name]["incorrect"]] for name in full_names]
        try:
            from scipy import stats  # type: ignore

            statistic, p_value, _, _ = stats.chi2_contingency(table)
            method = "scipy_chi2_contingency"
        except Exception:
            statistic, p_value, method = float("nan"), float("nan"), "scipy_unavailable_no_manual_fallback"
        groups_out = {
            name: {
                "n": full_counts[name]["correct"] + full_counts[name]["incorrect"],
                "n_correct": full_counts[name]["correct"],
                "accuracy": full_counts[name]["correct"] / (full_counts[name]["correct"] + full_counts[name]["incorrect"])
                if (full_counts[name]["correct"] + full_counts[name]["incorrect"])
                else float("nan"),
            }
            for name in full_names
        }
        results.append(
            {
                "scope": scope,
                "grouping": "full",
                "accuracy_field": accuracy_field,
                "groups": groups_out,
                "test": method,
                "statistic": statistic,
                "p_value": p_value,
                "n_total": sum(g["n"] for g in groups_out.values()),
                "confound_note": CONFOUND_NOTE,
            }
        )
    else:
        results.append(
            {
                "scope": scope,
                "grouping": "full",
                "accuracy_field": accuracy_field,
                "error": "fewer than 2 non-'unidentifiable' skin-tone groups present - cannot run test",
            }
        )

    return results


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def print_dry_run_summary(clips: List[ClipRecord], skipped: List[Dict[str, str]], skin_color_names: List[str]) -> None:
    n_classes = len({c.hmdb_class for c in clips})
    n_groups = len({c.k400_group for c in clips})
    print(f"Resolved {len(clips)} clips across {n_classes} HMDB51 classes ({n_groups} K400-mapped groups). Skipped: {len(skipped)}.")
    print()
    by_group: Dict[str, Counter] = defaultdict(Counter)
    target_for_group: Dict[str, str] = {}
    for c in clips:
        by_group[c.k400_group][c.skin_color] += 1
        target_for_group[c.k400_group] = c.k400_target

    short_names = [name[:4] for name in skin_color_names]
    print(f"{'k400_group':22s} {'k400_target':28s} " + " ".join(f"{n:>6s}" for n in short_names) + "  total")
    for group in sorted(by_group):
        counts = by_group[group]
        total = sum(counts.values())
        row = f"{group:22s} {target_for_group[group]:28s} " + " ".join(f"{counts.get(n, 0):6d}" for n in skin_color_names)
        print(row + f"  {total:5d}")
    if skipped:
        print(f"\n{len(skipped)} clip(s) skipped (missing video file) - see dry_run_clips.json for details.")


def print_console_summary(per_class: List[Dict[str, object]], pooled: List[Dict[str, object]]) -> None:
    print("\n=== Overall accuracy by model (top1_restricted / top1_400) ===")
    by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "c400": 0, "crestricted": 0})
    for row in per_class:
        m = by_model[str(row["model"])]
        m["n"] += int(row["n"])
        m["c400"] += int(row["n_correct_top1_400"])
        m["crestricted"] += int(row["n_correct_top1_restricted"])
    for model_name, m in sorted(by_model.items()):
        n = m["n"]
        if n:
            print(f"  {model_name:12s} n={n:4d}  top1_restricted={m['crestricted'] / n:.3f}  top1_400={m['c400'] / n:.3f}")

    print("\n=== Pooled binary (white vs non_white) skin-tone stats, all models combined ===")
    for record in pooled:
        if record.get("grouping") != "binary" or record.get("scope") != "pooled_all_models":
            continue
        if "error" in record:
            print(f"  [{record['accuracy_field']}] {record['error']}")
            continue
        groups = record["groups"]
        white, non_white = groups["white"], groups["non_white"]
        print(
            f"  [{record['accuracy_field']}] white acc={white['accuracy']:.3f} (n={white['n']}) "
            f"vs non_white acc={non_white['accuracy']:.3f} (n={non_white['n']}) "
            f"-> {record['test']} p={record['p_value']}"
        )
    print("\n(Read the confound_note in pooled_skin_tone_stats.json before treating this as a bias finding.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Restrict to a subset of HMDB51 class names (default: all classes in HMDB_TO_K400).",
    )
    parser.add_argument("--max_clips_per_class", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--hmdb51_root", type=Path, default=DEFAULT_HMDB51_ROOT)
    parser.add_argument("--pa_hmdb51_repo", type=Path, default=DEFAULT_PA_HMDB51_REPO)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run_tag", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--dry_run", action="store_true", help="Resolve mapping + load clips + print distribution, no inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_caches(args.cache_dir)

    class_map = dict(HMDB_TO_K400)
    clips, skipped, skin_color_names = load_annotated_clips(
        pa_hmdb51_repo=args.pa_hmdb51_repo,
        hmdb51_root=args.hmdb51_root,
        class_map=class_map,
        classes_filter=args.classes,
        max_clips_per_class=args.max_clips_per_class,
        seed=args.seed,
    )
    if not clips:
        print("No clips resolved - check --hmdb51_root / --pa_hmdb51_repo paths.", file=sys.stderr)
        raise SystemExit(1)

    out_dir = args.out_dir / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print_dry_run_summary(clips, skipped, skin_color_names)
        write_json(
            {
                "clips": [{**asdict(c), "abs_path": str(c.abs_path)} for c in clips],
                "skipped": skipped,
            },
            out_dir / "dry_run_clips.json",
        )
        return

    _check_runtime_dependencies()
    device = resolve_device(args.device)
    print(f"Resolved device: {device}")

    all_rows: List[Dict[str, object]] = []
    all_failed: List[Dict[str, str]] = list(skipped)
    for model_name in args.models:
        print(f"=== {model_name} ===")
        rows, failed = run_model_with_device_fallback(model_name, clips, class_map, device, args.num_frames, args.log_every)
        all_rows.extend(rows)
        all_failed.extend(failed)

    write_csv(all_rows, out_dir / "predictions.csv")

    per_class = summarize_per_class(all_rows, skin_color_names)
    write_csv(per_class, out_dir / "per_class_summary.csv")
    write_json(per_class, out_dir / "per_class_summary.json")

    pooled: List[Dict[str, object]] = []
    for accuracy_field in ("correct_top1_400", "correct_top1_restricted"):
        for model_name in args.models:
            pooled.extend(pooled_skin_tone_stats(all_rows, accuracy_field, scope_model=model_name))
        pooled.extend(pooled_skin_tone_stats(all_rows, accuracy_field, scope_model=None))
    write_json(pooled, out_dir / "pooled_skin_tone_stats.json")

    write_json(all_failed, out_dir / "skipped_clips.json")
    write_json(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "class_map": class_map,
            "n_clips": len(clips),
            "n_predictions": len(all_rows),
            "n_failed": len(all_failed),
        },
        out_dir / "run_config.json",
    )

    print_console_summary(per_class, pooled)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
