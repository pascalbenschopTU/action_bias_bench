from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from .schema import DARK_VARIANT_ORDER, LIGHT_VARIANT_ORDER, SPLIT_FAMILY_TO_SPLITS, tone_group_for_variant
except ImportError:  # pragma: no cover - direct script execution
    from schema import DARK_VARIANT_ORDER, LIGHT_VARIANT_ORDER, SPLIT_FAMILY_TO_SPLITS, tone_group_for_variant

# Every matched row's tone group has exactly one *opposite* group in the
# shifted split (see build_skin_tone_shortcut_probe.py's eval_shifted_* specs,
# which always swap dark_variants<->light_variants wholesale) -- so the
# shifted split already contains predictions for *both* colors of that
# opposite group, not just a single fixed counterpart. Pairing against all of
# them (below) is what makes all 4 achievable color pairs (african<->white,
# indian<->white, african<->asian, indian<->asian) available for analysis,
# not just the 2 that a fixed one-to-one VARIANT_SWAP mapping would give.
OPPOSITE_GROUP_VARIANTS = {
    "dark": tuple(LIGHT_VARIANT_ORDER),
    "light": tuple(DARK_VARIANT_ORDER),
}

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_actionbiasbench"))

FEATURE_COLUMNS: List[str] = [
    "top1_prob",
    "top2_prob",
    "margin",
    "entropy",
    "true_class_prob",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "hue_mean",
    "contrast",
    "r_mean",
    "g_mean",
    "b_mean",
]
NUMERIC_COLUMNS: List[str] = [
    "y_true",
    "y_pred",
    "correct",
    *FEATURE_COLUMNS,
]
BASE_ID_RE = re.compile(
    r"^(?P<action>.+)_(?P<base_id>\d+)_(?:modified_(?P<variant>[^.]+)|(?P<initial>initial))(?:\..+)?$",
    re.IGNORECASE,
)
VARIANT_RE = re.compile(r"_modified_([^/_]+?)(?:\.[^.]+|$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-sample skin-tone swap influence on benchmark prediction exports.")
    parser.add_argument("--root", type=Path, required=True, help="Benchmark output root (contains modality outputs).")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model names or 'all'.")
    parser.add_argument(
        "--split_families",
        type=str,
        default="seen,unseen",
        help="Comma-separated split families: seen,unseen",
    )
    parser.add_argument("--metric", type=str, default="f1_macro", help="Metric label to store in metadata.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory. Defaults to --root.")
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    return parser.parse_args()


def parse_csv_set(raw: str) -> set[str] | None:
    value = str(raw).strip()
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_split_families(raw: str) -> List[str]:
    families = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not families:
        return ["seen", "unseen"]
    bad = [family for family in families if family not in SPLIT_FAMILY_TO_SPLITS]
    if bad:
        raise ValueError(f"Unsupported split families: {bad}. Supported: {sorted(SPLIT_FAMILY_TO_SPLITS)}")
    return families


def to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def to_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


_SEED_FOLD_RE = re.compile(r"^seed_(?P<seed>\d+)(?:fold(?P<fold>\d+))?$")


def _parse_seed_fold_dirname(name: str) -> Tuple[str, str]:
    """Split a "seed_{seed}fold{fold}" (CV) or "seed_{seed}" (non-CV) directory
    name into (seed, fold). fold is "" for non-CV runs -- there is nothing to
    dedupe against in that case."""
    match = _SEED_FOLD_RE.match(name)
    if not match:
        return name.replace("seed_", "", 1), ""
    return match.group("seed"), (match.group("fold") or "")


def infer_context_from_prediction_path(root: Path, csv_path: Path) -> Dict[str, str]:
    context = {"model": "", "pair_tag": "", "seed": "", "fold": "", "eval_split": ""}
    try:
        rel_parts = csv_path.relative_to(root).parts
    except Exception:
        rel_parts = csv_path.parts
    rel_parts = tuple(str(part) for part in rel_parts)
    if "rgb_torchvision" in rel_parts:
        idx = rel_parts.index("rgb_torchvision")
        if idx + 1 < len(rel_parts):
            context["model"] = rel_parts[idx + 1]
        if idx + 2 < len(rel_parts):
            context["pair_tag"] = rel_parts[idx + 2]
        if idx + 3 < len(rel_parts):
            context["seed"], context["fold"] = _parse_seed_fold_dirname(rel_parts[idx + 3])
        if idx + 4 < len(rel_parts):
            context["eval_split"] = rel_parts[idx + 4]
        return context
    if "flow_i3d_external" in rel_parts:
        idx = rel_parts.index("flow_i3d_external")
        context["model"] = "flow_i3d_external_model"
        if idx + 1 < len(rel_parts):
            context["pair_tag"] = rel_parts[idx + 1]
        if idx + 2 < len(rel_parts):
            context["seed"], context["fold"] = _parse_seed_fold_dirname(rel_parts[idx + 2])
        if idx + 3 < len(rel_parts):
            context["eval_split"] = rel_parts[idx + 3]
    return context


def extract_variant(rel_path: str) -> str:
    rel = str(rel_path)
    match = VARIANT_RE.search(rel)
    if match:
        return str(match.group(1)).lower()
    if "_initial." in rel.lower():
        return "initial"
    return "unknown"


def parse_clip_identity(rel_path: str) -> Dict[str, object]:
    pure = PurePosixPath(str(rel_path))
    parts = list(pure.parts)
    background = parts[0] if parts else ""
    action = ""
    for idx, part in enumerate(parts):
        if part == "__generated_synthetic_videos" and idx + 1 < len(parts):
            action = parts[idx + 1]
            break
    if not action and len(parts) >= 2:
        action = parts[-2]

    variant = extract_variant(str(rel_path))
    base_id: int | None = None
    stem = pure.name
    for suffix in (".zst", ".npz", ".npy", ".mp4", ".avi", ".mov", ".mkv", ".webm"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = BASE_ID_RE.match(stem)
    if match:
        if not action:
            action = str(match.group("action") or "")
        try:
            base_id = int(match.group("base_id"))
        except Exception:
            base_id = None
        parsed_variant = str(match.group("variant") or ("initial" if match.group("initial") else "")).lower()
        if parsed_variant:
            variant = parsed_variant
    tone_group = tone_group_for_variant(variant)
    return {
        "background": str(background),
        "action": str(action),
        "base_id": base_id if base_id is not None else "",
        "variant": str(variant),
        "tone_group": str(tone_group),
    }


def normalize_prediction_row(
    root: Path,
    csv_path: Path,
    row: Dict[str, object],
) -> Dict[str, object]:
    inferred = infer_context_from_prediction_path(root, csv_path)
    rel_path = str(row.get("rel_path", "")).replace("\\", "/")
    clip_info = parse_clip_identity(rel_path)

    model = str(row.get("model", "") or inferred["model"]).strip()
    pair_tag = str(row.get("pair_tag", "") or inferred["pair_tag"]).strip()
    seed = str(row.get("seed", "") or inferred["seed"]).strip()
    # fold is CV-fold-only and never present in the per-clip prediction CSV
    # itself (that CSV only ever logs "seed") -- always taken from the
    # seed_{seed}fold{fold} directory name. Empty string for non-CV runs.
    fold = str(inferred["fold"]).strip()
    eval_split = str(row.get("eval_split", "") or inferred["eval_split"]).strip()

    normalized: Dict[str, object] = {
        "model": model,
        "pair_tag": pair_tag,
        "seed": seed,
        "fold": fold,
        "eval_split": eval_split,
        "rel_path": rel_path,
        "background": str(row.get("background", "") or clip_info["background"]),
        "action": str(row.get("action", "") or clip_info["action"]),
        "base_id": row.get("base_id", clip_info["base_id"]),
        "variant": str(row.get("variant", "") or clip_info["variant"]).lower(),
        "tone_group": str(row.get("tone_group", "") or clip_info["tone_group"]).lower(),
    }
    normalized["base_id"] = to_int(normalized["base_id"], default=-1)

    for column in NUMERIC_COLUMNS:
        if column in {"y_true", "y_pred", "correct"}:
            normalized[column] = to_int(row.get(column), default=-1)
        else:
            normalized[column] = to_float(row.get(column))
    return normalized


def load_prediction_rows(root: Path, models_filter: set[str] | None) -> Tuple[List[Dict[str, object]], List[Path]]:
    rgb_paths = sorted((root / "rgb_torchvision").rglob("predictions_rgb_*.csv")) if (root / "rgb_torchvision").exists() else []
    flow_paths = sorted((root / "flow_i3d_external").rglob("predictions_flow_i3d_external_model.csv")) if (root / "flow_i3d_external").exists() else []
    csv_paths = sorted(set(rgb_paths + flow_paths))
    if not csv_paths:
        csv_paths = sorted(root.rglob("predictions_*.csv"))
    rows: List[Dict[str, object]] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                normalized = normalize_prediction_row(root, csv_path, dict(raw_row))
                model_name = str(normalized["model"])
                if not model_name:
                    continue
                if models_filter is not None and model_name not in models_filter:
                    continue
                rows.append(normalized)
    return rows, csv_paths


def canonical_variant_pair(variant_a: str, variant_b: str) -> str:
    pair = {str(variant_a).lower(), str(variant_b).lower()}
    if pair == {"african", "white"}:
        return "african<->white"
    if pair == {"indian", "asian"}:
        return "indian<->asian"
    sorted_pair = sorted(pair)
    if len(sorted_pair) == 2:
        return f"{sorted_pair[0]}<->{sorted_pair[1]}"
    return f"{variant_a}<->{variant_b}"


def build_pair_rows(
    rows: Sequence[Dict[str, object]],
    split_families: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    # fold is part of the grouping key: seed values (0/1/2) repeat across all 3
    # CV folds, and without fold here, matched/shifted rows from *different
    # trained models* (fold0's vs fold2's) for the same clip could be pooled
    # together and cross-paired below -- silently comparing two different
    # models' predictions as if they were one matched/shifted pair. This
    # matters specifically for base_ids 0 and 1, the only IDs evaluated as
    # "unseen" in two different folds (fold0 and fold2).
    by_run_split: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model", "")),
            str(row.get("pair_tag", "")),
            str(row.get("seed", "")),
            str(row.get("fold", "")),
            str(row.get("eval_split", "")),
        )
        by_run_split[key].append(dict(row))

    pair_rows: List[Dict[str, object]] = []
    run_reports: List[Dict[str, object]] = []

    run_keys = {(model, pair_tag, seed, fold) for model, pair_tag, seed, fold, _split in by_run_split}
    for model, pair_tag, seed, fold in sorted(run_keys):
        for split_family in split_families:
            matched_split, shifted_split = SPLIT_FAMILY_TO_SPLITS[split_family]
            matched_rows = list(by_run_split.get((model, pair_tag, seed, fold, matched_split), []))
            shifted_rows = list(by_run_split.get((model, pair_tag, seed, fold, shifted_split), []))

            if not matched_rows and not shifted_rows:
                continue

            shifted_index: Dict[Tuple[str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
            for row in shifted_rows:
                key = (
                    str(row.get("background", "")),
                    str(row.get("action", "")),
                    int(row.get("base_id", -1)),
                    str(row.get("variant", "")).lower(),
                )
                shifted_index[key].append(row)
            for key in shifted_index:
                shifted_index[key].sort(key=lambda item: str(item.get("rel_path", "")))

            paired_count = 0
            missing_mapping_count = 0
            missing_counterpart_count = 0

            sorted_matched = sorted(
                matched_rows,
                key=lambda item: (
                    str(item.get("background", "")),
                    str(item.get("action", "")),
                    int(item.get("base_id", -1)),
                    str(item.get("variant", "")),
                    str(item.get("rel_path", "")),
                ),
            )

            for matched in sorted_matched:
                matched_variant = str(matched.get("variant", "")).lower()
                counterpart_variants = OPPOSITE_GROUP_VARIANTS.get(tone_group_for_variant(matched_variant), ())
                if not counterpart_variants:
                    missing_mapping_count += 1
                    continue
                matched_had_pair = False
                for counterpart_variant in counterpart_variants:
                    lookup_key = (
                        str(matched.get("background", "")),
                        str(matched.get("action", "")),
                        int(matched.get("base_id", -1)),
                        counterpart_variant,
                    )
                    # Not popped: the same shifted clip is the legitimate
                    # "after" observation for more than one matched variant
                    # (e.g. the "white" shifted clip pairs with both the
                    # african-matched row and the indian-matched row for the
                    # same base_id/background/action) -- these are distinct
                    # comparisons living in different variant_pair columns,
                    # not double use of one comparison.
                    candidates = shifted_index.get(lookup_key, [])
                    if not candidates:
                        continue
                    shifted = candidates[0]
                    matched_had_pair = True
                    paired_count += 1

                    pair_row: Dict[str, object] = {
                        "model": model,
                        "pair_tag": pair_tag,
                        "seed": seed,
                        "fold": fold,
                        "split_family": split_family,
                        "matched_split": matched_split,
                        "shifted_split": shifted_split,
                        "background": matched.get("background", ""),
                        "action": matched.get("action", ""),
                        "base_id": int(matched.get("base_id", -1)),
                        "variant_matched": matched_variant,
                        "variant_shifted": counterpart_variant,
                        "variant_pair": canonical_variant_pair(matched_variant, counterpart_variant),
                        "rel_path_matched": matched.get("rel_path", ""),
                        "rel_path_shifted": shifted.get("rel_path", ""),
                        "y_true_matched": int(matched.get("y_true", -1)),
                        "y_true_shifted": int(shifted.get("y_true", -1)),
                        "y_pred_matched": int(matched.get("y_pred", -1)),
                        "y_pred_shifted": int(shifted.get("y_pred", -1)),
                        "correct_matched": int(matched.get("correct", -1)),
                        "correct_shifted": int(shifted.get("correct", -1)),
                    }
                    pair_row["pred_flip"] = int(pair_row["y_pred_matched"] != pair_row["y_pred_shifted"])
                    pair_row["correctness_drop"] = int(
                        int(pair_row["correct_matched"]) == 1 and int(pair_row["correct_shifted"]) == 0
                    )
                    true_prob_matched = to_float(matched.get("true_class_prob"))
                    true_prob_shifted = to_float(shifted.get("true_class_prob"))
                    pair_row["true_class_prob_drop"] = (
                        float(true_prob_matched - true_prob_shifted)
                        if true_prob_matched == true_prob_matched and true_prob_shifted == true_prob_shifted
                        else float("nan")
                    )

                    for feature_name in FEATURE_COLUMNS:
                        matched_value = to_float(matched.get(feature_name))
                        shifted_value = to_float(shifted.get(feature_name))
                        pair_row[f"matched_{feature_name}"] = matched_value
                        pair_row[f"shifted_{feature_name}"] = shifted_value
                        if matched_value == matched_value and shifted_value == shifted_value:
                            delta = shifted_value - matched_value
                            pair_row[f"delta_{feature_name}"] = float(delta)
                            pair_row[f"abs_delta_{feature_name}"] = float(abs(delta))
                        else:
                            pair_row[f"delta_{feature_name}"] = float("nan")
                            pair_row[f"abs_delta_{feature_name}"] = float("nan")

                    pair_rows.append(pair_row)
                if not matched_had_pair:
                    missing_counterpart_count += 1

            unused_shifted = int(sum(len(values) for values in shifted_index.values()))
            run_reports.append(
                {
                    "model": model,
                    "pair_tag": pair_tag,
                    "seed": seed,
                    "fold": fold,
                    "split_family": split_family,
                    "matched_split": matched_split,
                    "shifted_split": shifted_split,
                    "matched_rows": len(matched_rows),
                    "shifted_rows": len(shifted_rows),
                    "paired_rows": paired_count,
                    "missing_variant_mapping": missing_mapping_count,
                    "missing_counterpart": missing_counterpart_count,
                    "unused_shifted_rows": unused_shifted,
                    "join_rate_vs_matched": (
                        float(paired_count / len(matched_rows)) if matched_rows else float("nan")
                    ),
                }
            )

    run_report_payload = {
        "num_runs": len(run_reports),
        "runs": run_reports,
        "total_matched_rows": int(sum(int(item["matched_rows"]) for item in run_reports)),
        "total_shifted_rows": int(sum(int(item["shifted_rows"]) for item in run_reports)),
        "total_paired_rows": int(sum(int(item["paired_rows"]) for item in run_reports)),
        "total_missing_variant_mapping": int(sum(int(item["missing_variant_mapping"]) for item in run_reports)),
        "total_missing_counterpart": int(sum(int(item["missing_counterpart"]) for item in run_reports)),
        "total_unused_shifted_rows": int(sum(int(item["unused_shifted_rows"]) for item in run_reports)),
    }
    return pair_rows, run_report_payload


def rank_biserial_effect(group_flip: np.ndarray, group_no_flip: np.ndarray) -> float:
    n_flip = len(group_flip)
    n_no_flip = len(group_no_flip)
    if n_flip == 0 or n_no_flip == 0:
        return float("nan")
    if scipy_stats is not None:
        try:
            u_stat, _p = scipy_stats.mannwhitneyu(group_flip, group_no_flip, alternative="two-sided")
            return float((2.0 * float(u_stat) / float(n_flip * n_no_flip)) - 1.0)
        except Exception:
            pass
    greater = 0.0
    equal = 0.0
    for flip_value in group_flip:
        greater += float(np.sum(flip_value > group_no_flip))
        equal += float(np.sum(flip_value == group_no_flip))
    u_stat = greater + 0.5 * equal
    return float((2.0 * u_stat / float(n_flip * n_no_flip)) - 1.0)


def mannwhitney_pvalue(group_flip: np.ndarray, group_no_flip: np.ndarray) -> float:
    if len(group_flip) == 0 or len(group_no_flip) == 0:
        return float("nan")
    if scipy_stats is not None:
        try:
            _u, p_value = scipy_stats.mannwhitneyu(group_flip, group_no_flip, alternative="two-sided")
            return float(p_value)
        except Exception:
            pass
    return float("nan")


def fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> float:
    if scipy_stats is not None:
        try:
            _or_value, p_value = scipy_stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
            return float(p_value)
        except Exception:
            pass
    return float("nan")


def bootstrap_rank_biserial_ci(
    group_flip: np.ndarray,
    group_no_flip: np.ndarray,
    *,
    iters: int,
    seed: int,
) -> Tuple[float, float]:
    n_flip = len(group_flip)
    n_no_flip = len(group_no_flip)
    if n_flip == 0 or n_no_flip == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples: List[float] = []
    for _ in range(max(0, int(iters))):
        resampled_flip = group_flip[rng.integers(0, n_flip, size=n_flip)]
        resampled_no_flip = group_no_flip[rng.integers(0, n_no_flip, size=n_no_flip)]
        samples.append(rank_biserial_effect(resampled_flip, resampled_no_flip))
    if not samples:
        return float("nan"), float("nan")
    arr = np.asarray(samples, dtype=float)
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def compute_feature_ranking(
    pair_rows: Sequence[Dict[str, object]],
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> List[Dict[str, object]]:
    feature_names = sorted(
        {
            key
            for row in pair_rows
            for key in row.keys()
            if key.startswith(("matched_", "shifted_", "delta_", "abs_delta_"))
        }
    )
    ranking: List[Dict[str, object]] = []
    event = np.asarray([to_int(row.get("pred_flip"), default=0) for row in pair_rows], dtype=np.int64)

    for feat_idx, feature_name in enumerate(feature_names):
        values = np.asarray([to_float(row.get(feature_name)) for row in pair_rows], dtype=float)
        valid = np.isfinite(values) & np.isfinite(event.astype(float))
        x = values[valid]
        y = event[valid]
        if x.size == 0:
            continue
        flip_mask = y == 1
        no_flip_mask = y == 0
        group_flip = x[flip_mask]
        group_no_flip = x[no_flip_mask]
        if group_flip.size < 2 or group_no_flip.size < 2:
            continue

        rank_biserial = rank_biserial_effect(group_flip, group_no_flip)
        mw_pvalue = mannwhitney_pvalue(group_flip, group_no_flip)
        ci_low, ci_high = bootstrap_rank_biserial_ci(
            group_flip,
            group_no_flip,
            iters=bootstrap_iters,
            seed=bootstrap_seed + feat_idx * 31,
        )

        q25 = float(np.nanpercentile(x, 25))
        q75 = float(np.nanpercentile(x, 75))
        high_mask = x >= q75
        low_mask = x <= q25
        high_flip = int(np.sum((y == 1) & high_mask))
        high_no = int(np.sum((y == 0) & high_mask))
        low_flip = int(np.sum((y == 1) & low_mask))
        low_no = int(np.sum((y == 0) & low_mask))
        odds_ratio = float(((high_flip + 0.5) * (low_no + 0.5)) / ((high_no + 0.5) * (low_flip + 0.5)))
        fisher_p = fisher_exact_pvalue(high_flip, high_no, low_flip, low_no)
        high_total = high_flip + high_no
        low_total = low_flip + low_no
        high_flip_rate = float(high_flip / high_total) if high_total > 0 else float("nan")
        low_flip_rate = float(low_flip / low_total) if low_total > 0 else float("nan")

        ranking.append(
            {
                "feature": feature_name,
                "n": int(x.size),
                "n_flip": int(group_flip.size),
                "n_no_flip": int(group_no_flip.size),
                "flip_rate": float(group_flip.size / x.size),
                "rank_biserial": float(rank_biserial),
                "rank_biserial_ci_low": float(ci_low),
                "rank_biserial_ci_high": float(ci_high),
                "mannwhitney_p": float(mw_pvalue),
                "quartile_q25": q25,
                "quartile_q75": q75,
                "quartile_odds_ratio": odds_ratio,
                "fisher_p": float(fisher_p),
                "flip_rate_high_q": high_flip_rate,
                "flip_rate_low_q": low_flip_rate,
                "direction": "higher_feature_more_flips" if odds_ratio >= 1.0 else "higher_feature_fewer_flips",
                "effect_abs": float(abs(rank_biserial)) if rank_biserial == rank_biserial else float("nan"),
            }
        )

    ranking.sort(
        key=lambda row: (
            -(to_float(row.get("effect_abs")) if to_float(row.get("effect_abs")) == to_float(row.get("effect_abs")) else -1.0),
            to_float(row.get("mannwhitney_p")),
            str(row.get("feature", "")),
        )
    )
    return ranking


def write_pair_level_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    lead_columns = [
        "model",
        "pair_tag",
        "seed",
        "fold",
        "split_family",
        "matched_split",
        "shifted_split",
        "background",
        "action",
        "base_id",
        "variant_pair",
        "variant_matched",
        "variant_shifted",
        "rel_path_matched",
        "rel_path_shifted",
        "pred_flip",
        "correctness_drop",
        "true_class_prob_drop",
        "y_true_matched",
        "y_true_shifted",
        "y_pred_matched",
        "y_pred_shifted",
        "correct_matched",
        "correct_shifted",
    ]
    dynamic_columns = sorted({key for row in rows for key in row.keys() if key not in set(lead_columns)})
    fieldnames = lead_columns + dynamic_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_ranking_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_feature_ranking_pdf(path: Path, ranking_rows: Sequence[Dict[str, object]], top_k: int = 16) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping {path.name}: {exc}", flush=True)
        return

    if not ranking_rows:
        return

    top = list(ranking_rows[: max(1, int(top_k))])
    top.reverse()
    labels = [str(row["feature"]) for row in top]
    values = np.asarray([to_float(row.get("rank_biserial")) for row in top], dtype=float)
    ci_low = np.asarray([to_float(row.get("rank_biserial_ci_low")) for row in top], dtype=float)
    ci_high = np.asarray([to_float(row.get("rank_biserial_ci_high")) for row in top], dtype=float)
    left_err = np.maximum(0.0, values - ci_low)
    right_err = np.maximum(0.0, ci_high - values)
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in values]

    fig_h = max(4.0, 0.38 * len(top) + 1.8)
    fig, ax = plt.subplots(figsize=(11.0, fig_h), dpi=200)
    y_pos = np.arange(len(top))
    ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="#333333", linewidth=0.6)
    ax.errorbar(values, y_pos, xerr=np.vstack([left_err, right_err]), fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Rank-biserial effect on prediction flips (flip vs no-flip)", fontsize=10)
    ax.set_title("Swap-influence feature ranking (higher |effect| = stronger association)", fontsize=13, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _mean_rate(rows: Iterable[Dict[str, object]]) -> float:
    values = [to_float(row.get("pred_flip")) for row in rows]
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def plot_flip_breakdown_pdf(path: Path, pair_rows: Sequence[Dict[str, object]], split_families: Sequence[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping {path.name}: {exc}", flush=True)
        return

    if not pair_rows:
        return

    dimensions = ["model", "pair_tag", "background"]
    family_list = list(split_families)
    n_rows = max(1, len(family_list))
    fig, axes = plt.subplots(
        n_rows,
        len(dimensions),
        figsize=(5.4 * len(dimensions), 3.2 * n_rows + 0.4),
        dpi=200,
        squeeze=False,
    )
    for r_idx, split_family in enumerate(family_list):
        family_rows = [row for row in pair_rows if str(row.get("split_family")) == split_family]
        for c_idx, dimension in enumerate(dimensions):
            ax = axes[r_idx][c_idx]
            grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for row in family_rows:
                grouped[str(row.get(dimension, ""))].append(dict(row))
            ranked = sorted(
                grouped.items(),
                key=lambda item: (_mean_rate(item[1]) if _mean_rate(item[1]) == _mean_rate(item[1]) else -1.0),
                reverse=True,
            )
            labels = [key for key, _rows in ranked]
            values = [_mean_rate(_rows) for _key, _rows in ranked]
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color="#4c72b0", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            ax.set_xlabel("Flip rate", fontsize=9)
            title = f"{split_family}: by {dimension}"
            ax.set_title(title, fontsize=10, weight="bold")
    fig.suptitle("Prediction flip-rate breakdowns", fontsize=14, weight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else root)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_filter = parse_csv_set(args.models)
    split_families = parse_split_families(args.split_families)

    prediction_rows, prediction_csvs = load_prediction_rows(root, models_filter)
    pair_rows, join_report = build_pair_rows(prediction_rows, split_families)
    ranking_rows = compute_feature_ranking(
        pair_rows,
        bootstrap_iters=int(args.bootstrap_iters),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    pair_csv_path = out_dir / "swap_pair_level_analysis.csv"
    ranking_csv_path = out_dir / "swap_influence_feature_ranking.csv"
    ranking_json_path = out_dir / "swap_influence_feature_ranking.json"
    join_report_path = out_dir / "swap_pair_join_report.json"
    influence_plot_path = out_dir / "swap_influence_features.pdf"
    breakdown_plot_path = out_dir / "swap_flip_rate_breakdowns.pdf"

    write_pair_level_csv(pair_csv_path, pair_rows)
    write_ranking_csv(ranking_csv_path, ranking_rows)
    ranking_json_path.write_text(json.dumps(ranking_rows, indent=2), encoding="utf-8")
    join_report_payload = {
        "root": str(root),
        "metric": str(args.metric),
        "models_filter": sorted(models_filter) if models_filter else "all",
        "split_families": split_families,
        "prediction_csv_count": len(prediction_csvs),
        "prediction_row_count": len(prediction_rows),
        "pair_row_count": len(pair_rows),
        "join_report": join_report,
    }
    join_report_path.write_text(json.dumps(join_report_payload, indent=2), encoding="utf-8")
    plot_feature_ranking_pdf(influence_plot_path, ranking_rows)
    plot_flip_breakdown_pdf(breakdown_plot_path, pair_rows, split_families)

    print(pair_csv_path)
    print(ranking_csv_path)
    print(ranking_json_path)
    print(join_report_path)
    if influence_plot_path.exists():
        print(influence_plot_path)
    if breakdown_plot_path.exists():
        print(breakdown_plot_path)


if __name__ == "__main__":
    main()
