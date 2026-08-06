from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from aggregate_skin_tone_probe import load_rows
try:
    from .schema import COLOR_PAIR_LABELS, COLOR_PAIR_ORDER, SWAP_LABELS, SWAP_ORDER, stable_seed
except ImportError:  # pragma: no cover - direct script execution
    from schema import COLOR_PAIR_LABELS, COLOR_PAIR_ORDER, SWAP_LABELS, SWAP_ORDER, stable_seed

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_actionbiasbench"))

DEFAULT_JITTER_ROOTS = ("cj0p0", "cj0p4", "cj0p8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired significance tests for skin-tone robustness and directional variant swaps."
    )
    parser.add_argument("--root", type=Path, required=True, help="Analysis root containing swap_pair_level_analysis.csv.")
    parser.add_argument(
        "--metric_roots",
        nargs="*",
        default=[],
        help="Optional label=path roots for jitter comparison. Example: cj0p0=/path cj0p4=/path cj0p8=/path",
    )
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--split_family", type=str, default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--correction",
        type=str,
        default="bh",
        choices=["bonferroni", "holm", "bh"],
        help="Multiple-comparison correction for directional McNemar tests (default: bh = Benjamini-Hochberg FDR).",
    )
    parser.add_argument(
        "--correction_scope",
        type=str,
        default="within_model",
        choices=["within_model", "pooled_rgb"],
        help="Family over which the correction is applied. 'within_model' (default) corrects the 4 "
        "directions per model; 'pooled_rgb' corrects across all RGB models jointly (i3d_flow always "
        "corrected within its own row as a control).",
    )
    return parser.parse_args()


def parse_root_specs(specs: Sequence[str], analysis_root: Path) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            continue
        label, raw_path = spec.split("=", 1)
        parsed.append((label.strip(), Path(raw_path).expanduser().resolve()))
    if parsed:
        return parsed

    parent = analysis_root.parent
    guessed: List[Tuple[str, Path]] = []
    for label in DEFAULT_JITTER_ROOTS:
        candidate = parent / f"skin_tone_probe_rgb_torchvision_v6_{label}"
        if candidate.exists():
            guessed.append((label, candidate))
    return guessed


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1))


def bootstrap_ci(values: Sequence[float], *, iters: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = []
    n = arr.size
    for _ in range(iters):
        samples.append(float(np.mean(arr[rng.integers(0, n, size=n)])))
    sample_arr = np.asarray(samples, dtype=float)
    return float(np.percentile(sample_arr, 2.5)), float(np.percentile(sample_arr, 97.5))


def paired_ttest(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, str]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if scipy_stats is not None:
        try:
            result = scipy_stats.ttest_rel(x_arr, y_arr, nan_policy="omit")
            return float(result.statistic), float(result.pvalue), "paired_t"
        except Exception:
            pass
    diffs = x_arr - y_arr
    n = diffs.size
    if n <= 1:
        return float("nan"), float("nan"), "paired_t_insufficient"
    diff_mean = float(np.mean(diffs))
    diff_std = float(np.std(diffs, ddof=1))
    if diff_std == 0.0:
        if abs(diff_mean) < 1e-12:
            return 0.0, 1.0, "paired_t_zero_variance"
        return float("inf") if diff_mean > 0 else float("-inf"), 0.0, "paired_t_zero_variance"
    t_stat = diff_mean / (diff_std / math.sqrt(n))
    if scipy_stats is not None:
        try:
            p = 2.0 * scipy_stats.t.sf(abs(t_stat), df=n - 1)
            return float(t_stat), float(p), "paired_t_fallback"
        except Exception:
            pass
    return float(t_stat), float("nan"), "paired_t_fallback"


def paired_wilcoxon(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, str]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diffs = x_arr - y_arr
    if np.allclose(diffs, 0.0):
        return 0.0, 1.0, "wilcoxon_all_zero"
    if scipy_stats is not None:
        try:
            result = scipy_stats.wilcoxon(x_arr, y_arr, zero_method="wilcox", alternative="two-sided")
            return float(result.statistic), float(result.pvalue), "wilcoxon"
        except Exception:
            pass
    return float("nan"), float("nan"), "wilcoxon_unavailable"


def binom_two_sided_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    if scipy_stats is not None and hasattr(scipy_stats, "binomtest"):
        try:
            return float(scipy_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
        except Exception:
            pass
    probs = [math.comb(n, i) * (0.5**n) for i in range(n + 1)]
    observed = probs[k]
    return float(sum(p for p in probs if p <= observed + 1e-15))


def mcnemar_exact_from_counts(b: int, c: int) -> float:
    if (b + c) == 0:
        return 1.0
    return binom_two_sided_pvalue(min(b, c), b + c)


def bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    finite = [p for p in p_values if p == p]
    m = len(finite)
    if m == 0:
        return [float("nan") for _ in p_values]
    adjusted: List[float] = []
    for p in p_values:
        adjusted.append(min(1.0, p * m) if p == p else float("nan"))
    return adjusted


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down FWER control (uniformly less conservative than Bonferroni)."""
    idx = [i for i, p in enumerate(p_values) if p == p]
    m = len(idx)
    out = [float("nan")] * len(p_values)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: p_values[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p_values[i])
        running = max(running, val)  # enforce monotonicity
        out[i] = running
    return out


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    idx = [i for i, p in enumerate(p_values) if p == p]
    m = len(idx)
    out = [float("nan")] * len(p_values)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: p_values[i])
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(1.0, p_values[i] * m / (rank + 1))
        prev = min(prev, val)  # enforce monotonicity from the top down
        out[i] = prev
    return out


CORRECTION_FUNCS = {
    "bonferroni": bonferroni_adjust,
    "holm": holm_adjust,
    "bh": benjamini_hochberg_adjust,
}

CORRECTION_TITLES = {
    "bonferroni": "Bonferroni-adjusted p-values",
    "holm": "Holm-adjusted p-values",
    "bh": "Benjamini-Hochberg FDR-adjusted p-values (q)",
}


def build_per_seed_rows(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    by_seed_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        experiment_tag = str(row.get("experiment_tag", row["pair_tag"]))
        key = (experiment_tag, str(row["modality"]), str(row["seed"]))
        item = by_seed_key.setdefault(
            key,
            {
                "pair_tag": key[0],
                "experiment_tag": key[0],
                "modality": key[1],
                "seed": key[2],
                "mode": row["mode"],
            },
        )
        split = str(row["eval_split"])
        item[f"{split}_{metric_name}_mean"] = float(row.get(f"{metric_name}_mean", float("nan")))

    out: List[Dict[str, object]] = []
    for item in by_seed_key.values():
        matched_unseen = float(item.get(f"eval_matched_unseen_ids_{metric_name}_mean", float("nan")))
        matched_seen = float(item.get(f"eval_matched_seen_ids_{metric_name}_mean", float("nan")))
        shifted_seen = float(item.get(f"eval_shifted_seen_ids_{metric_name}_mean", float("nan")))
        shifted_unseen = float(item.get(f"eval_shifted_unseen_ids_{metric_name}_mean", float("nan")))
        item[f"{metric_name}_matched_unseen_ids"] = matched_unseen
        item[f"{metric_name}_matched_seen_ids"] = matched_seen
        item[f"{metric_name}_shifted_seen_ids"] = shifted_seen
        item[f"{metric_name}_shifted_unseen_ids"] = shifted_unseen
        item[f"{metric_name}_drop_training_videos"] = (
            matched_seen - shifted_seen if matched_seen == matched_seen and shifted_seen == shifted_seen else float("nan")
        )
        item[f"{metric_name}_drop_testing_videos"] = (
            matched_unseen - shifted_unseen
            if matched_unseen == matched_unseen and shifted_unseen == shifted_unseen
            else float("nan")
        )
        out.append(item)
    return out


def load_metric_units(root: Path, metric: str) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    rows = build_per_seed_rows(load_rows(root), metric)
    by_model: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = defaultdict(dict)
    for row in rows:
        modality = str(row.get("modality", ""))
        if modality.startswith("rgb_torchvision:"):
            model = modality.split(":", 1)[1]
        elif modality == "flow_i3d_external":
            model = "i3d_flow"
        else:
            continue
        unit_key = (str(row["experiment_tag"]), str(row["seed"]))
        by_model[model][unit_key] = {
            "matched_seen": float(row.get(f"{metric}_matched_seen_ids", float("nan"))),
            "matched_unseen": float(row.get(f"{metric}_matched_unseen_ids", float("nan"))),
            "shifted_seen": float(row.get(f"{metric}_shifted_seen_ids", float("nan"))),
            "shifted_unseen": float(row.get(f"{metric}_shifted_unseen_ids", float("nan"))),
            "drop_seen": float(row.get(f"{metric}_drop_training_videos", float("nan"))),
            "drop_unseen": float(row.get(f"{metric}_drop_testing_videos", float("nan"))),
        }
    return by_model


def compute_model_significance_rows(metric_roots: Sequence[Tuple[str, Path]], metric: str) -> List[Dict[str, object]]:
    root_data = {label: load_metric_units(path, metric) for label, path in metric_roots}
    condition_labels = [label for label, _ in metric_roots]
    rows: List[Dict[str, object]] = []

    if not metric_roots:
        return rows

    all_models = sorted({model for data in root_data.values() for model in data.keys()})

    for condition_label in condition_labels:
        model_units = root_data[condition_label]
        for model in all_models:
            units = model_units.get(model, {})
            if not units:
                continue
            unseen_matched = []
            unseen_shifted = []
            seen_matched = []
            seen_shifted = []
            for values in units.values():
                if values["matched_unseen"] == values["matched_unseen"] and values["shifted_unseen"] == values["shifted_unseen"]:
                    unseen_matched.append(values["matched_unseen"])
                    unseen_shifted.append(values["shifted_unseen"])
                if values["matched_seen"] == values["matched_seen"] and values["shifted_seen"] == values["shifted_seen"]:
                    seen_matched.append(values["matched_seen"])
                    seen_shifted.append(values["shifted_seen"])
            for split_family, matched_values, shifted_values in (
                ("seen", seen_matched, seen_shifted),
                ("unseen", unseen_matched, unseen_shifted),
            ):
                if not matched_values or not shifted_values:
                    continue
                diffs = [m - s for m, s in zip(matched_values, shifted_values)]
                t_stat, t_p, _ = paired_ttest(matched_values, shifted_values)
                w_stat, w_p, _ = paired_wilcoxon(matched_values, shifted_values)
                ci_low, ci_high = bootstrap_ci(
                    diffs,
                    seed=stable_seed((condition_label, model, split_family), modulo=2**16),
                )
                rows.append(
                    {
                        "analysis_type": "matched_vs_shifted",
                        "condition_label": condition_label,
                        "model": model,
                        "split_family": split_family,
                        "n_units": len(diffs),
                        "matched_mean": mean(matched_values),
                        "matched_std": sample_std(matched_values),
                        "shifted_mean": mean(shifted_values),
                        "shifted_std": sample_std(shifted_values),
                        "drop_mean": mean(diffs),
                        "drop_std": sample_std(diffs),
                        "drop_ci_low": ci_low,
                        "drop_ci_high": ci_high,
                        "paired_t_stat": t_stat,
                        "paired_t_p": t_p,
                        "wilcoxon_stat": w_stat,
                        "wilcoxon_p": w_p,
                    }
                )

    if len(condition_labels) >= 2:
        first_label = condition_labels[0]
        all_models = sorted({model for data in root_data.values() for model in data.keys()})
        for idx in range(1, len(condition_labels)):
            comp_label = condition_labels[idx]
            for model in all_models:
                base_units = root_data[first_label].get(model, {})
                comp_units = root_data[comp_label].get(model, {})
                shared_units = sorted(set(base_units.keys()) & set(comp_units.keys()))
                if not shared_units:
                    continue
                for split_family, key in (("seen", "drop_seen"), ("unseen", "drop_unseen")):
                    base_values = []
                    comp_values = []
                    for unit in shared_units:
                        left = base_units[unit].get(key, float("nan"))
                        right = comp_units[unit].get(key, float("nan"))
                        if left == left and right == right:
                            base_values.append(left)
                            comp_values.append(right)
                    if not base_values:
                        continue
                    diffs = [b - c for b, c in zip(base_values, comp_values)]
                    t_stat, t_p, _ = paired_ttest(base_values, comp_values)
                    w_stat, w_p, _ = paired_wilcoxon(base_values, comp_values)
                    ci_low, ci_high = bootstrap_ci(
                        diffs,
                        seed=stable_seed((first_label, comp_label, model, split_family), modulo=2**16),
                    )
                    rows.append(
                        {
                            "analysis_type": "condition_drop_comparison",
                            "reference_condition": first_label,
                            "comparator_condition": comp_label,
                            "model": model,
                            "split_family": split_family,
                            "n_units": len(diffs),
                            "reference_drop_mean": mean(base_values),
                            "reference_drop_std": sample_std(base_values),
                            "comparator_drop_mean": mean(comp_values),
                            "comparator_drop_std": sample_std(comp_values),
                            "reference_minus_comparator_drop_mean": mean(diffs),
                            "reference_minus_comparator_drop_std": sample_std(diffs),
                            "diff_ci_low": ci_low,
                            "diff_ci_high": ci_high,
                            "paired_t_stat": t_stat,
                            "paired_t_p": t_p,
                            "wilcoxon_stat": w_stat,
                            "wilcoxon_p": w_p,
                        }
                    )

    return rows


def load_pair_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dedupe_multi_fold_clusters(pair_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Drop a base_id's rows from every CV fold but one, within whatever rows
    are passed in (call this *after* filtering to a single split_family).

    Base IDs 0 and 1 are the only ones evaluated as "unseen" in two different
    CV folds (fold0 and fold2: fold0 unseen={0,1,2,3}, fold2 unseen={8,9,0,1}
    -- every other ID appears in exactly one fold's unseen set). Without this,
    a cluster-level test's per-base_id statistic for IDs 0/1 pools predictions
    from two different trained models while every other base_id's statistic
    comes from exactly one -- an unequal, non-exchangeable cluster
    composition for what's supposed to be a set of independent
    motion-instance replicates. Keep only the lowest-numbered fold's rows per
    base_id so every cluster is "one trained model's behavior on one
    held-out person," matching every other ID.
    """
    folds_by_base_id: Dict[int, set] = defaultdict(set)
    for row in pair_rows:
        fold = str(row.get("fold", ""))
        if fold:
            folds_by_base_id[int(row.get("base_id", -1))].add(fold)

    keep_fold: Dict[int, str] = {
        base_id: min(folds, key=lambda f: int(f))
        for base_id, folds in folds_by_base_id.items()
        if len(folds) > 1
    }
    if not keep_fold:
        return list(pair_rows)

    filtered: List[Dict[str, object]] = []
    for row in pair_rows:
        base_id = int(row.get("base_id", -1))
        fold = str(row.get("fold", ""))
        if base_id in keep_fold and fold != keep_fold[base_id]:
            continue
        filtered.append(row)
    return filtered


def compute_variant_significance_rows(
    pair_rows: Sequence[Dict[str, object]],
    split_family: str,
    correction: str = "bh",
    correction_scope: str = "within_model",
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        if str(row.get("split_family", "")) != split_family:
            continue
        key = (
            str(row.get("model", "")),
            str(row.get("variant_matched", "")),
            str(row.get("variant_shifted", "")),
        )
        grouped[key].append(row)

    rows: List[Dict[str, object]] = []
    by_model: Dict[str, List[int]] = defaultdict(list)
    for (model, variant_matched, variant_shifted), items in sorted(grouped.items()):
        if not items:
            continue
        matched = np.asarray([int(item.get("correct_matched", 0)) for item in items], dtype=int)
        shifted = np.asarray([int(item.get("correct_shifted", 0)) for item in items], dtype=int)
        b = int(np.sum((matched == 1) & (shifted == 0)))
        c = int(np.sum((matched == 0) & (shifted == 1)))
        n = int(len(items))
        p_raw = mcnemar_exact_from_counts(b, c)
        acc_matched = float(np.mean(matched)) if n else float("nan")
        acc_shifted = float(np.mean(shifted)) if n else float("nan")
        delta = float(acc_matched - acc_shifted) if n else float("nan")
        row = {
            "model": model,
            "split_family": split_family,
            "variant_matched": variant_matched,
            "variant_shifted": variant_shifted,
            "n_pairs": n,
            "matched_accuracy": acc_matched,
            "shifted_accuracy": acc_shifted,
            "accuracy_drop": delta,
            "n_matched_correct_shifted_wrong": b,
            "n_matched_wrong_shifted_correct": c,
            "mcnemar_p_raw": p_raw,
        }
        rows.append(row)
        by_model[model].append(len(rows) - 1)

    adjust_fn = CORRECTION_FUNCS[correction]

    # Always record within-model Bonferroni for reference/back-compat.
    for model, indices in by_model.items():
        bonf = bonferroni_adjust([float(rows[idx]["mcnemar_p_raw"]) for idx in indices])
        for idx, p_bonf in zip(indices, bonf):
            rows[idx]["mcnemar_p_bonferroni"] = p_bonf

    # Chosen correction (drives the figure) — the i3d_flow control is always
    # corrected within its own row so it stays an independent false-positive gauge.
    if correction_scope == "pooled_rgb":
        rgb_indices = [idx for model, idxs in by_model.items() if model != "i3d_flow" for idx in idxs]
        flow_indices = [idx for model, idxs in by_model.items() if model == "i3d_flow" for idx in idxs]
        families = [rgb_indices, flow_indices]
    else:  # within_model
        families = [idxs for idxs in by_model.values()]

    for indices in families:
        if not indices:
            continue
        adjusted = adjust_fn([float(rows[idx]["mcnemar_p_raw"]) for idx in indices])
        for idx, p_adj in zip(indices, adjusted):
            rows[idx]["mcnemar_p_adj"] = p_adj
            rows[idx]["correction_method"] = correction
            rows[idx]["correction_scope"] = correction_scope
    return rows


def compute_variant_cluster_bootstrap_rows(
    pair_rows: Sequence[Dict[str, object]],
    split_family: str,
    *,
    n_boot: int = 5000,
) -> List[Dict[str, object]]:
    """Motion-instance-clustered bootstrap check for the same (model, variant_matched,
    variant_shifted) cells as ``compute_variant_significance_rows``.

    The pooled McNemar test treats every clip-level pair as an independent trial, but
    many pairs trace back to the same handful of motion instances (base_id) rendered
    across three backgrounds and, for some IDs, across more than one CV fold. This
    resamples whole motion instances with replacement (keeping each instance's clips
    together as a block) rather than individual clips, and reports whether the
    resulting 95% interval for the paired accuracy drop excludes zero.
    """
    split_rows = [row for row in pair_rows if str(row.get("split_family", "")) == split_family]
    split_rows = dedupe_multi_fold_clusters(split_rows)

    grouped: Dict[Tuple[str, str, str], Dict[int, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in split_rows:
        key = (
            str(row.get("model", "")),
            str(row.get("variant_matched", "")),
            str(row.get("variant_shifted", "")),
        )
        base_id = int(row.get("base_id", -1))
        grouped[key][base_id].append(row)

    rows: List[Dict[str, object]] = []
    for (model, variant_matched, variant_shifted), by_base_id in sorted(grouped.items()):
        base_ids = sorted(by_base_id.keys())
        n_clusters = len(base_ids)
        b_arr = np.zeros(n_clusters, dtype=float)
        c_arr = np.zeros(n_clusters, dtype=float)
        n_arr = np.zeros(n_clusters, dtype=float)
        for idx, base_id in enumerate(base_ids):
            items = by_base_id[base_id]
            matched = np.asarray([int(item.get("correct_matched", 0)) for item in items], dtype=int)
            shifted = np.asarray([int(item.get("correct_shifted", 0)) for item in items], dtype=int)
            b_arr[idx] = int(np.sum((matched == 1) & (shifted == 0)))
            c_arr[idx] = int(np.sum((matched == 0) & (shifted == 1)))
            n_arr[idx] = len(items)

        b_total, c_total, n_total = float(b_arr.sum()), float(c_arr.sum()), float(n_arr.sum())
        observed_drop = (b_total - c_total) / n_total if n_total else float("nan")

        if n_clusters > 0:
            rng = np.random.default_rng(stable_seed((model, variant_matched, variant_shifted), modulo=2**32))
            picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
            bb = b_arr[picks].sum(axis=1)
            cc = c_arr[picks].sum(axis=1)
            nn = n_arr[picks].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_drops = np.where(nn > 0, (bb - cc) / nn, np.nan)
            ci_low, ci_high = np.nanpercentile(boot_drops, [2.5, 97.5])
        else:
            ci_low = ci_high = float("nan")

        rows.append(
            {
                "model": model,
                "split_family": split_family,
                "variant_matched": variant_matched,
                "variant_shifted": variant_shifted,
                "n_clusters": n_clusters,
                "n_pairs": int(n_total),
                "n_matched_correct_shifted_wrong": int(b_total),
                "n_matched_wrong_shifted_correct": int(c_total),
                "accuracy_drop": observed_drop,
                "cluster_boot_ci_low": float(ci_low),
                "cluster_boot_ci_high": float(ci_high),
                "cluster_significant": bool(ci_low > 0 or ci_high < 0),
            }
        )
    return rows


def plot_variant_significance_with_cluster_bootstrap(
    path: Path,
    variant_rows: Sequence[Dict[str, object]],
    cluster_rows: Sequence[Dict[str, object]],
    split_family: str,
    alpha: float,
    correction: str = "bh",
) -> None:
    """Three-row figure: raw McNemar p-values, corrected q-values, and a
    motion-instance-clustered bootstrap check of the paired accuracy drop for the
    same (model, direction) cells.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot clustered variant heatmap: {exc}")
        return

    models = sorted({str(row["model"]) for row in variant_rows if str(row.get("split_family")) == split_family})
    if not models:
        return

    model_to_idx = {model: idx for idx, model in enumerate(models)}
    swap_to_idx = {swap: idx for idx, swap in enumerate(SWAP_ORDER)}

    raw_mat = np.ones((len(models), len(SWAP_ORDER)), dtype=float)
    adj_mat = np.ones((len(models), len(SWAP_ORDER)), dtype=float)
    for row in variant_rows:
        if str(row.get("split_family")) != split_family:
            continue
        swap = (str(row.get("variant_matched", "")), str(row.get("variant_shifted", "")))
        model = str(row.get("model", ""))
        if swap not in swap_to_idx or model not in model_to_idx:
            continue
        i, j = model_to_idx[model], swap_to_idx[swap]
        raw_mat[i, j] = float(row.get("mcnemar_p_raw", 1.0))
        adj_mat[i, j] = float(row.get("mcnemar_p_adj", row.get("mcnemar_p_bonferroni", 1.0)))

    drop_mat = np.full((len(models), len(SWAP_ORDER)), np.nan)
    ci_low_mat = np.full((len(models), len(SWAP_ORDER)), np.nan)
    ci_high_mat = np.full((len(models), len(SWAP_ORDER)), np.nan)
    sig_mat = np.zeros((len(models), len(SWAP_ORDER)), dtype=bool)
    for row in cluster_rows:
        if str(row.get("split_family")) != split_family:
            continue
        swap = (str(row.get("variant_matched", "")), str(row.get("variant_shifted", "")))
        model = str(row.get("model", ""))
        if swap not in swap_to_idx or model not in model_to_idx:
            continue
        i, j = model_to_idx[model], swap_to_idx[swap]
        drop_mat[i, j] = float(row.get("accuracy_drop", float("nan")))
        ci_low_mat[i, j] = float(row.get("cluster_boot_ci_low", float("nan")))
        ci_high_mat[i, j] = float(row.get("cluster_boot_ci_high", float("nan")))
        sig_mat[i, j] = bool(row.get("cluster_significant", False))

    p_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    base_cmap = plt.get_cmap("Reds_r")
    base_colors = base_cmap(np.linspace(0.0, 1.0, 256))
    white = np.ones_like(base_colors)
    white[:, 3] = 1.0
    softened = 0.60 * base_colors + 0.40 * white
    softened[:, 3] = 1.0
    p_cmap = mcolors.ListedColormap(softened)

    finite_drops = drop_mat[np.isfinite(drop_mat)]
    max_abs_drop = max(float(np.max(np.abs(finite_drops))) if finite_drops.size else 0.05, 0.05)
    drop_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_drop, vcenter=0.0, vmax=max_abs_drop)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, max(7.6, 0.62 * len(models) * 3 + 2.6)),
        dpi=220,
        constrained_layout=False,
    )

    for ax, title, p_mat in zip(
        axes[:2],
        ("p-values", CORRECTION_TITLES.get(correction, "adjusted p-values")),
        (raw_mat, adj_mat),
    ):
        im_p = ax.imshow(p_mat, cmap=p_cmap, norm=p_norm, aspect="auto")
        ax.set_xticks(range(len(SWAP_ORDER)))
        ax.set_xticklabels(SWAP_LABELS, fontsize=10)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")
        for i in range(len(models)):
            for j in range(len(SWAP_ORDER)):
                p_value = p_mat[i, j]
                color = "#ffffff" if p_value < 0.12 else "#1f1f1f"
                mark = "*" if p_value < alpha else ""
                label = f"{p_value:.0e}" if p_value < 0.005 else f"{p_value:.2f}"
                ax.text(j, i, f"{label}{mark}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xlim(-0.5, len(SWAP_ORDER) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_ylabel("Model", fontsize=11, fontweight="bold")
        ax.axvline(1.5, color="#333333", linewidth=1.0, alpha=0.5)

    ax3 = axes[2]
    im_drop = ax3.imshow(drop_mat, cmap="coolwarm", norm=drop_norm, aspect="auto")
    ax3.set_xticks(range(len(SWAP_ORDER)))
    ax3.set_xticklabels(SWAP_LABELS, fontsize=10)
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels(models, fontsize=10)
    ax3.text(
        0.5,
        1.05,
        "Motion-instance-clustered bootstrap (95% CI)",
        transform=ax3.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    for i in range(len(models)):
        for j in range(len(SWAP_ORDER)):
            drop = drop_mat[i, j]
            if not np.isfinite(drop):
                continue
            lo, hi = ci_low_mat[i, j], ci_high_mat[i, j]
            mark = "*" if sig_mat[i, j] else ""
            color = "#ffffff" if abs(drop) > max_abs_drop * 0.55 else "#1f1f1f"
            ax3.text(
                j,
                i,
                f"{drop:.3f}{mark}\n[{lo:.3f},{hi:.3f}]",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    ax3.set_xlim(-0.5, len(SWAP_ORDER) - 0.5)
    ax3.set_ylim(len(models) - 0.5, -0.5)
    ax3.set_ylabel("Model", fontsize=11, fontweight="bold")
    ax3.axvline(1.5, color="#333333", linewidth=1.0, alpha=0.5)

    axes[-1].set_xlabel("Directional skin-tone swap", fontsize=11, fontweight="bold")
    fig.suptitle("Skin-tone swap significance, with motion-instance clustering check", fontsize=15, fontweight="bold", y=0.975)

    cbar_p = fig.colorbar(
        plt.cm.ScalarMappable(norm=p_norm, cmap=p_cmap),
        ax=axes[:2],
        fraction=0.032,
        pad=0.03,
    )
    cbar_p.set_label("McNemar p-value (lower = stronger evidence)", fontsize=10)

    cbar_drop = fig.colorbar(
        plt.cm.ScalarMappable(norm=drop_norm, cmap="coolwarm"),
        ax=axes[2],
        fraction=0.032,
        pad=0.03,
    )
    cbar_drop.set_label("Paired accuracy drop (b-c)/n", fontsize=10)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08, hspace=0.55)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variant_heatmaps_with_aggregate_column(
    path: Path,
    variant_rows: Sequence[Dict[str, object]],
    model_cluster_rows: Sequence[Dict[str, object]],
    split_family: str,
    alpha: float,
    correction: str = "bh",
) -> None:
    """Same two-row per-direction heatmap as ``plot_variant_heatmaps``, with one
    extra column appended: a motion-instance-clustered aggregate check (Wilcoxon
    signed-rank across every independent motion instance feeding each model,
    pooled over all directions and action pairs, and deduped across CV folds --
    see ``compute_model_cluster_significance_rows``). The instance count shown
    in the column header is read directly from ``model_cluster_rows``, not
    hardcoded.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot variant heatmap with aggregate column: {exc}")
        return

    models = sorted({str(row["model"]) for row in variant_rows if str(row.get("split_family")) == split_family})
    if not models:
        return

    model_to_idx = {model: idx for idx, model in enumerate(models)}
    swap_to_idx = {swap: idx for idx, swap in enumerate(SWAP_ORDER)}
    n_cols = len(SWAP_ORDER) + 1
    agg_col = len(SWAP_ORDER)

    raw_mat = np.ones((len(models), n_cols), dtype=float)
    adj_mat = np.ones((len(models), n_cols), dtype=float)

    for row in variant_rows:
        if str(row.get("split_family")) != split_family:
            continue
        swap = (str(row.get("variant_matched", "")), str(row.get("variant_shifted", "")))
        model = str(row.get("model", ""))
        if swap not in swap_to_idx or model not in model_to_idx:
            continue
        i, j = model_to_idx[model], swap_to_idx[swap]
        raw_mat[i, j] = float(row.get("mcnemar_p_raw", 1.0))
        adj_mat[i, j] = float(row.get("mcnemar_p_adj", row.get("mcnemar_p_bonferroni", 1.0)))

    for row in model_cluster_rows:
        if str(row.get("split_family")) != split_family:
            continue
        model = str(row.get("model", ""))
        if model not in model_to_idx:
            continue
        i = model_to_idx[model]
        raw_mat[i, agg_col] = float(row.get("wilcoxon_p", 1.0))
        adj_mat[i, agg_col] = float(row.get("wilcoxon_q", row.get("wilcoxon_p", 1.0)))

    agg_n_clusters = sorted(
        {int(row["n_clusters"]) for row in model_cluster_rows if str(row.get("split_family")) == split_family}
    )
    if len(agg_n_clusters) == 1:
        agg_label = f"Aggregate\n({agg_n_clusters[0]} instances)"
    elif agg_n_clusters:
        agg_label = f"Aggregate\n({agg_n_clusters[0]}-{agg_n_clusters[-1]} instances)"
    else:
        agg_label = "Aggregate"
    col_labels = list(SWAP_LABELS) + [agg_label]

    p_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    base_cmap = plt.get_cmap("Reds_r")
    base_colors = base_cmap(np.linspace(0.0, 1.0, 256))
    white = np.ones_like(base_colors)
    white[:, 3] = 1.0
    softened = 0.60 * base_colors + 0.40 * white
    softened[:, 3] = 1.0
    p_cmap = mcolors.ListedColormap(softened)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.9, max(5.4, 0.62 * len(models) + 3.0)),
        dpi=220,
        constrained_layout=False,
    )

    for ax, title, p_mat in zip(
        axes,
        ("p-values", CORRECTION_TITLES.get(correction, "adjusted p-values")),
        (raw_mat, adj_mat),
    ):
        ax.imshow(p_mat, cmap=p_cmap, norm=p_norm, aspect="auto")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=9.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")
        for i in range(len(models)):
            for j in range(n_cols):
                p_value = p_mat[i, j]
                color = "#ffffff" if p_value < 0.12 else "#1f1f1f"
                # Only the aggregate column (j == agg_col) uses a valid
                # cluster-level test (Wilcoxon over motion instances) -- the
                # 4 per-direction columns are still the naive clip-level
                # McNemar test, which treats non-independent clips from the
                # same motion instance as independent trials. A "*" there
                # would claim more certainty than that test can support,
                # same reasoning as the star-free plot_variant_heatmaps.
                mark = "*" if (j == agg_col and p_value < alpha) else ""
                label = f"{p_value:.0e}" if p_value < 0.005 else f"{p_value:.2f}"
                ax.text(j, i, f"{label}{mark}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_ylabel("Model", fontsize=11, fontweight="bold")
        ax.axvline(1.5, color="#333333", linewidth=1.0, alpha=0.5)
        ax.axvline(agg_col - 0.5, color="#333333", linewidth=1.6, alpha=0.85)

    axes[-1].set_xlabel("Directional skin-tone swap", fontsize=11, fontweight="bold")
    fig.suptitle("Skin-tone swap significance, with a motion-instance-clustered aggregate check", fontsize=14.5, fontweight="bold", y=0.975)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=p_norm, cmap=p_cmap),
        ax=axes,
        fraction=0.032,
        pad=0.03,
    )
    cbar.set_label("p-value (lower = stronger evidence)", fontsize=11)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.11, hspace=0.40)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_model_cluster_significance_rows(
    pair_rows: Sequence[Dict[str, object]],
    split_family: str,
    *,
    n_boot: int = 5000,
) -> List[Dict[str, object]]:
    """Model-level, motion-instance-clustered significance test, pooling all action
    pairs and all four directional swaps per model.

    The per-direction test (``compute_variant_cluster_bootstrap_rows``) has too few
    independent motion instances behind any single cell for reliable cluster
    inference. Pooling over every pair and direction per model instead uses every
    motion instance that contributes to that model at all (up to all 10 IDs now
    that every base_id has complete 4-color coverage; previously capped at 4-7
    before that data-recovery fix), which is enough for a standard paired test
    (Wilcoxon signed-rank on the per-instance drop values, treating the motion
    instance as the unit of replication) to have real power, rather than a
    percentile bootstrap that is known to be unreliable with very few clusters.
    ``pair_rows`` is deduped via ``dedupe_multi_fold_clusters`` before clustering,
    so IDs 0 and 1 (the only ones held out in two different CV folds) contribute
    exactly one cluster each, same as every other ID.
    """
    split_rows = [row for row in pair_rows if str(row.get("split_family", "")) == split_family]
    split_rows = dedupe_multi_fold_clusters(split_rows)

    by_model_cluster: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    for row in split_rows:
        model = str(row.get("model", ""))
        base_id = int(row.get("base_id", -1))
        mc = int(row.get("correct_matched", 0))
        sc = int(row.get("correct_shifted", 0))
        item = by_model_cluster[model][base_id]
        item[2] += 1
        if mc == 1 and sc == 0:
            item[0] += 1
        elif mc == 0 and sc == 1:
            item[1] += 1

    rows: List[Dict[str, object]] = []
    for model, clusters in sorted(by_model_cluster.items()):
        base_ids = sorted(clusters.keys())
        n_clusters = len(base_ids)
        b_arr = np.array([clusters[bid][0] for bid in base_ids], dtype=float)
        c_arr = np.array([clusters[bid][1] for bid in base_ids], dtype=float)
        n_arr = np.array([clusters[bid][2] for bid in base_ids], dtype=float)
        per_cluster_drop = np.where(n_arr > 0, (b_arr - c_arr) / n_arr, np.nan)

        b_total, c_total, n_total = float(b_arr.sum()), float(c_arr.sum()), float(n_arr.sum())
        observed_drop = (b_total - c_total) / n_total if n_total else float("nan")

        w_stat = w_p = float("nan")
        if scipy_stats is not None and n_clusters > 1 and np.any(per_cluster_drop != 0):
            try:
                result = scipy_stats.wilcoxon(per_cluster_drop, alternative="two-sided")
                w_stat, w_p = float(result.statistic), float(result.pvalue)
            except Exception:
                pass

        t_p = float("nan")
        if scipy_stats is not None and n_clusters > 1:
            try:
                t_p = float(scipy_stats.ttest_1samp(per_cluster_drop, 0.0).pvalue)
            except Exception:
                pass

        if n_clusters > 0:
            rng = np.random.default_rng(stable_seed((model, "model_level_cluster"), modulo=2**32))
            picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
            bb = b_arr[picks].sum(axis=1)
            cc = c_arr[picks].sum(axis=1)
            nn = n_arr[picks].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_drops = np.where(nn > 0, (bb - cc) / nn, np.nan)
            ci_low, ci_high = np.nanpercentile(boot_drops, [2.5, 97.5])
        else:
            ci_low = ci_high = float("nan")

        rows.append(
            {
                "model": model,
                "split_family": split_family,
                "n_clusters": n_clusters,
                "n_pairs": int(n_total),
                "b": int(b_total),
                "c": int(c_total),
                "observed_drop": observed_drop,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "ttest_p": t_p,
                "cluster_boot_ci_low": float(ci_low),
                "cluster_boot_ci_high": float(ci_high),
            }
        )

    # BH-FDR across the RGB backbones only; the flow control is reported as an
    # independent reference and not folded into the RGB correction family.
    rgb_indices = [i for i, row in enumerate(rows) if row["model"] != "i3d_flow"]
    rgb_p = [float(rows[i]["wilcoxon_p"]) for i in rgb_indices]
    adjusted = benjamini_hochberg_adjust(rgb_p)
    for i, q in zip(rgb_indices, adjusted):
        rows[i]["wilcoxon_q"] = q
    for row in rows:
        if "wilcoxon_q" not in row:
            row["wilcoxon_q"] = row["wilcoxon_p"]
    return rows


def compute_variant_pair_cluster_significance_rows(
    pair_rows: Sequence[Dict[str, object]],
    split_family: str,
    *,
    correction_scope: str = "within_model",
    n_boot: int = 5000,
) -> List[Dict[str, object]]:
    """Undirected-color-pair, motion-instance-clustered significance test.

    ``build_pair_rows`` now pairs each matched clip against every opposite-
    tone-group shifted variant, not just a single fixed VARIANT_SWAP
    counterpart -- so ``variant_pair`` spans all 4 achievable dark/light color
    combinations (african<->white, indian<->white, african<->asian,
    indian<->asian; same-group pairs like african<->indian never occur, since
    matched/shifted always crosses the dark/light group boundary by
    construction of the shortcut probe). This pools both directions of a pair
    (e.g. african->white and white->african) into one cluster statistic per
    motion instance -- the same Wilcoxon-signed-rank-over-clusters design as
    ``compute_model_cluster_significance_rows``, just grouped by (model,
    color pair) instead of collapsing straight to (model). That gives each
    cell roughly twice the data of the old single-direction bootstrap cells
    (``compute_variant_cluster_bootstrap_rows``) and a real p-value instead of
    only a CI-excludes-zero check, at the cost of 4 cells per model instead of
    1.

    ``correction_scope`` mirrors ``compute_variant_significance_rows``:
    "within_model" (default) BH-corrects each model's own 4 color-pair tests
    as its own family (asking "which color pair matters for this model");
    "pooled_rgb" BH-corrects all 6*4=24 RGB cells as one family (the more
    conservative, harder-to-clear bar). The flow control is always corrected
    within its own row, independent of either family, as an unadjusted
    reference.
    """
    split_rows = [row for row in pair_rows if str(row.get("split_family", "")) == split_family]
    split_rows = dedupe_multi_fold_clusters(split_rows)

    by_cell_cluster: Dict[Tuple[str, str], Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0])
    )
    for row in split_rows:
        model = str(row.get("model", ""))
        variant_pair = str(row.get("variant_pair", ""))
        base_id = int(row.get("base_id", -1))
        mc = int(row.get("correct_matched", 0))
        sc = int(row.get("correct_shifted", 0))
        item = by_cell_cluster[(model, variant_pair)][base_id]
        item[2] += 1
        if mc == 1 and sc == 0:
            item[0] += 1
        elif mc == 0 and sc == 1:
            item[1] += 1

    rows: List[Dict[str, object]] = []
    for (model, variant_pair), clusters in sorted(by_cell_cluster.items()):
        base_ids = sorted(clusters.keys())
        n_clusters = len(base_ids)
        b_arr = np.array([clusters[bid][0] for bid in base_ids], dtype=float)
        c_arr = np.array([clusters[bid][1] for bid in base_ids], dtype=float)
        n_arr = np.array([clusters[bid][2] for bid in base_ids], dtype=float)
        per_cluster_drop = np.where(n_arr > 0, (b_arr - c_arr) / n_arr, np.nan)

        b_total, c_total, n_total = float(b_arr.sum()), float(c_arr.sum()), float(n_arr.sum())
        observed_drop = (b_total - c_total) / n_total if n_total else float("nan")

        w_stat = w_p = float("nan")
        if scipy_stats is not None and n_clusters > 1 and np.any(per_cluster_drop != 0):
            try:
                result = scipy_stats.wilcoxon(per_cluster_drop, alternative="two-sided")
                w_stat, w_p = float(result.statistic), float(result.pvalue)
            except Exception:
                pass

        if n_clusters > 0:
            rng = np.random.default_rng(stable_seed((model, variant_pair, "pair_level_cluster"), modulo=2**32))
            picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
            bb = b_arr[picks].sum(axis=1)
            cc = c_arr[picks].sum(axis=1)
            nn = n_arr[picks].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_drops = np.where(nn > 0, (bb - cc) / nn, np.nan)
            ci_low, ci_high = np.nanpercentile(boot_drops, [2.5, 97.5])
        else:
            ci_low = ci_high = float("nan")

        rows.append(
            {
                "model": model,
                "variant_pair": variant_pair,
                "split_family": split_family,
                "n_clusters": n_clusters,
                "n_pairs": int(n_total),
                "b": int(b_total),
                "c": int(c_total),
                "observed_drop": observed_drop,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "cluster_boot_ci_low": float(ci_low),
                "cluster_boot_ci_high": float(ci_high),
            }
        )

    if correction_scope == "pooled_rgb":
        families = [[i for i, row in enumerate(rows) if row["model"] != "i3d_flow"]]
    else:  # within_model
        by_model_idx: Dict[str, List[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            if row["model"] != "i3d_flow":
                by_model_idx[str(row["model"])].append(i)
        families = list(by_model_idx.values())

    for indices in families:
        if not indices:
            continue
        adjusted = benjamini_hochberg_adjust([float(rows[i]["wilcoxon_p"]) for i in indices])
        for i, q in zip(indices, adjusted):
            rows[i]["wilcoxon_q"] = q
            rows[i]["correction_scope"] = correction_scope
    for row in rows:
        if "wilcoxon_q" not in row:
            row["wilcoxon_q"] = row["wilcoxon_p"]
            row["correction_scope"] = correction_scope
    return rows


# Reference font-to-figure-width ratios (pt per inch of native figure width),
# calibrated on this heatmap so that all figures read at the same apparent
# size once uniformly scaled to the same column width in the paper. Reuse
# these via font_sizes_for_width() for every other figure in this repo.
FONT_RATIO_TITLE = 12.5 / 10.4
FONT_RATIO_LABEL = 10.5 / 10.4
FONT_RATIO_TICK = 10.0 / 10.4
FONT_RATIO_ANNOTATION = 8.7 / 10.4


def font_sizes_for_width(width_in: float) -> Dict[str, float]:
    """Title/label/tick/annotation font sizes (pt) scaled to a figure's native
    width (in), holding the same pt-per-inch ratios as the reference
    skin-tone color-pair heatmap -- so text reads at a consistent apparent
    size across figures once each is scaled to the same column width."""
    return {
        "title": FONT_RATIO_TITLE * width_in,
        "label": FONT_RATIO_LABEL * width_in,
        "tick": FONT_RATIO_TICK * width_in,
        "annotation": FONT_RATIO_ANNOTATION * width_in,
    }


def plot_variant_pair_cluster_heatmap(
    path: Path,
    pair_cluster_rows: Sequence[Dict[str, object]],
    split_family: str,
    alpha: float,
) -> None:
    """One cell per (model, undirected color pair): the signed accuracy drop
    and its BH-adjusted Wilcoxon q-value on a single line, starred at q<alpha.
    Every value is a valid motion-instance cluster test -- there is no naive
    clip-level fallback here. The per-model aggregate lives in the flip table,
    so it is not repeated as a column here.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot variant-pair cluster heatmap: {exc}")
        return

    rows = [row for row in pair_cluster_rows if str(row.get("split_family")) == split_family]
    if not rows:
        return

    def model_sort_key(model: str) -> Tuple[int, str]:
        return (0, "") if model == "i3d_flow" else (1, model)

    models = sorted({str(row["model"]) for row in rows}, key=model_sort_key)
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    pair_to_idx = {pair: idx for idx, pair in enumerate(COLOR_PAIR_ORDER)}
    n_rows, n_cols = len(models), len(COLOR_PAIR_ORDER)

    drop_mat = np.full((n_rows, n_cols), np.nan)
    q_mat = np.full((n_rows, n_cols), np.nan)
    for row in rows:
        pair = str(row.get("variant_pair", ""))
        model = str(row.get("model", ""))
        if pair not in pair_to_idx or model not in model_to_idx:
            continue
        i, j = model_to_idx[model], pair_to_idx[pair]
        drop_mat[i, j] = float(row.get("observed_drop", float("nan")))
        q_mat[i, j] = float(row.get("wilcoxon_q", float("nan")))

    # Scale runs from the actual minimum (slightly negative for the flow
    # control's noise-floor cells) up to the max. The low end still renders
    # near-white under the sequential map, so a small negative reads as "no
    # effect" rather than introducing a second hue.
    vmin = float(np.nanmin(drop_mat)) if np.any(~np.isnan(drop_mat)) else 0.0
    vmax = float(np.nanmax(drop_mat)) if np.any(~np.isnan(drop_mat)) else 0.02
    vmax = max(vmax, 0.01) * 1.08
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Reds")

    # Keep the aspect ratio of the previous five-column figure: with the
    # aggregate column gone, the four data cells each get wider, which is
    # what the single-line "drop (q=...)" annotation needs.
    fig_w = 11.3
    fig_h = max(2.6, 0.44 * n_rows + 1.2)
    fonts = font_sizes_for_width(fig_w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.imshow(drop_mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COLOR_PAIR_LABELS, fontsize=fonts["tick"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(["I3D_flow" if m == "i3d_flow" else m for m in models], fontsize=fonts["tick"])

    for i in range(n_rows):
        for j in range(n_cols):
            drop = drop_mat[i, j]
            if drop != drop:
                continue
            q = q_mat[i, j]
            star = "*" if (q == q and q < alpha) else ""
            color = "#ffffff" if drop > 0.6 * vmax else "#1f1f1f"
            label = f"{drop:+.3f}{star}  (q={q:.3f})" if q == q else f"{drop:+.3f}{star}"
            ax.text(j, i, label, ha="center", va="center", fontsize=fonts["annotation"], color=color)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlabel("Skin-tone color pair", fontsize=fonts["label"], fontweight="bold")
    ax.set_ylabel("Model", fontsize=fonts["label"], fontweight="bold")
    ax.set_title("Skin-tone shortcut by color pair", fontsize=fonts["title"], fontweight="bold")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Paired accuracy drop (b-c)/n", fontsize=fonts["tick"])
    cbar.ax.tick_params(labelsize=fonts["tick"] * 0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_cluster_significance(
    path: Path,
    cluster_rows: Sequence[Dict[str, object]],
    alpha: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot model-level cluster significance: {exc}")
        return
    if not cluster_rows:
        return

    def sort_key(row: Dict[str, object]) -> Tuple[int, str]:
        return (0, "") if row["model"] == "i3d_flow" else (1, str(row["model"]))

    ordered = sorted(cluster_rows, key=sort_key)
    labels = ["I3D_flow" if row["model"] == "i3d_flow" else str(row["model"]) for row in ordered]
    drops = np.array([float(row["observed_drop"]) for row in ordered])
    ci_low = np.array([float(row["cluster_boot_ci_low"]) for row in ordered])
    ci_high = np.array([float(row["cluster_boot_ci_high"]) for row in ordered])
    n_clusters = [int(row["n_clusters"]) for row in ordered]
    q_values = [float(row.get("wilcoxon_q", float("nan"))) for row in ordered]
    is_control = [row["model"] == "i3d_flow" for row in ordered]

    left_err = np.maximum(0.0, drops - ci_low)
    right_err = np.maximum(0.0, ci_high - drops)
    colors = ["#6A717D" if ctrl else ("#d62728" if d > 0 else "#1f77b4") for ctrl, d in zip(is_control, drops)]

    fig, ax = plt.subplots(figsize=(8.6, max(3.2, 0.62 * len(ordered) + 1.4)), dpi=200)
    y_pos = np.arange(len(ordered))
    for i, (d, lo_err, hi_err, c) in enumerate(zip(drops, left_err, right_err, colors)):
        ax.errorbar(
            [d],
            [i],
            xerr=[[lo_err], [hi_err]],
            fmt="o",
            markersize=9,
            color="#333333",
            ecolor=c,
            elinewidth=2.4,
            capsize=4,
            zorder=3,
        )
        ax.scatter([d], [i], color=c, s=90, zorder=4, edgecolor="#222222", linewidth=0.8)
    for i, q in enumerate(q_values):
        sig = "*" if (q == q and q < alpha) else ""
        label = f"n={n_clusters[i]} instances" + (f", q={q:.3f}{sig}" if q == q else "")
        ax.text(ci_high[i] + 0.003, i, label, va="center", fontsize=8.5, color="#333333")

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Paired accuracy drop (b-c)/n, pooled over all pairs and directions", fontsize=10)
    ax.set_title(
        "Aggregate skin-tone shortcut per backbone,\nWilcoxon signed-rank across motion instances (95% CI)",
        fontsize=12.5,
        weight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(
    path: Path,
    model_rows: Sequence[Dict[str, object]],
    variant_rows: Sequence[Dict[str, object]],
    alpha: float,
    correction: str = "bh",
    correction_scope: str = "within_model",
) -> None:
    lines = [
        "# Skin-Tone Significance Summary",
        "",
        "This file focuses on direct paired significance tests, rather than feature-ranking diagnostics.",
        "",
        "## Matched vs shifted paired tests",
        "",
        "Positive drop means performance is worse after the skin-tone swap.",
        "",
    ]
    matched_rows = [row for row in model_rows if row.get("analysis_type") == "matched_vs_shifted" and row.get("split_family") == "unseen"]
    matched_rows.sort(key=lambda row: float(row.get("drop_mean", 0.0)), reverse=True)
    for row in matched_rows:
        lines.append(
            f"- `{row['condition_label']}` `{row['model']}` unseen-ID: "
            f"matched={float(row['matched_mean']):.4f}, shifted={float(row['shifted_mean']):.4f}, "
            f"drop={float(row['drop_mean']):.4f} "
            f"[{float(row['drop_ci_low']):.4f}, {float(row['drop_ci_high']):.4f}], "
            f"paired t p={float(row['paired_t_p']):.4g}, Wilcoxon p={float(row['wilcoxon_p']):.4g}"
        )
    lines.extend(
        [
            "",
            "## Directional variant-swap tests",
            "",
            f"McNemar exact p-values are tested against `alpha={alpha:.3f}` and adjusted with "
            f"`{correction}` correction (scope: `{correction_scope}`).",
            "",
        ]
    )
    variant_focus = [row for row in variant_rows if row.get("split_family") == "unseen"]
    variant_focus.sort(
        key=lambda row: (
            float(row.get("mcnemar_p_adj", row.get("mcnemar_p_bonferroni", 1.0))),
            -float(row.get("accuracy_drop", 0.0)),
        )
    )
    for row in variant_focus[:16]:
        adj_p = float(row.get("mcnemar_p_adj", row.get("mcnemar_p_bonferroni", 1.0)))
        method = str(row.get("correction_method", "bonferroni"))
        lines.append(
            f"- `{row['model']}` `{row['variant_matched']}→{row['variant_shifted']}` unseen-ID: "
            f"matched_acc={float(row['matched_accuracy']):.4f}, shifted_acc={float(row['shifted_accuracy']):.4f}, "
            f"drop={float(row['accuracy_drop']):.4f}, raw p={float(row['mcnemar_p_raw']):.4g}, "
            f"adj p ({method})={adj_p:.4g}, n={int(row['n_pairs'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_variant_heatmaps(
    path: Path,
    variant_rows: Sequence[Dict[str, object]],
    split_family: str,
    alpha: float,
    correction: str = "bh",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot variant heatmaps: {exc}")
        return

    models = sorted({str(row["model"]) for row in variant_rows if str(row.get("split_family")) == split_family})
    if not models:
        return

    model_to_idx = {model: idx for idx, model in enumerate(models)}
    swap_to_idx = {swap: idx for idx, swap in enumerate(SWAP_ORDER)}

    raw_mat = np.ones((len(models), len(SWAP_ORDER)), dtype=float)
    adj_mat = np.ones((len(models), len(SWAP_ORDER)), dtype=float)
    effect_mat = np.zeros((len(models), len(SWAP_ORDER)), dtype=float)
    p_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    base_cmap = plt.get_cmap("Reds_r")
    base_colors = base_cmap(np.linspace(0.0, 1.0, 256))
    white = np.ones_like(base_colors)
    white[:, 3] = 1.0
    # Keep the same hue ordering, but mix with white so 0.00 and 1.00 do not
    # collapse into overly harsh extremes in the heatmap.
    softened = 0.60 * base_colors + 0.40 * white
    softened[:, 3] = 1.0
    p_cmap = mcolors.ListedColormap(softened)

    for row in variant_rows:
        if str(row.get("split_family")) != split_family:
            continue
        swap = (str(row.get("variant_matched", "")), str(row.get("variant_shifted", "")))
        if swap not in swap_to_idx:
            continue
        model = str(row.get("model", ""))
        if model not in model_to_idx:
            continue
        i = model_to_idx[model]
        j = swap_to_idx[swap]
        raw_mat[i, j] = float(row.get("mcnemar_p_raw", 1.0))
        adj_mat[i, j] = float(row.get("mcnemar_p_adj", row.get("mcnemar_p_bonferroni", 1.0)))
        effect_mat[i, j] = float(row.get("accuracy_drop", 0.0))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, max(5.4, 0.62 * len(models) + 3.0)),
        dpi=220,
        constrained_layout=False,
    )

    for ax, title, p_mat in zip(
        axes,
        ("p-values", CORRECTION_TITLES.get(correction, "adjusted p-values")),
        (raw_mat, adj_mat),
    ):
        ax.imshow(p_mat, cmap=p_cmap, norm=p_norm, aspect="auto")
        ax.set_xticks(range(len(SWAP_ORDER)))
        ax.set_xticklabels(SWAP_LABELS, fontsize=10)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.text(
            0.5,
            1.03,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        for i in range(len(models)):
            for j in range(len(SWAP_ORDER)):
                p_value = p_mat[i, j]
                color = "#ffffff" if p_value < 0.12 else "#1f1f1f"
                # Values below 0.005 round to "0.00" at 2 decimal places, which
                # hides how small they actually are; switch to scientific
                # notation instead of silently rounding to zero.
                label = f"{p_value:.0e}" if p_value < 0.005 else f"{p_value:.2f}"
                # No significance marker: at most 4-5 independent motion
                # instances sit behind any one cell (see
                # compute_variant_cluster_bootstrap_rows), so a per-cell "*"
                # would claim more certainty than that sample size can
                # support. These are reported as descriptive p-values only.
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)
        ax.set_xlim(-0.5, len(SWAP_ORDER) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_ylabel("Model", fontsize=11, fontweight="bold")
        ax.axvline(1.5, color="#333333", linewidth=1.0, alpha=0.5)

    axes[-1].set_xlabel("Directional skin-tone swap", fontsize=11, fontweight="bold")
    fig.suptitle("Skin-tone swap p-values (descriptive, not corrected for motion-instance clustering)", fontsize=14, fontweight="bold", y=0.975)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=p_norm, cmap=p_cmap),
        ax=axes,
        fraction=0.032,
        pad=0.03,
    )
    cbar.set_label("McNemar p-value (lower = stronger evidence)", fontsize=11)
    fig.subplots_adjust(left=0.15, right=0.87, top=0.86, bottom=0.11, hspace=0.40)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.root
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_roots = parse_root_specs(args.metric_roots, args.root)
    model_rows = compute_model_significance_rows(metric_roots, args.metric)

    pair_csv = args.root / "swap_pair_level_analysis.csv"
    if not pair_csv.exists():
        raise FileNotFoundError(f"Missing pair-level analysis file: {pair_csv}")
    pair_rows = load_pair_rows(pair_csv)
    variant_rows = compute_variant_significance_rows(
        pair_rows, args.split_family, correction=args.correction, correction_scope=args.correction_scope
    )
    cluster_rows = compute_variant_cluster_bootstrap_rows(pair_rows, args.split_family)
    model_cluster_rows = compute_model_cluster_significance_rows(pair_rows, args.split_family)
    pair_cluster_rows = compute_variant_pair_cluster_significance_rows(
        pair_rows, args.split_family, correction_scope=args.correction_scope
    )

    model_csv = out_dir / "skin_tone_significance_summary.csv"
    model_json = out_dir / "skin_tone_significance_summary.json"
    variant_csv = out_dir / f"skin_tone_variant_swap_significance_{args.split_family}.csv"
    variant_json = out_dir / f"skin_tone_variant_swap_significance_{args.split_family}.json"
    cluster_csv = out_dir / f"skin_tone_variant_swap_cluster_bootstrap_{args.split_family}.csv"
    cluster_json = out_dir / f"skin_tone_variant_swap_cluster_bootstrap_{args.split_family}.json"
    summary_md = out_dir / "skin_tone_significance_summary.md"
    heatmap_pdf = out_dir / f"skin_tone_variant_swap_significance_{args.split_family}.pdf"
    clustered_heatmap_pdf = out_dir / f"skin_tone_variant_swap_significance_clustered_{args.split_family}.pdf"
    model_cluster_csv = out_dir / f"skin_tone_model_cluster_significance_{args.split_family}.csv"
    model_cluster_json = out_dir / f"skin_tone_model_cluster_significance_{args.split_family}.json"
    model_cluster_pdf = out_dir / f"skin_tone_model_cluster_significance_{args.split_family}.pdf"
    aggregate_col_pdf = out_dir / f"skin_tone_variant_swap_significance_with_aggregate_{args.split_family}.pdf"
    pair_cluster_csv = out_dir / f"skin_tone_color_pair_cluster_significance_{args.split_family}.csv"
    pair_cluster_json = out_dir / f"skin_tone_color_pair_cluster_significance_{args.split_family}.json"
    pair_cluster_pdf = out_dir / f"skin_tone_color_pair_cluster_significance_{args.split_family}.pdf"

    write_csv(model_csv, model_rows)
    write_json(model_json, model_rows)
    write_csv(variant_csv, variant_rows)
    write_json(variant_json, variant_rows)
    write_csv(cluster_csv, cluster_rows)
    write_json(cluster_json, cluster_rows)
    write_csv(model_cluster_csv, model_cluster_rows)
    write_json(model_cluster_json, model_cluster_rows)
    write_md(summary_md, model_rows, variant_rows, float(args.alpha), args.correction, args.correction_scope)
    plot_variant_heatmaps(heatmap_pdf, variant_rows, str(args.split_family), float(args.alpha), args.correction)
    plot_variant_significance_with_cluster_bootstrap(
        clustered_heatmap_pdf, variant_rows, cluster_rows, str(args.split_family), float(args.alpha), args.correction
    )
    plot_model_cluster_significance(model_cluster_pdf, model_cluster_rows, float(args.alpha))
    plot_variant_heatmaps_with_aggregate_column(
        aggregate_col_pdf, variant_rows, model_cluster_rows, str(args.split_family), float(args.alpha), args.correction
    )
    write_csv(pair_cluster_csv, pair_cluster_rows)
    write_json(pair_cluster_json, pair_cluster_rows)
    plot_variant_pair_cluster_heatmap(
        pair_cluster_pdf, pair_cluster_rows, str(args.split_family), float(args.alpha)
    )

    print(model_csv)
    print(model_json)
    print(variant_csv)
    print(variant_json)
    print(cluster_csv)
    print(cluster_json)
    print(model_cluster_csv)
    print(model_cluster_json)
    print(summary_md)
    print(heatmap_pdf)
    print(clustered_heatmap_pdf)
    print(model_cluster_pdf)
    print(aggregate_col_pdf)
    print(pair_cluster_csv)
    print(pair_cluster_json)
    print(pair_cluster_pdf)


if __name__ == "__main__":
    main()
