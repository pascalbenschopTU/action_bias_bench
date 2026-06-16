from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from aggregate_skin_tone_probe import load_rows
try:
    from .schema import stable_seed
except ImportError:  # pragma: no cover - direct script execution
    from schema import stable_seed

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_actionbiasbench"))

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare skin-tone robustness across color-jitter conditions.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Condition roots in label=path format, e.g. cj0p0=out/..._cj0p0 cj0p4=out/..._cj0p4",
    )
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = [float(value) for value in values if float(value) == float(value)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    arr = np.asarray(clean, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def bootstrap_ci(values: Sequence[float], *, iters: int = 2000, seed: int = 0) -> Tuple[float, float]:
    clean = [float(value) for value in values if float(value) == float(value)]
    if not clean:
        return float("nan"), float("nan")
    arr = np.asarray(clean, dtype=float)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    n = arr.size
    samples = []
    for _ in range(iters):
        samples.append(float(np.mean(arr[rng.integers(0, n, size=n)])))
    sample_arr = np.asarray(samples, dtype=float)
    return float(np.percentile(sample_arr, 2.5)), float(np.percentile(sample_arr, 97.5))


def paired_ttest(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if scipy_stats is not None:
        try:
            result = scipy_stats.ttest_rel(x_arr, y_arr, nan_policy="omit")
            return float(result.statistic), float(result.pvalue)
        except Exception:
            pass
    diffs = x_arr - y_arr
    n = diffs.size
    if n <= 1:
        return float("nan"), float("nan")
    diff_mean = float(np.mean(diffs))
    diff_std = float(np.std(diffs, ddof=1))
    if diff_std == 0.0:
        if abs(diff_mean) < 1e-12:
            return 0.0, 1.0
        return float("inf") if diff_mean > 0 else float("-inf"), 0.0
    t_stat = diff_mean / (diff_std / np.sqrt(n))
    if scipy_stats is not None:
        try:
            p_value = 2.0 * scipy_stats.t.sf(abs(t_stat), df=n - 1)
            return float(t_stat), float(p_value)
        except Exception:
            pass
    return float(t_stat), float("nan")


def paired_wilcoxon(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diffs = x_arr - y_arr
    if np.allclose(diffs, 0.0):
        return 0.0, 1.0
    if scipy_stats is not None:
        try:
            result = scipy_stats.wilcoxon(x_arr, y_arr, zero_method="wilcox", alternative="two-sided")
            return float(result.statistic), float(result.pvalue)
        except Exception:
            pass
    return float("nan"), float("nan")


def bonferroni_adjust(values: Sequence[float]) -> List[float]:
    finite = [value for value in values if value == value]
    m = len(finite)
    if m == 0:
        return [float("nan") for _ in values]
    return [min(1.0, value * m) if value == value else float("nan") for value in values]


def parse_root_specs(raw_specs: Sequence[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for spec in raw_specs:
        text = str(spec).strip()
        if not text:
            continue
        if "=" in text:
            label, raw_path = text.split("=", 1)
            label = label.strip()
            path = Path(raw_path.strip()).expanduser().resolve()
        else:
            path = Path(text).expanduser().resolve()
            label = path.name
        if not label:
            raise ValueError(f"Invalid --roots item (missing label): {spec}")
        specs.append((label, path))
    if not specs:
        raise ValueError("No valid condition roots provided.")
    return specs


def infer_condition_value(label: str, path: Path, fallback_index: int) -> float:
    candidates = [str(label), str(path.name), str(path.as_posix())]
    for text in candidates:
        compact = text.lower().replace("-", "_")
        match_p = re.search(r"(\d+)p(\d+)", compact)
        if match_p:
            return float(f"{match_p.group(1)}.{match_p.group(2)}")
        match_dec = re.search(r"(-?\d+(?:\.\d+)?)", compact)
        if match_dec:
            try:
                return float(match_dec.group(1))
            except Exception:
                pass
    return float(fallback_index)


def build_per_seed_rows(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    by_seed_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        pair_tag = str(row.get("experiment_tag", row.get("pair_tag", "")))
        key = (pair_tag, str(row.get("modality", "")), str(row.get("seed", "")))
        item = by_seed_key.setdefault(
            key,
            {
                "pair_tag": key[0],
                "modality": key[1],
                "seed": key[2],
            },
        )
        split_name = str(row.get("eval_split", ""))
        item[f"{split_name}_{metric_name}_mean"] = to_float(row.get(f"{metric_name}_mean"))

    out: List[Dict[str, object]] = []
    for item in by_seed_key.values():
        matched_unseen = to_float(item.get(f"eval_matched_unseen_ids_{metric_name}_mean"))
        matched_seen = to_float(item.get(f"eval_matched_seen_ids_{metric_name}_mean"))
        shifted_seen = to_float(item.get(f"eval_shifted_seen_ids_{metric_name}_mean"))
        shifted_unseen = to_float(item.get(f"eval_shifted_unseen_ids_{metric_name}_mean"))
        item[f"{metric_name}_matched_seen_ids"] = matched_seen
        item[f"{metric_name}_matched_unseen_ids"] = matched_unseen
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


def load_pred_flip_rates(root: Path) -> Dict[Tuple[str, str], float]:
    pair_csv = root / "swap_pair_level_analysis.csv"
    if not pair_csv.exists():
        return {}
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    with pair_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = str(row.get("model", "")).strip()
            split_family = str(row.get("split_family", "")).strip()
            pred_flip = to_float(row.get("pred_flip"))
            if not model or not split_family or not (pred_flip == pred_flip):
                continue
            grouped[(model, split_family)].append(pred_flip)
    rates: Dict[Tuple[str, str], float] = {}
    for key, values in grouped.items():
        if not values:
            continue
        rates[key] = float(sum(values) / len(values))
    return rates


def summarize_condition(
    *,
    label: str,
    root: Path,
    condition_value: float,
    metric_name: str,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], float]]:
    rows = load_rows(root)
    per_seed_rows = build_per_seed_rows(rows, metric_name)
    pred_flip_rates = load_pred_flip_rates(root)

    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in per_seed_rows:
        modality = str(row.get("modality", ""))
        if not modality.startswith("rgb_torchvision:"):
            continue
        model = modality.split(":", 1)[1]
        normalized = dict(row)
        normalized["model"] = model
        by_model[model].append(normalized)

    summary_rows: List[Dict[str, object]] = []
    for model, model_rows in sorted(by_model.items()):
        item: Dict[str, object] = {
            "condition_label": label,
            "condition_value": float(condition_value),
            "root": str(root),
            "model": model,
            "num_units": len(model_rows),
        }
        for suffix in (
            "drop_training_videos",
            "drop_testing_videos",
            "shifted_seen_ids",
            "shifted_unseen_ids",
        ):
            values = [to_float(row.get(f"{metric_name}_{suffix}")) for row in model_rows]
            mean_value, std_value = mean_std(values)
            item[f"{metric_name}_{suffix}_mean"] = mean_value
            item[f"{metric_name}_{suffix}_std"] = std_value
        item["pred_flip_seen_rate"] = to_float(pred_flip_rates.get((model, "seen"), float("nan")))
        item["pred_flip_unseen_rate"] = to_float(pred_flip_rates.get((model, "unseen"), float("nan")))
        summary_rows.append(item)
    return summary_rows, pred_flip_rates


def collect_unit_metrics(root: Path, metric_name: str) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    rows = load_rows(root)
    per_seed_rows = build_per_seed_rows(rows, metric_name)
    by_model: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = defaultdict(dict)
    for row in per_seed_rows:
        modality = str(row.get("modality", ""))
        if not modality.startswith("rgb_torchvision:"):
            continue
        model = modality.split(":", 1)[1]
        unit_key = (str(row.get("pair_tag", "")), str(row.get("seed", "")))
        by_model[model][unit_key] = {
            "drop_training_videos": to_float(row.get(f"{metric_name}_drop_training_videos")),
            "drop_testing_videos": to_float(row.get(f"{metric_name}_drop_testing_videos")),
            "shifted_seen_ids": to_float(row.get(f"{metric_name}_shifted_seen_ids")),
            "shifted_unseen_ids": to_float(row.get(f"{metric_name}_shifted_unseen_ids")),
        }
    return by_model


def build_robustness_rows(root_specs: Sequence[Tuple[str, Path, float]], metric_name: str) -> List[Dict[str, object]]:
    if len(root_specs) < 2:
        return []
    by_condition = {
        label: collect_unit_metrics(root, metric_name)
        for label, root, _condition_value in root_specs
    }
    baseline_label = root_specs[0][0]
    comparison_rows: List[Dict[str, object]] = []
    all_models = sorted({model for data in by_condition.values() for model in data.keys()})

    for model in all_models:
        baseline_units = by_condition[baseline_label].get(model, {})
        for label, _root, _condition_value in root_specs[1:]:
            compare_units = by_condition[label].get(model, {})
            shared_units = sorted(set(baseline_units.keys()) & set(compare_units.keys()))
            if not shared_units:
                continue
            for metric_key in ("drop_training_videos", "drop_testing_videos", "shifted_unseen_ids"):
                baseline_values: List[float] = []
                compare_values: List[float] = []
                for unit_key in shared_units:
                    left = to_float(baseline_units[unit_key].get(metric_key))
                    right = to_float(compare_units[unit_key].get(metric_key))
                    if left == left and right == right:
                        baseline_values.append(left)
                        compare_values.append(right)
                if not baseline_values:
                    continue
                diffs = [right - left for left, right in zip(baseline_values, compare_values)]
                ci_low, ci_high = bootstrap_ci(
                    diffs,
                    seed=stable_seed((model, label, metric_key), modulo=2**16),
                )
                t_stat, t_p = paired_ttest(compare_values, baseline_values)
                w_stat, w_p = paired_wilcoxon(compare_values, baseline_values)
                comparison_rows.append(
                    {
                        "model": model,
                        "reference_condition": baseline_label,
                        "comparator_condition": label,
                        "metric": metric_key,
                        "num_shared_units": len(baseline_values),
                        "reference_mean": float(np.mean(baseline_values)),
                        "reference_std": float(np.std(baseline_values, ddof=1)) if len(baseline_values) > 1 else 0.0,
                        "comparator_mean": float(np.mean(compare_values)),
                        "comparator_std": float(np.std(compare_values, ddof=1)) if len(compare_values) > 1 else 0.0,
                        "comparator_minus_reference_mean": float(np.mean(diffs)),
                        "diff_ci_low": ci_low,
                        "diff_ci_high": ci_high,
                        "paired_t_stat": t_stat,
                        "paired_t_p": t_p,
                        "wilcoxon_stat": w_stat,
                        "wilcoxon_p": w_p,
                    }
                )
    grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(comparison_rows):
        grouped_indices[(str(row["metric"]), str(row["comparator_condition"]))].append(idx)
    for indices in grouped_indices.values():
        t_adjusted = bonferroni_adjust([to_float(comparison_rows[idx]["paired_t_p"]) for idx in indices])
        w_adjusted = bonferroni_adjust([to_float(comparison_rows[idx]["wilcoxon_p"]) for idx in indices])
        for idx, t_adj, w_adj in zip(indices, t_adjusted, w_adjusted):
            comparison_rows[idx]["paired_t_p_bonferroni"] = t_adj
            comparison_rows[idx]["wilcoxon_p_bonferroni"] = w_adj
    return comparison_rows


def collect_pair_sensitivity(
    *,
    root_specs: Sequence[Tuple[str, Path, float]],
    metric_name: str,
) -> Tuple[List[str], List[str], np.ndarray]:
    condition_labels = [label for label, _path, _value in root_specs]
    pair_model_keys: set[str] = set()
    values_by_condition: Dict[str, Dict[str, float]] = {}

    for label, root, _condition_value in root_specs:
        rows = load_rows(root)
        per_seed_rows = build_per_seed_rows(rows, metric_name)
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in per_seed_rows:
            modality = str(row.get("modality", ""))
            if not modality.startswith("rgb_torchvision:"):
                continue
            model = modality.split(":", 1)[1]
            pair_tag = str(row.get("pair_tag", ""))
            key = f"{model} | {pair_tag}"
            value = to_float(row.get(f"{metric_name}_drop_testing_videos"))
            if value == value:
                grouped[key].append(value)
        values_by_condition[label] = {}
        for key, values in grouped.items():
            if values:
                values_by_condition[label][key] = float(sum(values) / len(values))
                pair_model_keys.add(key)

    sorted_rows = sorted(pair_model_keys)
    if not sorted_rows:
        return condition_labels, [], np.zeros((0, len(condition_labels)), dtype=float)

    baseline_label = condition_labels[0]
    baseline_values = values_by_condition.get(baseline_label, {})
    matrix = np.full((len(sorted_rows), len(condition_labels)), np.nan, dtype=float)
    for row_idx, key in enumerate(sorted_rows):
        base = to_float(baseline_values.get(key, float("nan")))
        for col_idx, label in enumerate(condition_labels):
            current = to_float(values_by_condition.get(label, {}).get(key, float("nan")))
            if base == base and current == current:
                matrix[row_idx, col_idx] = current - base
    return condition_labels, sorted_rows, matrix


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_pvalue_label(value: float) -> str:
    if not (value == value):
        return "p=n/a"
    if value < 0.001:
        return "p<.001"
    return f"p={value:.3f}".replace("0.", ".")


def plot_condition_comparison_bars(
    path: Path,
    summary_rows: Sequence[Dict[str, object]],
    metric_name: str,
    robustness_rows: Sequence[Dict[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping {path.name}: {exc}", flush=True)
        return

    if not summary_rows:
        return

    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_model[str(row.get("model", ""))].append(dict(row))

    robustness_lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in robustness_rows:
        robustness_lookup[
            (
                str(row.get("model", "")),
                str(row.get("metric", "")),
                str(row.get("comparator_condition", "")),
            )
        ] = dict(row)

    metric_specs = [
        (
            f"{metric_name}_drop_training_videos_mean",
            f"{metric_name}_drop_training_videos_std",
            "drop_training_videos",
            "Color jitter effect on training-split skin-tone swap",
            "Model backbone",
            "Mean swap drop in F1 (matched - shifted, lower is better)",
        ),
        (
            f"{metric_name}_drop_testing_videos_mean",
            f"{metric_name}_drop_testing_videos_std",
            "drop_testing_videos",
            "Color jitter effect on test-split skin-tone swap",
            "Model backbone",
            "Mean swap drop in F1 (matched - shifted, lower is better)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=220)
    condition_values = sorted({to_float(row.get("condition_value")) for row in summary_rows})
    models_sorted = sorted(by_model.keys())
    y_positions = np.arange(len(models_sorted), dtype=float)
    width = 0.22 if len(condition_values) >= 3 else 0.28
    # For horizontal grouped bars, use top-to-bottom order inside each model:
    # jitter=0.0 (top), then 0.4, then 0.8 (bottom).
    offsets = np.linspace(width, -width, max(1, len(condition_values)))
    tab10 = plt.get_cmap("tab10")
    bar_colors = [tab10(idx % 10) for idx in range(len(condition_values))]

    for ax, (metric_key, std_key, robustness_metric_key, title, xlabel, ylabel) in zip(np.atleast_1d(axes), metric_specs):
        panel_max = -float("inf")
        panel_min = float("inf")
        for cond_idx, condition_value in enumerate(condition_values):
            label_rows = [
                row for row in summary_rows
                if to_float(row.get("condition_value")) == condition_value
            ]
            condition_label = str(label_rows[0].get("condition_label", "")) if label_rows else ""
            values_by_model = {str(row.get("model", "")): dict(row) for row in label_rows}
            y = [to_float(values_by_model.get(model, {}).get(metric_key)) for model in models_sorted]
            yerr = [to_float(values_by_model.get(model, {}).get(std_key)) for model in models_sorted]
            finite_tops = [value + err for value, err in zip(y, yerr) if value == value and err == err]
            finite_bottoms = [value - err for value, err in zip(y, yerr) if value == value and err == err]
            if finite_tops:
                panel_max = max(panel_max, max(finite_tops))
            if finite_bottoms:
                panel_min = min(panel_min, min(finite_bottoms))
            ax.barh(
                y_positions + offsets[cond_idx],
                y,
                height=width,
                xerr=yerr,
                capsize=2.5,
                label=f"jitter={condition_value:.1f}",
                color=bar_colors[cond_idx],
                edgecolor="#222222",
                linewidth=0.6,
                alpha=0.95,
            )
            if cond_idx > 0 and condition_label:
                for model_idx, model in enumerate(models_sorted):
                    value = y[model_idx]
                    err = yerr[model_idx]
                    if not (value == value and err == err):
                        continue
                    robust_row = robustness_lookup.get((model, robustness_metric_key, condition_label))
                    if not robust_row:
                        continue
                    # Use unadjusted paired p-values in the plot annotation.
                    p_value = to_float(robust_row.get("wilcoxon_p"))
                    if not (p_value == p_value):
                        p_value = to_float(robust_row.get("paired_t_p"))
                    ax.text(
                        value + err + 0.004,
                        y_positions[model_idx] + offsets[cond_idx],
                        _format_pvalue_label(p_value),
                        ha="left",
                        va="center",
                        fontsize=7,
                        color="#333333",
                    )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(models_sorted)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25)
        if panel_max > -float("inf") and panel_min < float("inf"):
            dynamic_range = panel_max - panel_min if panel_max > panel_min else max(abs(panel_max), 0.01)
            margin = max(0.01, 0.18 * dynamic_range)
            ax.set_xlim(panel_min - 0.05 * margin, panel_max + margin)

    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Color-jitter comparison for skin-tone swap sensitivity", fontsize=14, weight="bold", y=0.98)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pair_heatmap(path: Path, condition_labels: Sequence[str], row_labels: Sequence[str], matrix: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping {path.name}: {exc}", flush=True)
        return

    if matrix.size == 0:
        return

    finite_values = matrix[np.isfinite(matrix)]
    max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 0.05
    max_abs = max(max_abs, 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig_h = max(5.5, 0.28 * len(row_labels) + 1.8)
    fig_w = max(8.0, 1.1 * len(condition_labels) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_xticks(range(len(condition_labels)))
    ax.set_xticklabels(condition_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Color-jitter condition", fontsize=10, weight="bold")
    ax.set_ylabel("Model | action pair", fontsize=10, weight="bold")
    ax.set_title("Pair sensitivity change vs baseline jitter\n(delta drop_testing_videos)", fontsize=12, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Delta swap drop (positive = more sensitive)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root_specs_raw = parse_root_specs(args.roots)
    root_specs = [
        (label, root, infer_condition_value(label, root, idx))
        for idx, (label, root) in enumerate(root_specs_raw)
    ]
    root_specs.sort(key=lambda item: item[2])

    all_rows: List[Dict[str, object]] = []
    for label, root, condition_value in root_specs:
        condition_rows, _pred_flip = summarize_condition(
            label=label,
            root=root,
            condition_value=condition_value,
            metric_name=args.metric,
        )
        all_rows.extend(condition_rows)

    all_rows.sort(key=lambda row: (to_float(row.get("condition_value")), str(row.get("model", ""))))
    csv_path = out_dir / "color_jitter_comparison.csv"
    json_path = out_dir / "color_jitter_comparison.json"
    plot_path = out_dir / "color_jitter_comparison.pdf"
    heatmap_path = out_dir / "color_jitter_pair_heatmap.pdf"
    robustness_csv = out_dir / "color_jitter_robustness_checks.csv"
    robustness_json = out_dir / "color_jitter_robustness_checks.json"

    robustness_rows = build_robustness_rows(root_specs, args.metric)
    write_csv(csv_path, all_rows)
    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    plot_condition_comparison_bars(plot_path, all_rows, args.metric, robustness_rows)
    write_csv(robustness_csv, robustness_rows)
    robustness_json.write_text(json.dumps(robustness_rows, indent=2), encoding="utf-8")

    condition_labels, row_labels, heatmap_matrix = collect_pair_sensitivity(
        root_specs=root_specs,
        metric_name=args.metric,
    )
    plot_pair_heatmap(heatmap_path, condition_labels, row_labels, heatmap_matrix)

    print(csv_path)
    print(json_path)
    if plot_path.exists():
        print(plot_path)
    if heatmap_path.exists():
        print(heatmap_path)
    if robustness_csv.exists():
        print(robustness_csv)
    if robustness_json.exists():
        print(robustness_json)


if __name__ == "__main__":
    main()
