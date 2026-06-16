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
    from .schema import SWAP_LABELS, SWAP_ORDER, stable_seed
except ImportError:  # pragma: no cover - direct script execution
    from schema import SWAP_LABELS, SWAP_ORDER, stable_seed

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


def compute_variant_significance_rows(pair_rows: Sequence[Dict[str, object]], split_family: str) -> List[Dict[str, object]]:
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

    for model, indices in by_model.items():
        adjusted = bonferroni_adjust([float(rows[idx]["mcnemar_p_raw"]) for idx in indices])
        for idx, p_adj in zip(indices, adjusted):
            rows[idx]["mcnemar_p_bonferroni"] = p_adj
    return rows


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


def write_md(path: Path, model_rows: Sequence[Dict[str, object]], variant_rows: Sequence[Dict[str, object]], alpha: float) -> None:
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
            f"McNemar exact p-values are tested against `alpha={alpha:.3f}` and Bonferroni-adjusted within each model.",
            "",
        ]
    )
    variant_focus = [row for row in variant_rows if row.get("split_family") == "unseen"]
    variant_focus.sort(
        key=lambda row: (
            float(row.get("mcnemar_p_bonferroni", 1.0)),
            -float(row.get("accuracy_drop", 0.0)),
        )
    )
    for row in variant_focus[:16]:
        lines.append(
            f"- `{row['model']}` `{row['variant_matched']}→{row['variant_shifted']}` unseen-ID: "
            f"matched_acc={float(row['matched_accuracy']):.4f}, shifted_acc={float(row['shifted_accuracy']):.4f}, "
            f"drop={float(row['accuracy_drop']):.4f}, raw p={float(row['mcnemar_p_raw']):.4g}, "
            f"adj p={float(row['mcnemar_p_bonferroni']):.4g}, n={int(row['n_pairs'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_variant_heatmaps(path: Path, variant_rows: Sequence[Dict[str, object]], split_family: str, alpha: float) -> None:
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
        adj_mat[i, j] = float(row.get("mcnemar_p_bonferroni", 1.0))
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
        ("p-values", "Bonferroni-adjusted p-values"),
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
                mark = "*" if p_value < alpha else ""
                ax.text(j, i, f"{p_value:.2f}{mark}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xlim(-0.5, len(SWAP_ORDER) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_ylabel("Model", fontsize=11, fontweight="bold")
        ax.axvline(1.5, color="#333333", linewidth=1.0, alpha=0.5)

    axes[-1].set_xlabel("Directional skin-tone swap", fontsize=11, fontweight="bold")
    fig.suptitle("Skin-tone swap significance", fontsize=18, fontweight="bold", y=0.965)

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
    variant_rows = compute_variant_significance_rows(pair_rows, args.split_family)

    model_csv = out_dir / "skin_tone_significance_summary.csv"
    model_json = out_dir / "skin_tone_significance_summary.json"
    variant_csv = out_dir / f"skin_tone_variant_swap_significance_{args.split_family}.csv"
    variant_json = out_dir / f"skin_tone_variant_swap_significance_{args.split_family}.json"
    summary_md = out_dir / "skin_tone_significance_summary.md"
    heatmap_pdf = out_dir / f"skin_tone_variant_swap_significance_{args.split_family}.pdf"

    write_csv(model_csv, model_rows)
    write_json(model_json, model_rows)
    write_csv(variant_csv, variant_rows)
    write_json(variant_json, variant_rows)
    write_md(summary_md, model_rows, variant_rows, float(args.alpha))
    plot_variant_heatmaps(heatmap_pdf, variant_rows, str(args.split_family), float(args.alpha))

    print(model_csv)
    print(model_json)
    print(variant_csv)
    print(variant_json)
    print(summary_md)
    print(heatmap_pdf)


if __name__ == "__main__":
    main()
