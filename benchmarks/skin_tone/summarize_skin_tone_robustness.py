from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

from aggregate_skin_tone_probe import load_rows
try:
    from .schema import RGB_MODEL_COLORS
except ImportError:  # pragma: no cover - direct script execution
    from schema import RGB_MODEL_COLORS
try:
    from .analyze_skin_tone_swap_influence import load_prediction_rows, build_pair_rows
except ImportError:  # pragma: no cover - direct script execution
    from analyze_skin_tone_swap_influence import load_prediction_rows, build_pair_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize skin-tone robustness across all models and create a figure.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="f1_macro")
    return parser.parse_args()


_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*"]
_PAIR_ORDER = [
    "squat_vs_tie",
    "tie_vs_squat",
    "clap_vs_celebrate",
    "celebrate_vs_clap",
    "dribble_vs_golf",
    "golf_vs_dribble",
    "lunge_vs_cartwheel",
    "cartwheel_vs_lunge",
    "yawn_vs_fish",
    "fish_vs_yawn",
]


def display_name(modality: str) -> str:
    if modality == "flow_i3d_external":
        return "I3D_flow"
    if modality.startswith("rgb_torchvision:"):
        return modality.split(":", 1)[1]
    return modality


def color_for_modality(modality: str) -> str:
    palette = {
        "motion": "#2C6BA0",
        "rgb": "#D17A22",
        "flow_i3d_external": "#e377c2",
    }
    if modality.startswith("rgb_torchvision:"):
        model_name = modality.split(":", 1)[1]
        return RGB_MODEL_COLORS.get(model_name, "#6A717D")
    return palette.get(modality, "#6A717D")


def modality_sort_key(modality: str) -> tuple[int, str]:
    if modality == "flow_i3d_external":
        return (0, display_name(modality))
    if modality.startswith("rgb_torchvision:"):
        return (1, display_name(modality))
    return (2, display_name(modality))


def pair_sort_key(pair_tag: str) -> tuple[int, str]:
    try:
        return (_PAIR_ORDER.index(pair_tag), pair_tag)
    except ValueError:
        return (len(_PAIR_ORDER), pair_tag)


def pretty_pair_label(pair_tag: str, *, multiline: bool = False) -> str:
    left, _, right = pair_tag.partition("_vs_")
    if not right:
        return pair_tag
    sep = "\nvs\n" if multiline else " vs "
    return f"{left}{sep}{right}"


def build_per_seed_rows(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    by_seed_key: Dict[tuple[str, str, str], Dict[str, object]] = {}
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
        drop_training = matched_seen - shifted_seen if matched_seen == matched_seen and shifted_seen == shifted_seen else float("nan")
        drop_testing = matched_unseen - shifted_unseen if matched_unseen == matched_unseen and shifted_unseen == shifted_unseen else float("nan")
        item[f"{metric_name}_matched_seen_ids"] = matched_seen
        item[f"{metric_name}_matched_unseen_ids"] = matched_unseen
        item[f"{metric_name}_shifted_seen_ids"] = shifted_seen
        item[f"{metric_name}_shifted_unseen_ids"] = shifted_unseen
        item[f"{metric_name}_drop_training_videos"] = drop_training
        item[f"{metric_name}_drop_testing_videos"] = drop_testing
        out.append(item)
    return out


def mean_std(values: List[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if float(value) == float(value)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    mean_value = sum(clean) / len(clean)
    variance = sum((value - mean_value) ** 2 for value in clean) / (len(clean) - 1)
    return mean_value, variance ** 0.5


def summarize_modalities(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["modality"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for modality, modality_rows in sorted(grouped.items(), key=lambda item: modality_sort_key(item[0])):
        item: Dict[str, object] = {
            "modality": modality,
            "display_name": display_name(modality),
            "num_units": len(modality_rows),
            "color": color_for_modality(modality),
        }
        for suffix in (
            "matched_seen_ids",
            "matched_unseen_ids",
            "shifted_seen_ids",
            "shifted_unseen_ids",
            "drop_training_videos",
            "drop_testing_videos",
        ):
            values = [float(row.get(f"{metric_name}_{suffix}", float("nan"))) for row in modality_rows]
            mean_value, std_value = mean_std(values)
            item[f"{metric_name}_{suffix}_mean"] = mean_value
            item[f"{metric_name}_{suffix}_std"] = std_value
        summary_rows.append(item)
    return summary_rows


def summarize_pairs(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        pair_tag = str(row.get("experiment_tag", row["pair_tag"]))
        grouped.setdefault((pair_tag, str(row["modality"])), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (pair_tag, modality), pair_rows in sorted(
        grouped.items(),
        key=lambda item: (pair_sort_key(item[0][0]), modality_sort_key(item[0][1])),
    ):
        item: Dict[str, object] = {
            "pair_tag": pair_tag,
            "pair_display": pretty_pair_label(pair_tag),
            "modality": modality,
            "display_name": display_name(modality),
            "color": color_for_modality(modality),
            "num_units": len(pair_rows),
        }
        for suffix in (
            "matched_seen_ids",
            "matched_unseen_ids",
            "shifted_seen_ids",
            "shifted_unseen_ids",
            "drop_training_videos",
            "drop_testing_videos",
        ):
            values = [float(row.get(f"{metric_name}_{suffix}", float("nan"))) for row in pair_rows]
            mean_value, std_value = mean_std(values)
            item[f"{metric_name}_{suffix}_mean"] = mean_value
            item[f"{metric_name}_{suffix}_std"] = std_value
        summary_rows.append(item)
    return summary_rows


def write_csv(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / f"skin_tone_robustness_summary_{metric_name}.csv"
    fieldnames = [
        "modality",
        "display_name",
        "num_units",
        "color",
        f"{metric_name}_matched_seen_ids_mean",
        f"{metric_name}_matched_seen_ids_std",
        f"{metric_name}_matched_unseen_ids_mean",
        f"{metric_name}_matched_unseen_ids_std",
        f"{metric_name}_shifted_seen_ids_mean",
        f"{metric_name}_shifted_seen_ids_std",
        f"{metric_name}_shifted_unseen_ids_mean",
        f"{metric_name}_shifted_unseen_ids_std",
        f"{metric_name}_drop_training_videos_mean",
        f"{metric_name}_drop_training_videos_std",
        f"{metric_name}_drop_testing_videos_mean",
        f"{metric_name}_drop_testing_videos_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_pair_csv(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / f"skin_tone_pair_robustness_summary_{metric_name}.csv"
    fieldnames = [
        "pair_tag",
        "pair_display",
        "modality",
        "display_name",
        "num_units",
        "color",
        f"{metric_name}_matched_seen_ids_mean",
        f"{metric_name}_matched_seen_ids_std",
        f"{metric_name}_matched_unseen_ids_mean",
        f"{metric_name}_matched_unseen_ids_std",
        f"{metric_name}_shifted_seen_ids_mean",
        f"{metric_name}_shifted_seen_ids_std",
        f"{metric_name}_shifted_unseen_ids_mean",
        f"{metric_name}_shifted_unseen_ids_std",
        f"{metric_name}_drop_training_videos_mean",
        f"{metric_name}_drop_training_videos_std",
        f"{metric_name}_drop_testing_videos_mean",
        f"{metric_name}_drop_testing_videos_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_json(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / f"skin_tone_robustness_summary_{metric_name}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_path


def write_pair_json(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / f"skin_tone_pair_robustness_summary_{metric_name}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_path


def write_plot(root: Path, rows: List[Dict[str, object]], metric_name: str) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skipping robustness figure: {exc}", flush=True)
        return []

    if not rows:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=180)
    panel_specs = [
        {
            "title": "Train-ID split (seen IDs)",
            "x_key": f"{metric_name}_drop_training_videos_mean",
            "x_std_key": f"{metric_name}_drop_training_videos_std",
            "y_key": f"{metric_name}_shifted_seen_ids_mean",
            "y_std_key": f"{metric_name}_shifted_seen_ids_std",
            "x_label": "Swap drop  (matched − shifted F1)       0 = no bias",
            "y_label": f"Shifted F1 macro  (higher is better)",
        },
        {
            "title": "Test-ID split (held-out IDs)",
            "x_key": f"{metric_name}_drop_testing_videos_mean",
            "x_std_key": f"{metric_name}_drop_testing_videos_std",
            "y_key": f"{metric_name}_shifted_unseen_ids_mean",
            "y_std_key": f"{metric_name}_shifted_unseen_ids_std",
            "x_label": "Swap drop  (matched − shifted F1)       0 = no bias",
            "y_label": f"Shifted F1 macro  (higher is better)",
        },
    ]

    for ax, spec in zip(axes, panel_specs):
        x_values = []
        y_values = []
        for row in rows:
            x = float(row[spec["x_key"]])
            y = float(row[spec["y_key"]])
            x_std = float(row[spec["x_std_key"]])
            y_std = float(row[spec["y_std_key"]])
            if x == x:
                x_values.append(x)
                if x_std == x_std and x_std > 0:
                    x_values.extend([x - x_std, x + x_std])
            if y == y:
                y_values.append(y)
                if y_std == y_std and y_std > 0:
                    y_values.extend([y - y_std, y + y_std])
        x_min = min(x_values) if x_values else -0.05
        x_max = max(x_values) if x_values else 0.1
        y_min = min(y_values) if y_values else 0.0
        y_max = max(y_values) if y_values else 1.0
        x_pad = max(0.012, (x_max - x_min) * 0.14 if x_max > x_min else 0.03)
        y_pad = max(0.02, (y_max - y_min) * 0.22 if y_max > y_min else 0.05)
        ax_xlim = (x_min - x_pad, x_max + x_pad)
        ax_ylim = (max(0.0, y_min - y_pad), min(1.05, y_max + y_pad))
        ax.set_xlim(*ax_xlim)
        ax.set_ylim(*ax_ylim)
        ax.set_title(spec["title"], fontsize=12, weight="bold")
        ax.set_xlabel(spec["x_label"], fontsize=10)
        ax.set_ylabel(spec["y_label"], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)
        ax.set_facecolor("#fbfbfd")

        # Vertical reference line at x=0 (no bias)
        ax.axvline(0.0, color="#2c6e49", linestyle="-", linewidth=1.4, alpha=0.5, zorder=1)
        ax.text(
            0.0,
            ax_ylim[0] + (ax_ylim[1] - ax_ylim[0]) * 0.01,
            "  ideal",
            fontsize=8,
            color="#2c6e49",
            va="bottom",
            ha="left",
            alpha=0.85,
        )

        for idx, row in enumerate(rows):
            x = float(row[spec["x_key"]])
            y = float(row[spec["y_key"]])
            x_std = float(row[spec["x_std_key"]])
            y_std = float(row[spec["y_std_key"]])
            if not (x == x and y == y):
                continue
            color = str(row["color"])
            label = str(row["display_name"])
            marker = _MARKERS[idx % len(_MARKERS)]
            ax.errorbar(
                x,
                y,
                xerr=None if not (x_std == x_std and x_std > 0) else x_std,
                yerr=None if not (y_std == y_std and y_std > 0) else y_std,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=10,
                markeredgewidth=0.8,
                markeredgecolor="#333333",
                alpha=0.92,
                label=label,
                zorder=3,
            )

        # "better" annotation: high y (accuracy), x≈0 (no bias)
        ax.annotate(
            "better →",
            xy=(0.02, 0.97),
            xycoords="axes fraction",
            fontsize=10,
            weight="bold",
            color="#4b4f56",
            va="top",
        )
        ax.text(
            0.02,
            0.89,
            "x<0: reversed bias\n(not better than 0)",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#888888",
            va="top",
            style="italic",
        )

    # Shared legend below the two panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(rows), 4),
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.suptitle("Skin-tone shortcut robustness summary", fontsize=15, weight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0.13, 1, 0.98))

    png_path = root / f"skin_tone_robustness_summary_{metric_name}.png"
    pdf_path = root / f"skin_tone_robustness_summary_{metric_name}.pdf"
    svg_path = root / f"skin_tone_robustness_summary_{metric_name}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def _matrix_from_rows(
    rows: Sequence[Dict[str, object]],
    pair_tags: Sequence[str],
    modalities: Sequence[str],
    value_key: str,
) -> List[List[float]]:
    lookup = {
        (str(row["pair_tag"]), str(row["modality"])): float(row.get(value_key, float("nan")))
        for row in rows
    }
    return [[lookup.get((pair_tag, modality), float("nan")) for pair_tag in pair_tags] for modality in modalities]


def write_pair_heatmap(root: Path, rows: List[Dict[str, object]], metric_name: str) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skipping pair heatmap: {exc}", flush=True)
        return []

    if not rows:
        return []

    # Order both axes by a fixed canonical key so the heatmap layout is identical
    # across conditions (e.g. color-jitter sweeps). Using per-condition scores here
    # would place the same pair/model in different rows/columns per figure.
    modalities = sorted(
        {str(row["modality"]) for row in rows},
        key=modality_sort_key,
    )
    pair_tags = sorted(
        {str(row["pair_tag"]) for row in rows},
        key=pair_sort_key,
    )

    matrices = [
        (
            f"{metric_name}_drop_training_videos_mean",
            "training split skin color swap effect on F1",
        ),
        (
            f"{metric_name}_drop_testing_videos_mean",
            "test split skin color swap effect on F1",
        ),
    ]

    all_values = []
    for key, _ in matrices:
        matrix = _matrix_from_rows(rows, pair_tags, modalities, key)
        for line in matrix:
            all_values.extend([value for value in line if value == value])
    max_abs = max((abs(value) for value in all_values), default=0.05)
    max_abs = max(max_abs, 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig = plt.figure(figsize=(max(10.5, len(pair_tags) * 1.0), 7.6), dpi=200)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[40, 1.4],
        height_ratios=[1, 1],
        wspace=0.12,
        hspace=0.54,
    )
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[1, 0]),
        ]
    )
    cax = fig.add_subplot(grid[:, 1])

    last_im = None
    for ax, (value_key, title) in zip(axes, matrices):
        matrix = np.array(_matrix_from_rows(rows, pair_tags, modalities, value_key), dtype=float)
        last_im = ax.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(range(len(pair_tags)))
        ax.set_xticklabels([pretty_pair_label(pair_tag, multiline=True) for pair_tag in pair_tags], fontsize=9)
        ax.set_yticks(range(len(modalities)))
        ax.set_yticklabels([display_name(modality) for modality in modalities], fontsize=10)
        ax.set_xlabel("Action pair", fontsize=10, fontweight="bold")
        ax.set_ylabel("Model", fontsize=10, fontweight="bold")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if not (value == value):
                    continue
                text_color = "#111111" if abs(value) < max_abs * 0.45 else "white"
                ax.text(col_idx, row_idx, f"{value:.03f}", ha="center", va="center", fontsize=7.5, color=text_color)

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Matched $-$ shifted F1", fontsize=10)
    fig.suptitle("Skin-tone shortcut sensitivity is pair- and architecture-dependent", fontsize=15, weight="bold", y=0.995)
    fig.subplots_adjust(top=0.88, left=0.12, right=0.94, bottom=0.08)

    png_path = root / f"skin_tone_pair_heatmap_{metric_name}.png"
    pdf_path = root / f"skin_tone_pair_heatmap_{metric_name}.pdf"
    svg_path = root / f"skin_tone_pair_heatmap_{metric_name}.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def write_pair_heatmap_test_only(root: Path, rows: List[Dict[str, object]], metric_name: str) -> List[Path]:
    """Single-panel heatmap of the held-out (test) split swap effect on F1."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skipping test-only pair heatmap: {exc}", flush=True)
        return []

    if not rows:
        return []

    modalities = sorted({str(row["modality"]) for row in rows}, key=modality_sort_key)
    pair_tags = sorted({str(row["pair_tag"]) for row in rows}, key=pair_sort_key)

    value_key = f"{metric_name}_drop_testing_videos_mean"
    matrix = np.array(_matrix_from_rows(rows, pair_tags, modalities, value_key), dtype=float)

    finite = [v for line in matrix for v in line if v == v]
    max_abs = max((abs(v) for v in finite), default=0.05)
    max_abs = max(max_abs, 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig = plt.figure(figsize=(max(10.5, len(pair_tags) * 1.0), 4.6), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[40, 1.4], wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    im = ax.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(pair_tags)))
    ax.set_xticklabels([pretty_pair_label(pair_tag, multiline=True) for pair_tag in pair_tags], fontsize=9)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels([display_name(modality) for modality in modalities], fontsize=10)
    ax.set_xlabel("Action pair", fontsize=10, fontweight="bold")
    ax.set_ylabel("Model", fontsize=10, fontweight="bold")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not (value == value):
                continue
            text_color = "#111111" if abs(value) < max_abs * 0.45 else "white"
            ax.text(col_idx, row_idx, f"{value:.03f}", ha="center", va="center", fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Matched $-$ shifted F1", fontsize=10)
    ax.set_title("Skin-tone swap effect on F1", fontsize=15, weight="bold", pad=12)
    fig.subplots_adjust(top=0.86, left=0.12, right=0.94, bottom=0.16)

    png_path = root / f"skin_tone_pair_heatmap_{metric_name}_testonly.png"
    pdf_path = root / f"skin_tone_pair_heatmap_{metric_name}_testonly.pdf"
    svg_path = root / f"skin_tone_pair_heatmap_{metric_name}_testonly.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def _paired_flip_display_name(model: str) -> str:
    return "I3D_flow" if model == "i3d_flow" else model


def _paired_flip_sort_key(model: str) -> tuple[int, str]:
    return (0, model) if model == "i3d_flow" else (1, model)


def write_pair_heatmap_paired_flip_rate(root: Path) -> List[Path]:
    """Single-panel heatmap of the held-out (test) split swap effect, computed
    from paired clip-level correctness rather than from separately-averaged
    split F1 scores.

    ``write_pair_heatmap_test_only`` compares F1_macro(matched) against
    F1_macro(shifted) as two independently-averaged splits. Those splits do
    not weight every performer identity's clips equally (the shifted split
    draws on the whole opposite tone group, not one specific counterpart), so
    a single identity misclassified under every skin tone can be weighted
    differently in each average and produce an apparent gap with no real
    directional cause. Here each matched clip is instead joined to its
    specific shifted counterpart (same identity, background, and action),
    and the plotted value is (b-c)/n over those pairs, exactly the accuracy
    drop computed on the correctly-paired set: a clip wrong under both tones
    contributes to neither b nor c.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skipping paired flip-rate heatmap: {exc}", flush=True)
        return []

    prediction_rows, _csv_paths = load_prediction_rows(root, None)
    if not prediction_rows:
        return []
    joined_rows, _report = build_pair_rows(prediction_rows, ["unseen"])
    if not joined_rows:
        return []

    counts: Dict[tuple[str, str], Dict[str, int]] = {}
    for row in joined_rows:
        key = (str(row["model"]), str(row["pair_tag"]))
        item = counts.setdefault(key, {"b": 0, "c": 0, "n": 0})
        item["n"] += 1
        matched_correct = int(row["correct_matched"])
        shifted_correct = int(row["correct_shifted"])
        if matched_correct == 1 and shifted_correct == 0:
            item["b"] += 1
        elif matched_correct == 0 and shifted_correct == 1:
            item["c"] += 1

    models = sorted({model for model, _pair in counts}, key=_paired_flip_sort_key)
    pair_tags = sorted({pair for _model, pair in counts}, key=pair_sort_key)
    if not models or not pair_tags:
        return []

    matrix = np.full((len(models), len(pair_tags)), np.nan)
    for i, model in enumerate(models):
        for j, pair_tag in enumerate(pair_tags):
            item = counts.get((model, pair_tag))
            if item is None or item["n"] == 0:
                continue
            matrix[i, j] = (item["b"] - item["c"]) / item["n"]

    finite = [v for line in matrix for v in line if v == v]
    max_abs = max((abs(v) for v in finite), default=0.05)
    max_abs = max(max_abs, 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig = plt.figure(figsize=(max(10.5, len(pair_tags) * 1.0), 4.6), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[40, 1.4], wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    im = ax.imshow(matrix, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(pair_tags)))
    ax.set_xticklabels([pretty_pair_label(pair_tag, multiline=True) for pair_tag in pair_tags], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_paired_flip_display_name(model) for model in models], fontsize=10)
    ax.set_xlabel("Action pair", fontsize=10, fontweight="bold")
    ax.set_ylabel("Model", fontsize=10, fontweight="bold")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not (value == value):
                continue
            text_color = "#111111" if abs(value) < max_abs * 0.45 else "white"
            ax.text(col_idx, row_idx, f"{value:.03f}", ha="center", va="center", fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Paired accuracy drop (b-c)/n", fontsize=9)
    ax.set_title("Paired accuracy drop", fontsize=15, weight="bold", pad=12)
    fig.subplots_adjust(top=0.86, left=0.12, right=0.94, bottom=0.16)

    png_path = root / "skin_tone_pair_heatmap_paired_flip_rate_testonly.png"
    pdf_path = root / "skin_tone_pair_heatmap_paired_flip_rate_testonly.pdf"
    svg_path = root / "skin_tone_pair_heatmap_paired_flip_rate_testonly.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def _compute_raw_accuracy_rows(root: Path) -> List[Dict[str, object]]:
    """Raw matched/shifted accuracy per (model, action pair, tone-swap direction),
    pooled over folds and seeds on the held-out ID split. This is the
    unaggregated backing data for the paired-drop heatmap: plain accuracy in
    each condition, not F1 and not a derived delta, so it can be checked
    directly against the summary statistics built on top of it.
    """
    prediction_rows, _csv_paths = load_prediction_rows(root, None)
    if not prediction_rows:
        return []
    joined_rows, _report = build_pair_rows(prediction_rows, ["unseen"])
    if not joined_rows:
        return []

    counts: Dict[tuple[str, str, str], Dict[str, int]] = {}
    for row in joined_rows:
        direction = f"{row['variant_matched']}->{row['variant_shifted']}"
        key = (str(row["model"]), str(row["pair_tag"]), direction)
        item = counts.setdefault(key, {"n": 0, "correct_matched": 0, "correct_shifted": 0})
        item["n"] += 1
        item["correct_matched"] += int(row["correct_matched"])
        item["correct_shifted"] += int(row["correct_shifted"])

    rows: List[Dict[str, object]] = []
    for (model, pair_tag, direction), item in counts.items():
        n = item["n"]
        rows.append(
            {
                "model": model,
                "pair_tag": pair_tag,
                "direction": direction,
                "n": n,
                "correct_matched": item["correct_matched"],
                "correct_shifted": item["correct_shifted"],
                "acc_matched": item["correct_matched"] / n if n else float("nan"),
                "acc_shifted": item["correct_shifted"] / n if n else float("nan"),
            }
        )
    rows.sort(
        key=lambda r: (
            _paired_flip_sort_key(str(r["model"])),
            pair_sort_key(str(r["pair_tag"])),
            str(r["direction"]),
        )
    )
    return rows


def write_raw_accuracy_csv(root: Path) -> List[Path]:
    rows = _compute_raw_accuracy_rows(root)
    if not rows:
        return []
    out_path = root / "skin_tone_raw_accuracy_by_direction_testonly.csv"
    fieldnames = ["model", "pair_tag", "direction", "n", "correct_matched", "correct_shifted", "acc_matched", "acc_shifted"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return [out_path]


def write_raw_accuracy_plot(root: Path) -> List[Path]:
    """Grouped bar chart of raw matched vs. shifted accuracy per action pair,
    one panel per model, pooled over tone-swap direction within each pair.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skipping raw accuracy plot: {exc}", flush=True)
        return []

    direction_rows = _compute_raw_accuracy_rows(root)
    if not direction_rows:
        return []

    pooled: Dict[tuple[str, str], Dict[str, int]] = {}
    for row in direction_rows:
        key = (str(row["model"]), str(row["pair_tag"]))
        item = pooled.setdefault(key, {"n": 0, "correct_matched": 0, "correct_shifted": 0})
        item["n"] += int(row["n"])
        item["correct_matched"] += int(row["correct_matched"])
        item["correct_shifted"] += int(row["correct_shifted"])

    models = sorted({model for model, _pair in pooled}, key=_paired_flip_sort_key)
    pair_tags = sorted({pair for _model, pair in pooled}, key=pair_sort_key)
    if not models or not pair_tags:
        return []

    n_cols = 2
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.6 * n_cols, 3.4 * n_rows), dpi=180, squeeze=False)

    x = np.arange(len(pair_tags))
    width = 0.36

    for idx, model in enumerate(models):
        ax = axes[idx // n_cols][idx % n_cols]
        matched_vals = []
        shifted_vals = []
        for pair_tag in pair_tags:
            item = pooled.get((model, pair_tag))
            if item is None or item["n"] == 0:
                matched_vals.append(float("nan"))
                shifted_vals.append(float("nan"))
                continue
            matched_vals.append(item["correct_matched"] / item["n"])
            shifted_vals.append(item["correct_shifted"] / item["n"])

        ax.bar(x - width / 2, matched_vals, width, label="Matched", color="#4c72b0")
        ax.bar(x + width / 2, shifted_vals, width, label="Shifted", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_pair_label(pair_tag) for pair_tag in pair_tags], fontsize=7.5, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(_paired_flip_display_name(model), fontsize=11, weight="bold")
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx in range(len(models), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Raw matched vs. shifted accuracy per action pair (held-out IDs)", fontsize=14, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    png_path = root / "skin_tone_raw_accuracy_by_pair_testonly.png"
    pdf_path = root / "skin_tone_raw_accuracy_by_pair_testonly.pdf"
    svg_path = root / "skin_tone_raw_accuracy_by_pair_testonly.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def main() -> None:
    args = parse_args()
    raw_rows = load_rows(args.root)
    per_seed_rows = build_per_seed_rows(raw_rows, args.metric)
    summary_rows = summarize_modalities(per_seed_rows, args.metric)
    pair_rows = summarize_pairs(per_seed_rows, args.metric)
    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(args.root, summary_rows, args.metric)
    json_path = write_json(args.root, summary_rows, args.metric)
    pair_csv_path = write_pair_csv(args.root, pair_rows, args.metric)
    pair_json_path = write_pair_json(args.root, pair_rows, args.metric)
    print(csv_path)
    print(json_path)
    print(pair_csv_path)
    print(pair_json_path)
    for plot_path in write_plot(args.root, summary_rows, args.metric):
        print(plot_path)
    for plot_path in write_pair_heatmap(args.root, pair_rows, args.metric):
        print(plot_path)
    for plot_path in write_pair_heatmap_test_only(args.root, pair_rows, args.metric):
        print(plot_path)
    for plot_path in write_pair_heatmap_paired_flip_rate(args.root):
        print(plot_path)
    for plot_path in write_raw_accuracy_csv(args.root):
        print(plot_path)
    for plot_path in write_raw_accuracy_plot(args.root):
        print(plot_path)


if __name__ == "__main__":
    main()
