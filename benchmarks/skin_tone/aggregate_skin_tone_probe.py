from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    from .schema import RGB_MODEL_COLORS
except ImportError:  # pragma: no cover - direct script execution
    from schema import RGB_MODEL_COLORS


SPLIT_ORDER = [
    "eval_matched_unseen_ids",
    "eval_matched_seen_ids",
    "eval_shifted_seen_ids",
    "eval_shifted_unseen_ids",
]
MODE_BY_MODALITY = {
    "motion": "motion_only",
    "rgb": "rgb_model",
    "rgb_r2plus1d": "rgb_r2plus1d_model",
    "flow_i3d_external": "flow_i3d_external_model",
    "tc_clip": "tc_clip_model",
}
COLOR_BY_MODALITY = {
    "motion": "#1f77b4",
    "rgb": "#ff7f0e",
    "rgb_r2plus1d": "#2ca02c",
    "flow_i3d_external": "#e377c2",
    "tc_clip": "#9467bd",
}
DISPLAY_NAME_BY_MODALITY = {
    "motion": "motion",
    "rgb": "rgb",
    "rgb_r2plus1d": "rgb_r2plus1d",
    "flow_i3d_external": "I3D_flow",
    "tc_clip": "tc_clip",
}
MODALITY_ORDER = ["motion", "rgb", "rgb_r2plus1d", "flow_i3d_external", "tc_clip"]
GOOD_COLOR = (46, 125, 50)
BAD_COLOR = (198, 40, 40)


@dataclass
class LoadRowsReport:
    root: str
    scanned_summary_files: int = 0
    accepted_rows: int = 0
    skipped_parse_failures: List[str] = field(default_factory=list)
    skipped_mode_mismatches: List[Dict[str, str]] = field(default_factory=list)
    skipped_duplicate_keys: List[str] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return (
            len(self.skipped_parse_failures)
            + len(self.skipped_mode_mismatches)
            + len(self.skipped_duplicate_keys)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate skin-tone shortcut probe summaries.")
    parser.add_argument("--root", type=Path, default=Path("out/skin_tone_probe"))
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        choices=[
            "top1",
            "top5",
            "mean_class_acc",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ],
    )
    return parser.parse_args()


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mean_std(values: List[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def lerp_color(t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    r = round(GOOD_COLOR[0] + (BAD_COLOR[0] - GOOD_COLOR[0]) * t)
    g = round(GOOD_COLOR[1] + (BAD_COLOR[1] - GOOD_COLOR[1]) * t)
    b = round(GOOD_COLOR[2] + (BAD_COLOR[2] - GOOD_COLOR[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_hex(hex_color: str, factor: float = 0.72) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, round(r * factor)))
    g = max(0, min(255, round(g * factor)))
    b = max(0, min(255, round(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_seed_name(seed_name: str) -> str:
    raw = str(seed_name).replace("seed_", "", 1)
    return raw.split("_", 1)[0]


def normalize_experiment_tag(raw_tag: str) -> str:
    return str(raw_tag)


def expected_mode_for_modality(modality: str) -> str | None:
    if modality.startswith("rgb_torchvision:"):
        model_name = modality.split(":", 1)[1]
        return f"rgb_{model_name.lower()}_model"
    return MODE_BY_MODALITY.get(modality)


def display_name_for_modality(modality: str) -> str:
    if modality.startswith("rgb_torchvision:"):
        return modality.split(":", 1)[1]
    return DISPLAY_NAME_BY_MODALITY.get(modality, modality)


def color_for_modality(modality: str) -> str:
    if modality.startswith("rgb_torchvision:"):
        model_name = modality.split(":", 1)[1]
        return RGB_MODEL_COLORS.get(model_name, "#555555")
    return COLOR_BY_MODALITY.get(modality, "#555555")


def modality_sort_key(modality: str) -> tuple[int, str]:
    if modality in MODALITY_ORDER:
        return (MODALITY_ORDER.index(modality), modality)
    if modality.startswith("rgb_torchvision:"):
        return (MODALITY_ORDER.index("rgb_r2plus1d"), modality.split(":", 1)[1])
    return (len(MODALITY_ORDER) + 1, modality)


def parse_summary_location(root: Path, summary_path: Path) -> tuple[str, str, str, str | None] | None:
    try:
        rel_parts = summary_path.relative_to(root).parts
    except ValueError:
        return None
    if len(rel_parts) < 4:
        return None

    if rel_parts[0] == "rgb_torchvision":
        if len(rel_parts) < 5:
            return None
        modality = f"rgb_torchvision:{rel_parts[1]}"
        experiment_tag = normalize_experiment_tag(rel_parts[2])
        seed_name = rel_parts[3]
        extra_parts = rel_parts[4:-1]
    else:
        modality = rel_parts[0]
        experiment_tag = normalize_experiment_tag(rel_parts[1])
        seed_name = rel_parts[2]
        extra_parts = rel_parts[3:-1]

    if not seed_name.startswith("seed_"):
        return None
    eval_split = extra_parts[0] if extra_parts and str(extra_parts[0]).startswith("eval_") else None
    return modality, experiment_tag, seed_name, eval_split


def load_rows_with_report(root: Path) -> tuple[List[Dict[str, object]], LoadRowsReport]:
    rows: List[Dict[str, object]] = []
    report = LoadRowsReport(root=str(root))
    if not root.exists():
        return rows, report

    seen_keys = set()
    for summary_path in sorted(root.rglob("summary_*.json")):
        report.scanned_summary_files += 1
        parsed = parse_summary_location(root, summary_path)
        if parsed is None:
            report.skipped_parse_failures.append(str(summary_path))
            continue
        modality, experiment_tag, seed_name, eval_split = parsed
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw_mode = str(summary.get("mode", ""))
        normalized_mode = "rgb_model" if modality == "rgb" and raw_mode == "motion_only" else raw_mode
        expected_mode = expected_mode_for_modality(modality)
        if normalized_mode != expected_mode:
            report.skipped_mode_mismatches.append(
                {
                    "path": str(summary_path),
                    "modality": str(modality),
                    "mode": str(normalized_mode),
                    "expected_mode": str(expected_mode),
                }
            )
            continue
        per_variant_splits = summary.get("per_variant_splits", {})

        if eval_split is None:
            for split_name, metrics in summary.get("splits", {}).items():
                key = (modality, experiment_tag, seed_name, str(split_name), normalized_mode)
                if key in seen_keys:
                    report.skipped_duplicate_keys.append(str(summary_path))
                    continue
                seen_keys.add(key)
                row: Dict[str, object] = {
                    "modality": modality,
                    "pair_tag": experiment_tag,
                    "experiment_tag": experiment_tag,
                    "seed": normalize_seed_name(seed_name),
                    "eval_split": str(split_name),
                    "mode": normalized_mode,
                    "summary_file": str(summary_path),
                }
                for metric_name, value in dict(metrics).items():
                    row[f"{metric_name}_mean"] = float(value)
                    row[f"{metric_name}_std"] = 0.0
                # Per tone-group gap (optional — only present when re-run with updated code)
                tone_group = per_variant_splits.get(str(split_name), {}).get("per_tone_group", {})
                gap = tone_group.get("gap_light_minus_dark")
                if gap is not None:
                    row["group_gap_mean"] = float(gap)
                rows.append(row)
                report.accepted_rows += 1
            continue

        key = (modality, experiment_tag, seed_name, eval_split, normalized_mode)
        if key in seen_keys:
            report.skipped_duplicate_keys.append(str(summary_path))
            continue
        seen_keys.add(key)
        row = {
            "modality": modality,
            "pair_tag": experiment_tag,
            "experiment_tag": experiment_tag,
            "seed": normalize_seed_name(seed_name),
            "eval_split": eval_split,
            "mode": normalized_mode,
            "summary_file": str(summary_path),
        }
        for metric_name, stats in summary.get("aggregate", {}).items():
            row[f"{metric_name}_mean"] = float(stats.get("mean", float("nan")))
            row[f"{metric_name}_std"] = float(stats.get("std", float("nan")))
        # Per tone-group gap for single-split summaries (rgb_torchvision eval subdirs)
        tone_group = per_variant_splits.get(str(eval_split), {}).get("per_tone_group", {})
        gap = tone_group.get("gap_light_minus_dark")
        if gap is not None:
            row["group_gap_mean"] = float(gap)
        rows.append(row)
        report.accepted_rows += 1
    return rows, report


def load_rows(root: Path) -> List[Dict[str, object]]:
    rows, _report = load_rows_with_report(root)
    return rows


def build_compact_rows(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
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
                "mode": row["mode"],
            },
        )
        split = str(row["eval_split"])
        item[f"{split}_{metric_name}_mean"] = row.get(f"{metric_name}_mean")
        item[f"{split}_{metric_name}_std"] = row.get(f"{metric_name}_std")
        # carry per-variant group gap if present
        gap = row.get("group_gap_mean")
        if gap is not None:
            item[f"{split}_group_gap"] = float(gap)

    per_seed_rows: List[Dict[str, object]] = []
    for item in by_seed_key.values():
        matched_unseen = float(item.get(f"eval_matched_unseen_ids_{metric_name}_mean", float("nan")))
        matched_seen = float(item.get(f"eval_matched_seen_ids_{metric_name}_mean", float("nan")))
        shifted_seen = float(item.get(f"eval_shifted_seen_ids_{metric_name}_mean", float("nan")))
        shifted_unseen = float(item.get(f"eval_shifted_unseen_ids_{metric_name}_mean", float("nan")))
        item[f"{metric_name}_matched_unseen_ids"] = matched_unseen
        item[f"{metric_name}_matched_seen_ids"] = matched_seen
        item[f"{metric_name}_shifted_seen_ids"] = shifted_seen
        item[f"{metric_name}_shifted_unseen_ids"] = shifted_unseen
        item[f"{metric_name}_drop_training_videos"] = matched_seen - shifted_seen if matched_seen == matched_seen and shifted_seen == shifted_seen else float("nan")
        item[f"{metric_name}_drop_testing_videos"] = matched_unseen - shifted_unseen if matched_unseen == matched_unseen and shifted_unseen == shifted_unseen else float("nan")
        # group gap on the shifted splits (light accuracy minus dark accuracy)
        item["group_gap_shifted_seen_ids"] = float(item.get("eval_shifted_seen_ids_group_gap", float("nan")))
        item["group_gap_shifted_unseen_ids"] = float(item.get("eval_shifted_unseen_ids_group_gap", float("nan")))
        per_seed_rows.append(item)

    by_pair_key: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in per_seed_rows:
        experiment_tag = str(row.get("experiment_tag", row["pair_tag"]))
        by_pair_key.setdefault((experiment_tag, str(row["modality"])), []).append(row)

    compact_rows: List[Dict[str, object]] = []
    for key, seed_rows in by_pair_key.items():
        item: Dict[str, object] = {
            "pair_tag": key[0],
            "experiment_tag": key[0],
            "modality": key[1],
            "mode": seed_rows[0]["mode"],
            "num_seeds": len(seed_rows),
            "seed_ids": ",".join(sorted(str(seed_row["seed"]) for seed_row in seed_rows)),
        }
        for label in (
            "matched_unseen_ids",
            "matched_seen_ids",
            "shifted_seen_ids",
            "shifted_unseen_ids",
            "drop_training_videos",
            "drop_testing_videos",
        ):
            values = [float(seed_row.get(f"{metric_name}_{label}", float("nan"))) for seed_row in seed_rows]
            mean_value, std_value = mean_std(values)
            item[f"{metric_name}_{label}"] = mean_value
            item[f"{metric_name}_{label}_seed_std"] = std_value
        # Group gap columns (light top1 − dark top1 on shifted evals)
        for gap_label in ("group_gap_shifted_seen_ids", "group_gap_shifted_unseen_ids"):
            gap_values = [float(seed_row.get(gap_label, float("nan"))) for seed_row in seed_rows]
            gap_mean, gap_std = mean_std(gap_values)
            item[gap_label] = gap_mean
            item[f"{gap_label}_seed_std"] = gap_std
        compact_rows.append(item)

    compact_rows.sort(
        key=lambda row: (
            str(row.get("experiment_tag", row["pair_tag"])),
            modality_sort_key(str(row["modality"])),
        )
    )
    return compact_rows


def write_csv(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / "shortcut_probe_summary.csv"
    fieldnames = [
        "experiment_tag",
        "pair_tag",
        "modality",
        "mode",
        "num_seeds",
        "seed_ids",
        f"{metric_name}_matched_unseen_ids",
        f"{metric_name}_matched_unseen_ids_seed_std",
        f"{metric_name}_matched_seen_ids",
        f"{metric_name}_matched_seen_ids_seed_std",
        f"{metric_name}_shifted_seen_ids",
        f"{metric_name}_shifted_seen_ids_seed_std",
        f"{metric_name}_shifted_unseen_ids",
        f"{metric_name}_shifted_unseen_ids_seed_std",
        f"{metric_name}_drop_training_videos",
        f"{metric_name}_drop_training_videos_seed_std",
        f"{metric_name}_drop_testing_videos",
        f"{metric_name}_drop_testing_videos_seed_std",
        "group_gap_shifted_seen_ids",
        "group_gap_shifted_seen_ids_seed_std",
        "group_gap_shifted_unseen_ids",
        "group_gap_shifted_unseen_ids_seed_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_svg(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / f"shortcut_probe_{metric_name}.svg"
    if not rows:
        out_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='120'><text x='20' y='60' font-family='Arial' font-size='24'>No probe results found.</text></svg>",
            encoding="utf-8",
        )
        return out_path

    pair_order = sorted({str(row["pair_tag"]) for row in rows})
    present_modalities = sorted({str(row["modality"]) for row in rows}, key=modality_sort_key)
    modality_order = list(present_modalities)
    row_items: List[Dict[str, object]] = []
    for pair_tag in pair_order:
        for modality in modality_order:
            row = next((r for r in rows if r["pair_tag"] == pair_tag and r["modality"] == modality), None)
            if row is not None:
                row_items.append(row)

    drop_extents: List[float] = []
    abs_drop_values: List[float] = []
    for row in row_items:
        for key, std_key in (
            (f"{metric_name}_drop_training_videos", f"{metric_name}_drop_training_videos_seed_std"),
            (f"{metric_name}_drop_testing_videos", f"{metric_name}_drop_testing_videos_seed_std"),
        ):
            value = float(row.get(key, float("nan")))
            if value == value:
                abs_drop_values.append(abs(value))
                std_value = float(row.get(std_key, float("nan")))
                extent = abs(value) + (std_value if std_value == std_value else 0.0)
                drop_extents.append(extent)

    max_extent = max(drop_extents) if drop_extents else 0.0
    max_abs_drop = max(abs_drop_values) if abs_drop_values else 1.0
    x_radius = max(0.12, max_extent + 0.03)
    x_radius = math.ceil(x_radius * 10.0) / 10.0
    x_min = -x_radius
    x_max = x_radius

    width = 1400
    row_height = 48
    height = 106 + row_height * len(row_items)
    margin_left = 360
    margin_right = 60
    margin_top = 24
    margin_bottom = 82
    plot_width = width - margin_left - margin_right

    def x_to_px(value: float) -> float:
        return margin_left + plot_width * ((value - x_min) / (x_max - x_min))

    def y_to_px(idx: int) -> float:
        return margin_top + row_height * idx + row_height / 2

    tick_values = []
    tick = x_min
    while tick <= x_max + 1e-9:
        tick_values.append(round(tick, 1))
        tick += 0.1

    n_model_cols = 2
    n_model_rows = (len(modality_order) + n_model_cols - 1) // n_model_cols
    legend_box_width = 310
    legend_box_height = 80 + 12 + 14 + 4 + n_model_rows * 18 + 8
    legend_x = width - margin_right - legend_box_width
    legend_y = 18
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    lines.append("<rect width='100%' height='100%' fill='#ffffff'/>")

    for tick in tick_values:
        x = x_to_px(tick)
        stroke = "#666666" if abs(tick) < 1e-9 else "#d9d9d9"
        dash = "none" if abs(tick) < 1e-9 else "3 4"
        lines.append(f"<line x1='{x:.1f}' y1='{margin_top}' x2='{x:.1f}' y2='{height-margin_bottom}' stroke='{stroke}' stroke-dasharray='{dash}'/>")
        lines.append(f"<text x='{x:.1f}' y='{height-margin_bottom+24}' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='12' fill='#555'>{tick:.1f}</text>")

    lines.append(f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height-margin_bottom}' stroke='#333' stroke-width='1.2'/>")
    lines.append(f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{width-margin_right}' y2='{height-margin_bottom}' stroke='#333' stroke-width='1.2'/>")

    lines.append(f"<rect x='{legend_x}' y='{legend_y}' width='{legend_box_width}' height='{legend_box_height}' rx='8' fill='#ffffff' stroke='#cccccc'/>")
    lines.append(f"<circle cx='{legend_x+18}' cy='{legend_y+22}' r='6' fill='#888' stroke='#444'/>")
    lines.append(f"<text x='{legend_x+34}' y='{legend_y+27}' font-family='Arial, Helvetica, sans-serif' font-size='13' fill='#222'>circle = training videos</text>")
    lines.append(f"<rect x='{legend_x+12}' y='{legend_y+40}' width='12' height='12' fill='#888' stroke='#444'/>")
    lines.append(f"<text x='{legend_x+34}' y='{legend_y+51}' font-family='Arial, Helvetica, sans-serif' font-size='13' fill='#222'>square = testing videos</text>")
    lines.append(f"<text x='{legend_x+12}' y='{legend_y+67}' font-family='Arial, Helvetica, sans-serif' font-size='11' fill='#555'>darker = larger |bias|; x&lt;0: reversed bias</text>")
    sep_y = legend_y + 77
    lines.append(f"<line x1='{legend_x+8}' y1='{sep_y}' x2='{legend_x+legend_box_width-8}' y2='{sep_y}' stroke='#e0e0e0'/>")
    lines.append(f"<text x='{legend_x+12}' y='{sep_y+15}' font-family='Arial, Helvetica, sans-serif' font-size='12' font-weight='600' fill='#444'>Models:</text>")
    model_col_width = legend_box_width // n_model_cols
    for m_idx, modality in enumerate(modality_order):
        col = m_idx % n_model_cols
        row_num = m_idx // n_model_cols
        mx = legend_x + 12 + col * model_col_width
        my = sep_y + 22 + row_num * 18
        mcolor = color_for_modality(modality)
        mlabel = display_name_for_modality(modality)
        lines.append(f"<rect x='{mx}' y='{my-9}' width='10' height='10' fill='{mcolor}' stroke='#333' stroke-width='0.5'/>")
        lines.append(f"<text x='{mx+14}' y='{my+1}' font-family='Arial, Helvetica, sans-serif' font-size='11' fill='{mcolor}'>{esc(mlabel)}</text>")

    last_pair = None
    for idx, row in enumerate(row_items):
        pair_tag = str(row["pair_tag"])
        modality = str(row["modality"])
        y = y_to_px(idx)
        training_y = y - 6.0
        testing_y = y + 6.0
        if last_pair is not None and pair_tag != last_pair:
            sep_y = y - row_height / 2
            lines.append(f"<line x1='{margin_left-110}' y1='{sep_y:.1f}' x2='{width-margin_right}' y2='{sep_y:.1f}' stroke='#e8e8e8'/>")
        last_pair = pair_tag

        pretty_pair_label = pair_tag.replace("_vs_", " vs ")
        if idx == 0 or row_items[idx - 1]["pair_tag"] != pair_tag:
            lines.append(f"<text x='{margin_left-120}' y='{y+5:.1f}' text-anchor='end' font-family='Arial, Helvetica, sans-serif' font-size='13' font-weight='600' fill='#222'>{esc(pretty_pair_label)}</text>")
        lines.append(f"<text x='{margin_left-12}' y='{training_y+4:.1f}' text-anchor='end' font-family='Arial, Helvetica, sans-serif' font-size='12' fill='{color_for_modality(modality)}'>{esc(display_name_for_modality(modality))}</text>")

        training_drop = float(row.get(f"{metric_name}_drop_training_videos", float("nan")))
        testing_drop = float(row.get(f"{metric_name}_drop_testing_videos", float("nan")))
        training_std = float(row.get(f"{metric_name}_drop_training_videos_seed_std", float("nan")))
        testing_std = float(row.get(f"{metric_name}_drop_testing_videos_seed_std", float("nan")))

        if training_drop == training_drop:
            x = x_to_px(training_drop)
            training_color = lerp_color(abs(training_drop) / max(max_abs_drop, 1e-9))
            training_stroke = darken_hex(training_color)
            if training_std == training_std and training_std > 0:
                x0 = x_to_px(training_drop - training_std)
                x1 = x_to_px(training_drop + training_std)
                lines.append(f"<line x1='{x0:.1f}' y1='{training_y:.1f}' x2='{x1:.1f}' y2='{training_y:.1f}' stroke='#555' stroke-width='1.4'/>")
                lines.append(f"<line x1='{x0:.1f}' y1='{training_y-5:.1f}' x2='{x0:.1f}' y2='{training_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
                lines.append(f"<line x1='{x1:.1f}' y1='{training_y-5:.1f}' x2='{x1:.1f}' y2='{training_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
            lines.append(f"<circle cx='{x:.1f}' cy='{training_y:.1f}' r='6' fill='{training_color}' stroke='{training_stroke}' stroke-width='1.2'/>")
            lines.append(f"<text x='{x+10:.1f}' y='{training_y-4:.1f}' font-family='Arial, Helvetica, sans-serif' font-size='11' fill='{training_stroke}'>{training_drop:.2f}</text>")

        if testing_drop == testing_drop:
            x = x_to_px(testing_drop)
            testing_color = lerp_color(abs(testing_drop) / max(max_abs_drop, 1e-9))
            testing_stroke = darken_hex(testing_color)
            if testing_std == testing_std and testing_std > 0:
                x0 = x_to_px(testing_drop - testing_std)
                x1 = x_to_px(testing_drop + testing_std)
                lines.append(f"<line x1='{x0:.1f}' y1='{testing_y:.1f}' x2='{x1:.1f}' y2='{testing_y:.1f}' stroke='#555' stroke-width='1.4'/>")
                lines.append(f"<line x1='{x0:.1f}' y1='{testing_y-5:.1f}' x2='{x0:.1f}' y2='{testing_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
                lines.append(f"<line x1='{x1:.1f}' y1='{testing_y-5:.1f}' x2='{x1:.1f}' y2='{testing_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
            lines.append(f"<rect x='{x-6:.1f}' y='{testing_y-6:.1f}' width='12' height='12' fill='{testing_color}' stroke='{testing_stroke}' stroke-width='1.2'/>")
            lines.append(f"<text x='{x+10:.1f}' y='{testing_y+13:.1f}' font-family='Arial, Helvetica, sans-serif' font-size='11' fill='{testing_stroke}'>{testing_drop:.2f}</text>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    raw_rows = load_rows(args.root)
    compact_rows = build_compact_rows(raw_rows, args.metric)
    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(args.root, compact_rows, args.metric)
    svg_path = write_svg(args.root, compact_rows, args.metric)
    print(csv_path)
    print(svg_path)


if __name__ == "__main__":
    main()
