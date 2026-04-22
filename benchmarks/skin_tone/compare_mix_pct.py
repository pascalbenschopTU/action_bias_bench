"""Compare shortcut-probe results across multiple mix_pct conditions.

Usage
-----
python compare_mix_pct.py \\
    --roots mix0=/path/to/out_mix0 mix50=/path/to/out_mix50 \\
    --metric f1_macro \\
    --out_dir /path/to/comparison_output

Each ``--roots`` entry is ``label=path``.  The script reads the aggregate rows
from each root via ``aggregate_skin_tone_probe.load_rows``, computes the mean
shortcut drop per (modality, pair_tag) for every condition, and writes:

* ``mix_pct_comparison.csv``   — machine-readable table
* ``mix_pct_comparison.svg``   — bar/dot chart: drop vs condition for each modality
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from aggregate_skin_tone_probe import load_rows, build_compact_rows, modality_sort_key, display_name_for_modality, color_for_modality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare shortcut probe drops across mix_pct conditions.")
    parser.add_argument(
        "--roots",
        nargs="+",
        metavar="LABEL=PATH",
        required=True,
        help="One or more label=path pairs, e.g. mix0=/path/to/out mix50=/path/to/out_mix50",
    )
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--out_dir", type=Path, default=Path("."))
    return parser.parse_args()


def parse_roots(roots: List[str]) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    for entry in roots:
        if "=" not in entry:
            raise ValueError(f"--roots entries must be in label=path format, got: {entry!r}")
        label, path_str = entry.split("=", 1)
        result.append((label.strip(), Path(path_str.strip())))
    return result


def mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [v for v in values if v == v]  # drop NaN
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_comparison(
    conditions: List[Tuple[str, Path]],
    metric_name: str,
) -> List[Dict[str, object]]:
    """Return rows keyed by (modality, pair_tag) with drop values per condition."""
    # {(modality, pair_tag): {condition_label: [drop_testing_videos per seed]}}
    by_key: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for label, root in conditions:
        raw_rows = load_rows(root)
        compact = build_compact_rows(raw_rows, metric_name)
        for row in compact:
            modality = str(row["modality"])
            pair_tag = str(row["pair_tag"])
            key = (modality, pair_tag)
            by_key.setdefault(key, {})
            drop = float(row.get(f"{metric_name}_drop_testing_videos", float("nan")))
            by_key[key].setdefault(label, []).append(drop)

    condition_labels = [label for label, _ in conditions]
    rows: List[Dict[str, object]] = []
    for (modality, pair_tag), cond_drops in sorted(by_key.items(), key=lambda item: (item[0][1], modality_sort_key(item[0][0]))):
        row: Dict[str, object] = {"modality": modality, "pair_tag": pair_tag}
        for label in condition_labels:
            values = cond_drops.get(label, [])
            m, s = mean_std(values)
            row[f"drop_testing_{label}"] = m
            row[f"drop_testing_{label}_std"] = s
        rows.append(row)
    return rows, condition_labels


def write_comparison_csv(out_path: Path, rows: List[Dict[str, object]], condition_labels: List[str]) -> None:
    fieldnames = ["modality", "pair_tag"]
    for label in condition_labels:
        fieldnames += [f"drop_testing_{label}", f"drop_testing_{label}_std"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_svg(out_path: Path, rows: List[Dict[str, object]], condition_labels: List[str], metric_name: str) -> None:
    if not rows:
        out_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='80'>"
            "<text x='20' y='50' font-family='Arial' font-size='20'>No comparison data found.</text></svg>",
            encoding="utf-8",
        )
        return

    # Collect all drop values for axis range
    all_drops: List[float] = []
    for row in rows:
        for label in condition_labels:
            v = float(row.get(f"drop_testing_{label}", float("nan")))
            if v == v:
                all_drops.append(abs(v))

    max_drop = max(all_drops) if all_drops else 0.5
    x_radius = max(0.15, math.ceil((max_drop + 0.05) * 10) / 10)
    x_min, x_max = -x_radius, x_radius

    n_conditions = len(condition_labels)
    row_height = max(28, 24 * n_conditions + 8)
    width = 1200
    margin_left = 320
    margin_right = 60
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    height = margin_top + margin_bottom + row_height * len(rows)

    condition_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f",
    ]

    def x_to_px(v: float) -> float:
        return margin_left + plot_width * ((v - x_min) / (x_max - x_min))

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect width='100%' height='100%' fill='#ffffff'/>")

    # Grid lines
    tick = x_min
    while tick <= x_max + 1e-9:
        tick = round(tick, 2)
        x = x_to_px(tick)
        stroke = "#666" if abs(tick) < 1e-9 else "#ddd"
        dash = "none" if abs(tick) < 1e-9 else "3 4"
        lines.append(f"<line x1='{x:.1f}' y1='{margin_top}' x2='{x:.1f}' y2='{height - margin_bottom}' stroke='{stroke}' stroke-dasharray='{dash}'/>")
        lines.append(f"<text x='{x:.1f}' y='{height - margin_bottom + 18}' text-anchor='middle' font-family='Arial' font-size='11' fill='#555'>{tick:.2f}</text>")
        tick += 0.1

    # Axes
    lines.append(f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#333' stroke-width='1.2'/>")
    lines.append(f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#333' stroke-width='1.2'/>")

    # Legend
    lx, ly = width - margin_right - 200, margin_top
    lines.append(f"<rect x='{lx - 8}' y='{ly - 4}' width='200' height='{20 * n_conditions + 12}' rx='6' fill='#fff' stroke='#ccc'/>")
    for ci, label in enumerate(condition_labels):
        color = condition_colors[ci % len(condition_colors)]
        cy = ly + 10 + 20 * ci
        lines.append(f"<circle cx='{lx + 8}' cy='{cy}' r='6' fill='{color}'/>")
        lines.append(f"<text x='{lx + 20}' y='{cy + 4}' font-family='Arial' font-size='13' fill='#222'>{esc(label)}</text>")

    # Rows
    last_modality = None
    for ri, row in enumerate(rows):
        modality = str(row["modality"])
        pair_tag = str(row["pair_tag"])
        y_center = margin_top + ri * row_height + row_height / 2

        if modality != last_modality:
            lines.append(f"<text x='{margin_left - 12}' y='{y_center - 6:.1f}' text-anchor='end' font-family='Arial' font-size='12' font-weight='600' fill='{color_for_modality(modality)}'>{esc(display_name_for_modality(modality))}</text>")
            last_modality = modality

        lines.append(f"<text x='{margin_left - 140}' y='{y_center + 4:.1f}' text-anchor='end' font-family='Arial' font-size='11' fill='#444'>{esc(pair_tag.replace('_vs_', ' vs '))}</text>")

        for ci, label in enumerate(condition_labels):
            color = condition_colors[ci % len(condition_colors)]
            drop = float(row.get(f"drop_testing_{label}", float("nan")))
            std = float(row.get(f"drop_testing_{label}_std", float("nan")))
            offset = (ci - (n_conditions - 1) / 2) * 8
            cy = y_center + offset
            if drop != drop:
                continue
            x = x_to_px(drop)
            if std == std and std > 0:
                x0, x1 = x_to_px(drop - std), x_to_px(drop + std)
                lines.append(f"<line x1='{x0:.1f}' y1='{cy:.1f}' x2='{x1:.1f}' y2='{cy:.1f}' stroke='#555' stroke-width='1.2'/>")
            lines.append(f"<circle cx='{x:.1f}' cy='{cy:.1f}' r='5' fill='{color}' stroke='#333' stroke-width='0.8'/>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    conditions = parse_roots(args.roots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, condition_labels = build_comparison(conditions, args.metric)

    csv_path = args.out_dir / "mix_pct_comparison.csv"
    svg_path = args.out_dir / "mix_pct_comparison.svg"

    write_comparison_csv(csv_path, rows, condition_labels)
    write_comparison_svg(svg_path, rows, condition_labels, args.metric)

    print(f"[wrote] {csv_path}")
    print(f"[wrote] {svg_path}")
    print(f"\nConditions compared: {', '.join(condition_labels)}")
    print(f"Pairs found: {len(rows)}")


if __name__ == "__main__":
    main()
