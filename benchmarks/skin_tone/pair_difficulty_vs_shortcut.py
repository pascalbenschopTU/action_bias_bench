"""
Tests whether the concentration of the skin-tone shortcut in a few action
pairs (Figure~\\ref{fig:shortcut_heatmap}: celebrate/clap and lunge/cartwheel
elevated, yawn/fish flat) is explained by how hard the underlying binary task
already is without any tone intervention.

Hypothesis: pairs that are already harder to separate by motion alone (lower
matched, pre-swap accuracy) leave more room for an appearance-based shortcut
to matter, so their swap effect should be larger.

Uses the existing per-clip shortcut-probe predictions
(swap_pair_level_analysis.csv, unseen split, 6 RGB backbones) -- no new
training runs. For each (model, undirected action pair) cell we already have
the matched/shifted correctness columns; we just aggregate and correlate.

Usage (run from the ActionBiasBench directory):
    python benchmarks/skin_tone/pair_difficulty_vs_shortcut.py \
        --root out/skin_tone_probe_v7_cv_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

UNDIRECTED_PAIR = {
    "squat_vs_tie": "squat/tie", "tie_vs_squat": "squat/tie",
    "clap_vs_celebrate": "clap/celebrate", "celebrate_vs_clap": "clap/celebrate",
    "dribble_vs_golf": "dribble/golf", "golf_vs_dribble": "dribble/golf",
    "lunge_vs_cartwheel": "lunge/cartwheel", "cartwheel_vs_lunge": "lunge/cartwheel",
    "yawn_vs_fish": "yawn/fish", "fish_vs_yawn": "yawn/fish",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("out/skin_tone_probe_v7_cv_analysis"))
    ap.add_argument("--split_family", default="unseen")
    ap.add_argument("--out_csv", type=Path, default=None,
                     help="Defaults to <root>/pair_difficulty_vs_shortcut.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_csv = args.out_csv or (args.root / "pair_difficulty_vs_shortcut.csv")

    df = pd.read_csv(args.root / "swap_pair_level_analysis.csv")
    df = df[(df["split_family"] == args.split_family) & (df["model"] != "i3d_flow")].copy()
    df["pair"] = df["pair_tag"].map(UNDIRECTED_PAIR)

    cell = df.groupby(["model", "pair"]).agg(
        matched_acc=("correct_matched", "mean"),
        shifted_acc=("correct_shifted", "mean"),
        n=("correct_matched", "size"),
    ).reset_index()
    cell["swap_effect"] = cell["matched_acc"] - cell["shifted_acc"]
    cell["baseline_error_rate"] = 1.0 - cell["matched_acc"]
    cell.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    r_cell, p_cell = stats.pearsonr(cell["baseline_error_rate"], cell["swap_effect"])
    print(f"\ncell-level (model x pair, n={len(cell)}): "
          f"baseline_error_rate vs swap_effect  r={r_cell:.3f}  p={p_cell:.4f}")

    pair_level = cell.groupby("pair").agg(
        matched_acc=("matched_acc", "mean"), swap_effect=("swap_effect", "mean")
    ).reset_index()
    pair_level["baseline_error_rate"] = 1.0 - pair_level["matched_acc"]
    pair_level = pair_level.sort_values("swap_effect", ascending=False)
    print(f"\npair-level (n={len(pair_level)}):")
    print(pair_level.to_string(index=False))

    r_pair, p_pair = stats.pearsonr(pair_level["baseline_error_rate"], pair_level["swap_effect"])
    rho_pair, p_rho = stats.spearmanr(pair_level["matched_acc"], pair_level["swap_effect"])
    print(f"\npair-level Pearson r={r_pair:.3f}  p={p_pair:.4f}")
    print(f"pair-level Spearman rho(matched_acc, swap_effect)={rho_pair:.3f}  p={p_rho:.4f}")

    pair_csv = out_csv.with_name(out_csv.stem + "_by_pair.csv")
    pair_level.to_csv(pair_csv, index=False)
    print(f"wrote {pair_csv}")


if __name__ == "__main__":
    main()
