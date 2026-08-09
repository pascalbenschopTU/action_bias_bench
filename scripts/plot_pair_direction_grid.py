"""
Supplementary figure: full (action assignment x directional tone swap) grid of
skin-tone-swap prediction instability, one panel per augmentation condition.

For each (action assignment, direction) cell and each seed, we compute the net
flip count b - c pooled over the six RGB backbones, where b counts clips
correct under the matched tone assignment and incorrect under the shifted one
and c counts the reverse. The cell plotted is the mean over seeds of
|b - c| (per seed), not the signed mean. Averaging the signed value directly
would be misleading here: two seeds that disagree in sign for the same cell
(seed 0 net +3, seed 1 net -3) would average to ~0 and read as "no effect",
when the model was in fact unstable in both seeds, just in opposite
directions. Taking the absolute value first, then averaging, reports the
magnitude of that instability instead of letting it cancel out. One
consequence: with direction no longer signed, the colorbar is sequential
(magnitude only) rather than diverging red/blue.

The baseline condition has 3 seeds and the four augmentation conditions have
2, so the mean (not the sum) is what makes panels comparable across
conditions with different seed counts; each panel's subtitle reports how many
seeds it was averaged over.

Two conventions are inherited from the main analysis so this figure and Table 2
count the same events:

  * rows are restricted to one split family (default `unseen`), and
  * `dedupe_multi_fold_clusters` drops base_ids 0 and 1 from their second CV
    fold, since those are the only motion instances held out twice (applied
    per seed -- the fold/base_id structure it corrects for is identical
    across seeds, so deduping each seed's rows independently is equivalent to
    deduping the pooled multi-seed set).

This replaces an earlier ad-hoc version of the figure that was (a) generated
from the v6 benchmark run, which covered only 4 of the 8 directional swaps,
and (b) single-seed, signed, unaveraged.

Usage (run from the ActionBiasBench directory):
    python scripts/plot_pair_direction_grid.py \
        --roots "no augmentation=out/skin_tone_probe_v7_cv_analysis" \
                "weak jitter=out/skin_tone_probe_v7_cjweak_cv_analysis" \
                "strong jitter=out/skin_tone_probe_v7_cjstrong_cv_analysis" \
                "strong + grayscale=out/skin_tone_probe_v7_cjstronggray_cv_analysis" \
                "planckian jitter=out/skin_tone_probe_v7_planckian_cv_analysis" \
        --out_dir out/skin_tone_probe_v7_augmentation_conditions
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks" / "skin_tone"))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_actionbiasbench"))

from summarize_skin_tone_significance import dedupe_multi_fold_clusters  # noqa: E402

PAIR_ORDER = [
    "squat_vs_tie", "tie_vs_squat",
    "clap_vs_celebrate", "celebrate_vs_clap",
    "dribble_vs_golf", "golf_vs_dribble",
    "lunge_vs_cartwheel", "cartwheel_vs_lunge",
    "yawn_vs_fish", "fish_vs_yawn",
]
DIRECTION_ORDER = [
    "african->white", "african->asian", "indian->white", "indian->asian",
    "white->african", "white->indian", "asian->african", "asian->indian",
]
FLOW_MODELS = {"i3d_flow", "flow_i3d_external"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="label=analysis_dir pairs, in panel order.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_name", default="full_grid_pair_x_direction")
    ap.add_argument("--split_family", default="unseen")
    return ap.parse_args()


def parse_roots(specs):
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--roots entry must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def net_flip_grid(rows) -> np.ndarray:
    grid = np.zeros((len(PAIR_ORDER), len(DIRECTION_ORDER)), dtype=float)
    pair_idx = {p: i for i, p in enumerate(PAIR_ORDER)}
    dir_idx = {d: i for i, d in enumerate(DIRECTION_ORDER)}
    skipped = set()
    for row in rows:
        direction = f"{row['variant_matched']}->{row['variant_shifted']}"
        if row["pair_tag"] not in pair_idx or direction not in dir_idx:
            skipped.add((row["pair_tag"], direction))
            continue
        matched = int(float(row["correct_matched"]))
        shifted = int(float(row["correct_shifted"]))
        if matched == 1 and shifted == 0:
            grid[pair_idx[row["pair_tag"]], dir_idx[direction]] += 1
        elif matched == 0 and shifted == 1:
            grid[pair_idx[row["pair_tag"]], dir_idx[direction]] -= 1
    if skipped:
        print(f"[WARN] skipped unexpected cells {sorted(skipped)}", flush=True)
    return grid


def load_mean_abs_grid(analysis_dir: Path, split_family: str):
    """Mean over seeds of |net flips| per cell. See module docstring for why
    the absolute value is taken before averaging rather than after."""
    csv_path = analysis_dir / "swap_pair_level_analysis.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"missing {csv_path}\nRun analyze_skin_tone_swap_influence.py on the matching "
            f"benchmark root first (see README Experiment 3)."
        )

    rows_by_seed = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split_family"] != split_family:
                continue
            if row["model"] in FLOW_MODELS:
                continue
            rows_by_seed[int(row["seed"])].append(row)

    if not rows_by_seed:
        raise SystemExit(f"no rows for split_family={split_family!r} in {csv_path}")

    per_seed_grids = []
    for seed in sorted(rows_by_seed):
        deduped = dedupe_multi_fold_clusters(rows_by_seed[seed])
        per_seed_grids.append(net_flip_grid(deduped))

    stacked = np.abs(np.stack(per_seed_grids, axis=0))
    return stacked.mean(axis=0), sorted(rows_by_seed), sum(len(v) for v in rows_by_seed.values())


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = parse_roots(args.roots)
    grids = []
    for label, path in specs:
        grid, seeds, n = load_mean_abs_grid(path, args.split_family)
        print(f"{label:22s} mean |net flips| total = {grid.sum():6.1f}   "
              f"(seeds={seeds}, {n} paired clips)", flush=True)
        grids.append((label, grid, seeds))

    max_val = max(float(g.max()) for _, g, _ in grids)
    max_val = max(max_val, 1.0)

    n_panels = len(grids)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.1 * n_panels + 1.6, 5.4), dpi=200, sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (label, grid, seeds) in zip(axes, grids):
        im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0.0, vmax=max_val)
        ax.set_xticks(range(len(DIRECTION_ORDER)))
        ax.set_xticklabels([d.replace("->", "\n→") for d in DIRECTION_ORDER], fontsize=6)
        ax.set_xlabel("skin-tone swap direction", fontsize=8)
        ax.set_title(
            f"{label}\nmean total = {grid.sum():.1f}  ({len(seeds)} seeds)",
            fontsize=9, weight="bold",
        )
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                ax.text(j, i, "0" if value == 0 else f"{value:.1f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if value > 0.6 * max_val else "black")

    axes[0].set_yticks(range(len(PAIR_ORDER)))
    axes[0].set_yticklabels(PAIR_ORDER, fontsize=7)
    axes[0].set_ylabel("action pair", fontsize=9)

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.015)
    cbar.set_label("mean |net flips| across seeds (b - c, absolute), pooled over models", fontsize=8)
    fig.suptitle(
        "Full (action pair × skin-tone direction) grid — no marginalization\n"
        "(mean of |b - c| over each condition's seeds; 6 RGB backbones pooled, flow control excluded)",
        fontsize=10, weight="bold", y=1.04,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.out_dir / f"{args.out_name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"[OK] wrote {path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
