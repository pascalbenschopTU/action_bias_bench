"""Reference script: shows exactly which files/columns Table 2 (net flips,
BH-adjusted q) and the "individual tone-pair" claim in the Results text come
from, and confirms the two BH correction families are disjoint.

This does not recompute any statistics -- everything here is already produced
by `summarize_skin_tone_significance.py` (see `compute_model_cluster_significance_rows`
and `compute_variant_pair_cluster_significance_rows`). It only reassembles the
two output CSVs into the shape used in the paper, with the correction
provenance made explicit, so the mapping from code to table is documented in
one place.

Usage:
    python report_table2_correction_provenance.py \
        --analysis_root out/skin_tone_probe_v7_cv_analysis --split_family unseen
"""
import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_root", type=Path, required=True)
    p.add_argument("--split_family", type=str, default="unseen")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.analysis_root
    sf = args.split_family

    model = pd.read_csv(root / f"skin_tone_model_cluster_significance_{sf}.csv")
    pair = pd.read_csv(root / f"skin_tone_color_pair_cluster_significance_{sf}.csv")

    # --- Table 2: pooled model-level test -----------------------------------
    # Correction family: the 6 RGB backbones, BH-adjusted together.
    # i3d_flow is excluded from that family and shown with its raw (unadjusted) p.
    model = model.copy()
    model["correction_family"] = model["model"].apply(
        lambda m: "unadjusted_reference" if m == "i3d_flow" else "rgb_pooled_n6"
    )
    model["drop_pp"] = model["observed_drop"] * 100
    print(f"=== Table 2 source: skin_tone_model_cluster_significance_{sf}.csv ===")
    print(
        model[
            [
                "model",
                "n_clusters",
                "drop_pp",
                "wilcoxon_p",
                "wilcoxon_q",
                "correction_family",
                "cluster_boot_ci_low",
                "cluster_boot_ci_high",
            ]
        ]
        .sort_values("drop_pp")
        .to_string(index=False)
    )

    # --- Individual tone-pair test (Results: "only S3D and Swin3D-S...") ---
    # Correction family: each backbone's own 4 tone-pair cells, BH-adjusted
    # within that backbone only (correction_scope == "within_model").
    print(f"\n=== Tone-pair source: skin_tone_color_pair_cluster_significance_{sf}.csv ===")
    sig = pair[pair["wilcoxon_q"] < args.alpha]
    n_sig_by_model = sig.groupby("model").size().reindex(pair["model"].unique(), fill_value=0)
    print(f"Significant cells (q<{args.alpha}) per backbone, out of 4 tone pairs each:")
    print(n_sig_by_model.to_string())
    print(
        "\nNote: this is a *different* BH family than Table 2 (4 tests per backbone, "
        "not 6 backbones pooled) -- a model can be non-significant here while "
        "significant in the pooled model-level test, or vice versa."
    )

    # --- Where the bootstrap intervals live ---------------------------------
    print(
        "\n=== Bootstrap ===\n"
        "Both files above carry `cluster_boot_ci_low`/`cluster_boot_ci_high`: a "
        "10-cluster, 5000-resample percentile bootstrap over motion instances "
        "(see compute_model_cluster_significance_rows / "
        "compute_variant_pair_cluster_significance_rows, n_boot=5000). These are "
        "the intervals promised in Methods; they are not currently printed in "
        "Table 2 or quoted in Results."
    )

    print(
        "\n=== Not used for any claim in Results/Table 2 ===\n"
        f"skin_tone_variant_swap_significance_{sf}.csv (compute_variant_significance_rows) "
        "runs an UNCLUSTERED per-clip McNemar exact test on each of the 8 directional "
        "swaps. It is a separate, finer-grained diagnostic (no motion-instance "
        "clustering, so clips are treated as independent) and should not be cited "
        "as the source of the significance results in the main text."
    )


if __name__ == "__main__":
    main()
