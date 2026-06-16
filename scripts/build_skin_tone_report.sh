#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$ROOT_DIR/benchmarks/skin_tone"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASELINE_ROOT="${SKIN_TONE_REPORT_BASELINE_ROOT:-$ROOT_DIR/out/skin_tone_probe_rgb_torchvision_v6_cj0p0}"
REPORT_ROOT="${SKIN_TONE_REPORT_ROOT:-$ROOT_DIR/out/skin_tone_report}"
JITTER_ROOTS_RAW="${SKIN_TONE_REPORT_JITTER_ROOTS:-cj0p0=$ROOT_DIR/out/skin_tone_probe_rgb_torchvision_v6_cj0p0,cj0p4=$ROOT_DIR/out/skin_tone_probe_rgb_torchvision_v6_cj0p4,cj0p8=$ROOT_DIR/out/skin_tone_probe_rgb_torchvision_v6_cj0p8}"
METRIC="${SKIN_TONE_REPORT_METRIC:-f1_macro}"
SPLIT_FAMILY="${SKIN_TONE_REPORT_SPLIT_FAMILY:-unseen}"

print_usage() {
  cat <<'EOF'
Usage: bash scripts/build_skin_tone_report.sh

Environment:
  SKIN_TONE_REPORT_BASELINE_ROOT=/path/to/baseline/run/root
  SKIN_TONE_REPORT_JITTER_ROOTS=cj0p0=/path,cj0p4=/path,cj0p8=/path
  SKIN_TONE_REPORT_ROOT=/path/to/report/root
  SKIN_TONE_REPORT_METRIC=f1_macro
  SKIN_TONE_REPORT_SPLIT_FAMILY=unseen
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || {
    echo "Missing ${label}: $path" >&2
    exit 1
  }
}

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
  else
    echo "[WARN] Missing expected artifact: $src" >&2
  fi
}

IFS=',' read -r -a JITTER_ROOT_SPECS <<< "$JITTER_ROOTS_RAW"

require_dir "$BASELINE_ROOT" "baseline root"
mkdir -p "$REPORT_ROOT/figures" "$REPORT_ROOT/tables" "$REPORT_ROOT/analysis/swap" "$REPORT_ROOT/analysis/color_jitter"

"$PYTHON_BIN" "$BENCHMARK_DIR/summarize_skin_tone_robustness.py" \
  --root "$BASELINE_ROOT" \
  --metric "$METRIC"

"$PYTHON_BIN" "$BENCHMARK_DIR/analyze_skin_tone_swap_influence.py" \
  --root "$BASELINE_ROOT" \
  --metric "$METRIC" \
  --out_dir "$REPORT_ROOT/analysis/swap"

"$PYTHON_BIN" "$BENCHMARK_DIR/summarize_skin_tone_significance.py" \
  --root "$REPORT_ROOT/analysis/swap" \
  --metric_roots "${JITTER_ROOT_SPECS[@]}" \
  --metric "$METRIC" \
  --split_family "$SPLIT_FAMILY"

"$PYTHON_BIN" "$BENCHMARK_DIR/compare_color_jitter_conditions.py" \
  --roots "${JITTER_ROOT_SPECS[@]}" \
  --metric "$METRIC" \
  --out_dir "$REPORT_ROOT/analysis/color_jitter"

copy_if_exists "$BASELINE_ROOT/skin_tone_pair_heatmap_${METRIC}.png" "$REPORT_ROOT/figures/skin_tone_pair_heatmap_${METRIC}.png"
copy_if_exists "$BASELINE_ROOT/skin_tone_pair_heatmap_${METRIC}.pdf" "$REPORT_ROOT/figures/skin_tone_pair_heatmap_${METRIC}.pdf"
copy_if_exists "$BASELINE_ROOT/skin_tone_robustness_summary_${METRIC}.png" "$REPORT_ROOT/figures/skin_tone_robustness_summary_${METRIC}.png"
copy_if_exists "$BASELINE_ROOT/skin_tone_robustness_summary_${METRIC}.pdf" "$REPORT_ROOT/figures/skin_tone_robustness_summary_${METRIC}.pdf"
copy_if_exists "$REPORT_ROOT/analysis/color_jitter/color_jitter_comparison.pdf" "$REPORT_ROOT/figures/color_jitter_comparison.pdf"
copy_if_exists "$REPORT_ROOT/analysis/swap/skin_tone_variant_swap_significance_${SPLIT_FAMILY}.pdf" "$REPORT_ROOT/figures/skin_tone_variant_swap_significance_${SPLIT_FAMILY}.pdf"

copy_if_exists "$BASELINE_ROOT/skin_tone_pair_robustness_summary_${METRIC}.csv" "$REPORT_ROOT/tables/skin_tone_pair_robustness_summary_${METRIC}.csv"
copy_if_exists "$BASELINE_ROOT/skin_tone_robustness_summary_${METRIC}.csv" "$REPORT_ROOT/tables/skin_tone_robustness_summary_${METRIC}.csv"
copy_if_exists "$REPORT_ROOT/analysis/color_jitter/color_jitter_comparison.csv" "$REPORT_ROOT/tables/color_jitter_comparison.csv"
copy_if_exists "$REPORT_ROOT/analysis/color_jitter/color_jitter_robustness_checks.csv" "$REPORT_ROOT/tables/color_jitter_robustness_checks.csv"
copy_if_exists "$REPORT_ROOT/analysis/swap/skin_tone_variant_swap_significance_${SPLIT_FAMILY}.csv" "$REPORT_ROOT/tables/skin_tone_variant_swap_significance_${SPLIT_FAMILY}.csv"
copy_if_exists "$REPORT_ROOT/analysis/swap/skin_tone_significance_summary.csv" "$REPORT_ROOT/tables/skin_tone_significance_summary.csv"
copy_if_exists "$REPORT_ROOT/analysis/swap/swap_pair_join_report.json" "$REPORT_ROOT/tables/swap_pair_join_report.json"

cat > "$REPORT_ROOT/README.txt" <<EOF
Skin-tone report bundle

baseline_root: $BASELINE_ROOT
jitter_roots: $JITTER_ROOTS_RAW
metric: $METRIC
split_family: $SPLIT_FAMILY

Canonical figures are in figures/.
Canonical source tables are in tables/.
Full intermediate analysis outputs are in analysis/.
EOF

echo "Done. Report bundle: $REPORT_ROOT"
