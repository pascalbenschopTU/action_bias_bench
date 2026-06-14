#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_action_bias_bench.sh"

if [[ ! -f "$RUNNER" ]]; then
  echo "Missing runner script: $RUNNER" >&2
  exit 1
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_skin_tone_color_jitter_sweep.sh [runner flags]

Environment:
  SKIN_TONE_COLOR_JITTER_VALUES=0.0,0.4,0.8
  SKIN_TONE_SWEEP_BASE_OUT_ROOT=/path/to/out_prefix
  SKIN_TONE_DATASET_ROOT=...
  SKIN_TONE_RGB_TORCHVISION_ROOT_DIR=...

This launcher forces:
  SKIN_TONE_MODALITIES=rgb_torchvision
  MODALITIES=rgb_torchvision
EOF
  exit 0
fi

COLOR_JITTER_VALUES_RAW="${SKIN_TONE_COLOR_JITTER_VALUES:-0.0,0.4,0.8}"
BASE_OUT_ROOT="${SKIN_TONE_SWEEP_BASE_OUT_ROOT:-${SKIN_TONE_OUT_ROOT:-$ROOT_DIR/out/skin_tone_probe_rgb_torchvision}}"

IFS=',' read -r -a COLOR_JITTER_VALUES <<< "$COLOR_JITTER_VALUES_RAW"
if [[ "${#COLOR_JITTER_VALUES[@]}" -eq 0 ]]; then
  echo "No color-jitter values resolved. Check SKIN_TONE_COLOR_JITTER_VALUES." >&2
  exit 1
fi

for COLOR_JITTER_VALUE in "${COLOR_JITTER_VALUES[@]}"; do
  value="$(echo "$COLOR_JITTER_VALUE" | xargs)"
  [[ -z "$value" ]] && continue
  suffix="${value//./p}"
  suffix="${suffix//-/_m_}"
  OUT_ROOT="${BASE_OUT_ROOT}_cj${suffix}"

  echo "[SWEEP] jitter=${value} out=${OUT_ROOT}"
  SKIN_TONE_MODALITIES="rgb_torchvision" \
  MODALITIES="rgb_torchvision" \
  SKIN_TONE_COLOR_JITTER="$value" \
  SKIN_TONE_OUT_ROOT="$OUT_ROOT" \
  bash "$RUNNER" "$@"
done

echo "Color-jitter sweep complete."
