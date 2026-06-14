#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$ROOT_DIR/benchmarks/skin_tone"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK="${BENCHMARK:-skin_tone}"
RUN_PREFLIGHT=0
DRY_RUN=0

print_usage() {
  cat <<'EOF'
Usage: bash scripts/run_action_bias_bench.sh [--preflight] [--dry-run]

Environment:
  BENCHMARK=skin_tone
  MODALITIES=motion,rgb,rgb_torchvision,flow_i3d_external
  SKIN_TONE_DATASET_ROOT=/path/to/video/root
  SKIN_TONE_MOTION_ROOT_DIR=/path/to/zstd/root
  SKIN_TONE_FLOW_TVL1_ROOT_DIR=/path/to/tvl1/root
  SKIN_TONE_RGB_TORCHVISION_MODELS=r3d_18,mc3_18 or all
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --preflight)
      RUN_PREFLIGHT=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
  shift
done

[[ "$BENCHMARK" == "skin_tone" ]] || {
  echo "Unsupported BENCHMARK: $BENCHMARK" >&2
  exit 1
}

print_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || {
    echo "Missing ${label}: $path" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || {
    echo "Missing ${label}: $path" >&2
    exit 1
  }
}

python_check_modules() {
  local label="$1"
  shift
  if ! "$PYTHON_BIN" - "$@" <<'PY'
import importlib
import sys

missing = []
for module_name in sys.argv[1:]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name}: {exc}")

if missing:
    raise SystemExit("\n".join(missing))
PY
  then
    echo "Python dependency check failed for ${label}." >&2
    exit 1
  fi
}

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  local latest=""
  local old_nullglob=""
  old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  local ckpts=()
  if [[ -d "$ckpt_dir" ]]; then
    ckpts=("$ckpt_dir"/checkpoint*.pt)
  fi
  eval "$old_nullglob"
  if [[ "${#ckpts[@]}" -gt 0 ]]; then
    latest="$(ls -t "${ckpts[@]}" 2>/dev/null | head -n 1 || true)"
  fi
  printf '%s' "$latest"
}

run_already_done() {
  local summary_path="$1"
  local expected_splits="${2:-0}"
  [[ -f "$summary_path" ]] || return 1
  if [[ "$expected_splits" -le 0 ]]; then
    return 0
  fi
  "$PYTHON_BIN" - "$summary_path" "$expected_splits" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_splits = int(sys.argv[2])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
num_splits = int(payload.get("num_splits", 0))
raise SystemExit(0 if num_splits >= expected_splits else 1)
PY
}

manifest_root_has_entries() {
  local root_dir="$1"
  local manifest_path="$2"
  local mode="$3"
  "$PYTHON_BIN" - "$root_dir" "$manifest_path" "$mode" <<'PY'
from pathlib import Path
import sys

root_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
mode = sys.argv[3]

def candidates(rel_path: str):
    raw_path = root_dir / rel_path
    options = [raw_path]
    if mode == "flow":
        options.extend([
            raw_path.with_suffix(".npz"),
            raw_path.with_suffix(".npy"),
            Path(str(raw_path) + ".npz"),
            Path(str(raw_path) + ".npy"),
        ])
    return options

missing = []
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        rel_path, _label = line.rsplit(" ", 1)
        if any(path.exists() for path in candidates(rel_path)):
            continue
        missing.append(rel_path)
        if len(missing) >= 5:
            break

if missing:
    print(f"{mode} root is missing files referenced by {manifest_path}:")
    for rel_path in missing:
        print(f"  - {rel_path}")
    raise SystemExit(1)
PY
}

validate_manifest_coverage() {
  local root_dir="$1"
  local mode="$2"
  local manifest_root="$3"
  local manifests=("$manifest_root/train_in_domain.txt")
  local eval_name
  for eval_name in "${EVAL_SPLITS[@]}"; do
    manifests+=("$manifest_root/${eval_name}.txt")
  done
  local manifest_path
  for manifest_path in "${manifests[@]}"; do
    manifest_root_has_entries "$root_dir" "$manifest_path" "$mode" || {
      echo "Manifest coverage check failed for mode=${mode} root=${root_dir}" >&2
      exit 1
    }
  done
}

modality_selected() {
  local needle="$1"
  local modality
  for modality in "${MODALITIES[@]}"; do
    if [[ "$modality" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

build_action_pairs() {
  local raw_pairs="$1"
  local raw_actions="$2"
  local include_reversed="$3"
  local -a pairs=()
  if [[ -n "$raw_actions" ]]; then
    IFS=',' read -r -a actions <<< "$raw_actions"
    local n="${#actions[@]}"
    for ((i=0; i<n; i++)); do
      local ai="${actions[$i]}"
      [[ -z "$ai" ]] && continue
      for ((j=i+1; j<n; j++)); do
        local aj="${actions[$j]}"
        [[ -z "$aj" ]] && continue
        pairs+=("${ai}:${aj}")
        if [[ "$include_reversed" == "1" ]]; then
          pairs+=("${aj}:${ai}")
        fi
      done
    done
  else
    IFS=',' read -r -a base_pairs <<< "$raw_pairs"
    for pair_spec in "${base_pairs[@]}"; do
      [[ -z "$pair_spec" ]] && continue
      pairs+=("$pair_spec")
      if [[ "$include_reversed" == "1" ]]; then
        IFS=':' read -r a b <<< "$pair_spec"
        if [[ -n "${a:-}" && -n "${b:-}" ]]; then
          pairs+=("${b}:${a}")
        fi
      fi
    done
  fi
  printf '%s\n' "${pairs[@]}" | awk 'NF && !seen[$0]++'
}

resolve_or_empty() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf '%s' "$path"
  fi
}

expand_torchvision_models() {
  local raw_models="$1"
  if [[ "$raw_models" == "all" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" list_models
    return 0
  fi
  printf '%s\n' "$raw_models" | tr ',' '\n' | awk 'NF && !seen[$0]++'
}

torchvision_batch_size_for_model() {
  local model_name="$1"
  local default_bs="$RGB_TORCHVISION_BATCH_SIZE"
  case "$model_name" in
    swin3d_b)
      printf '%s' "${SKIN_TONE_RGB_TORCHVISION_BATCH_SIZE_SWIN3D_B:-2}"
      ;;
    swin3d_s|swin3d_t)
      printf '%s' "${SKIN_TONE_RGB_TORCHVISION_BATCH_SIZE_SWIN3D:-4}"
      ;;
    mvit_v2_s|mvit_v1_b)
      printf '%s' "${SKIN_TONE_RGB_TORCHVISION_BATCH_SIZE_MVIT:-8}"
      ;;
    s3d)
      printf '%s' "${SKIN_TONE_RGB_TORCHVISION_BATCH_SIZE_S3D:-8}"
      ;;
    r3d_18|mc3_18|r2plus1d_18)
      printf '%s' "${SKIN_TONE_RGB_TORCHVISION_BATCH_SIZE_RESNET:-16}"
      ;;
    *)
      printf '%s' "$default_bs"
      ;;
  esac
}

BACKGROUNDS="${SKIN_TONE_BACKGROUNDS:-autumn_hockey,konzerthaus,stadium_01}"
DARK_VARIANTS="${SKIN_TONE_DARK_VARIANTS:-african,indian}"
LIGHT_VARIANTS="${SKIN_TONE_LIGHT_VARIANTS:-white,asian}"
TRAIN_IDS="${SKIN_TONE_TRAIN_IDS:-0,1,2,3,7,8}"
VAL_IDS="${SKIN_TONE_VAL_IDS:-}"
SAME_ID_EVAL_IDS="${SKIN_TONE_SAME_ID_EVAL_IDS:-0,1,2,3,7,8}"
DISJOINT_EVAL_IDS="${SKIN_TONE_DISJOINT_EVAL_IDS:-4,5,6,9}"
ACTION_PAIRS_RAW="${SKIN_TONE_ACTION_PAIRS:-squat:tie,clap:celebrate,dribble:golf,lunge:cartwheel,yawn:fish}"
ACTIONS_RAW="${SKIN_TONE_ACTIONS:-}"
INCLUDE_REVERSED_PAIRS="${SKIN_TONE_INCLUDE_REVERSED_PAIRS:-0}"
PROBE_MODE="${SKIN_TONE_PROBE_MODE:-binary}"
SEEDS_RAW="${SKIN_TONE_SEEDS:-0,1,2}"
OUT_ROOT="${SKIN_TONE_OUT_ROOT:-$ROOT_DIR/out/skin_tone_probe_seeded_v7}"
TRAIN_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_TRAIN_MAX_SAMPLES_PER_CLASS:-12}"
VAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_VAL_MAX_SAMPLES_PER_CLASS:-6}"
EVAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_EVAL_MAX_SAMPLES_PER_CLASS:-0}"
MIX_PCT="${SKIN_TONE_MIX_PCT:-0}"
MODALITIES_RAW="${MODALITIES:-${SKIN_TONE_MODALITIES:-motion,rgb,rgb_torchvision,flow_i3d_external}}"

COLOR_JITTER="${SKIN_TONE_COLOR_JITTER:-0.8}"
MOTION_NOISE_STD="${SKIN_TONE_MOTION_NOISE_STD:-0.0}"

SKIN_TONE_DATASET_ROOT="${SKIN_TONE_DATASET_ROOT:-}"
MOTION_ROOT_DIR="${SKIN_TONE_MOTION_ROOT_DIR:-}"
RGB_ROOT_DIR="${SKIN_TONE_RGB_ROOT_DIR:-$SKIN_TONE_DATASET_ROOT}"

MOTION_PRETRAINED_CKPT="${SKIN_TONE_MOTION_PRETRAINED_CKPT:-$(resolve_or_empty "$ROOT_DIR/out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_033_loss3.5884.pt")}"
MOTION_PRESET="${SKIN_TONE_MOTION_PRESET:-default}"
HEAD_MODES_RAW="${SKIN_TONE_HEAD_MODES:-language}"
RGB_FRAMES="${SKIN_TONE_RGB_FRAMES:-64}"
RGB_SAMPLING="${SKIN_TONE_RGB_SAMPLING:-uniform}"
RGB_NORM="${SKIN_TONE_RGB_NORM:-i3d}"

RGB_PRETRAINED_CKPT="${SKIN_TONE_RGB_PRETRAINED_CKPT:-$(resolve_or_empty "$ROOT_DIR/out/rgb_checkpoint_epoch_019_loss0.6533.pt")}"

RGB_TORCHVISION_MODELS_RAW="${SKIN_TONE_RGB_TORCHVISION_MODELS:-${SKIN_TONE_RGB_R2PLUS1D_MODEL:-r3d_18}}"
RGB_TORCHVISION_ROOT_DIR="${SKIN_TONE_RGB_TORCHVISION_ROOT_DIR:-${SKIN_TONE_RGB_R2PLUS1D_ROOT_DIR:-$RGB_ROOT_DIR}}"
RGB_TORCHVISION_FRAMES="${SKIN_TONE_RGB_TORCHVISION_FRAMES:-${SKIN_TONE_RGB_R2PLUS1D_FRAMES:-16}}"
RGB_TORCHVISION_IMG_SIZE="${SKIN_TONE_RGB_TORCHVISION_IMG_SIZE:-${SKIN_TONE_RGB_R2PLUS1D_IMG_SIZE:-224}}"
RGB_TORCHVISION_BATCH_SIZE="${SKIN_TONE_RGB_TORCHVISION_BATCH_SIZE:-${SKIN_TONE_RGB_R2PLUS1D_BATCH_SIZE:-16}}"
RGB_TORCHVISION_EPOCHS="${SKIN_TONE_RGB_TORCHVISION_EPOCHS:-${SKIN_TONE_RGB_R2PLUS1D_EPOCHS:-10}}"
RGB_TORCHVISION_LR="${SKIN_TONE_RGB_TORCHVISION_LR:-${SKIN_TONE_RGB_R2PLUS1D_LR:-0.0002}}"
RGB_TORCHVISION_WEIGHT_DECAY="${SKIN_TONE_RGB_TORCHVISION_WEIGHT_DECAY:-${SKIN_TONE_RGB_R2PLUS1D_WEIGHT_DECAY:-0.0001}}"
RGB_TORCHVISION_NUM_WORKERS="${SKIN_TONE_RGB_TORCHVISION_NUM_WORKERS:-${SKIN_TONE_RGB_R2PLUS1D_NUM_WORKERS:-8}}"
RGB_TORCHVISION_DEVICE="${SKIN_TONE_RGB_TORCHVISION_DEVICE:-${SKIN_TONE_RGB_R2PLUS1D_DEVICE:-cuda}}"

FLOW_ROOT_DIR="${SKIN_TONE_FLOW_TVL1_ROOT_DIR:-}"
FLOW_PRETRAINED_CKPT="${SKIN_TONE_FLOW_PRETRAINED_CKPT:-$ROOT_DIR/third_party/pytorch-i3d/models/flow_imagenet.pt}"
FLOW_FRAMES="${SKIN_TONE_FLOW_FRAMES:-64}"
FLOW_IMG_SIZE="${SKIN_TONE_FLOW_IMG_SIZE:-224}"
FLOW_BATCH_SIZE="${SKIN_TONE_FLOW_BATCH_SIZE:-4}"
FLOW_EPOCHS="${SKIN_TONE_FLOW_EPOCHS:-10}"
FLOW_LR="${SKIN_TONE_FLOW_LR:-0.0002}"
FLOW_WEIGHT_DECAY="${SKIN_TONE_FLOW_WEIGHT_DECAY:-0.0001}"
FLOW_NUM_WORKERS="${SKIN_TONE_FLOW_NUM_WORKERS:-4}"
FLOW_DEVICE="${SKIN_TONE_FLOW_DEVICE:-cuda}"
FLOW_SAMPLING="${SKIN_TONE_FLOW_SAMPLING:-random}"
FLOW_FREEZE_UNTIL="${SKIN_TONE_FLOW_FREEZE_UNTIL:-none}"

IFS=',' read -r -a MODALITIES <<< "$MODALITIES_RAW"
IFS=',' read -r -a HEAD_MODES <<< "$HEAD_MODES_RAW"
IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"
IFS=',' read -r -a BACKGROUND_LIST <<< "$BACKGROUNDS"
ACTION_PAIRS=()
while IFS= read -r pair; do
  [[ -n "$pair" ]] && ACTION_PAIRS+=("$pair")
done < <(build_action_pairs "$ACTION_PAIRS_RAW" "$ACTIONS_RAW" "$INCLUDE_REVERSED_PAIRS")
RGB_TORCHVISION_MODELS=()
while IFS= read -r model_name; do
  [[ -n "$model_name" ]] && RGB_TORCHVISION_MODELS+=("$model_name")
done < <(expand_torchvision_models "$RGB_TORCHVISION_MODELS_RAW")

EVAL_SPLITS=(
  eval_matched_unseen_ids
  eval_matched_seen_ids
  eval_shifted_seen_ids
  eval_shifted_unseen_ids
)

[[ "$PROBE_MODE" == "binary" ]] || {
  echo "Skin-tone milestone supports only SKIN_TONE_PROBE_MODE=binary." >&2
  exit 1
}
[[ "${#ACTION_PAIRS[@]}" -gt 0 ]] || {
  echo "No action pairs resolved. Check SKIN_TONE_ACTION_PAIRS or SKIN_TONE_ACTIONS." >&2
  exit 1
}
[[ "${#RGB_TORCHVISION_MODELS[@]}" -gt 0 ]] || {
  echo "No torchvision RGB models resolved. Check SKIN_TONE_RGB_TORCHVISION_MODELS." >&2
  exit 1
}

if [[ "$MIX_PCT" -gt 0 ]]; then
  OUT_ROOT="${OUT_ROOT}_mix${MIX_PCT}"
  DATASET_SUBDIR_BASE="skin_tone_camera_far_binary_mix${MIX_PCT}"
else
  DATASET_SUBDIR_BASE="skin_tone_camera_far_binary"
fi

preflight() {
  require_file "$ROOT_DIR/finetune.py" "repo entrypoint"
  require_file "$ROOT_DIR/eval.py" "repo entrypoint"
  require_file "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" "torchvision RGB helper"
  require_file "$BENCHMARK_DIR/build_skin_tone_shortcut_probe.py" "skin-tone manifest builder"
  require_file "$BENCHMARK_DIR/aggregate_skin_tone_probe.py" "skin-tone aggregator"
  require_file "$BENCHMARK_DIR/compute_skin_tone_probe_stats.py" "skin-tone stats helper"
  require_file "$BENCHMARK_DIR/summarize_skin_tone_robustness.py" "skin-tone robustness summarizer"
  require_dir "$SKIN_TONE_DATASET_ROOT" "SKIN_TONE_DATASET_ROOT"
  mkdir -p "$OUT_ROOT" "$BENCHMARK_DIR/generated/manifests" "$BENCHMARK_DIR/generated/labels"
  python_check_modules "base benchmark runtime" numpy

  for modality in "${MODALITIES[@]}"; do
    case "$modality" in
      motion)
        require_dir "$MOTION_ROOT_DIR" "SKIN_TONE_MOTION_ROOT_DIR"
        require_file "$MOTION_PRETRAINED_CKPT" "SKIN_TONE_MOTION_PRETRAINED_CKPT"
        python_check_modules "motion modality" torch clip cv2 zstandard tensorboard
        ;;
      rgb)
        require_dir "$RGB_ROOT_DIR" "SKIN_TONE_RGB_ROOT_DIR"
        require_file "$RGB_PRETRAINED_CKPT" "SKIN_TONE_RGB_PRETRAINED_CKPT"
        python_check_modules "rgb modality" torch clip cv2 tensorboard
        ;;
      rgb_torchvision|rgb_r2plus1d)
        require_dir "$RGB_TORCHVISION_ROOT_DIR" "SKIN_TONE_RGB_TORCHVISION_ROOT_DIR"
        python_check_modules "rgb_torchvision modality" torch torchvision cv2
        ;;
      flow_i3d_external)
        require_dir "$FLOW_ROOT_DIR" "SKIN_TONE_FLOW_TVL1_ROOT_DIR"
        require_file "$FLOW_PRETRAINED_CKPT" "SKIN_TONE_FLOW_PRETRAINED_CKPT"
        local background
        for background in "${BACKGROUND_LIST[@]}"; do
          require_dir "$FLOW_ROOT_DIR/$background" "SKIN_TONE_FLOW_TVL1_ROOT_DIR background (${background})"
        done
        python_check_modules "flow_i3d_external modality" torch cv2
        ;;
      tc_clip)
        echo "tc_clip is recognized but not vendored in this milestone yet." >&2
        exit 1
        ;;
      *)
        echo "Unsupported modality: $modality" >&2
        exit 1
        ;;
    esac
  done

  echo "Preflight OK"
  echo "  benchmark: skin_tone"
  echo "  modalities: ${MODALITIES[*]}"
  echo "  torchvision_models: ${RGB_TORCHVISION_MODELS[*]}"
  echo "  dataset_root: $SKIN_TONE_DATASET_ROOT"
  echo "  output_root: $OUT_ROOT"
}

preflight
if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  exit 0
fi

for pair_spec in "${ACTION_PAIRS[@]}"; do
  IFS=':' read -r dark_action light_action <<< "$pair_spec"
  if [[ -z "${dark_action:-}" || -z "${light_action:-}" ]]; then
    echo "Invalid action pair spec: $pair_spec (expected dark:light)" >&2
    exit 1
  fi
  pair_tag="${dark_action}_vs_${light_action}"

  for seed in "${SEEDS[@]}"; do
    run_cmd "$PYTHON_BIN" "$BENCHMARK_DIR/build_skin_tone_shortcut_probe.py" \
      --dataset_root "$SKIN_TONE_DATASET_ROOT" \
      --pair_tag "$pair_tag" \
      --dark_action "$dark_action" \
      --light_action "$light_action" \
      --backgrounds "$BACKGROUNDS" \
      --dark_variants "$DARK_VARIANTS" \
      --light_variants "$LIGHT_VARIANTS" \
      --train_ids "$TRAIN_IDS" \
      --val_ids "$VAL_IDS" \
      --same_id_eval_ids "$SAME_ID_EVAL_IDS" \
      --disjoint_eval_ids "$DISJOINT_EVAL_IDS" \
      --train_max_samples_per_class "$TRAIN_MAX_SAMPLES_PER_CLASS" \
      --val_max_samples_per_class "$VAL_MAX_SAMPLES_PER_CLASS" \
      --eval_max_samples_per_class "$EVAL_MAX_SAMPLES_PER_CLASS" \
      --mix_pct "$MIX_PCT" \
      --mix_seed "$seed"

    if [[ "$MIX_PCT" -gt 0 ]]; then
      DATASET_SUBDIR="${DATASET_SUBDIR_BASE}_seed${seed}"
    else
      DATASET_SUBDIR="$DATASET_SUBDIR_BASE"
    fi
    manifest_root="$BENCHMARK_DIR/generated/manifests/${DATASET_SUBDIR}/${pair_tag}"
    label_csv="$BENCHMARK_DIR/generated/labels/${DATASET_SUBDIR}/${pair_tag}_labels.csv"

    if modality_selected "rgb_torchvision" || modality_selected "rgb_r2plus1d"; then
      validate_manifest_coverage "$RGB_TORCHVISION_ROOT_DIR" "rgb" "$manifest_root"
    fi
    if modality_selected "flow_i3d_external"; then
      validate_manifest_coverage "$FLOW_ROOT_DIR" "flow" "$manifest_root"
    fi

    for modality in "${MODALITIES[@]}"; do
      case "$modality" in
        motion)
          motion_out_root="${OUT_ROOT}/motion"
          motion_finetune_configs=(configs/benchmarks/skin_tone/finetune/common.toml)
          motion_eval_configs=(configs/benchmarks/skin_tone/eval/common.toml)
          motion_active_branch="both"
          if [[ "$MOTION_PRESET" == "x3d_flow_only" ]]; then
            motion_out_root="${OUT_ROOT}/motion_x3d_flow"
            motion_finetune_configs+=(configs/benchmarks/skin_tone/finetune/x3d_flow_only.toml)
            motion_eval_configs+=(configs/benchmarks/skin_tone/eval/x3d_flow_only.toml)
            motion_active_branch="second"
          fi

          for head_mode in "${HEAD_MODES[@]}"; do
            [[ -z "$head_mode" ]] && continue
            out_dir="${motion_out_root}/${pair_tag}/seed_${seed}"
            if [[ "$head_mode" != "legacy" ]]; then
              out_dir="${out_dir}_${head_mode}"
            fi
            run_already_done "$out_dir/summary_motion_only.json" && continue

            finetune_cmd=("$PYTHON_BIN" finetune.py)
            for config_path in "${motion_finetune_configs[@]}"; do
              finetune_cmd+=(--config "$config_path")
            done
            finetune_cmd+=(
              --root_dir "$MOTION_ROOT_DIR"
              --train_modality motion
              --val_modality motion
              --motion_data_source zstd
              --manifest "${manifest_root}/train_in_domain.txt"
              --class_id_to_label_csv "$label_csv"
              --out_dir "$out_dir"
              --seed "$seed"
              --val_subset_seed "$seed"
              --rgb_frames "$RGB_FRAMES"
              --rgb_sampling "$RGB_SAMPLING"
              --rgb_norm "$RGB_NORM"
              --finetune_head_mode "$head_mode"
              --motion_noise_std "$MOTION_NOISE_STD"
              --pretrained_ckpt "$MOTION_PRETRAINED_CKPT"
            )
            run_cmd "${finetune_cmd[@]}"

            ckpt="$(latest_ckpt "$out_dir")"
            [[ -n "${ckpt:-}" ]] || {
              echo "No checkpoint found in $out_dir/checkpoints" >&2
              exit 1
            }

            eval_cmd=("$PYTHON_BIN" eval.py)
            for config_path in "${motion_eval_configs[@]}"; do
              eval_cmd+=(--config "$config_path")
            done
            eval_cmd+=(
              --root_dir "$MOTION_ROOT_DIR"
              --input_modality motion
              --motion_data_source zstd
              --summary_only
              --no_clip
              --ckpt "$ckpt"
              --class_id_to_label_csv "$label_csv"
              --out_dir "$out_dir"
              --model_rgb_frames "$RGB_FRAMES"
              --model_rgb_sampling "$RGB_SAMPLING"
              --model_rgb_norm "$RGB_NORM"
            )
            if [[ "$motion_active_branch" != "both" ]]; then
              eval_cmd+=(--active_branch "$motion_active_branch")
            fi
            for eval_name in "${EVAL_SPLITS[@]}"; do
              eval_cmd+=(--manifests "${manifest_root}/${eval_name}.txt")
            done
            run_cmd "${eval_cmd[@]}"
          done
          ;;
        rgb)
          out_dir="${OUT_ROOT}/rgb/${pair_tag}/seed_${seed}"
          run_already_done "$out_dir/summary_rgb_model.json" && continue

          run_cmd "$PYTHON_BIN" finetune.py \
            --config configs/benchmarks/skin_tone/finetune/common.toml \
            --root_dir "$RGB_ROOT_DIR" \
            --train_modality rgb \
            --val_modality rgb \
            --manifest "${manifest_root}/train_in_domain.txt" \
            --class_id_to_label_csv "$label_csv" \
            --out_dir "$out_dir" \
            --seed "$seed" \
            --val_subset_seed "$seed" \
            --rgb_frames "$RGB_FRAMES" \
            --rgb_sampling "$RGB_SAMPLING" \
            --rgb_norm "$RGB_NORM" \
            --color_jitter "$COLOR_JITTER" \
            --pretrained_ckpt "$RGB_PRETRAINED_CKPT"

          ckpt="$(latest_ckpt "$out_dir")"
          [[ -n "${ckpt:-}" ]] || {
            echo "No checkpoint found in $out_dir/checkpoints" >&2
            exit 1
          }

          eval_cmd=(
            "$PYTHON_BIN" eval.py
            --config configs/benchmarks/skin_tone/eval/common.toml
            --root_dir "$RGB_ROOT_DIR"
            --input_modality rgb
            --summary_only
            --no_clip
            --ckpt "$ckpt"
            --class_id_to_label_csv "$label_csv"
            --out_dir "$out_dir"
            --model_rgb_frames "$RGB_FRAMES"
            --model_rgb_sampling "$RGB_SAMPLING"
            --model_rgb_norm "$RGB_NORM"
          )
          for eval_name in "${EVAL_SPLITS[@]}"; do
            eval_cmd+=(--manifests "${manifest_root}/${eval_name}.txt")
          done
          run_cmd "${eval_cmd[@]}"
          ;;
        rgb_torchvision|rgb_r2plus1d)
          for rgb_model in "${RGB_TORCHVISION_MODELS[@]}"; do
            out_dir="${OUT_ROOT}/rgb_torchvision/${rgb_model}/${pair_tag}/seed_${seed}"
            summary_path="$out_dir/summary_rgb_${rgb_model}_model.json"
            predictions_complete=1
            for eval_name in "${EVAL_SPLITS[@]}"; do
              pred_csv="$out_dir/${eval_name}/predictions_rgb_${rgb_model}.csv"
              if [[ ! -f "$pred_csv" ]]; then
                predictions_complete=0
                break
              fi
            done
            if run_already_done "$summary_path" "${#EVAL_SPLITS[@]}" && [[ "$predictions_complete" == "1" ]]; then
              continue
            fi
            rgb_batch_size="$(torchvision_batch_size_for_model "$rgb_model")"

            resume_ckpt="$(latest_ckpt "$out_dir")"
            train_cmd=(
              "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py"
              train
              --root_dir "$RGB_TORCHVISION_ROOT_DIR"
              --manifest "${manifest_root}/train_in_domain.txt"
              --class_id_to_label_csv "$label_csv"
              --out_dir "$out_dir"
              --seed "$seed"
              --model "$rgb_model"
              --rgb_frames "$RGB_TORCHVISION_FRAMES"
              --img_size "$RGB_TORCHVISION_IMG_SIZE"
              --rgb_sampling "$RGB_SAMPLING"
              --batch_size "$rgb_batch_size"
              --epochs "$RGB_TORCHVISION_EPOCHS"
              --lr "$RGB_TORCHVISION_LR"
              --weight_decay "$RGB_TORCHVISION_WEIGHT_DECAY"
              --num_workers "$RGB_TORCHVISION_NUM_WORKERS"
              --device "$RGB_TORCHVISION_DEVICE"
              --color_jitter "$COLOR_JITTER"
            )
            if [[ -n "$resume_ckpt" ]]; then
              train_cmd+=(--resume_ckpt "$resume_ckpt")
            fi
            run_cmd "${train_cmd[@]}"

            ckpt="$(latest_ckpt "$out_dir")"
            [[ -n "${ckpt:-}" ]] || {
              echo "No checkpoint found in $out_dir/checkpoints" >&2
              exit 1
            }

            eval_ran=0
            for eval_name in "${EVAL_SPLITS[@]}"; do
              split_summary="$out_dir/${eval_name}/summary_rgb_${rgb_model}_model.json"
              split_pred_csv="$out_dir/${eval_name}/predictions_rgb_${rgb_model}.csv"
              if run_already_done "$split_summary" && [[ -f "$split_pred_csv" ]]; then
                continue
              fi
              run_cmd "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" \
                eval \
                --root_dir "$RGB_TORCHVISION_ROOT_DIR" \
                --manifest "${manifest_root}/${eval_name}.txt" \
                --class_id_to_label_csv "$label_csv" \
                --ckpt "$ckpt" \
                --out_dir "$out_dir/${eval_name}" \
                --split_name "$eval_name" \
                --model "$rgb_model" \
                --pair_tag "$pair_tag" \
                --rgb_frames "$RGB_TORCHVISION_FRAMES" \
                --img_size "$RGB_TORCHVISION_IMG_SIZE" \
                --rgb_sampling uniform \
                --batch_size "$rgb_batch_size" \
                --num_workers "$RGB_TORCHVISION_NUM_WORKERS" \
                --device "$RGB_TORCHVISION_DEVICE" \
                --seed "$seed" \
                --summary_only
              eval_ran=1
            done
            if [[ "$eval_ran" == "1" ]] || [[ ! -f "$summary_path" ]]; then
              run_cmd "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" \
                aggregate \
                --out_dir "$out_dir" \
                --model "$rgb_model"
            fi
          done
          ;;
        flow_i3d_external)
          out_dir="${OUT_ROOT}/flow_i3d_external/${pair_tag}/seed_${seed}"
          summary_path="$out_dir/summary_flow_i3d_external_model.json"
          predictions_complete=1
          for eval_name in "${EVAL_SPLITS[@]}"; do
            pred_csv="$out_dir/${eval_name}/predictions_flow_i3d_external_model.csv"
            if [[ ! -f "$pred_csv" ]]; then
              predictions_complete=0
              break
            fi
          done
          if run_already_done "$summary_path" "${#EVAL_SPLITS[@]}" && [[ "$predictions_complete" == "1" ]]; then
            continue
          fi

          resume_ckpt="$(latest_ckpt "$out_dir")"
          train_cmd=(
            "$PYTHON_BIN" "$BENCHMARK_DIR/train_skin_tone_pytorch_i3d_flow_probe.py"
            train
            --root_dir "$FLOW_ROOT_DIR"
            --manifest "${manifest_root}/train_in_domain.txt"
            --class_id_to_label_csv "$label_csv"
            --pretrained_ckpt "$FLOW_PRETRAINED_CKPT"
            --out_dir "$out_dir"
            --flow_frames "$FLOW_FRAMES"
            --img_size "$FLOW_IMG_SIZE"
            --sampling "$FLOW_SAMPLING"
            --batch_size "$FLOW_BATCH_SIZE"
            --epochs "$FLOW_EPOCHS"
            --lr "$FLOW_LR"
            --weight_decay "$FLOW_WEIGHT_DECAY"
            --num_workers "$FLOW_NUM_WORKERS"
            --device "$FLOW_DEVICE"
            --seed "$seed"
            --freeze_until "$FLOW_FREEZE_UNTIL"
            --motion_noise_std "$MOTION_NOISE_STD"
          )
          if [[ -n "$resume_ckpt" ]]; then
            train_cmd+=(--resume_ckpt "$resume_ckpt")
          fi
          run_cmd "${train_cmd[@]}"

          ckpt="$(latest_ckpt "$out_dir")"
          [[ -n "${ckpt:-}" ]] || {
            echo "No checkpoint found in $out_dir/checkpoints" >&2
            exit 1
          }

          eval_ran=0
          for eval_name in "${EVAL_SPLITS[@]}"; do
            split_summary="$out_dir/${eval_name}/summary_flow_i3d_external_model.json"
            split_pred_csv="$out_dir/${eval_name}/predictions_flow_i3d_external_model.csv"
            if run_already_done "$split_summary" && [[ -f "$split_pred_csv" ]]; then
              continue
            fi
            run_cmd "$PYTHON_BIN" "$BENCHMARK_DIR/train_skin_tone_pytorch_i3d_flow_probe.py" \
              eval \
              --root_dir "$FLOW_ROOT_DIR" \
              --ckpt "$ckpt" \
              --manifest "${manifest_root}/${eval_name}.txt" \
              --class_id_to_label_csv "$label_csv" \
              --out_dir "$out_dir/${eval_name}" \
              --split_name "$eval_name" \
              --pair_tag "$pair_tag" \
              --flow_frames "$FLOW_FRAMES" \
              --img_size "$FLOW_IMG_SIZE" \
              --batch_size "$FLOW_BATCH_SIZE" \
              --num_workers "$FLOW_NUM_WORKERS" \
              --device "$FLOW_DEVICE" \
              --seed "$seed" \
              --summary_only
            eval_ran=1
          done
          if [[ "$eval_ran" == "1" ]] || [[ ! -f "$summary_path" ]]; then
            run_cmd "$PYTHON_BIN" "$BENCHMARK_DIR/train_skin_tone_pytorch_i3d_flow_probe.py" \
              aggregate \
              --out_dir "$out_dir"
          fi
          ;;
        *)
          echo "Unsupported modality: $modality" >&2
          exit 1
          ;;
      esac
    done
  done
done

run_cmd "$PYTHON_BIN" "$BENCHMARK_DIR/aggregate_skin_tone_probe.py" --root "$OUT_ROOT"
run_cmd "$PYTHON_BIN" "$BENCHMARK_DIR/compute_skin_tone_probe_stats.py" --root "$OUT_ROOT" --metric f1_macro
run_cmd "$PYTHON_BIN" "$BENCHMARK_DIR/summarize_skin_tone_robustness.py" --root "$OUT_ROOT" --metric f1_macro

echo "Done. Results in: $OUT_ROOT"
