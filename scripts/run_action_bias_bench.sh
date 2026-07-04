#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$ROOT_DIR/benchmarks/skin_tone"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── paths ──────────────────────────────────────────────────────────────────────
SKIN_TONE_DATASET_ROOT="${SKIN_TONE_DATASET_ROOT:-}"
MOTION_ROOT_DIR="${SKIN_TONE_MOTION_ROOT_DIR:-}"
RGB_ROOT_DIR="${SKIN_TONE_RGB_ROOT_DIR:-$SKIN_TONE_DATASET_ROOT}"
FLOW_ROOT_DIR="${SKIN_TONE_FLOW_TVL1_ROOT_DIR:-}"
RGB_TORCHVISION_ROOT_DIR="${SKIN_TONE_RGB_TORCHVISION_ROOT_DIR:-$RGB_ROOT_DIR}"
OUT_ROOT="${SKIN_TONE_OUT_ROOT:-$ROOT_DIR/out/skin_tone_probe_seeded_v7}"

# auto-detect default checkpoints if they exist
_findfile() { [[ -f "$1" ]] && echo "$1" || true; }
MOTION_PRETRAINED_CKPT="${SKIN_TONE_MOTION_PRETRAINED_CKPT:-$(_findfile "$ROOT_DIR/out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_033_loss3.5884.pt")}"
RGB_PRETRAINED_CKPT="${SKIN_TONE_RGB_PRETRAINED_CKPT:-$(_findfile "$ROOT_DIR/out/rgb_checkpoint_epoch_019_loss0.6533.pt")}"

# ── experiment config ──────────────────────────────────────────────────────────
BACKGROUNDS="${SKIN_TONE_BACKGROUNDS:-autumn_hockey,konzerthaus,stadium_01}"
DARK_VARIANTS="${SKIN_TONE_DARK_VARIANTS:-african,indian}"
LIGHT_VARIANTS="${SKIN_TONE_LIGHT_VARIANTS:-white,asian}"
TRAIN_IDS="${SKIN_TONE_TRAIN_IDS:-0,1,2,3,7,8}"
VAL_IDS="${SKIN_TONE_VAL_IDS:-}"
SAME_ID_EVAL_IDS="${SKIN_TONE_SAME_ID_EVAL_IDS:-0,1,2,3,7,8}"
DISJOINT_EVAL_IDS="${SKIN_TONE_DISJOINT_EVAL_IDS:-4,5,6,9}"
ACTION_PAIRS="${SKIN_TONE_ACTION_PAIRS:-squat:tie,clap:celebrate,dribble:golf,lunge:cartwheel,yawn:fish}"
SEEDS="${SKIN_TONE_SEEDS:-0,1,2}"
MIX_PCT="${SKIN_TONE_MIX_PCT:-0}"
MODALITIES="${MODALITIES:-${SKIN_TONE_MODALITIES:-motion,rgb,rgb_torchvision,flow_i3d_external}}"
HEAD_MODES="${SKIN_TONE_HEAD_MODES:-language}"
MOTION_PRESET="${SKIN_TONE_MOTION_PRESET:-default}"
RGB_TORCHVISION_MODELS="${SKIN_TONE_RGB_TORCHVISION_MODELS:-r3d_18}"
COLOR_JITTER="${SKIN_TONE_COLOR_JITTER:-0.8}"
# ColorJitter strength (defaults reproduce the original augmentation).
COLOR_JITTER_BRIGHTNESS="${SKIN_TONE_COLOR_JITTER_BRIGHTNESS:-0.4}"
COLOR_JITTER_CONTRAST="${SKIN_TONE_COLOR_JITTER_CONTRAST:-0.4}"
COLOR_JITTER_SATURATION="${SKIN_TONE_COLOR_JITTER_SATURATION:-0.2}"
COLOR_JITTER_HUE="${SKIN_TONE_COLOR_JITTER_HUE:-0.1}"
# Grayscale probability: independent of color jitter, removes chroma entirely
# (vs. jitter which perturbs it) for a qualitatively different mitigation test.
GRAYSCALE_PROB="${SKIN_TONE_GRAYSCALE_PROB:-0.0}"

# ── derived ────────────────────────────────────────────────────────────────────
if [[ "$MIX_PCT" -gt 0 ]]; then
  OUT_ROOT="${OUT_ROOT}_mix${MIX_PCT}"
  DATASET_SUBDIR_BASE="skin_tone_camera_far_binary_mix${MIX_PCT}"
else
  DATASET_SUBDIR_BASE="skin_tone_camera_far_binary"
fi

EVAL_SPLITS=(eval_matched_unseen_ids eval_matched_seen_ids eval_shifted_seen_ids eval_shifted_unseen_ids)

latest_ckpt() { ls -t "$1/checkpoints"/checkpoint*.pt 2>/dev/null | head -n 1 || true; }

mkdir -p "$OUT_ROOT" "$BENCHMARK_DIR/generated/manifests" "$BENCHMARK_DIR/generated/labels"

IFS=',' read -r -a _pairs  <<< "$ACTION_PAIRS"
IFS=',' read -r -a _seeds  <<< "$SEEDS"
IFS=',' read -r -a _heads  <<< "$HEAD_MODES"
IFS=',' read -r -a _mods   <<< "$MODALITIES"
IFS=',' read -r -a _tvmods <<< "$RGB_TORCHVISION_MODELS"

for pair_spec in "${_pairs[@]}"; do
  IFS=':' read -r dark_action light_action <<< "$pair_spec"
  pair_tag="${dark_action}_vs_${light_action}"

  for seed in "${_seeds[@]}"; do
    "$PYTHON_BIN" "$BENCHMARK_DIR/build_skin_tone_shortcut_probe.py" \
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
      --train_max_samples_per_class 0 \
      --val_max_samples_per_class 6 \
      --eval_max_samples_per_class 0 \
      --mix_pct "$MIX_PCT" \
      --mix_seed "$seed"

    if [[ "$MIX_PCT" -gt 0 ]]; then
      DATASET_SUBDIR="${DATASET_SUBDIR_BASE}_seed${seed}"
    else
      DATASET_SUBDIR="$DATASET_SUBDIR_BASE"
    fi
    manifest_root="$BENCHMARK_DIR/generated/manifests/${DATASET_SUBDIR}/${pair_tag}"
    label_csv="$BENCHMARK_DIR/generated/labels/${DATASET_SUBDIR}/${pair_tag}_labels.csv"

    for modality in "${_mods[@]}"; do
      case "$modality" in

        motion)
          ft_configs=(configs/benchmarks/skin_tone/finetune/common.toml)
          ev_configs=(configs/benchmarks/skin_tone/eval/common.toml)
          motion_out="${OUT_ROOT}/motion"
          active_branch=""
          if [[ "$MOTION_PRESET" == "x3d_flow_only" ]]; then
            ft_configs+=(configs/benchmarks/skin_tone/finetune/x3d_flow_only.toml)
            ev_configs+=(configs/benchmarks/skin_tone/eval/x3d_flow_only.toml)
            motion_out="${OUT_ROOT}/motion_x3d_flow"
            active_branch="second"
          fi

          for head_mode in "${_heads[@]}"; do
            out_dir="${motion_out}/${pair_tag}/seed_${seed}"
            [[ "$head_mode" != "legacy" ]] && out_dir="${out_dir}_${head_mode}"
            [[ -f "$out_dir/summary_motion_only.json" ]] && continue

            ft_cmd=("$PYTHON_BIN" finetune.py)
            for cfg in "${ft_configs[@]}"; do ft_cmd+=(--config "$cfg"); done
            ft_cmd+=(
              --root_dir "$MOTION_ROOT_DIR"
              --manifest "${manifest_root}/train_in_domain.txt"
              --class_id_to_label_csv "$label_csv"
              --out_dir "$out_dir"
              --seed "$seed"
              --val_subset_seed "$seed"
              --finetune_head_mode "$head_mode"
              --pretrained_ckpt "$MOTION_PRETRAINED_CKPT"
            )
            "${ft_cmd[@]}"

            ckpt="$(latest_ckpt "$out_dir")"
            [[ -n "$ckpt" ]] || { echo "No checkpoint in $out_dir/checkpoints" >&2; exit 1; }

            ev_cmd=("$PYTHON_BIN" eval.py)
            for cfg in "${ev_configs[@]}"; do ev_cmd+=(--config "$cfg"); done
            ev_cmd+=(
              --root_dir "$MOTION_ROOT_DIR"
              --ckpt "$ckpt"
              --class_id_to_label_csv "$label_csv"
              --out_dir "$out_dir"
              --summary_only
            )
            [[ -n "$active_branch" ]] && ev_cmd+=(--active_branch "$active_branch")
            for split in "${EVAL_SPLITS[@]}"; do ev_cmd+=(--manifests "${manifest_root}/${split}.txt"); done
            "${ev_cmd[@]}"
          done
          ;;

        rgb)
          out_dir="${OUT_ROOT}/rgb/${pair_tag}/seed_${seed}"
          [[ -f "$out_dir/summary_rgb_model.json" ]] && continue

          "$PYTHON_BIN" finetune.py \
            --config configs/benchmarks/skin_tone/finetune/common.toml \
            --root_dir "$RGB_ROOT_DIR" \
            --train_modality rgb \
            --val_modality rgb \
            --manifest "${manifest_root}/train_in_domain.txt" \
            --class_id_to_label_csv "$label_csv" \
            --out_dir "$out_dir" \
            --seed "$seed" \
            --val_subset_seed "$seed" \
            --color_jitter "$COLOR_JITTER" \
            --pretrained_ckpt "$RGB_PRETRAINED_CKPT"

          ckpt="$(latest_ckpt "$out_dir")"
          [[ -n "$ckpt" ]] || { echo "No checkpoint in $out_dir/checkpoints" >&2; exit 1; }

          ev_cmd=(
            "$PYTHON_BIN" eval.py
            --config configs/benchmarks/skin_tone/eval/common.toml
            --root_dir "$RGB_ROOT_DIR"
            --input_modality rgb
            --ckpt "$ckpt"
            --class_id_to_label_csv "$label_csv"
            --out_dir "$out_dir"
            --summary_only
          )
          for split in "${EVAL_SPLITS[@]}"; do ev_cmd+=(--manifests "${manifest_root}/${split}.txt"); done
          "${ev_cmd[@]}"
          ;;

        rgb_torchvision|rgb_r2plus1d)
          for rgb_model in "${_tvmods[@]}"; do
            out_dir="${OUT_ROOT}/rgb_torchvision/${rgb_model}/${pair_tag}/seed_${seed}"
            [[ -f "$out_dir/summary_rgb_${rgb_model}_model.json" ]] && continue

            resume_ckpt="$(latest_ckpt "$out_dir")"
            train_cmd=(
              "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" train
              --root_dir "$RGB_TORCHVISION_ROOT_DIR"
              --manifest "${manifest_root}/train_in_domain.txt"
              --class_id_to_label_csv "$label_csv"
              --out_dir "$out_dir"
              --seed "$seed"
              --model "$rgb_model"
              --num_workers 8
              --color_jitter "$COLOR_JITTER"
              --color_jitter_brightness "$COLOR_JITTER_BRIGHTNESS"
              --color_jitter_contrast "$COLOR_JITTER_CONTRAST"
              --color_jitter_saturation "$COLOR_JITTER_SATURATION"
              --color_jitter_hue "$COLOR_JITTER_HUE"
              --grayscale_prob "$GRAYSCALE_PROB"
            )
            [[ -n "$resume_ckpt" ]] && train_cmd+=(--resume_ckpt "$resume_ckpt")
            "${train_cmd[@]}"

            ckpt="$(latest_ckpt "$out_dir")"
            [[ -n "$ckpt" ]] || { echo "No checkpoint in $out_dir/checkpoints" >&2; exit 1; }

            for split in "${EVAL_SPLITS[@]}"; do
              "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" eval \
                --root_dir "$RGB_TORCHVISION_ROOT_DIR" \
                --manifest "${manifest_root}/${split}.txt" \
                --class_id_to_label_csv "$label_csv" \
                --ckpt "$ckpt" \
                --out_dir "$out_dir/${split}" \
                --split_name "$split" \
                --model "$rgb_model" \
                --pair_tag "$pair_tag" \
                --seed "$seed" \
                --num_workers 8 \
                --summary_only
            done

            "$PYTHON_BIN" "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py" aggregate \
              --out_dir "$out_dir" --model "$rgb_model"
          done
          ;;

        flow_i3d_external)
          out_dir="${OUT_ROOT}/flow_i3d_external/${pair_tag}/seed_${seed}"
          [[ -f "$out_dir/summary_flow_i3d_external_model.json" ]] && continue

          resume_ckpt="$(latest_ckpt "$out_dir")"
          train_cmd=(
            "$PYTHON_BIN" "$BENCHMARK_DIR/train_skin_tone_pytorch_i3d_flow_probe.py" train
            --root_dir "$FLOW_ROOT_DIR"
            --manifest "${manifest_root}/train_in_domain.txt"
            --class_id_to_label_csv "$label_csv"
            --out_dir "$out_dir"
            --seed "$seed"
            --sampling random
          )
          [[ -n "$resume_ckpt" ]] && train_cmd+=(--resume_ckpt "$resume_ckpt")
          "${train_cmd[@]}"

          ckpt="$(latest_ckpt "$out_dir")"
          [[ -n "$ckpt" ]] || { echo "No checkpoint in $out_dir/checkpoints" >&2; exit 1; }

          for split in "${EVAL_SPLITS[@]}"; do
            "$PYTHON_BIN" "$BENCHMARK_DIR/train_skin_tone_pytorch_i3d_flow_probe.py" eval \
              --root_dir "$FLOW_ROOT_DIR" \
              --ckpt "$ckpt" \
              --manifest "${manifest_root}/${split}.txt" \
              --class_id_to_label_csv "$label_csv" \
              --out_dir "$out_dir/${split}" \
              --split_name "$split" \
              --pair_tag "$pair_tag" \
              --seed "$seed" \
              --summary_only
          done

          "$PYTHON_BIN" "$BENCHMARK_DIR/train_skin_tone_pytorch_i3d_flow_probe.py" aggregate \
            --out_dir "$out_dir"
          ;;

        *)
          echo "Unsupported modality: $modality" >&2; exit 1
          ;;

      esac
    done
  done
done

"$PYTHON_BIN" "$BENCHMARK_DIR/aggregate_skin_tone_probe.py" --root "$OUT_ROOT"
"$PYTHON_BIN" "$BENCHMARK_DIR/compute_skin_tone_probe_stats.py" --root "$OUT_ROOT" --metric f1_macro
"$PYTHON_BIN" "$BENCHMARK_DIR/summarize_skin_tone_robustness.py" --root "$OUT_ROOT" --metric f1_macro

echo "Done. Results in: $OUT_ROOT"
