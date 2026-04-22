# ActionBiasBench

Standalone bias-benchmark repo for action-recognition models.

This milestone packages the runtime needed for:

- `motion`
- `rgb`
- `rgb_r2plus1d`
- `flow_i3d_external`

`tc_clip` is recognized by the launcher but not vendored yet. The launcher will fail early with a clear message if you request it.

## Layout

- `finetune.py`, `eval.py`, `config.py`, `dataset.py`, `augment.py`, `model.py`, `e2s_x3d.py`
- `cli/`, `data/`, `models/`, `utils/`
- `benchmarks/skin_tone/`
- `configs/benchmarks/skin_tone/`
- `scripts/run_action_bias_bench.sh`
- `third_party/pytorch-i3d/`

## Main Entry Point

Run the centralized launcher from the repo root:

```bash
bash scripts/run_action_bias_bench.sh --preflight
```

## Required Environment

You must supply external dataset/checkpoint locations explicitly.

Required for skin-tone runs:

- `SKIN_TONE_DATASET_ROOT`: video-root for manifest generation
- `SKIN_TONE_MOTION_ROOT_DIR`: zstd motion root for the motion model
- `SKIN_TONE_FLOW_TVL1_ROOT_DIR`: TV-L1 flow root for the external flow I3D baseline

Common optional overrides:

- `MODALITIES=motion,rgb,rgb_torchvision,flow_i3d_external`
- `SKIN_TONE_MOTION_PRETRAINED_CKPT=/path/to/checkpoint.pt`
- `SKIN_TONE_RGB_PRETRAINED_CKPT=/path/to/checkpoint.pt`
- `SKIN_TONE_FLOW_PRETRAINED_CKPT=/path/to/flow_imagenet.pt`
- `SKIN_TONE_RGB_TORCHVISION_MODELS=r3d_18,mc3_18,mvit_v2_s`
- `SKIN_TONE_RGB_TORCHVISION_MODELS=all`
- `SKIN_TONE_OUT_ROOT=/path/to/output_root`

Example:

```bash
export SKIN_TONE_DATASET_ROOT=/data/skin_tone_actions/camera_far
export SKIN_TONE_MOTION_ROOT_DIR=/data/skin_tone_actions/camera_far_motion_zst
export SKIN_TONE_FLOW_TVL1_ROOT_DIR=/data/skin_tone_actions/camera_far_flow_tvl1_fast_npz
bash scripts/run_action_bias_bench.sh --preflight
```

Generated manifests and label CSVs are written under `benchmarks/skin_tone/generated/`.
