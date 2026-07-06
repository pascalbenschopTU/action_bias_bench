# ActionBiasBench

Standalone bias-benchmark repo for action-recognition models.

This milestone packages the runtime needed for:

- `motion`
- `rgb`
- `rgb_r2plus1d`
- `flow_i3d_external`

`tc_clip` (CLIP ViT-B/16 + temporal context, Kinetics-400 pretrained) is vendored via `models/huggingface_models.py::load_tc_clip/encode_tc_clip`, sourced from `appearance_free_cross_domain_action_recognition/tc-clip`. It needs its own apptainer image (`huggingface_yolo_clip_tcclip_cuda.sif` — einops, mmcv-full, timm==0.4.12); see `jobs/bias/embed_all_models.sbatch` for the isolated container invocation.

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
- `SKIN_TONE_ACTION_PAIRS=squat:tie,clap:celebrate,...` (dark:light; default covers only
  the 5 base directions — pass the reversed 5 as a second run into the same
  `SKIN_TONE_OUT_ROOT` for the full 10-pair benchmark)
- `SKIN_TONE_COLOR_JITTER=0.8` (fraction of RGB-torchvision training clips jittered)
- `SKIN_TONE_COLOR_JITTER_BRIGHTNESS/CONTRAST/SATURATION/HUE` (jitter strength;
  default 0.4/0.4/0.2/0.1 — the original augmentation)
- `SKIN_TONE_GRAYSCALE_PROB=0.5` (fraction of clips converted to grayscale;
  independent of `SKIN_TONE_COLOR_JITTER`, removes chroma entirely)

Example:

```bash
export SKIN_TONE_DATASET_ROOT=/data/skin_tone_actions/camera_far
export SKIN_TONE_MOTION_ROOT_DIR=/data/skin_tone_actions/camera_far_motion_zst
export SKIN_TONE_FLOW_TVL1_ROOT_DIR=/data/skin_tone_actions/camera_far_flow_tvl1_fast_npz
bash scripts/run_action_bias_bench.sh --preflight
```

Generated manifests and label CSVs are written under `benchmarks/skin_tone/generated/`.

## Figures

### Foundation-model bias: linear probe + SSM

```bash
# 1. cache per-frame embeddings (GPU; tc_clip needs its own apptainer, see above)
#    clip caches to out/bias_analysis/clip_embeddings, not the default
#    out/bias_analysis/embeddings — pass --cache_dir explicitly for clip, both
#    when caching and in steps 2-3 below, or ssm_frobenius_analysis.py will
#    silently load 0 clips for that model.
python scripts/skin_tone_bias_analysis.py --model <model> --frames 64
python scripts/skin_tone_bias_analysis.py --model clip --frames 64 --cache_dir out/bias_analysis/clip_embeddings

# 2. linear probe on the cached embeddings (CPU)
python scripts/train_embedding_linear_probe.py --model <model> --seeds 0,1,2

# 3. SSM representational-similarity diagnostic (CPU; only meaningful for
#    models that emit a per-frame token sequence — see SSM_VALID in the plot script)
python scripts/ssm_frobenius_analysis.py --model <model> --metric rsa
python scripts/ssm_frobenius_analysis.py --model clip --metric rsa --cache_dir out/bias_analysis/clip_embeddings
#    --pairs all instead of the default "matching" runs the diagnostic over all
#    C(10,2)=45 action-pair combinations among the 10 actions, instead of just
#    the 5 curated pairs used for the fine-tune/probe experiments (cheap, CPU-only)
python scripts/ssm_frobenius_analysis.py --model <model> --metric rsa --pairs all

# 4. regenerate the figures
python scripts/plot_probe_ssm.py
```
Steps 1–2 can also be run for every model via `jobs/bias/embed_all_models.sbatch`
and `jobs/bias/run_all_linear_probes.sbatch`. Writes to `out/linear_probes/`:
`_probe_drop_by_model.pdf` (bar chart; per-bar `matched F1=...` is shown in red
when a probe scored below 0.75 on its matched split, meaning it never learned
the task and its shortcut-drop is not interpretable either way),
`_probe_vs_ssm.pdf` (scatter, probe drop vs. SSM ratio, restricted to the
SSM-valid models), `_probe_ssm_by_pair.pdf` (per-action-pair breakdown).
`ssm_frobenius_analysis.py` writes `out/bias_analysis/ssm_<metric>_<model>.csv`
(e.g. `ssm_rsa_clip.csv`); `plot_probe_ssm.py`'s `METRIC` constant selects which
metric's files it reads (default `"rsa"` — preferred over `"frobenius"`, which
is dominated by a few clips with unusually large SSM magnitude).

### Color-jitter / grayscale augmentation radar

Requires one or more completed `run_skin_tone_shortcut_probe.sbatch` output
roots (`rgb_torchvision` modality; see `SKIN_TONE_COLOR_JITTER*` /
`SKIN_TONE_GRAYSCALE_PROB` above to produce new conditions). Use
`SKIN_TONE_COLOR_JITTER_CONSISTENT=1` for any jitter condition — without it,
jitter parameters are resampled independently per frame instead of once per
clip, producing unintended frame-to-frame flicker on top of the intended
color shift (see `data/rgb.py` and
`llm_reports/color_jitter_strength_experiment.md`):

```bash
python scripts/plot_augmentation_radar.py \
  --roots none=out/skin_tone_probe_rgb_torchvision_v6_cj0p0 \
          "jitter 40%=out/skin_tone_probe_rgb_torchvision_v6_cj0p4_consistent" \
          "jitter 80%=out/skin_tone_probe_rgb_torchvision_v6_cj0p8_consistent" \
          "jitter 80% strong=out/skin_tone_probe_rgb_torchvision_v6_cj0p8_strong_consistent" \
          "grayscale 50%=out/skin_tone_probe_rgb_torchvision_v6_grayscale0p5" \
  --baseline none \
  --out_dir out/skin_tone_probe_rgb_torchvision_v6_analysis/augmentation_conditions
```
Writes `augmentation_radar_delta.{pdf,png}`: a baseline-anchored radar where
every model's no-augmentation level collapses onto one shared circle, so
deviation from the circle is the effect of that augmentation (each model gets
a fixed color and a distinct marker shape). Each `--roots` entry must cover
the same pair-tags across conditions, or the script silently restricts to the
shared (pair, seed) subset and prints the resulting unit count per model to
the console — see `llm_reports/color_jitter_strength_experiment.md` for a
worked example of the mismatch this catches.
