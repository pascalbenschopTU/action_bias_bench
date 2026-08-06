# ActionBiasBench

Standalone bias-benchmark repo for action-recognition models. Code and reproduction
commands for **"Controlled Auditing of Skin-Tone Shortcut Sensitivity in Human Action
Recognition"**.

This milestone packages the runtime needed for:

- `motion`
- `rgb`
- `rgb_r2plus1d`
- `flow_i3d_external`

`tc_clip` (CLIP ViT-B/16 + temporal context, Kinetics-400 pretrained) is vendored via `models/huggingface_models.py::load_tc_clip/encode_tc_clip`, sourced from `appearance_free_cross_domain_action_recognition/tc-clip`. It needs its own apptainer image (`huggingface_yolo_clip_tcclip_cuda.sif` — einops, mmcv-full, timm==0.4.12); see `jobs/bias/embed_all_models.sbatch` for the isolated container invocation.

## Layout

- `finetune.py`, `eval.py`, `config.py`, `dataset.py`, `augment.py`, `model.py`, `e2s_x3d.py`
- `cli/`, `data/`, `models/`, `utils/`
- `benchmarks/skin_tone/` — manifest generation, aggregation, significance testing, and the diagnostic scripts below
- `configs/benchmarks/skin_tone/`
- `scripts/run_action_bias_bench.sh` — central launcher for the fine-tuning benchmark
- `third_party/pytorch-i3d/`

Cluster job scripts (`.sbatch`) live **outside this repo**, under `motion_only_AR/jobs/bias/`, and are submitted from `Pascal/` (the parent of `motion_only_AR/`), not from inside `ActionBiasBench/`. Each experiment section below shows the plain Python command first (runnable locally, no SLURM) and collapses the actual cluster invocation used to produce the paper's numbers where one exists.

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
  `SKIN_TONE_OUT_ROOT` for the full 10-pair benchmark, or set
  `SKIN_TONE_INCLUDE_REVERSED_PAIRS=1` to generate both directions in one run)
- `SKIN_TONE_COLOR_JITTER=0.8` (fraction of RGB-torchvision training clips jittered)
- `SKIN_TONE_COLOR_JITTER_BRIGHTNESS/CONTRAST/SATURATION/HUE` (jitter strength)
- `SKIN_TONE_COLOR_JITTER_CONSISTENT=1` (one jitter draw per clip, applied to every
  frame — **use this** for any jitter condition; without it, parameters are resampled
  independently per frame, producing unintended flicker on top of the intended color
  shift, see `data/rgb.py` and `llm_reports/color_jitter_strength_experiment.md`)
- `SKIN_TONE_GRAYSCALE_PROB=0.2` (fraction of clips converted to grayscale;
  independent of `SKIN_TONE_COLOR_JITTER`, removes chroma entirely)
- `SKIN_TONE_PLANCKIAN_JITTER=0.8` (+ `_MIN_K`/`_MAX_K`/`_REFERENCE_K`, default
  3000/12000/6504K) — physically-motivated illuminant jitter, see [48] in the paper

**Cross-validation (used for every number in the paper):**

- `SKIN_TONE_SPLIT_MODE=cv` (vs. the older `original` fixed 6-train/4-held-out split;
  `run_action_bias_bench.sh` auto-appends `_cv` to `SKIN_TONE_OUT_ROOT` in this mode)
- `SKIN_TONE_CV_FOLDS=3`, `SKIN_TONE_CV_IDS=0,1,2,3,4,5,6,7,8,9` — 3-fold CV over the
  10 motion instances (cyclic blocks: fold0={0,1,2,3}, fold1={4,5,6,7}, fold2={8,9,0,1})
- `SKIN_TONE_SEEDS=0,1,2` — training seeds, layered on top of the CV folds (3 seeds
  for the main shortcut probe and frozen-feature probes; 2 seeds for the augmentation
  sweep, see below)

Example:

```bash
export SKIN_TONE_DATASET_ROOT=/data/skin_tone_actions/camera_far
export SKIN_TONE_MOTION_ROOT_DIR=/data/skin_tone_actions/camera_far_motion_zst
export SKIN_TONE_FLOW_TVL1_ROOT_DIR=/data/skin_tone_actions/camera_far_flow_tvl1_fast_npz
bash scripts/run_action_bias_bench.sh --preflight
```

Generated manifests and label CSVs are written under `benchmarks/skin_tone/generated/`.

---

## Experiments

Each subsection maps to one part of the paper, in the order results appear.

### 0. Synthetic-vs-real domain gap (Section 3.1)

Quantifies how far the synthetic clips sit from real Kinetics-400 footage of the
matching action, using DINOv2 embeddings: `gap_ratio = d_cross(real, synth) /
d_real_intra(real, real)`. Reuses the cached synthetic DINOv2 embeddings from
Experiment 2 below; embeds and caches real K400 clips on first run.

```bash
python scripts/measure_synthetic_real_gap.py --n_real 50 --device cuda
```

Writes `out/bias_analysis/synthetic_real_domain_gap.csv` (per-action `gap_ratio`) and
prints the mean/median/range over all 10 actions (paper reports mean ≈ 1.8).

<details>
<summary>Cluster job used to produce the paper's numbers</summary>

```bash
# submitted from Pascal/, not from inside motion_only_AR/
sbatch motion_only_AR/jobs/bias/run_synthetic_real_gap.sbatch
```
</details>

### 1. Biased fine-tuning stress test (Table 2, Figure 2 — Section 4.1)

The positive-control shortcut: skin tone is made predictive of a binary action label
during fine-tuning, then the model is evaluated with the tone assignment matched
(preserved) or shifted (reversed).

**Manifest generation + training** (`train_torchvision_rgb_probe.py`, driven by
`scripts/run_action_bias_bench.sh`; also trains the I3D-flow control via
`benchmarks/skin_tone/train_skin_tone_pytorch_i3d_flow_probe.py`):

<details>
<summary>Cluster job (3-fold CV × 3 seeds × 6 RGB backbones + I3D-flow control)</summary>

```bash
SKIN_TONE_OUT_ROOT=out/skin_tone_probe_v7 \
SKIN_TONE_SPLIT_MODE=cv \
SKIN_TONE_SEEDS=0,1,2 \
SKIN_TONE_MODALITIES=rgb_torchvision,flow_i3d_external \
SKIN_TONE_RGB_TORCHVISION_MODELS=mc3_18,mvit_v2_s,r2plus1d_18,r3d_18,s3d,swin3d_s \
SKIN_TONE_INCLUDE_REVERSED_PAIRS=1 \
sbatch motion_only_AR/jobs/bias/run_skin_tone_shortcut_probe.sbatch
```

This same job script can also run the downstream analysis steps below in one go
(`SKIN_TONE_RUN_ANALYSIS=1`) — see the script for the full toggle list
(`SKIN_TONE_ANALYSIS_ONLY`, `SKIN_TONE_RUN_SIGNIFICANCE`, etc.).
</details>

**Aggregation, Figure 2 heatmap, and per-pair flip CSVs**
(`benchmarks/skin_tone/analyze_skin_tone_swap_influence.py`):

```bash
python benchmarks/skin_tone/analyze_skin_tone_swap_influence.py \
  --root out/skin_tone_probe_v7_cv \
  --models all \
  --split_families seen,unseen \
  --out_dir out/skin_tone_probe_v7_cv_analysis
```

**Table 2 significance** — motion-instance-clustered Wilcoxon signed-rank test,
95% cluster-bootstrap CIs, and Benjamini–Hochberg correction (two separate families:
6 RGB backbones pooled at the model level, 4 tone-pairs within each backbone):

```bash
python benchmarks/skin_tone/summarize_skin_tone_significance.py \
  --root out/skin_tone_probe_v7_cv_analysis \
  --split_family unseen \
  --alpha 0.05
```

Reference-only script confirming exactly which files/columns feed Table 2 and that
the two BH families are disjoint (does not recompute anything):

```bash
python benchmarks/skin_tone/report_table2_correction_provenance.py \
  --analysis_root out/skin_tone_probe_v7_cv_analysis --split_family unseen
```

**Why celebrate/clap and lunge/cartwheel show the largest effect while yawn/fish is
flat** (Figure 2 discussion) — correlates each pair's pre-swap (matched) accuracy
against its swap effect, no new training runs:

```bash
python benchmarks/skin_tone/pair_difficulty_vs_shortcut.py \
  --root out/skin_tone_probe_v7_cv_analysis --split_family unseen
```

### 2. Frozen-feature probes + temporal self-similarity (Figures 3–4 — Section 4.2)

**Cache per-frame embeddings** (GPU; `tc_clip` needs its own apptainer, see above).
CLIP caches to `out/bias_analysis/clip_embeddings`, not the default
`out/bias_analysis/embeddings` — pass `--cache_dir` explicitly for CLIP everywhere
below, or downstream scripts silently load 0 clips for that model.

```bash
python scripts/skin_tone_bias_analysis.py --model <model> --frames 64
python scripts/skin_tone_bias_analysis.py --model clip --frames 64 --cache_dir out/bias_analysis/clip_embeddings
```

Models: `clip, siglip, dinov2, dinov3, eva02, hiera, hiera_large, vjepa2, tc_clip` +
the six RGB torchvision backbones. Can also be run for every model via
`jobs/bias/embed_all_models.sbatch`.

**Linear probe, 3-fold CV × 3 seeds** (`scripts/train_embedding_linear_probe.py`):

```bash
python scripts/train_embedding_linear_probe.py --model <model> --seeds 0,1,2 --folds 0,1,2 \
  --cache_dir out/bias_analysis/embeddings \
  --out_root out/linear_probes/skin_tone_probe_<model>_linear_cv \
  --subsample_frames 64   # 0 for models with fewer than 64 native frames (see job below)
```

<details>
<summary>Cluster job running this for every model with the correct per-model cache_dir / subsample_frames</summary>

```bash
sbatch motion_only_AR/jobs/bias/run_all_linear_probes_cv.sbatch
```
</details>

Build the flat probe-drop summary that the plotting script reads:

```bash
python scripts/build_probe_summary.py --suffix _linear_cv --out out/linear_probes/_probe_summary_cv.json
```

**Temporal self-similarity (SSM/RSA) diagnostic** — no model/GPU needed, runs
entirely on the cached NPZ embeddings:

```bash
python scripts/ssm_frobenius_analysis.py --model <model> --metric rsa
python scripts/ssm_frobenius_analysis.py --model clip --metric rsa --cache_dir out/bias_analysis/clip_embeddings
```

**Regenerate Figures 3–4:**

```bash
PROBE_SUMMARY_PATH=out/linear_probes/_probe_summary_cv.json \
PROBE_OUT_PREFIX=out/linear_probes/_probe_cv \
python scripts/plot_probe_ssm.py
```

Writes `_probe_cv_drop_by_model.pdf` (Fig. 3), `_probe_cv_vs_ssm.pdf` (Fig. 4),
`_probe_cv_ssm_by_pair.pdf` (per-action-pair breakdown, supplementary).

**Why TC-CLIP's Figure 3 and Figure 4 positions disagree** (Section 4.2 discussion) —
two diagnostics run in sequence:

1. Max-pool vs. mean-pool ablation — tests whether mean-pooling cancels a
   sign-flipping tone signal across frames (it doesn't: TC-CLIP's drop got smaller
   under max-pooling, not larger):

   ```bash
   python benchmarks/skin_tone/pooling_ablation_probe.py --models clip,tc_clip
   ```

2. Decision-direction alignment — for every same-clip tone swap, projects the
   embedding shift onto the probe's own decision direction and measures what
   fraction of the shift's squared magnitude lies along it vs. orthogonal to it.
   This is the test that actually explains the disagreement: TC-CLIP's tone-shift is
   *larger* in raw magnitude than CLIP's, but only ~4% of it lies along TC-CLIP's
   decision direction vs. ~14% for CLIP — the signal isn't weaker after fine-tuning,
   it just points somewhere the classifier isn't looking.

   ```bash
   python benchmarks/skin_tone/probe_direction_alignment.py --models clip,tc_clip
   ```

   Writes `out/skin_tone_probe_v7_cv_analysis/probe_direction_alignment.csv`.

Both are CPU-only, reuse the embeddings already cached in step 1 above, and take
roughly 20–30 minutes each (dominated by per-pair LBFGS logistic-regression
training, not I/O).

### 3. Color-jitter / grayscale / Planckian mitigation (Figure 5, Supplementary — Section 4.3)

Four conditions — weak jitter, strong jitter, strong jitter + grayscale, and
Planckian illuminant jitter — each retrained with the same 3-fold CV, 2 seeds
(seeds beyond the single original run were added specifically to address the
single-seed reviewer concern; the two seeds disagree on whether jitter helps, which
is the paper's actual finding here).

<details>
<summary>Cluster job (one condition per submission; conditions write to independent
output roots and are safely resumable/parallelizable)</summary>

```bash
SKIN_TONE_AUG_CONDITIONS=cjweak \
SKIN_TONE_SEEDS=0,1 \
sbatch motion_only_AR/jobs/bias/run_skin_tone_augmentation_sweep.sbatch

# repeat with SKIN_TONE_AUG_CONDITIONS=cjstrong / cjstronggray / planckian
```

Condition → jitter/grayscale/Planckian parameter mapping is hardcoded in the job
script (weak: jitter 0.8 @ ±0.2 brightness/contrast/saturation, hue ±0.05; strong:
same probability @ ±0.8/±0.8/±0.8/±0.2; strong+grayscale: strong + grayscale
p=0.2; planckian: jitter/grayscale off, Planckian p=0.8).
</details>

Per-condition analysis (repeat Experiment 1's `analyze_skin_tone_swap_influence.py`
+ `summarize_skin_tone_significance.py` against each condition's own output root),
then the summary figure:

```bash
python scripts/plot_augmentation_radar.py \
  --roots "none=out/skin_tone_probe_v7_cv" \
          "weak jitter=out/skin_tone_probe_v7_cjweak_cv" \
          "strong jitter=out/skin_tone_probe_v7_cjstrong_cv" \
          "strong jitter + grayscale=out/skin_tone_probe_v7_cjstronggray_cv" \
          "planckian=out/skin_tone_probe_v7_planckian_cv" \
  --baseline none \
  --out_dir out/skin_tone_probe_v7_augmentation_conditions \
  --out_name augmentation_radar_delta
```

Writes `augmentation_radar_delta.{pdf,png}` (Fig. 5): a baseline-anchored radar
where every model's no-augmentation level collapses onto one shared circle, so
deviation from the circle is the effect of that condition. Each `--roots` entry must
cover the same pair-tags across conditions, or the script silently restricts to the
shared (pair, seed) subset and prints the resulting unit count per model to the
console.

The supplementary per-condition, per-pair, per-direction grid figure is produced by
the same `analyze_skin_tone_swap_influence.py` flip-breakdown output as Experiment 1,
run against each condition's output root individually.

### 4. Real-world observational audit (Section 4, PA-HMDB51 + Kinetics dribbling)

Zero-shot (no fine-tuning) Kinetics-400-pretrained torchvision models, evaluated on
real video stratified by an existing/hand-assigned skin-tone label. Purely
observational — tone is confounded with performer, background, and recording
conditions here, which is exactly what motivates the synthetic audit; see each
script's docstring for the specific caveats.

**PA-HMDB51**, all classes with a clean Kinetics-400 counterpart:

```bash
python scripts/eval_pahmdb51_zero_shot.py --dry_run          # sanity-check class mapping first
python scripts/eval_pahmdb51_zero_shot.py --models r3d_18 --max_clips_per_class 2   # smoke test
python scripts/eval_pahmdb51_zero_shot.py                    # full run, all 6 models
```

<details>
<summary>Cluster job</summary>

```bash
sbatch motion_only_AR/jobs/bias/run_pa_hmdb51_zero_shot.sbatch
```
</details>

**PA-HMDB51 "dribble" scaffold** (145 clips) — the source-video-dependence check
referenced in the abstract/conclusion. Reports the naive clip-level test alongside a
background-controlled test that treats repeated clips from the same source video as
one unit, since 145 clips resolve to only 29 distinct source videos:

```bash
python scripts/eval_dribble_scaffold.py --dry_run
python scripts/eval_dribble_scaffold.py
```

<details>
<summary>Cluster job</summary>

```bash
sbatch motion_only_AR/jobs/bias/run_dribble_scaffold_zero_shot.sbatch
```
</details>

**Kinetics-400 "dribbling_basketball"** (806 clips, 806 distinct source videos — no
pseudo-replication correction needed, unlike the HMDB51 scaffold above; skin-tone
labels are hand-assigned from one representative frame per clip):

```bash
python scripts/eval_kinetics_dribble_zero_shot.py --dry_run
python scripts/eval_kinetics_dribble_zero_shot.py
```

<details>
<summary>Cluster job</summary>

```bash
sbatch motion_only_AR/jobs/bias/run_kinetics_dribble_zero_shot.sbatch
```
</details>

**Supplementary: pooled McNemar re-analysis of the zero-shot synthetic-audit
predictions**, using the same clustering/BH machinery as the main paper instead of
the per-pair Bonferroni comparison:

```bash
python scripts/zero_shot_pooled_mcnemar.py --root <zero_shot_predictions_root> --alpha 0.05
```

---

## Notes on filenames

- `_cv` suffix on an output root or figure name means the 3-fold cross-validation
  protocol was used (every result in the paper); its absence means the older fixed
  6-train/4-held-out split.
- `out/skin_tone_probe_v7*` is the naming convention used throughout this README and
  matches the roots referenced in `llm_reports/`; older `v5`/`v6` roots referenced in
  some job-script defaults predate the CV protocol and the current 4-condition
  augmentation naming (`cjweak`/`cjstrong`/`cjstronggray`/`planckian`) — don't reuse
  those defaults without overriding `SKIN_TONE_OUT_ROOT` / `SKIN_TONE_AUG_CONDITIONS`.
