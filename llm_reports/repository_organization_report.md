# ActionBiasBench repository organization report

Date: 2026-07-05

Scope: repository-local review of `models/ActionBiasBench`, the related Slurm
jobs in `jobs/bias`, and the figure artifacts currently being used from
`models/ActionBiasBench/out`.

This report is intentionally non-invasive. It describes how to organize the
repository and reproducibility workflow better; it does not implement those
changes.

## Executive summary

The current codebase has the important pieces needed for reproducibility, but
they are not organized as one reproducible experiment pipeline. Training,
analysis, plotting, and report bundling are split across:

- Slurm jobs outside the `ActionBiasBench` repo: `jobs/bias/*.sbatch`
- first-stage benchmark runners: `scripts/run_action_bias_bench.sh`
- second-stage analysis scripts under `benchmarks/skin_tone/`
- one-off plotting scripts under `scripts/`
- ignored generated artifacts under `out/`
- LLM notes under `llm_reports/`

The main cleanup should be to make the skin-tone experiment a named,
config-driven pipeline with explicit stages:

1. generate manifests
2. train/evaluate shortcut probes
3. embed all foundation models
4. compute SSM metrics
5. train embedding linear probes
6. generate all paper/report figures
7. bundle canonical figures, tables, commands, environment, and provenance

The highest-value immediate fix is not a large refactor. It is to add a
single reproducibility README plus one canonical report/figure entry point
that generates every figure currently in use.

## Current repository shape

Top-level source layout:

```text
models/ActionBiasBench/
  README.md
  train.py, finetune.py, eval.py, config.py, dataset.py, ...
  cli/
  data/
  models/
  utils/
  benchmarks/skin_tone/
  configs/benchmarks/skin_tone/
  scripts/
  test/
  third_party/
  llm_reports/
  out/                 # ignored generated artifacts
  .cache/              # ignored local cache
```

Related Slurm jobs currently live outside the nested repo:

```text
jobs/bias/bias_test.sbatch
jobs/bias/embed_all_models.sbatch
jobs/bias/run_all_linear_probes.sbatch
jobs/bias/run_skin_tone_shortcut_probe.sbatch
```

Observed issue: because `ActionBiasBench` is its own Git repository, the Slurm
jobs in `jobs/bias` are not naturally versioned with the exact benchmark code
unless the parent repo is also tracked carefully. For reproducibility, the
canonical Slurm wrappers should live inside `models/ActionBiasBench/jobs/slurm/`,
or the parent jobs should become thin wrappers that call tracked scripts inside
`ActionBiasBench`.

## Current figure provenance

These are the figures you said you are currently using and the observed source
of each.

| Figure | Current location | Generator | Current automation status |
|---|---|---|---|
| skin-tone pair heatmap | `out/skin_tone_probe_rgb_torchvision_v6_cj0p0/skin_tone_pair_heatmap_f1_macro.pdf` | `benchmarks/skin_tone/summarize_skin_tone_robustness.py` | Generated automatically at the end of `scripts/run_action_bias_bench.sh`, if the correct `SKIN_TONE_OUT_ROOT` and augmentation env are used |
| swap significance heatmap | `out/skin_tone_probe_rgb_torchvision_v6_analysis_combined/skin_tone_variant_swap_significance_unseen.pdf` | `benchmarks/skin_tone/analyze_skin_tone_swap_influence.py` followed by `benchmarks/skin_tone/summarize_skin_tone_significance.py` | Not produced by the default benchmark run; can be produced by the analysis path in `run_skin_tone_shortcut_probe.sbatch` if env vars are set, or by `scripts/build_skin_tone_report.sh` |
| linear probe drop by model | `out/linear_probes/_probe_drop_by_model.pdf` | `scripts/plot_probe_ssm.py` | Manual or ad hoc; not called by `run_all_linear_probes.sbatch` |
| linear probe versus SSM | `out/linear_probes/_probe_vs_ssm.pdf` | `scripts/plot_probe_ssm.py` | Manual or ad hoc; depends on `out/linear_probes/_probe_summary.json` |
| linear probe SSM by pair | `out/linear_probes/_probe_ssm_by_pair.pdf` | `scripts/plot_probe_ssm.py` | Manual or ad hoc; depends on SSM CSVs under `out/bias_analysis/` |
| augmentation radar delta | `out/skin_tone_probe_rgb_torchvision_v6_analysis/augmentation_conditions/augmentation_radar_delta.pdf` | `scripts/plot_augmentation_radar.py` | Manual or ad hoc; not called by current Slurm jobs |

Important reproducibility gap: `_probe_summary.json` is read by
`scripts/plot_probe_ssm.py`, but I did not find a tracked script or Slurm job
that creates it. It appears to be a manually assembled derived artifact. That
should be replaced by a deterministic script that derives the summary from the
per-model linear-probe outputs.

## What is already good

- The skin-tone benchmark scripts mostly write source tables next to figures:
  CSV, JSON, and sometimes Markdown summaries are already available.
- `scripts/run_action_bias_bench.sh` is a useful central first-stage launcher.
- `scripts/build_skin_tone_report.sh` is a good start for a report bundle.
- `benchmarks/skin_tone/schema.py` has started centralizing split names,
  variants, swap mappings, colors, and stable seeds.
- `.gitignore` already excludes generated outputs, caches, checkpoints, assets,
  and generated manifests.
- The existing `skin_tone_visual_pipeline_trace.txt` is useful provenance
  documentation for the original skin-tone report figures.

## Main problems

### 1. There is no single canonical runbook

The README explains the general launcher, but not how to reproduce the exact
figures currently being used. The current figure set spans multiple roots:

```text
out/skin_tone_probe_rgb_torchvision_v6_cj0p0/
out/skin_tone_probe_rgb_torchvision_v6_analysis_combined/
out/linear_probes/
out/skin_tone_probe_rgb_torchvision_v6_analysis/augmentation_conditions/
```

A new user cannot tell which Slurm jobs to submit, which environment variables
to set, which analysis scripts remain manual, and which outputs are canonical.

### 2. Jobs live outside the nested repository

The canonical jobs are currently in `jobs/bias`, not under
`models/ActionBiasBench`. This makes the benchmark repository incomplete on its
own. It also makes it harder to version a job file with the exact script
version that produced a result.

### 3. Output roots mix raw runs, analysis outputs, debug artifacts, and report figures

Current `out/` contains at least:

```text
out/bias_analysis/
out/frame_grids/
out/hmdb_pahmdb51_zero_shot/
out/linear_probes/
out/skin_tone_probe_rgb_torchvision_v6_cj0p0/
out/skin_tone_probe_rgb_torchvision_v6_cj0p4/
out/skin_tone_probe_rgb_torchvision_v6_cj0p8/
out/skin_tone_probe_rgb_torchvision_v6_analysis/
out/skin_tone_probe_rgb_torchvision_v6_analysis_combined/
out/skin_tone_probe_rgb_torchvision_v6_cj0p8_strong/
out/skin_tone_probe_rgb_torchvision_v6_grayscale0p5/
out/ssm_debug/
```

The names are understandable historically, but not as a reproducible release
layout. `analysis`, `analysis_combined`, `linear_probes`, `bias_analysis`, and
`augmentation_conditions` are all separate conventions.

### 4. Some current figures are generated by one-off plotting scripts

The linear-probe/SSM figures and augmentation radar figures are currently not
first-class pipeline outputs. They are easy to lose because they are generated
by direct script calls rather than by a named stage.

### 5. Configuration is spread across shell env, scripts, and hard-coded constants

Examples:

- `run_skin_tone_shortcut_probe.sbatch` sets many `SKIN_TONE_*` env vars.
- `run_action_bias_bench.sh` has defaults for action pairs, variants, IDs,
  models, color jitter, grayscale, and output roots.
- `skin_tone_bias_analysis.py`, `ssm_frobenius_analysis.py`,
  `train_embedding_linear_probe.py`, and plotting scripts duplicate action
  pairs, variants, model lists, and split names.
- `SKIN_TONE_INCLUDE_REVERSED_PAIRS` is exported by the Slurm job but is not
  consumed by `run_action_bias_bench.sh`. To get reversed pairs, the actual
  mechanism is to run again with `SKIN_TONE_ACTION_PAIRS` set to the reversed
  list. This should become explicit config, not institutional memory.

### 6. Defaults do not match current figure roots

Examples:

- `README.md` mentions default output roots like `skin_tone_probe_seeded_v7`.
- `run_skin_tone_shortcut_probe.sbatch` defaults to
  `out/skin_tone_probe_seeded_v5`.
- The current figures use `skin_tone_probe_rgb_torchvision_v6_*`.

That is fine historically, but a reproducibility README should make the `v6`
run matrix explicit.

### 7. Container and model cache provenance is not first-class

The jobs reference Apptainer images such as:

```text
Video_LLM_testing/apptainer/huggingface_yolo_cuda.sif
Video_LLM_testing/apptainer/huggingface_yolo_clip_tcclip_cuda.sif
```

There is also an Apptainer `.def` and `.sif` under `out/`, which is ignored.
The binary SIF should remain outside Git, but the recipe or environment
specification should be tracked in a dedicated location such as `containers/`.

Also, `models/ActionBiasBench/models/huggingface/` currently holds downloaded
model artifacts inside a directory that otherwise looks like Python source.
That works, but it is confusing. A cache path under `.cache/` or an external
artifact root would be clearer.

## Recommended target layout

Keep the existing Python modules for now. The cleanup should focus first on
jobs, configs, outputs, and docs.

```text
models/ActionBiasBench/
  README.md
  docs/
    skin_tone_v6_reproduction.md
    output_layout.md
  configs/
    benchmarks/
      skin_tone/
        ...
    experiments/
      skin_tone_v6.yaml
      skin_tone_v6_reversed_pairs.yaml
    reports/
      skin_tone_v6.yaml
  jobs/
    slurm/
      skin_tone/
        01_train_shortcut_probe.sbatch
        02_embed_foundation_models.sbatch
        03_compute_ssm.sbatch
        04_train_embedding_linear_probes.sbatch
        05_make_figures.sbatch
  scripts/
    run_action_bias_bench.sh
    run_skin_tone_color_jitter_sweep.sh
    build_skin_tone_report.sh
    make_skin_tone_v6_figures.sh
  benchmarks/
    skin_tone/
      ...
  containers/
    huggingface_yolo_cuda.def
    tcclip_cuda.def
  llm_reports/
  out/
    runs/
    analysis/
    reports/
    debug/
```

Suggested generated output layout:

```text
out/
  runs/
    skin_tone/
      v6/
        cj0p0/
        cj0p4/
        cj0p8/
        cj0p8_strong/
        grayscale0p5/
    foundation_embeddings/
      v1/
        embeddings/
        clip_embeddings/
    linear_probes/
      v1/
        clip/
        dinov2/
        dinov3/
        ...
  analysis/
    skin_tone/
      v6/
        swap/
        color_jitter/
        augmentation_conditions/
    foundation_bias/
      v1/
        ssm/
  reports/
    skin_tone_v6/
      README.md
      manifest.json
      figures/
      tables/
      logs/
      commands/
  debug/
    frame_grids/
    ssm_debug/
```

This separates:

- raw training/eval runs: `out/runs/...`
- derived analysis: `out/analysis/...`
- final canonical figures/tables: `out/reports/...`
- disposable diagnostics: `out/debug/...`

## Recommended canonical pipeline

### Stage 1: train/evaluate skin-tone shortcut probes

Canonical job:

```text
jobs/slurm/skin_tone/01_train_shortcut_probe.sbatch
```

This should call `scripts/run_action_bias_bench.sh` with an explicit config or
env file. For `skin_tone_v6`, define every condition in one place:

```yaml
experiment: skin_tone_v6
dataset_root: ${SKIN_TONE_DATASET_ROOT}
modalities:
  - rgb_torchvision
  - flow_i3d_external
rgb_torchvision_models:
  - mc3_18
  - mvit_v2_s
  - r2plus1d_18
  - r3d_18
  - s3d
  - swin3d_s
seeds: [0, 1, 2]
action_pairs:
  - squat:tie
  - clap:celebrate
  - dribble:golf
  - lunge:cartwheel
  - yawn:fish
  - tie:squat
  - celebrate:clap
  - golf:dribble
  - cartwheel:lunge
  - fish:yawn
conditions:
  cj0p0:
    color_jitter: 0.0
    grayscale_prob: 0.0
  cj0p4:
    color_jitter: 0.4
    grayscale_prob: 0.0
  cj0p8:
    color_jitter: 0.8
    grayscale_prob: 0.0
  cj0p8_strong:
    color_jitter: 0.8
    color_jitter_brightness: 0.8
    color_jitter_contrast: 0.8
    color_jitter_saturation: 0.5
    color_jitter_hue: 0.2
  grayscale0p5:
    color_jitter: 0.0
    grayscale_prob: 0.5
```

The important part is that reversed pairs are explicit. Do not rely on
`SKIN_TONE_INCLUDE_REVERSED_PAIRS` unless the launcher actually implements it.

### Stage 2: embed foundation models

Canonical job:

```text
jobs/slurm/skin_tone/02_embed_foundation_models.sbatch
```

This corresponds to the current `jobs/bias/embed_all_models.sbatch`.

Outputs should go to a versioned root such as:

```text
out/runs/foundation_embeddings/v1/
```

The job should record:

- model list
- frames per model
- dataset root
- cache root
- SIF path
- Git commit
- command line

### Stage 3: compute SSM metrics

Canonical job:

```text
jobs/slurm/skin_tone/03_compute_ssm.sbatch
```

This is currently missing as a first-class job. It should call:

```bash
python scripts/ssm_frobenius_analysis.py --model <model> --metric rsa
```

for the SSM models used by `scripts/plot_probe_ssm.py`.

Expected outputs:

```text
out/analysis/foundation_bias/v1/ssm/ssm_rsa_clip.csv
out/analysis/foundation_bias/v1/ssm/ssm_rsa_dinov2.csv
...
```

### Stage 4: train embedding linear probes

Canonical job:

```text
jobs/slurm/skin_tone/04_train_embedding_linear_probes.sbatch
```

This corresponds to the current `jobs/bias/run_all_linear_probes.sbatch`.

Recommended improvement: either pass `--run_analysis` for each model or run a
separate aggregation step after all models finish. The pipeline should generate
one deterministic probe summary table and JSON from the per-model outputs. Do
not keep `_probe_summary.json` as a manually maintained artifact.

Suggested deterministic output:

```text
out/analysis/foundation_bias/v1/linear_probe_summary.csv
out/analysis/foundation_bias/v1/linear_probe_summary.json
```

### Stage 5: generate all report figures

Canonical job:

```text
jobs/slurm/skin_tone/05_make_figures.sbatch
```

or a local command:

```bash
bash scripts/make_skin_tone_v6_figures.sh
```

This stage should generate every figure currently being used:

```text
skin_tone_pair_heatmap_f1_macro.pdf
skin_tone_variant_swap_significance_unseen.pdf
_probe_drop_by_model.pdf
_probe_vs_ssm.pdf
_probe_ssm_by_pair.pdf
augmentation_radar_delta.pdf
augmentation_radar_overlay.pdf
```

The existing `scripts/build_skin_tone_report.sh` should either be extended to
cover the linear-probe/SSM and augmentation-radar figures, or a new
`make_skin_tone_v6_figures.sh` should call it and then run the missing figure
scripts.

### Stage 6: bundle final report artifacts

Final output should be:

```text
out/reports/skin_tone_v6/
  README.md
  manifest.json
  figures/
    skin_tone_pair_heatmap_f1_macro.pdf
    skin_tone_variant_swap_significance_unseen.pdf
    probe_drop_by_model.pdf
    probe_vs_ssm.pdf
    probe_ssm_by_pair.pdf
    augmentation_radar_delta.pdf
    augmentation_radar_overlay.pdf
  tables/
    skin_tone_pair_robustness_summary_f1_macro.csv
    skin_tone_variant_swap_significance_unseen.csv
    skin_tone_significance_summary.csv
    linear_probe_summary.csv
    ssm_model_summary.csv
  commands/
    01_train_shortcut_probe.sh
    02_embed_foundation_models.sh
    03_compute_ssm.sh
    04_train_embedding_linear_probes.sh
    05_make_figures.sh
  logs/
```

The bundle should be the only location used by a paper, thesis, or slide deck.
Raw `out/runs/...` directories should remain source data, not presentation
targets.

## Recommended README structure

Add a focused reproduction guide, for example:

```text
docs/skin_tone_v6_reproduction.md
```

Suggested sections:

1. Purpose
   - one paragraph explaining what the skin-tone benchmark measures

2. Required data
   - `SKIN_TONE_DATASET_ROOT`
   - `SKIN_TONE_RGB_TORCHVISION_ROOT_DIR`
   - `SKIN_TONE_FLOW_TVL1_ROOT_DIR`
   - expected dataset substructure

3. Required containers
   - main SIF path
   - TC-CLIP SIF path
   - how to rebuild from tracked `.def` files, if available

4. Run matrix
   - models
   - seeds
   - action pairs, including reversed pairs
   - augmentation conditions

5. Commands
   - Slurm command for each stage
   - local/non-Slurm command when feasible

6. Expected outputs
   - raw run roots
   - analysis roots
   - final report bundle root

7. Figure provenance
   - one row per canonical figure
   - generator script
   - input tables/files
   - output path

8. Troubleshooting
   - missing reversed pairs
   - missing `_probe_summary.json`
   - missing SSM CSVs
   - cache/model download issues
   - path assumptions around `SLURM_SUBMIT_DIR`

## Specific cleanup recommendations

### A. Move or mirror Slurm jobs into ActionBiasBench

Recommended:

```text
models/ActionBiasBench/jobs/slurm/skin_tone/
```

Keep the old parent paths temporarily if convenient, but make them wrappers:

```bash
sbatch motion_only_AR/models/ActionBiasBench/jobs/slurm/skin_tone/01_train_shortcut_probe.sbatch
```

This makes the benchmark repo self-contained.

### B. Make the report pipeline config-driven

Create:

```text
configs/reports/skin_tone_v6.yaml
```

containing all input roots, labels, baseline condition, model lists, split
family, and output bundle location. Then plotting scripts can accept
`--config configs/reports/skin_tone_v6.yaml`.

This removes hard-coded paths like:

```python
Path("out/linear_probes/_probe_summary.json")
Path("out/bias_analysis/ssm_rsa_<model>.csv")
fig.savefig("out/linear_probes/_probe_vs_ssm.pdf")
```

### C. Generate `_probe_summary.json` deterministically

Current problem:

```text
scripts/plot_probe_ssm.py reads out/linear_probes/_probe_summary.json
```

but the current Slurm jobs do not create that file.

Fix:

- add `scripts/build_linear_probe_summary.py`, or
- extend `train_embedding_linear_probe.py --run_analysis`, or
- teach `plot_probe_ssm.py` to read each `shortcut_probe_summary.csv`

Preferred output:

```text
out/analysis/foundation_bias/v1/linear_probe_summary.{csv,json}
```

### D. Promote augmentation radar generation into the report stage

Current script (consolidated from three earlier variants — small-multiples,
absolute-overlay, and line-chart views — once the baseline-anchored delta
radar superseded all of them):

```text
scripts/plot_augmentation_radar.py
```

Recommended:

- keep it, but call it from `make_skin_tone_v6_figures.sh`
- write outputs under `out/analysis/skin_tone/v6/augmentation_conditions/`
- copy canonical PDFs into `out/reports/skin_tone_v6/figures/`

### E. Centralize skin-tone constants

Continue expanding:

```text
benchmarks/skin_tone/schema.py
```

Move repeated constants there:

- action pairs
- reversed action pairs
- background names
- ID splits
- variant groups
- split names
- model display names
- model family names
- plotting colors

Scripts that currently duplicate this information should import it.

### F. Make output validation noisy

`aggregate_skin_tone_probe.py` already has `load_rows_with_report()`. Use that
report in CLI output and fail, or at least warn, when:

- zero summaries are accepted
- expected models are missing
- expected pair count is wrong
- expected seed count is wrong
- expected reversed pairs are missing
- prediction CSVs required for significance are missing

This would catch partial roots such as a condition with only the five base
pairs when the report expects ten directional pairs.

### G. Track environment and command provenance

Each stage should write a small manifest:

```json
{
  "experiment": "skin_tone_v6",
  "stage": "train_shortcut_probe",
  "git_commit": "...",
  "dirty_git_status": "...",
  "command": "...",
  "slurm_job_id": "...",
  "sif_path": "...",
  "dataset_roots": {},
  "env": {},
  "started_at": "...",
  "finished_at": "..."
}
```

The final report bundle should aggregate those manifests.

### H. Keep containers and caches out of source paths

Recommended:

- tracked recipes: `containers/*.def`
- binary SIFs: external artifact storage or ignored `out/containers/`
- Hugging Face cache: `.cache/huggingface` or external cache root
- downloaded model snapshots should not live under the Python `models/`
  package unless there is a strong reason

### I. Clarify old, smoke, and debug scripts

Current `bias_test.sbatch` appears to be a CLIP embedding smoke/initial run, not
a canonical stage for the current figure set. Rename or document it as:

```text
jobs/slurm/skin_tone/smoke_embed_clip.sbatch
```

Move frame grids and SSM debug outputs under:

```text
out/debug/
```

### J. Leave raw historical outputs in place until the new bundle reproduces them

Do not immediately move or delete existing `out/skin_tone_probe_rgb_torchvision_v6_*`
directories. First create a new report bundle that reads them in place and
reproduces the current figures. After that, optionally reorganize future outputs
under the new layout.

## Suggested implementation order

### Pass 1: documentation and no behavior change

1. Add `docs/skin_tone_v6_reproduction.md`.
2. Add `configs/reports/skin_tone_v6.yaml` listing current roots and canonical
   figure outputs.
3. Add a `make_skin_tone_v6_figures.sh` script that calls existing scripts with
   current roots.
4. Write all final figures into `out/reports/skin_tone_v6/figures/`.

This pass should be fast and should not risk changing results.

### Pass 2: make manual artifacts reproducible

1. Add a deterministic builder for the linear probe summary.
2. Add a first-class SSM computation job.
3. Make `plot_probe_ssm.py` accept input/output arguments instead of fixed
   `out/...` paths.
4. Add validation for expected models, seeds, pairs, and inputs.

### Pass 3: consolidate jobs

1. Copy canonical Slurm jobs into `models/ActionBiasBench/jobs/slurm/skin_tone/`.
2. Convert parent `jobs/bias/*.sbatch` into wrappers or archive them.
3. Ensure every job writes a stage manifest.

### Pass 4: clean output conventions for future runs

1. New runs go under `out/runs/...`.
2. New analyses go under `out/analysis/...`.
3. New report bundles go under `out/reports/...`.
4. Debug images go under `out/debug/...`.

### Pass 5: refactor internals only after the pipeline is stable

1. Centralize constants in `benchmarks/skin_tone/schema.py`.
2. Split data collection from plotting.
3. Replace implicit directory parsing with a clear manifest where practical.
4. Rename `test/` to `tests/` only if desired; this is low priority.

## Minimal README command outline

This is the kind of command outline that should appear in the reproduction
README. Exact paths should be filled from your cluster environment.

```bash
# From the parent directory that contains motion_only_AR.
cd /Volumes/MoDDL/Pascal

export SIF_PATH=Video_LLM_testing/apptainer/huggingface_yolo_cuda.sif
export TC_CLIP_SIF_PATH=Video_LLM_testing/apptainer/huggingface_yolo_clip_tcclip_cuda.sif
export SKIN_TONE_DATASET_ROOT=motion_only_AR/datasets/skin_tone_actions/camera_far
export SKIN_TONE_RGB_TORCHVISION_ROOT_DIR=$SKIN_TONE_DATASET_ROOT
export SKIN_TONE_FLOW_TVL1_ROOT_DIR=motion_only_AR/datasets/skin_tone_actions/camera_far_flow_tvl1_npz

# 1. Train/evaluate the skin-tone shortcut probes for each condition.
sbatch motion_only_AR/models/ActionBiasBench/jobs/slurm/skin_tone/01_train_shortcut_probe.sbatch

# 2. Embed all foundation models.
sbatch motion_only_AR/models/ActionBiasBench/jobs/slurm/skin_tone/02_embed_foundation_models.sbatch

# 3. Compute SSM metrics from cached embeddings.
sbatch motion_only_AR/models/ActionBiasBench/jobs/slurm/skin_tone/03_compute_ssm.sbatch

# 4. Train embedding linear probes.
sbatch motion_only_AR/models/ActionBiasBench/jobs/slurm/skin_tone/04_train_embedding_linear_probes.sbatch

# 5. Generate and bundle all figures.
sbatch motion_only_AR/models/ActionBiasBench/jobs/slurm/skin_tone/05_make_figures.sbatch
```

Expected final artifacts:

```text
motion_only_AR/models/ActionBiasBench/out/reports/skin_tone_v6/figures/
motion_only_AR/models/ActionBiasBench/out/reports/skin_tone_v6/tables/
motion_only_AR/models/ActionBiasBench/out/reports/skin_tone_v6/manifest.json
motion_only_AR/models/ActionBiasBench/out/reports/skin_tone_v6/README.md
```

## Bottom line

The repository does not need a full rewrite. It needs a reproducibility layer:
tracked Slurm wrappers, a named experiment config, deterministic builders for
currently manual artifacts, a single figure/report command, and a clear
`out/runs`, `out/analysis`, `out/reports`, `out/debug` separation.

The first practical milestone should be:

```text
out/reports/skin_tone_v6/
```

containing every currently used PDF plus the exact tables, commands, and
provenance needed to regenerate them.
