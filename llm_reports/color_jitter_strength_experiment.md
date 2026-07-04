## Color-jitter strength experiment (cj0p8_strong)

### Motivation
The original jitter sweep (`cj0p0`, `cj0p4`, `cj0p8`) only varied *how often* ColorJitter
is applied per clip (0%, 40%, 80%), at a fixed strength (brightness/contrast 0.4,
saturation 0.2, hue 0.1). This leaves open the question a reviewer would ask: maybe
the augmentation just wasn't strong enough. This experiment adds a second knob —
jitter *strength* — while keeping application frequency fixed at 80%, to test
whether stronger appearance perturbation is a more effective mitigation.

### How it's run

**Code changes** (all backward-compatible; omitting the new flags reproduces the
original augmentation exactly):
- `data/rgb.py`: `RGBVideoClipDataset` now takes `color_jitter_brightness/contrast/
  saturation/hue` (defaults 0.4/0.4/0.2/0.1 — the original values).
- `scripts/train_torchvision_rgb_probe.py`: exposes `--color_jitter_brightness/
  contrast/saturation/hue` CLI flags, threaded into the dataset.
- `scripts/run_action_bias_bench.sh`: reads `SKIN_TONE_COLOR_JITTER_{BRIGHTNESS,
  CONTRAST,SATURATION,HUE}` env vars and passes them to the torchvision probe
  training command.
- `jobs/bias/run_skin_tone_shortcut_probe.sbatch`: exports those vars (defaulting
  to the originals) and echoes them into the job log for traceability.

**Command used** (from the project root, i.e. the directory containing
`motion_only_AR/`):
```bash
export SKIN_TONE_COLOR_JITTER=0.8                 # 80% of clips (unchanged)
export SKIN_TONE_COLOR_JITTER_BRIGHTNESS=0.8      # was 0.4
export SKIN_TONE_COLOR_JITTER_CONTRAST=0.8        # was 0.4
export SKIN_TONE_COLOR_JITTER_SATURATION=0.5      # was 0.2
export SKIN_TONE_COLOR_JITTER_HUE=0.2             # was 0.1
export SKIN_TONE_MODALITIES=rgb_torchvision
export SKIN_TONE_OUT_ROOT=motion_only_AR/models/ActionBiasBench/out/skin_tone_probe_rgb_torchvision_v6_cj0p8_strong

sbatch motion_only_AR/jobs/bias/run_skin_tone_shortcut_probe.sbatch
```
Ran as SLURM job `271446`, completed cleanly (all 6 backbones × 10 pair-tags ×
3 seeds = 180 trained probes).

### Where it's stored
- Training/eval outputs: `out/skin_tone_probe_rgb_torchvision_v6_cj0p8_strong/`
  (same structure as the other `cj0pX` roots — per-model, per-pair, per-seed
  checkpoints and `summary_*.json` files, plus `skin_tone_robustness_summary_f1_macro.json`,
  `skin_tone_pair_heatmap_f1_macro.pdf`, etc.)
- Job log: `logs/skin_tone_bias/out/run_skin_tone_shortcut_probe_271446.out`
- Paired significance test (cj0p8 vs cj0p8_strong) output: `/tmp/cj_strong_compare/`
  (`color_jitter_comparison.{csv,json,pdf}`, `color_jitter_robustness_checks.{csv,json}`,
  `color_jitter_pair_heatmap.pdf`) — generated via
  `benchmarks/skin_tone/compare_color_jitter_conditions.py --roots cj0p8=... cj0p8_strong=...`

### What it means

**Probes still learn the task.** Matched-unseen F1 stays in the same 0.85–0.96
range as the milder jitter conditions, so the stronger augmentation didn't break
training — the drop comparisons below are meaningful.

**Test-split skin-tone drop (matched − shifted F1, unseen identities):**

| model | cj0p0 | cj0p4 | cj0p8 | cj0p8_strong |
|---|---|---|---|---|
| mc3_18 | 0.018 | 0.018 | 0.023 | 0.010 |
| mvit_v2_s | 0.012 | 0.017 | 0.015 | 0.012 |
| r2plus1d_18 | 0.006 | 0.011 | 0.010 | 0.022 |
| r3d_18 | 0.004 | 0.009 | 0.016 | 0.012 |
| s3d | 0.009 | 0.018 | 0.024 | 0.062 |
| swin3d_s | 0.031 | 0.014 | 0.032 | 0.008 |

**Paired significance (cj0p8 vs cj0p8_strong, n=15 shared units per model —
10 pair-tags × avg over seeds where applicable, paired t-test + Wilcoxon,
Bonferroni-corrected across the 6 models):**

| model | Δ (strong − cj0p8) | raw p (t) | Bonferroni p |
|---|---|---|---|
| swin3d_s | −0.053 (improved) | 0.045 | 0.27 (n.s.) |
| s3d | +0.024 (worse) | 0.13 | n.s. |
| mc3_18 | −0.009 | 0.41 | n.s. |
| mvit_v2_s | −0.003 | 0.66 | n.s. |
| r2plus1d_18 | +0.006 | 0.71 | n.s. |
| r3d_18 | +0.000 | 0.98 | n.s. |

**Conclusion.** Neither the frequency sweep nor a ~2× strength increase at fixed
80% frequency produces a mitigation that survives multiple-comparison correction.
The two largest raw effects (swin3d_s improves, s3d worsens) point in opposite
directions — the same qualitative signature as the original R3D-18 exception in
the frequency sweep. This strengthens the "not a reliable mitigation" claim: it
now holds across two independent axes of the same augmentation family (how often
vs. how strong), not just one.

**Known inconsistency to resolve before the numbers go in a table/figure:** the
`compare_color_jitter_conditions.py` script and `skin_tone_robustness_summary_f1_macro.json`
report different reference means for the same condition (e.g. swin3d_s cj0p8
test-split drop: 0.032 in the robustness summary vs 0.061 as the compare script's
"reference_mean"). Both claim n=15 shared units. The discrepancy is in how the
two pipelines aggregate seed/pair units — not yet root-caused. Same directional
conclusion either way, but pick one aggregation before finalizing a paper table.

### Not yet tested
- Only one "strength" point was tried (~2× baseline). No dose-response curve.
- No test of a qualitatively different augmentation (e.g. random grayscale,
  which removes chroma entirely rather than perturbing it) — see the grayscale
  experiment added alongside this report.
