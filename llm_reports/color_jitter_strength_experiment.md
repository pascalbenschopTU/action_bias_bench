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
Ran as SLURM job `271446`, completed cleanly. **Caveat found later:** this run
only covers the 5 base pair-tags, not the 5 reversed directions the original
`cj0p0/cj0p4/cj0p8` roots have (those were evidently produced by two separate
runs — one with the default `SKIN_TONE_ACTION_PAIRS`, one with the pairs
reversed — both writing into the same `OUT_ROOT`). So this run is 6 backbones
× 5 pair-tags × 3 seeds = 90 trained probes, not 180. See "Missing reversed
pairs" below for the fix.

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
the 5 base pair-tags × 3 seeds, since cj0p8_strong only has those 5; paired
t-test + Wilcoxon, Bonferroni-corrected across the 6 models):**

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

**Resolved: the compare-script vs robustness-summary discrepancy noted earlier
was the missing-pairs issue, not an aggregation bug.**
`compare_color_jitter_conditions.py` restricts each condition to the (pair, seed)
units *shared* with the other root being compared. Since `cj0p8_strong` only has
the 5 base pairs, the compare script's "reference_mean" for `cj0p8` in that
comparison is `cj0p8` computed on those same 5 pairs (e.g. swin3d_s = 0.061),
while `skin_tone_robustness_summary_f1_macro.json` reports `cj0p8`'s full
10-pair mean (0.032). Confirmed directly: `cj0p8` restricted to the 5 base
pairs gives exactly 0.061 for swin3d_s. Both numbers are correct for what they
measure; once the reversed pairs are backfilled for `cj0p8_strong` (below), the
compare script will use all 10 pairs and the two numbers will agree.

### Grayscale experiment (follow-up, completed)

Random grayscale (`SKIN_TONE_GRAYSCALE_PROB=0.5`, jitter disabled) removes chroma
entirely rather than perturbing it. Run as the `grayscale0p5` condition; outputs
in `out/skin_tone_probe_rgb_torchvision_v6_grayscale0p5/`. Same caveat as
`cj0p8_strong`: this run only has the 5 base pair tags, not the 5 reversed
directions, so all comparisons below are restricted to shared units (15 per
model). See "Missing reversed pairs" for the fix.

**Result (test-split drop, shared units, paired t vs no augmentation):**

| model | none (base-5) | grayscale 50% | Δ | raw p |
|---|---|---|---|---|
| mc3_18 | 0.021 | 0.006 | −0.015 | 0.040 |
| mvit_v2_s | 0.015 | 0.010 | −0.006 | 0.44 |
| r2plus1d_18 | 0.009 | 0.004 | −0.005 | 0.45 |
| r3d_18 | 0.004 | 0.000 | −0.004 | 0.16 |
| s3d | 0.015 | 0.035 | +0.020 | 0.23 |
| swin3d_s | 0.041 | 0.015 | −0.026 | 0.16 |

Matched-unseen F1 stays at 0.86–0.95 (same range as the other conditions), so
the probes still learn the task — the reductions are genuine robustness gains.

**Interpretation.** Grayscale is the first augmentation with a *consistent
direction* of effect: it reduces the swap drop for 5 of 6 backbones (mc3_18
nominally significant; r3d_18's drop goes to exactly zero), consistent with a
primarily chroma-carried shortcut. However, individual effects do not survive
Bonferroni correction at n=15 units, and s3d again moves the opposite way —
its drop increases under grayscale just as under strong jitter, making s3d the
backbone for which *every* tested augmentation increases skin-tone sensitivity.
Even chroma removal is therefore not a universal fix.

### Figures
`out/skin_tone_probe_rgb_torchvision_v6_analysis/augmentation_conditions/`:
- `augmentation_radar_delta.{pdf,png}` — the figure: baseline-anchored radar,
  radius = Δ drop vs no augmentation, bold circle = baseline; outside = worse.
  All conditions restricted to shared (pair, seed) units. Each model has a
  fixed color and a distinct marker shape.
- Generated by `scripts/plot_augmentation_radar.py` (the earlier small-multiples,
  absolute-overlay, and line-chart variants were consolidated into this one
  script and removed once the delta figure superseded them).

### Missing reversed pairs (cj0p8_strong and grayscale0p5)

Both `cj0p8_strong` and `grayscale0p5` only have the 5 base pair-tags
(`squat_vs_tie`, `clap_vs_celebrate`, `dribble_vs_golf`, `lunge_vs_cartwheel`,
`yawn_vs_fish`); the original `cj0p0/cj0p4/cj0p8` roots additionally have the
5 reversed-direction tags (`tie_vs_squat`, `celebrate_vs_clap`, `golf_vs_dribble`,
`cartwheel_vs_lunge`, `fish_vs_yawn`).

**Root cause:** `run_action_bias_bench.sh` reads pairs from `SKIN_TONE_ACTION_PAIRS`,
which defaults to only the 5 base directions. There is no code path — despite
what an earlier version of this report claimed — that reads
`SKIN_TONE_INCLUDE_REVERSED_PAIRS` (exported by the sbatch job but never
consumed anywhere); it is dead configuration. The original 10-pair roots must
have been produced by two separate submissions targeting the same `OUT_ROOT`,
the second with `SKIN_TONE_ACTION_PAIRS` set to the reversed list.

**Fix — before trusting any comparison against the 10-pair conditions, backfill
the reversed 5 pairs into both new roots** (safe to re-run: per-pair/seed/model
summaries already present are skipped):
```bash
export SKIN_TONE_ACTION_PAIRS="tie:squat,celebrate:clap,golf:dribble,cartwheel:lunge,fish:yawn"

# cj0p8_strong
export SKIN_TONE_COLOR_JITTER=0.8
export SKIN_TONE_COLOR_JITTER_BRIGHTNESS=0.8
export SKIN_TONE_COLOR_JITTER_CONTRAST=0.8
export SKIN_TONE_COLOR_JITTER_SATURATION=0.5
export SKIN_TONE_COLOR_JITTER_HUE=0.2
export SKIN_TONE_MODALITIES=rgb_torchvision
export SKIN_TONE_OUT_ROOT=motion_only_AR/models/ActionBiasBench/out/skin_tone_probe_rgb_torchvision_v6_cj0p8_strong
sbatch motion_only_AR/jobs/bias/run_skin_tone_shortcut_probe.sbatch

# grayscale0p5 (unset jitter strength vars first if reusing the shell)
export SKIN_TONE_COLOR_JITTER=0.0
export SKIN_TONE_GRAYSCALE_PROB=0.5
export SKIN_TONE_OUT_ROOT=motion_only_AR/models/ActionBiasBench/out/skin_tone_probe_rgb_torchvision_v6_grayscale0p5
sbatch motion_only_AR/jobs/bias/run_skin_tone_shortcut_probe.sbatch
```
After both complete, regenerate the radar figures (they'll no longer need the
shared-unit restriction) and re-run `compare_color_jitter_conditions.py` — the
compare script's reference means will then match the robustness-summary means
for `cj0p8` (see "Resolved" note above).

### Not yet tested
- Only one jitter-strength point (~2× baseline); no dose-response curve.
- Grayscale at other probabilities (e.g. 1.0), and grayscale + jitter combined.
