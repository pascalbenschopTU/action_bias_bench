# Poster / paper reference stats

Verified numbers used while writing the ECCV poster and later paper-text passes,
with the exact source so they don't need to be re-derived. Re-verify before
reuse if the underlying `out/` run directories have been regenerated since the
date below.

Last assembled: 2026-08-19.

## Dataset scale

Dataset root: `/Volumes/MoDDL/Pascal/motion_only_AR/datasets/skin_tone_actions/camera_far`
(local mount; cluster path is `/tudelft.net/staff-umbrella/MoDDL/Pascal/motion_only_AR/datasets/skin_tone_actions/camera_far`).

- **4,380** total rendered `.mp4` clips (`find camera_far -iname "*.mp4" | wc -l`).
- **3 backgrounds**: `autumn_hockey`, `konzerthaus`, `stadium_01` (1,460 clips each).
- **20 actions** per background (identical set across backgrounds), e.g. cartwheel,
  celebrate, clap, dribble, golf, lunge, squat, tie, yawn, fish, ...
- **7 skin-tone variants** rendered per clip: `african, asian, hispanic, indian,
  middle_eastern, south_east_asian, white` (plus one `_initial` baseline render,
  which is not one of the 7 — 8 files on disk per clip in total). Source:
  `llm_reports/pa_hmdb51_real_data_plan.md:44`.
- **4 variants actually used** in the shortcut/probe studies: `african, asian,
  indian, white` — `benchmarks/skin_tone/schema.py:11` (`VARIANT_ORDER`).
- **Paper's display order** (light→dark, used in all tables/figure columns):
  White, Asian, Indian, African — note this differs from `schema.py`'s
  `VARIANT_ORDER`, which is not light-to-dark.
- **10 motion instances per action** (cyclic fold blocks) — `README.md:58`.

## Main swap-test ("positive-control shortcut test") experiment scale

Backing CSV: `out/skin_tone_probe_v7_cv_analysis/swap_pair_level_analysis.csv`
(199 MB; baseline "no augmentation" condition, read by
`scripts/plot_pair_direction_grid.py`).

- **151,200 rows** = one matched/shifted clip **pair** per row → **302,400**
  individual clip-level predictions total.
- Exact factorial design (verified by multiplying distinct per-column value
  counts): 7 models (`i3d_flow` + 6 RGB) × 10 `pair_tag`s (5 action pairs × 2
  directions) × 3 seeds × 3 folds × 2 split families (`seen`/`unseen`) × 4
  `variant_pair`s × 3 backgrounds × 10 `base_id`s = 151,200.
- **1,200 unique underlying clips** = 10 actions × 3 backgrounds × 10 base_ids ×
  4 tones (counted via unique `rel_path_matched` ∪ `rel_path_shifted`).
- **"7,200"**: filtering the CSV to one model + one seed (e.g. `r3d_18`, seed 0)
  gives exactly 7,200 rows — the number of matched/shifted clip-pair
  evaluations per trained model/seed (10 pair_tags × 3 folds × 2 split
  families × 4 variant_pairs × 3 backgrounds × 10 base_ids). Not written
  anywhere as a stated figure; derived but exactly reproducible.

## Linear-probe CV experiment (frozen-feature probes)

- Script: `scripts/train_embedding_linear_probe.py` (added `--folds` CLI arg,
  reuses manifests already built for the main CV experiment under
  `generated/manifests/skin_tone_camera_far_binary_fold{0,1,2}`).
- sbatch: `jobs/bias/run_all_linear_probes_cv.sbatch` (submit from `Pascal/`,
  not from `motion_only_AR/` — see `feedback_sbatch_submit_dir` memory). Job
  273001 completed cleanly.
- **3 seeds × 3 folds = 90 runs per model**, 15 models total.
- Aggregation: `scripts/build_probe_summary.py` — flattens per-model matched/
  unseen F1, plus a run-level 95% CI on the unseen drop (2.5/97.5 percentile
  across the 90 runs' per-run mean drop). Output:
  `out/linear_probes/_probe_summary_cv.json`.
- Precision vs. validity: a tight/zero-excluding CI on a probe's drop is only
  run-to-run *precision* — it says nothing about whether the probe learned the
  task at all (*validity*, judged by matched-F1 vs. chance). Example: `mc3_18`
  has CI=[0.002, 0.002] (excludes zero) despite matched-F1=0.602 (near chance
  for a binary task). Both diagnostics must be reported, never conflated.

## Probe-drop vs. SSM correlation

Computed via `scipy.stats.pearsonr` on the CV-aggregated probe summary against
the training-free temporal self-similarity (SSM) ranking.

- All 7 SSM-valid models: **r = 0.790, p = 0.0345, n = 7**.
- Excluding TC-CLIP: **r = 0.965, p = 0.0018, n = 6**.

## Table 2 — cluster-bootstrap 95% CIs (fine-tuned backbones, unseen split)

Source: `skin_tone_model_cluster_significance_unseen.csv`, 5,000-resample
percentile CI, Wilcoxon signed-rank over the 10 motion instances (the unit of
replication), Benjamini–Hochberg FDR correction. Sign convention: positive =
accuracy lost under the tone swap.

| Model | Drop (pp) | 95% CI | q |
|---|---|---|---|
| I3D-flow | 0.28 | [−0.50, 1.03] | 0.54 |
| R3D-18 | 0.19 | [−0.11, 0.58] | 0.50 |
| R(2+1)D-18 | 0.53 | [0.17, 0.94] | 0.075 |
| MViT-v2-S | 0.72 | [0.22, 1.42] | 0.047* |
| MC3-18 | 1.31 | [0.47, 2.33] | 0.031* |
| S3D | 1.36 | [0.67, 2.14] | 0.023* |
| Swin3D-S | 1.64 | [0.64, 2.94] | 0.023* |

4 of 6 RGB backbones significant (q<0.05) — this is the poster's "4/6" claim.
Largest drop: Swin3D-S at 1.6 pp — the poster's "1.6 pp" claim.

## HMDB51 `dribble` clip analysis (PA-HMDB51 extension)

- Extended PA-HMDB51 annotations from 6 to all **145** HMDB51 `dribble` clips,
  **23 source videos**.
- Clips treated independently (naive/wrong unit): light-tone acc 0.82 (n=450)
  vs. other 0.66 (n=414), **p = 8.1×10⁻⁸**.
- Grouped by source video (correct unit — repeated clips share performer/
  background/recording conditions): light-tone acc 0.67 (n=16 videos) vs.
  other 0.61 (n=7 videos), **p = 0.49**.

## Kinetics-400 "dribbling basketball" zero-shot replication

- Script: `scripts/eval_kinetics_dribble_zero_shot.py`.
- Headline (pooled over 6 RGB backbones) run:
  `out/kinetics_dribble_zero_shot/cluster_276813/skin_tone_stats.json`
  (n=806 clips = 806 distinct source videos — no clip/source-video
  aggregation needed here, unlike HMDB51; the only pseudo-replication is
  pooling 6 backbones' predictions per clip). Also documented in
  `appendix_revised.tex`, "Scaling to Kinetics-400" (~line 298), tables
  `tab:kinetics_dribble_pooled` / `tab:kinetics_dribble_permodel`.
- Group sizes: 345 White, 240 Black, 5 Mixed, 216 unidentifiable (excluded)
  source videos. "Other" = Black+Mixed = 245.
- **Exact-metric** (top-1 == "dribbling basketball"), naive: White 0.8676
  (n=2070) vs. Other 0.8408 (n=1470), Fisher exact **p = 0.0253**.
- **Family-metric** (credits any of 4 basketball sibling classes), corrected:
  White 0.9512 (n=2070) vs. Other 0.9463 (n=1470), Fisher exact **p = 0.5348**.
  This is the number to use as Kinetics' "corrected" result alongside HMDB51's
  source-video-grouped number — both represent the paper's considered/robust
  comparison, just correcting for different confounds (per-identity clip
  imbalance for HMDB51, sibling-class over-penalization for Kinetics).
- Siblings excluded entirely: White 0.95 vs. Other 0.94, p = 0.44.
- Per-backbone (siblings excluded), none individually significant: R3D-18
  0.90/0.88 p=0.32; MC3-18 0.98/0.98 p=1.00; R(2+1)D-18 0.95/0.97 p=0.22;
  MViT-v2-S 0.99/0.99 p=1.00; S3D 0.89/0.86 p=0.30; Swin3D-S 0.98/0.97 p=0.56.
- A separate single-model-only run
  (`out/kinetics_dribble_zero_shot/local_full_r3d18/skin_tone_stats.json`,
  r3d_18 only) exists but is **not** the paper's headline number — don't cite
  it as "the" Kinetics result.

## Paired flip-rate heatmap (`skin_tone_pair_heatmap_paired_flip_rate_testonly.pdf`)

Generated by `write_pair_heatmap_paired_flip_rate()` in
`benchmarks/skin_tone/summarize_skin_tone_robustness.py` (~lines 604–701),
called from `main()`.

- Rows = model, columns = `pair_tag` (action pair). Cell = `(b − c) / n` on the
  held-out (`unseen`) split only, diverging `coolwarm` colormap centered at 0.
- `b` = matched-correct & shifted-wrong; `c` = matched-wrong & shifted-correct;
  `n` = total joined pairs for that (model, pair_tag). A clip wrong under both
  tones contributes to neither.
- Each matched clip is joined to its *specific* shifted counterpart (same
  identity/background/action) — deliberately not the same thing as comparing
  two independently-averaged F1 splits, which can look biased purely from
  uneven per-identity clip counts (same motivation as the HMDB51 source-video
  grouping fix above).
- The same `b`/`c` counts feed `mcnemar_exact_from_counts()` in
  `summarize_skin_tone_significance.py` — the significance test behind Table 2.
- Backing per-cell CSV: `out/skin_tone_probe_v6_cv/skin_tone_raw_accuracy_by_direction_testonly.csv`.
  `i3d_flow` (motion-only, no color signal) sits at exactly 0 in 27/40 rows and
  within ±1 clip elsewhere — quantitative support for "shortcut barely
  exploited" even where Table 2 finds statistical significance.

## Poster hero figure: cartwheel skin-tone pair comparison

Script: `scripts/make_cartwheel_tone_pairs.py` — crops the white/african
`konzerthaus`/`cartwheel`/`base_id=0` renders directly from full-resolution
decoded frames (no manifest/cache), using a shared bounding box (union across
chosen frames, via background-subtraction) so all panels use an identical crop
window and the two tones are pixel-aligned within a panel.

Final invocation used for the current `out/frame_grids/cartwheel_konzerthaus_tone_pairs.{png,pdf}`:

```
python scripts/make_cartwheel_tone_pairs.py \
  --dataset_root .../datasets/skin_tone_actions/camera_far \
  --background konzerthaus --action cartwheel --base_id 0 \
  --frames 2,22,40 --gap_px 0 \
  --width_crop_left_frac 0.06 --width_crop_right_frac 0.145 \
  --out_dir out/frame_grids
```

- Frames 2/22/40 chosen (via a background-subtraction bbox scan across the
  whole clip, see the frame-height/center-y profile) to show, within a single
  cartwheel repetition (the clip contains ~2.5 reps): wind-up → peak inverted
  extension → mirrored recovery — in that chronological order.
- `--width_crop_left_frac`/`--width_crop_right_frac` are asymmetric on purpose:
  the shared bbox's geometric center (~x=794) doesn't match the actor's actual
  per-frame center (~x=723–799, avg ≈762), so more is trimmed from the right
  to recenter the actor without clipping the widest pose (frame 22's extended
  leg, raw bbox up to x≈1038).
- `--gap_px 0` puts the two tones flush against each other within a panel
  while `--panel_gap_px` (default 36) keeps whitespace between panels.
