# Real-data skin-tone check at scale: Kinetics-400 `dribbling_basketball`

Date: 2026-08-03.

## Why

The PA-HMDB51 real-data check (`llm_reports/pa_hmdb51_real_data_plan.md`,
`scripts/eval_dribble_scaffold.py`) found that a naive clip-level skin-tone-vs-accuracy
gap on HMDB51 "dribble" clips (n=145) disappeared once corrected for pseudo-replication:
the 145 clips were cut from only 29 distinct source videos (16 white-performer sources,
7 black-performer sources), so the naive test was treating repeated clips of the same
person/court/camera as independent evidence. With only 7 independent black-performer
sources, no real conclusion was possible either way.

Kinetics-400 is sourced one clip per distinct YouTube upload (in principle), so it may
not have this problem, and it has an order of magnitude more `dribbling_basketball`
clips than PA-HMDB51's dribble scaffold. If the videos are genuinely independent, this
is a much better-powered real-data check than PA-HMDB51 allowed.

## Step 1 — duplication/corruption check (done)

Parsed the standard Kinetics filename convention (`{youtube_id}_{start:06d}_{end:06d}.mp4`)
across `datasets/Kinetics/k400/{train,val}/dribbling_basketball/`:

- train: 756 clips, val: 50 clips, test: 0 (class not present in test split)
- **806 distinct YouTube source-video IDs, zero duplicates** (no video ID appears more
  than once, including across train/val)
- 0 zero-byte / corrupted files

Conclusion: unlike the HMDB51 dribble scaffold, every clip here is an independently
sourced video. No source-video aggregation/control step is needed - the direct
clip-level test is already the valid one. `scripts/eval_kinetics_dribble_zero_shot.py`
reflects this (no `per_source_*` aggregation, unlike `eval_dribble_scaffold.py`).

## Step 2 — skin-tone labeling

Kinetics-400 has no built-in performer attribute annotations (unlike PA-HMDB51), so
labels had to be produced from scratch. Options considered with the user: (a) tile
representative frames into contact sheets for human labeling, (b) an automated
face-detection + Individual Typology Angle pipeline, (c) direct LLM visual judgment
from extracted frames, (d) defer labeling and ship scaffolding only. The user chose
**(c): LLM (Claude) visual judgment**, scoped to `dribbling_basketball` only (not the
sibling `shooting_basketball`/`playing_basketball`/`dunking_basketball` classes, for
this pass).

Method:
1. Extract one representative frame per clip (mid-duration timestamp via
   `ffprobe`/`ffmpeg`, `scale=240:-2`).
2. Tile into 5x5 contact sheets (25 clips/sheet, ~33 sheets for 806 clips), each cell
   captioned with its integer index.
3. Visually review each sheet and assign one of
   `unidentifiable/white/yellow/black/mixed_skin_color` per clip (matching the
   PA-HMDB51 `ATTRIBUTE_CLASS_NAMES["skin_color"]` vocabulary for compatibility with
   the existing eval code), flagging low-confidence calls with a `review` flag.
4. Labels are written to
   `benchmarks/skin_tone/generated/kinetics_dribble_skin_tone_labels.json`
   (`{video_name: {split, skin_color, review, index}}`).

**Important limitation to carry into any writeup**: this is a single-frame,
single-reviewer (LLM, not human) visual call, not a validated annotation protocol like
PA-HMDB51's. It is noisier and more subjective than the PA-HMDB51 labels used
elsewhere in this project, and an LLM assigning race/skin-tone categories from a still
image is itself a debatable methodology - flag this explicitly rather than presenting
these labels with the same confidence as PA-HMDB51's. Treat this pass as a scoping/
pilot check on whether the effect the user hypothesized (darker skin tones more
prevalent and recognized *better* in `dribbling_basketball`, opposite direction from
the naive HMDB51 result) shows up at Kinetics scale - not a publication-ready
annotation.

## Step 3 — zero-shot eval

`scripts/eval_kinetics_dribble_zero_shot.py` runs the same 6 pretrained,
Kinetics-400-native torchvision video models (`r3d_18, mc3_18, r2plus1d_18, mvit_v2_s,
s3d, swin3d_s`) used throughout this project, no fine-tuning, against the labeled
clips. Reports a binary white-vs-non_white Fisher exact test and a full
5-group chi-square breakdown, both at the clip level (valid here, unlike the HMDB51
scaffold script, since clips are already independent - see Step 1).

## Labeling outcome

All 806 clips labeled (756 train + 50 val), from 33 contact sheets of 25 clips each
(one partial sheet of 6). Final distribution:

| skin_color | n |
|---|---|
| white | 345 |
| black | 240 |
| unidentifiable | 216 |
| mixed_skin_color | 5 |
| yellow | 0 |

280 clips (35%) flagged `review` (low-confidence call - usually small/distant
figures, motion blur, backlighting, or B&W footage). This is a much larger and
better-balanced white/black comparison than PA-HMDB51's post-correction 16 vs. 7
source videos.

**Known labeling inconsistency to disclose**: several clips during review looked
plausibly East/South Asian ("light-medium/asian" in the reasoning at the time) but
were recorded as `white` rather than `yellow`, since the PA-HMDB51 `yellow` category
wasn't being actively tracked as a separate bucket during most of the pass. As a
result **no clips ended up labeled `yellow`**, and the `white` count is likely
inflated by an unknown small number (rough guess: order 10-20 clips) that should
have been `yellow`. This only matters for interpretation if the eventual finding is
close to the significance boundary - it does not affect the black vs. white
comparison directly (yellow was never going to be counted as "black" either way) but
does mean "white" here is somewhat broader than a strict PA-HMDB51-style call would
give. Flag this rather than silently presenting the white count as clean.

## Full 6-model cluster result (job 276813, 2026-08-03)

All 6 models completed cleanly (0 skipped/failed clips) against all 806 labeled
clips. Pooled across models and clips (n = n_clips x 6):

| test | white | non_white (black+mixed) | stat | p |
|---|---|---|---|---|
| exact top-1 ("dribbling basketball") | 86.8% (n=2070) | 84.1% (n=1470) | Fisher | **0.025** |
| basketball-family tolerant match | 95.1% (n=2070) | 94.6% (n=1470) | Fisher | 0.535 |
| exact, full breakdown (white/black/mixed) | - | - | chi2 | **0.0068** |
| family, full breakdown | - | - | chi2 | 0.763 |

Per-model breakdown (white_acc - non_white_acc gap) on the strict exact-match metric:
r3d_18 +0.043, mc3_18 +0.030, r2plus1d_18 +0.011, mvit_v2_s +0.018, s3d +0.011,
swin3d_s +0.047 - **positive (white higher) in all 6 independently-trained
architectures**, a consistent direction even though individually small. On the
tolerant family metric the same per-model gaps are near-zero and flip sign
(+0.023, -0.004, -0.024, -0.003, +0.030, +0.007) - three positive, three negative,
consistent with the pooled non-significant p=0.535.

**Interpretation**: the "statistically significant" pooled result is driven almost
entirely by the strict top-1 metric, and a meaningful chunk of it appears to be
models confusing "dribbling basketball" with a sibling K400 label (shooting/playing/
dunking basketball) rather than gross misclassification - once that's counted as
"correct," the gap nearly vanishes. This mirrors the exact caveat already flagged in
`eval_dribble_scaffold.py`'s docstring for the HMDB51 version ("models often pick a
sibling basketball label").

**Statistical caveat on the pooled p-values**: the pooled test (n=2070 vs n=1470)
treats each (clip, model) pair as an independent Bernoulli trial, but the 6
predictions per clip are correlated (an easy clip tends to be right across most
models, a hard one wrong across most) - so p=0.025/0.0068 likely overstate the true
independent evidence. The more defensible signal is the **consistent sign** of the
gap across all 6 architectures under the strict metric, not the pooled p-value
itself.

**Direction relative to the original hypothesis**: opposite of what was
hypothesized going in (expected darker skin tones recognized *better*, given
assumed prevalence in Kinetics basketball footage). Instead, under the strict
metric, white-labeled clips are recognized slightly *more* reliably as exactly
"dribbling basketball" - same direction as the naive/uncorrected HMDB51 finding,
though far smaller in magnitude (1-5 points vs. HMDB51's large naive gap) and not
present at all once sibling-class confusions are tolerated.

## Strict sibling-excluded, per-model re-analysis (2026-08-04)

The pooled "significant" result above conflates two different kinds of "wrong":
(a) genuinely misclassified outside the whole basketball family, and (b) picked a
sibling K400 label (shooting/playing/dunking basketball) - a label-vocabulary
ambiguity, not evidence the model failed to recognize the action. (b) is 556/4836
(11.5%) of all predictions. Re-ran the comparison **excluding sibling predictions
entirely** (neither correct nor wrong - just dropped) and **per model, not pooled**,
using the project's existing manual Fisher-exact fallback
(`eval_pahmdb51_zero_shot.fisher_exact_2x2_fallback`, no scipy dependency needed):

| model | n kept (white/non_white) | white acc | non_white acc | gap | p (Fisher) | wrong counts (white/non_white) |
|---|---|---|---|---|---|---|
| r3d_18 | 311 / 216 | 0.904 | 0.875 | +0.029 | 0.320 | 30/311, 27/216 |
| mc3_18 | 309 / 211 | 0.977 | 0.981 | -0.004 | 1.000 | 7/309, 4/211 |
| r2plus1d_18 | 326 / 223 | 0.945 | 0.969 | -0.024 | 0.216 | 18/326, 7/223 |
| mvit_v2_s | 320 / 222 | 0.988 | 0.991 | -0.003 | 1.000 | 4/320, 2/222 |
| s3d | 313 / 227 | 0.885 | 0.855 | +0.030 | 0.300 | 36/313, 33/227 |
| swin3d_s | 318 / 216 | 0.981 | 0.972 | +0.009 | 0.558 | 6/318, 6/216 |
| pooled (6 models, correlated rows - caveat still applies) | 1897 / 1315 | 0.947 | 0.940 | +0.007 | 0.436 | - |

**No model reaches significance, and the direction is inconsistent** (3 models
white-higher, 3 non_white-higher) - unlike the unfiltered strict metric, which was
positive in all 6. That consistency-across-models was itself apparently an artifact
of sibling-label confusion correlating with skin-tone group, not a real recognition
gap. The pooled gap also shrank (0.007 vs. 0.027 unfiltered) and is far from
significant (p=0.436).

The user's prediction that per-model n would be "too small" for significance is
correct in practice - not so much for total n (all models keep 500-550 clips after
exclusion) but for the **wrong-outcome count**, which is what actually powers a
Fisher test: mvit_v2_s/mc3_18/swin3d_s have single-digit wrong counts per group (as
low as 2-7), since these models are already >97% accurate on this class once
sibling confusion is set aside. Only r3d_18 and s3d have enough genuine errors
(27-36 per group) to say anything at all, and neither reaches significance.

**Conclusion**: once sibling-label confusion is properly excluded (rather than
either "counted as correct" or "counted as wrong"), there is no detectable
per-model or pooled effect of skin tone on whether these 6 K400-pretrained models
correctly recognize "dribbling basketball" in real Kinetics footage - in either
direction. This is a genuine null result at reasonable power for the two models
with enough errors to test (r3d_18, s3d), and underpowered-by-construction for the
other four (they're just very good at this class, sibling-confusion aside). Given
the earlier PA-HMDB51 real-data check was also a null/small-effect result once
pseudo-replication was controlled, `dribbling_basketball` seems to be a class where,
if there is a real skin-tone-linked shortcut in these off-the-shelf K400 models, it
either isn't present or is too small to detect at n~800 real clips - worth explicit
framing as "no effect found here" rather than "no effect exists," and worth
comparing to whether a different class (or the synthetic Ctrl-A-Bias setup) shows a
larger, more testable effect.

## Deeper checks: train-split contamination + continuous-confidence test (2026-08-04)

Two things the binary accuracy analysis could not answer.

### Train-split contamination (the biggest methodological issue found)

**756 of the 806 clips (94%) are from the Kinetics-400 *train* split - exactly the data
these torchvision models were trained on.** Accuracy on the 50 held-out val clips is
substantially lower for half the models:

| model | train acc (n=756) | val acc (n=50) | drop |
|---|---|---|---|
| r3d_18 | 0.765 | 0.540 | **0.225** |
| mc3_18 | 0.839 | 0.680 | 0.159 |
| r2plus1d_18 | 0.856 | 0.640 | **0.216** |
| mvit_v2_s | 0.882 | 0.880 | 0.002 |
| s3d | 0.745 | 0.760 | -0.015 |
| swin3d_s | 0.857 | 0.800 | 0.057 |

So the near-ceiling accuracy that made the Fisher tests underpowered is *partly
memorisation*, not genuine generalisation. On truly held-out data these models sit at
54-88%, i.e. there would be plenty of errors to test - but only 50 val clips exist for
this class. Any future real-data check of this kind should be run on held-out data only.

### Continuous true-class probability (avoids the accuracy ceiling)

Binary correct/wrong throws away almost all information when a model is >97% accurate.
Testing the probability the model assigns to "dribbling basketball" instead lets every
clip contribute. Mann-Whitney U, white vs non_white, per model:

| model | white median | non_white median | p (sibling-excluded) | p (all clips) |
|---|---|---|---|---|
| r3d_18 | 0.963 | 0.949 | 0.116 | 0.072 |
| mc3_18 | 0.979 | 0.976 | 0.389 | 0.128 |
| r2plus1d_18 | 0.970 | 0.972 | 0.915 | 0.562 |
| mvit_v2_s | 0.864 | 0.863 | 0.713 | 0.419 |
| s3d | 0.981 | 0.977 | 0.722 | 0.907 |
| swin3d_s | 0.994 | 0.992 | 0.655 | 0.216 |

**No model shows a significant difference under the far more powerful continuous test
either.** This materially strengthens the null: it is no longer just "too few errors to
tell" - even using all 590 labeled clips per model as continuous observations, no
skin-tone effect is detectable.

### Was the sibling confusion itself skin-tone-linked?

| model | white sibling-rate | non_white sibling-rate | diff | p |
|---|---|---|---|---|
| r3d_18 | 9.9% | 11.8% | -2.0% | 0.499 |
| mc3_18 | 10.4% | 13.9% | -3.4% | 0.245 |
| r2plus1d_18 | 5.5% | 9.0% | -3.5% | 0.138 |
| mvit_v2_s | 7.2% | 9.4% | -2.1% | 0.363 |
| s3d | 9.3% | 7.3% | +1.9% | 0.455 |
| swin3d_s | 7.8% | 11.8% | -4.0% | 0.117 |

Non-white clips get a sibling label slightly more often in 5/6 models (2-4 points), none
individually significant. This is the mechanism behind the original "significant" pooled
result: it was not that models failed to recognise the action on darker-skinned
performers, but that they slightly more often chose *playing/shooting basketball* over
*dribbling basketball*. That is plausibly a scene/context difference (pickup-game vs.
instructional-drill footage) rather than a skin-tone effect per se, and it is exactly
the kind of confound an observational dataset cannot separate.

## Open follow-ups

- This pass only labeled/evaluated `dribbling_basketball`. Sibling classes
  (`shooting_basketball`, `playing_basketball`, `dunking_basketball`) exist in the same
  train/val layout and could extend the sample if the user wants more power or wants to
  check whether the effect (if any) is dribble-specific.
- Consider a second human-reviewed pass (or spot-check) over a sample of the LLM
  labels before treating results as more than a pilot signal, especially the 280
  clips flagged `review`.
- Full 6-model zero-shot eval needs the cluster (`jobs/bias/run_kinetics_dribble_zero_shot.sbatch`)
  - CPU-only local inference measured ~4s/clip for `r3d_18` alone (806 clips ≈ 55min
    for one model; all 6 would be several hours). MPS (Apple Silicon GPU) does not
    support the Conv3D ops these models use, so local Mac runs must pass `--device cpu`.
