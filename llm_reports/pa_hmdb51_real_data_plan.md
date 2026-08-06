# Real-data skin-tone bias check via PA-HMDB51 — research notes + plan

Date: 2026-07-05. No code changed yet — this is a scoping doc.

## 0. Literature check — has anyone else done this?

Web search turned up one important, directly-relevant hit and one loosely-relevant one.
No one appears to have run a real-video (HMDB51/Kinetics-style) skin-tone-vs-accuracy
audit the way you're planning — this looks like a genuinely open gap, not a repeat of
someone else's study.

- **"Identifying Ethical Biases in Action Recognition Models"** (arXiv 2604.17971,
  April 2026) — **authors: Ana Băltărețu, Pascal Benschop, Jan van Gemert, TU Delft.**
  This is your own published paper — it looks like the Ctrl-A-Bias thesis work
  (`[[project_actionbiasbench]]` memory: "unpublished master's thesis, merge before
  publishing") is now public. Worth reconciling: the memory note about talking to the
  thesis author/advisor before publishing may now be moot, or the terms may have
  changed — worth a quick sanity check that this is the finalized/agreed version.
  Confirms the abstract-level finding already in memory (e.g. the cartwheel-vs-capoeira
  skin-tone flip example) but is still synthetic/BEDLAM-only — it does not answer the
  real-data question either, so the gap you're pointing at is real.
- **"Investigating Racial and Skin Tone Biases in Automated Classification of
  Teachers' Activities in Classroom Videos"** (AIED 2025, Springer) — the closest real-
  video precedent found: real classroom video, checks whether an activity classifier's
  accuracy differs by teacher race/skin tone, and whether training-set skin-tone
  balance changes that. Finding: **no significant bias under good classroom lighting**;
  training-set balance had a small, inconsistent effect; poor lighting hurt
  darker-skinned subjects more. Different domain (classroom instructional activities,
  not sports/HMDB-style actions) but useful as a methodological precedent for how to
  frame a real-data null/small-effect result honestly.
- Broader hits (Kinetics-400 gender-imbalance analysis, Mimetics background-bias
  benchmark, "Mitigating Representation Bias in Action Recognition" / SMAD, ALBAR) are
  about **scene/background/gender** bias in real action datasets, not skin tone — related
  motivation, not the same question.

Net: real-data skin-tone-vs-accuracy for action recognition looks like an actual gap
worth filling, and the closest precedent (classroom study) found a small/near-null
effect — useful as a prior to calibrate expectations, not a reason not to try.

## 1. Is there an existing real-data experiment?

No. Everything documented so far (probe, SSM Frobenius, linear probes, counterfactual
flip analysis — see `EXPERIMENT_NOTES.md` and prior `llm_reports/*.md`) runs on the
**synthetic Ctrl-A-Bias dataset** (BEDLAM/SMPL-X renders, 7 skin tones, matched vs.
shifted pairs). There is no real-video skin-tone-vs-accuracy experiment in
ActionBiasBench yet.

There **is** a PA-HMDB51 integration already in the sibling project
`appearance_free_cross_domain_action_recognition/privacy/`, but it answers a
different question: it trains an *attacker* model to **predict** `skin_color` (and
gender/face/nudity/relationship) from RGB/motion/flow video — i.e. privacy leakage,
not "does an action classifier's accuracy vary by skin tone." Results live in
`privacy/out/pa_hmdb51_privacy_cv/skin_color/`. Useful as infra/precedent, not as the
answer to your question.

## 2. What's actually available for the real-data question

- **Real HMDB51 videos are on disk**: `/Volumes/MoDDL/Pascal/motion_only_AR/datasets/hmdb51/<action_class>/*.avi` (51 class folders, standard HMDB51 layout).
- **PA-HMDB51 skin-tone labels already parsed**: `privacy/pa_hmdb51.py::load_pa_hmdb51_records()` reads the JSONs at `privacy/data/pa_hmdb51/PrivacyAttributes/*.json` (per-segment labels, majority-vote already implemented) and returns one `skin_color` label per video, values:
  - `unidentifiable` (22), `white` (352), `yellow` (69), `black` (61), `mixed_skin_color` (11) — **515 videos total**, from a real prior run's `dataset_metadata.json`.
  - This is the standard PA-HMDB51 annotation subset (~10 clips/class across all 51 HMDB51 classes).
- **Official HMDB51 train/test splits** (`val1.txt`/`val2.txt`/`val3.txt`) and the **51-class label CSV** already exist under `appearance_free_cross_domain_action_recognition/tc-clip/datasets_splits/hmdb_splits/` and `tc-clip/labels/hmdb_51_labels.csv`.
- **A zero-shot eval path that needs no training already exists in ActionBiasBench**: `cli/eval_cli.py` evaluates any `root/class_name/*.{mp4,avi,...}` folder against a **CLIP text bank** (docstring literally says "No retraining required"), with three logged modes per run — `motion_only` / `rgb_model` (your trained branch), `clip_rgb_only` (pretrained CLIP vision encoder), and an ensemble. It already emits `per_class_*.csv` and confusion matrices. Config knobs (`configs/benchmarks/skin_tone/eval/common.toml`) take an arbitrary `root_dir` + `manifests` + label source, so pointing it at HMDB51 clips with the HMDB51 label bank instead of the 10 synthetic action-pair classes should be a small config/manifest change, not new model code.

Net: a real-data correlational check is very feasible and, per your direction below,
**both phases can be training-free.**

## 3. Proposed plan

**Phase A — zero-shot foundation models (CLIP/SigLIP text-bank), no training:**
1. Build a manifest restricted to the 515 PA-HMDB51-annotated clips (rel_path + HMDB51 class id), reusing `load_pa_hmdb51_records()` for the video list and `tc-clip/labels/hmdb_51_labels.csv` for the 51-way text bank.
2. Run the existing `eval_cli.py` zero-shot mode (`clip_rgb_only`, and SigLIP/other foundation models the same way the linear-probe work already covered 13 models) over these 515 clips against `datasets/hmdb51/`.
3. Join per-clip correct/incorrect with the `skin_color` label from `pa_hmdb51.py`.
4. Compute: (a) pooled accuracy by skin-tone group, (b) per-class accuracy where sample size allows, (c) a pooled significance test (see §4) analogous to the validated counterfactual-flip methodology already used on synthetic data.

**Phase B — off-the-shelf Kinetics-400-pretrained video models, no training, class-matched (your proposed approach):**

Instead of training a 51-way HMDB51 head, just run the existing torchvision video
models (`models/torchvision_models.py`, already Kinetics-400-pretrained, already
wrapped for ActionBiasBench) directly on the 515 PA-HMDB51 clips and take their
native K400 softmax. No training needed. The catch is that K400's 400 classes don't
line up 1:1 with HMDB51's 51 classes, so this only works on the subset of HMDB51
classes that have a clean K400 counterpart — needs a small manual class-mapping
table, e.g.:

| HMDB51 class | plausible K400 match | why interesting |
|---|---|---|
| `dribble` | `dribbling basketball` | your flagged example — Kinetics basketball footage plausibly skews darker-skinned given real-world US sports-media demographics, so K400-pretrained features may be entangled with skin tone for this class |
| `shoot_ball` | `shooting basketball` | same hypothesis as dribble, second basketball-adjacent class |
| `golf` | `playing golf` | plausible opposite skew (recreational golf skews whiter in typical web footage) — good contrast case to `dribble` |
| `ride_bike` | `riding a bike` | clean match, probably low skew — useful as a near-null control class |
| `ride_horse` | `riding or walking with horse` | clean match |
| `pullup` / `pushup` / `situp` | `pull ups` / `push up` / `sit ups` (verify exact K400 names) | clean matches, gym-context |
| `climb` / `climb_stairs` | `rock climbing` / `climbing ladder` (approx.) | rough match, verify |
| `swing_baseball` | `swing baseball` (approx.) | check exact K400 phrasing |
| `fencing`, `sword`, `sword_exercise` | `fencing` | only one has an obvious K400 match |

Many HMDB51 classes (`clap`, `wave`, `smile`, `kiss`, `talk`, `hug`, `laugh`, `chew`,
`smoke`, `stand`, `sit`, `turn`) have no clean K400 counterpart and should just be
excluded from Phase B rather than force-matched.

Steps:
1. Finalize the class-mapping table above (verify exact K400 class strings against `tc-clip/labels/kinetics_400_base_labels.csv`, which is already in the repo).
2. Run the pretrained torchvision K400 models over the PA-HMDB51 clips in the matched classes only, restricting/renormalizing the softmax to the matched-class subset (or just taking top-1 over all 400 and checking whether it lands on the matched class).
3. Join with `skin_color` per clip, same as Phase A.
4. Treat `dribble`/`shoot_ball` (predicted skew toward worse accuracy for lighter-skinned or better for darker-skinned performers, or vice versa — state the direction as a pre-registered hypothesis before looking at results) as the flagship test case, with `golf`/`ride_bike` as contrast/control classes.
5. This phase is now roughly as cheap as Phase A (no training, just inference + a class-mapping table), so no need to gate it on Phase A results — can run both in parallel.

## 4. Analysis methodology — carry over lessons from the synthetic work

- **Sample sizes are small and imbalanced**: 352 white / 69 yellow / 61 black / 11 mixed / 22 unidentifiable, spread over 51 classes (~1-2 black-labeled clips per class on average). Per-class, per-skin-tone breakdowns will mostly be noise — the same lesson already learned from the synthetic thesis data (per-pair Bonferroni hid the signal; pooling found it, per the `[[project_actionbiasbench]]` memory). Use the same **pooled** approach here: pool correct/incorrect counts across all classes, not per-class significance tests.
- Recommend collapsing to **white vs. non-white** (352 vs. 141) as the primary axis for power, with black-only (61) as a secondary check when framing findings around "darker vs. lighter" (consistent with the existing bias-direction framing in `bias_direction.md`).
- Drop or separately footnote `unidentifiable` (22) — no visible skin tone, not a fair comparison group.
- Recommend a simple 2x2 (or 2xK) contingency test (Fisher exact / chi-square on pooled correct-vs-incorrect by group), plus reporting the raw accuracy gap, mirroring how the counterfactual-flip analysis reported flip counts + directional skew rather than per-pair p-values.

## 5. Important caveat to lead with in any writeup

Unlike Ctrl-A-Bias (counterfactual pixel recoloring holding pose/background/motion
fixed), PA-HMDB51 skin tone is **observational, not causal**. It's confounded with:
performer identity, video source/era/quality (many HMDB51 clips are old, low-res,
grainy YouTube/movie rips), background/scene, and possibly the action class itself
(self-selected web videos — e.g. who happens to upload "swim" vs. "sword fighting"
clips). A real-data accuracy gap by skin tone is a **correlational sanity check** that
the synthetic causal finding shows up in the wild — not itself proof of a causal skin-
tone effect. Worth explicitly stating this in any report so it isn't over-claimed
relative to the causal counterfactual-flip result.

## 6. Suggested first concrete step

Both phases are now training-free, so the cheapest useful slice is: (a) build the
515-clip manifest + `skin_color` join once (shared by both phases), (b) run Phase B's
`dribble`/`shoot_ball` vs. `golf`/`ride_bike` class-matched K400 comparison first since
it's a pre-registered, directional hypothesis and needs no CLIP text-bank plumbing —
just the existing torchvision K400 model wrappers plus the class-mapping table above.
Run Phase A (CLIP/SigLIP zero-shot, full 51-class pooled analysis) alongside or right
after for the broader picture.
