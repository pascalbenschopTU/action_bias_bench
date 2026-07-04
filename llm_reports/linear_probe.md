# ActionBiasBench — Experiment Notes

## Current results (rgb_torchvision_v6_cj0p0)

Full fine-tune of 6 Kinetics-400 pretrained torchvision models (mc3_18, mvit_v2_s,
r2plus1d_18, r3d_18, s3d, swin3d_s) on the skin-tone probe with 48 training clips
per action pair. Color jitter = 0.0.

**Main finding:** Near-zero F1 drops for almost all pairs/models. The one exception
(`lunge_vs_cartwheel`, consistent ~0.127 drop) traces back to a single performer
(ID 7, jumping lunge) whose videos are OOD within the lunge class and whose
white-skin variant happens to coincide with the only white-skin clips the model
saw during training (all labeled as cartwheel, class 1). Not a general skin-tone
bias finding.

## Open question: SSM analysis vs training results

The SSM Frobenius analysis (DINOv2, CLIP) and the probe training are measuring
different things and do not obviously correlate:

- Yawn has the highest CLIP SSM r (0.53) but near-zero F1 drop in the probe.
- Lunge has low SSM r (0.21) but the largest F1 drop.

This means the structural skin-tone sensitivity of pretrained features does not
predict the classifier shortcut behaviour under fine-tuning. The lunge_vs_cartwheel
bias is driven by a data artifact (performer 7, OOD motion + missing white-skin
training examples), not by the model encoding skin tone in its temporal structure.

To properly connect the two experiments, the most tractable option is a **linear
probe on cached DINOv2/CLIP embeddings** using the same train/eval split manifests.
This requires no GPU — the embeddings are already cached.

## Planned experiment: frozen backbone (linear probe)

**Motivation:**
The full fine-tune result ("no shortcuts") could mean either:
  (a) pretrained features don't encode skin tone → no signal to exploit
  (b) 48 clips are too few to adapt the full network → shortcut can't be learned
A linear probe disambiguates: if a single linear layer trained on 48 clips finds a
skin-tone shortcut, the signal is already in the pretrained representation.

**Implementation:**
`--freeze_backbone` already exists in `scripts/train_torchvision_rgb_probe.py`.
When set, all backbone parameters are frozen and only the final classification
layer is trained. This is equivalent to a linear probe on the pretrained features.

**How to run (from ActionBiasBench directory):**
Pass `SKIN_TONE_FREEZE_BACKBONE=1` when launching `scripts/run_action_bias_bench.sh`
and point `SKIN_TONE_OUT_ROOT` to a new directory to keep the original results intact.

Example sbatch addition:
```bash
export SKIN_TONE_FREEZE_BACKBONE="${SKIN_TONE_FREEZE_BACKBONE:-0}"
# in the rgb_torchvision train command block:
[[ "$SKIN_TONE_FREEZE_BACKBONE" == "1" ]] && train_cmd+=(--freeze_backbone)
```

Suggested output root: `out/skin_tone_probe_rgb_torchvision_v6_cj0p0_frozen`

**Expected outcomes:**
- If frozen > full fine-tune drops: pretrained features already encode skin tone;
  full fine-tune was smoothing it out via feature adaptation.
- If frozen ≈ full fine-tune drops (both near zero): features don't encode skin tone
  at all — the probe is genuinely robust regardless of fine-tune depth.
- If frozen > full fine-tune AND concentrated on specific actions/performers:
  supports H3 (bias conditional on motion ambiguity in pretrained features).

**What to keep from existing experiments:**
All existing results in `out/skin_tone_probe_rgb_torchvision_v6_cj0p*/` are
untouched. The frozen experiment writes to a new directory.

## Notes on CLIP vs DINOv2 SSM sensitivity

CLIP (language-supervised) shows ~3× higher mean SSM Frobenius r than DINOv2
(self-supervised) across all actions. This is consistent with CLIP encoding
visual appearance attributes (including skin tone) more strongly due to its
language supervision objective. DINOv2's representations are more
texture/appearance-agnostic.

Performer 7's lunge stands out in CLIP specifically (mean_r=0.51 vs next highest
0.27 for other lunge performers), connecting the SSM finding to the classifier
failure independently of the training experiment.


The short answer is no — and importantly, for the wrong reason.

| Condition | Matched F1 | Drop |
|---|---|---|
| Full fine-tune (r3d_18) | ~1.000 | ~0.000 |
| Linear probe (r3d_18) | 0.486–0.833 | ~0.000 |
| Linear probe (CLIP) | 0.916–1.000 | 0.064–0.501 |

The linear probe drops for r3d_18 are near-zero just like the full fine-tune — but that's because the classifier barely learned anything at all. `clap_vs_celebrate` and `yawn_vs_fish` sit at 0.486 — binary chance. The r3d_18 Kinetics-400 features, designed for 400-class discrimination, simply aren't linearly separable for these specific binary pairs from 48 clips.

So the full fine-tune and the linear probe reach the same conclusion (no skin-tone bias) via completely different mechanisms:

- **Full fine-tune:** achieves matched=1.0, learns a good motion-based classifier, and doesn't rely on skin tone.
- **Linear probe:** can't even learn the task — the pretrained features aren't informative enough for a linear head on 48 clips regardless of skin tone.

This actually weakens the linear probe as a tool for torchvision models. CLIP works because it's an enormously expressive image representation that linearly separates almost anything. R3d_18 is optimised for 400-way Kinetics action recognition — binary discrimination of these specific pairs from global average pooled features is genuinely hard.

The full fine-tune result remains the more meaningful statement: the model CAN learn to classify these pairs, but chooses not to exploit skin tone when doing so.

---

## CLIP Linear Probe — Where Does the Bias Come From?

### Why the SSM patterns look completely different yet models "confuse" the actions

They don't — not on motion. The SSMs confirm **cartwheel and lunge are structurally completely different in CLIP's representation**: the cartwheel has large oscillating off-diagonal blocks (periodic full-body rotation, frames far apart are dissimilar), while the lunge has a fine-grained checkerboard (repetitive alternating steps, high-frequency similarity changes). A motion-based classifier would have no trouble separating them.

The confusion is not about motion at all. The linear probe is trained on a dataset where **skin tone is perfectly correlated with action**: all cartwheelers are dark-skinned, all lungers are light-skinned. So the probe learns "dark skin → cartwheel, light skin → lunge" rather than using the motion structure. On the shifted split that shortcut inverts and performance collapses. The SSM structure being distinct is actually *why the drop is so large* — the probe ignores the perfectly good motion signal and uses skin tone instead.

### CLIP per-pair results (unseen performers, sorted by drop)

| Pair | Matched F1 | Shifted F1 | Drop |
|---|---|---|---|
| dribble_vs_golf | 1.000 | 0.138 | **0.862** |
| golf_vs_dribble | 0.972 | 0.156 | **0.816** |
| cartwheel_vs_lunge | 1.000 | 0.331 | **0.669** |
| lunge_vs_cartwheel | 0.917 | 0.292 | **0.625** |
| fish_vs_yawn | 0.972 | 0.405 | **0.567** |
| yawn_vs_fish | 1.000 | 0.579 | 0.421 |
| celebrate_vs_clap | 1.000 | 0.602 | 0.398 |
| clap_vs_celebrate | 0.858 | 0.663 | 0.195 |
| tie_vs_squat | 0.887 | 0.766 | 0.121 |
| squat_vs_tie | 1.000 | 1.000 | 0.000 |

The matched F1 is near-perfect for almost every pair — the probe successfully learned the skin-tone shortcut. The drop is purely about what happens when the shortcut is reversed.

`squat_vs_tie` is the exception: zero drop in both directions, meaning the probe can't distinguish them even with the skin-tone shortcut available. These are likely too visually similar for any shortcut to form reliably in 48 clips.

### Cross-model comparison on the worst pairs (drop = matched − shifted F1, unseen performers)

| Pair | CLIP | SigLIP | EVA-02 | V-JEPA2 | swin3d_s | mvit_v2_s |
|---|---|---|---|---|---|---|
| dribble_vs_golf | **0.862** | 0.722 | 0.376 | −0.083 | −0.056 | 0.000 |
| golf_vs_dribble | **0.816** | 0.582 | 0.652 | 0.056 | 0.112 | 0.000 |
| cartwheel_vs_lunge | **0.669** | 0.219 | 0.027 | 0.000 | 0.000 | 0.000 |
| lunge_vs_cartwheel | **0.625** | 0.305 | 0.027 | 0.000 | 0.000 | 0.000 |
| fish_vs_yawn | **0.567** | 0.482 | 0.385 | 0.000 | 0.028 | 0.142 |
| yawn_vs_fish | 0.421 | 0.223 | 0.057 | 0.000 | 0.000 | 0.028 |
| celebrate_vs_clap | 0.398 | 0.188 | 0.226 | −0.080 | 0.028 | −0.001 |
| clap_vs_celebrate | 0.195 | **0.392** | 0.204 | 0.000 | 0.146 | 0.057 |

**dribble vs golf is the most egregious pair for CLIP** — drop of 0.86, meaning the probe essentially collapses to chance on the shifted split. Golf is heavily context/appearance-coded in CLIP's training data (a specific outdoor aesthetic, particular clothing), making skin tone as a secondary shortcut very learnable.

The negative values for V-JEPA2 (−0.083 on dribble_vs_golf) mean it actually does slightly better on the shifted split than matched — it's not learning any shortcut in either direction, just noise around chance because its matched F1 is already low.

### Why these pairs dominate

The training manifests have the same assignment structure for every pair: dark variants (african, indian) → action A; light variants (asian, white) → action B. The pairs that produce the largest drops are those where CLIP's features carry a strong enough per-frame appearance difference between the two actions that 48 clips is enough to learn "dark-toned actor → this action" as a linear rule. Dribble/golf and cartwheel/lunge are the cleanest cases — visually distinct enough for the shortcut to be near-perfectly learnable (matched ≈ 1.0), and distinct enough that the shift fully inverts performance.


# Why is Hiera unaffected?
Hiera isn't "unaffected" — the probe simply failed on it
Look back at its matched F1 in the table: 0.47, 0.49, 0.60, 0.61, 0.64, 0.67, 0.72, 0.77… — mostly below 0.75, several near binary chance. That's why almost every Hiera cell carries a †. It's the same situation as the torchvision CNNs (matched ≈ 0.62): the probe never built a working classifier, so there's nothing for a skin-tone shift to break. A ~0 drop there means "uninterpretable," not "robust/fair."

The reason the probe fails on Hiera is mechanistic, and it's the one thing that separates it from DINOv2/CLIP:

MAE objective. Hiera is trained with masked autoencoding. MAE is famous for producing features with low linear-probe accuracy but high fine-tune accuracy — the information isn't linearly separable without adaptation. DINOv2 (joint-embedding SSL) and CLIP (contrastive) are, by construction, trained to make features linearly readable. So "image SSL" is not one category for linear probing — MAE-SSL ≠ DINO-SSL.
Capacity/resolution. It's the only base, 224px model in the set; everything else is large (and EVA-02 is 448). Smaller backbone, less separable features.
Consistent with our SSM finding: Hiera's per-frame embeddings had ~0 temporal variation (0.035) — its pooled token barely moves, another symptom of features that don't expose much to a linear reader.
So Hiera belongs in the "probe can't learn it" bucket with the CNNs, not in the "low bias" story. The daggers already encode this, but it's worth a sentence in the text so a reader doesn't misread Hiera's 0.03 as fairness.

EVA-02 — you're exactly right, and it actually strengthens the story
mim_in22k_ft_in22k means: masked-image-modeling pretraining then supervised fine-tuning on ImageNet-22k classification. So it is not a clean SSL datapoint — calling it "img-ssl" alongside DINOv2/DINOv3 is misleading. Two consequences:

The supervised 22k-classification finetune is exactly what makes its features (a) linearly separable (matched F1 reaches 0.73–1.0, unlike Hiera) and (b) appearance/object-discriminative — which is why it carries a real skin-tone drop (0.38, 0.65 on dribble/golf).
Crucially, EVA-02's MIM stage distills from EVA-CLIP (a CLIP-like teacher). So it inherits CLIP-style appearance features and gets a supervised classification head. That places EVA-02 much closer to the language/supervised end than to pure SSL — which is precisely why it sits above DINOv2/DINOv3 in the gradient. It's not a counterexample to "language/CLIP-derived ⇒ more skin-tone signal"; it's another instance of it.
Suggested fix for the figures/table
Relabel the family axis so the grouping reflects the actual training, e.g.:

CLIP, SigLIP → language-supervised
EVA-02 → CLIP-distilled + supervised-ft (its own category, not "img-ssl")
DINOv2, DINOv3 → pure image SSL
Hiera → MAE-SSL (mark unreliable — probe doesn't fit)
V-JEPA2 → video SSL; the rest → Kinetics-supervised video
That regrouping makes the real axis legible: the more an objective optimizes for appearance/semantic alignment (language, CLIP-distillation, supervised classification), the more skin-tone signal a linear head can extract — and pure reconstruction/SSL (MAE) or motion-pretrained pooled features just aren't linearly probe-friendly.

One honest caveat to put in the paper: because EVA-02 (large/448/supervised) and Hiera (base/224/MAE) differ from the others in capacity and resolution too, the cross-family comparison isn't perfectly controlled. The clean within-family contrasts are CLIP vs SigLIP (both language) and DINOv2 vs DINOv3 (both pure SSL).