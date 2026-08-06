# Hiera-Large scaling experiment: does capacity fix the degenerate Hiera result?

## Motivation

Hiera-Base (51M params) failed both the linear-probe and SSM diagnostics: its
probe never reliably learned the shortcut task (matched F1 mostly < 0.75),
and its per-frame embeddings showed almost no frame-to-frame variation,
producing a near-constant self-similarity matrix (SSM off-diagonal entries
in [0.997, 1.000], std ≈ 0.0006 — compare CLIP's [0.70, 0.995], std ≈ 0.055).

Two candidate explanations were on the table:
1. **Capacity**: Hiera-Base is much smaller (51M) than the other
   foundation models tested (~300-430M for CLIP/DINOv2/DINOv3/EVA-02/SigLIP/
   V-JEPA2). Maybe a larger Hiera would behave more like them.
2. **Objective**: Hiera is trained with masked autoencoding (MAE) —
   reconstructing raw pixels with no teacher distillation — unlike DINOv2
   (joint-embedding self-distillation) or EVA-02 (MIM distilled from an
   EVA-CLIP teacher, then supervised-finetuned on ImageNet-22k). MAE is
   documented to produce representations that need further adaptation to
   become linearly separable, regardless of scale.

To distinguish these, we reran the full pipeline (embedding caching, linear
probe, SSM) on `facebook/hiera-large-224-hf` — same MAE-only recipe as the
base checkpoint (`architectures: ["HieraModel"]`, no classification head,
`mask_ratio`/`norm_pix_loss` present, no supervised fine-tuning stage), just
~213M params instead of 51M (a ~4x increase).

## What was run

```bash
python scripts/download_foundation_models.py --only hiera-l
python scripts/skin_tone_bias_analysis.py --model hiera_large --frames 64
python scripts/train_embedding_linear_probe.py --model hiera_large --seeds 0,1,2
python scripts/ssm_frobenius_analysis.py --model hiera_large --metric rsa
```

Required adding `load_hiera_large`/`encode_hiera_large` to
`models/huggingface_models.py` (encode function is identical to
`encode_hiera` — pooling logic doesn't depend on hidden size — so
`encode_hiera_large = encode_hiera` is a plain alias) and registering
`hiera-l -> facebook/hiera-large-224-hf` in
`scripts/download_foundation_models.py`.

## Result: capacity did not fix either diagnostic

| | Hiera-Base (51M) | Hiera-Large (213M) | CLIP (for reference) |
|---|---|---|---|
| matched F1 (linear probe) | 0.658 | **0.646** (no improvement) | 0.961 |
| shifted F1 | 0.647 | 0.643 | 0.493 |
| drop | 0.010 | 0.004 | 0.467 |
| temporal variation (mean, 5 clips) | 0.030 | **0.048** (still ~7x below CLIP) | 0.324 |
| SSM off-diagonal std | 0.00062 | **0.00092** (still ~60x below CLIP) | 0.055 |

- **Linear probe**: matched F1 stayed essentially flat (0.646 vs 0.658),
  nowhere near the 0.75 reliability threshold. 4x the parameters produced no
  measurable improvement in linear separability.
- **SSM**: temporal variation and off-diagonal spread both moved only
  marginally, remaining an order of magnitude below every other model
  tested. The self-similarity matrix is still effectively constant.
- The SSM ratio computes to 0.41 for Hiera-Large (higher than CLIP's 0.24),
  but this number is **not meaningful** — it's the ratio of two RSA
  distances computed from near-constant, noise-dominated matrices (std
  ≈ 0.0009), so it reflects floating-point noise rather than genuine
  structural sensitivity to skin tone. Reporting it at face value would be
  misleading.

## Conclusion

This is evidence *against* the capacity explanation and *for* the objective
explanation. If the degenerate behavior were a capacity limitation, scaling
4x should have helped at least somewhat. It didn't. This points more
specifically at the MAE objective itself (raw-pixel reconstruction, no
teacher distillation) as the cause, independent of model size.

## Decision: Hiera-Large replaces Hiera-Base as *the* Hiera result

Since Hiera-Large is strictly the better-controlled data point (larger,
closer in scale to the other foundation models, same conclusion either way),
the paper now reports only one Hiera row, using the Hiera-Large numbers.
`out/linear_probes/_probe_summary.json`'s `"hiera"` key was overwritten with
the Hiera-Large results (base-scale numbers are preserved only in this
report, for the record); `"hiera_large"` is no longer a separate row.

It remains excluded from the reliable-probe set in `scripts/plot_probe_ssm.py`
(`MATCHED_FLOOR = 0.75`, matched F1 = 0.646) and from `SSM_VALID` (still
degenerate SSM at this scale). It is shown in `_probe_drop_by_model.png`
with a red-flagged annotation (matched F1 < 0.75), under the plain
`"img-ssl"` family (same blue/square styling as DINOv2/DINOv3) — **not** a
separate `"img-mae"` category. We considered giving Hiera its own
MAE-specific tag, but every other family label in this taxonomy names the
training *signal* (language / image-SSL / image-SSL+supervised-finetune /
video-SSL / Kinetics-supervised) without naming the specific SSL algorithm
(DINOv2/DINOv3 aren't tagged "img-dino" either), so a one-off "img-mae" tag
for Hiera alone would be an inconsistent level of detail. EVA-02 keeps its
separate `"img-ssl+ImgNet"` tag because that distinction is real at the
recipe level (EVA-02's checkpoint genuinely has a supervised fine-tuning
stage, `eva02_large_patch14_448.mim_in22k_ft_in22k`; Hiera's does not) and
is already present as a distinct category in the paper's model table.

Suggested paper framing: *"Scaling Hiera 4x (51M -> 213M) did not improve
linear separability or resolve the degenerate self-similarity structure,
suggesting the effect is attributable to the masked-autoencoding objective
rather than model capacity."*
