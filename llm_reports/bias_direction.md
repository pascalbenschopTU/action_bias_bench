The pooled result (unseen IDs)
harmful	helpful	p
RGB models pooled	95	16	3.8×10⁻¹⁵
→ toward darker (→african/indian)	56	1	4×10⁻¹⁶
→ toward lighter (→white/asian)	39	15	7×10⁻⁴
i3d_flow (control)	9	16	n.s. (0.95)
The thing that was invisible in the per-cell tests is now unambiguous:

The asymmetry is overwhelming and directional. Swapping a clip to a darker skin tone breaks the prediction 56 times and fixes it exactly once. That's about as clean a counterfactual bias signal as you'll ever see.
The flow control behaves exactly as a control should — 9 vs 16, symmetric, non-significant. The one modality blind to appearance shows no directional bias. This is your strongest single piece of evidence that the effect is appearance-driven, not an artifact of the swap procedure.
Per-model, mc3_18, mvit_v2_s, swin3d_s are all *** ; s3d is * ; r3d_18/r2plus1d_18 trend (p≈0.06) but just have few flips. Every RGB model leans harmful; none leans helpful.
One honesty note on framing: it's not purely toward-darker — swapping toward lighter is also net-harmful (39 vs 15), just ~2× weaker and much noisier. So the accurate claim is "any skin-tone swap degrades, but degradation toward darker variants is far stronger and almost perfectly one-directional," not "bias only goes one way."

(The seen split is mostly empty of flips because the models have memorized those IDs — matched≈shifted≈ceiling. unseen is the right primary split, which is also what your significance files used.)

The one robustness check I'd add before trusting the direction
These are synthetic recolorings (_modified_african.mp4 etc.). So a skeptical reviewer will ask: is "toward darker" harming predictions because of bias, or because the darker rendering changes more pixels / introduces more artifacts? You can pre-empt this directly — the CSV already has delta_luma_mean, delta_saturation_mean, delta_contrast, etc. If the magnitude of image change is comparable in both directions but only the darker direction harms predictions, the artifact explanation is dead and the bias claim is solid. If the darker rendering also moves pixels much more, you'll want to control for it.

Want me to run that check (compare the pixel-change deltas across swap directions, and ideally regress harmful on the swap direction while controlling for delta magnitude)? It's the natural companion analysis and closes the obvious reviewer objection — again, entirely from the existing file.

Minor: the Panel A legend overlaps the swin3d_s bar; I'll nudge it when we finalize the figure.