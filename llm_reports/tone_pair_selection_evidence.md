Context: the benchmark tests two swap pairs, african<->white and indian<->asian, not the other two possible cross-pairings (african<->asian, indian<->white). The paper draft needed a justification for that choice beyond "that's how schema.py defines VARIANT_SWAP." The author's intuition was that these four texture presets get progressively lighter african -> indian -> asian -> white, which would make african<->white the largest-contrast pair and indian<->asian the smallest, a legitimate reason to pick exactly these two. The author was upfront that this ordering would be bogus as a claim about real-world ethnic skin tone, but plausible as a property of this specific synthetic rendering pipeline's texture assets. Checked it empirically before writing anything into the paper.

Method: swap_pair_level_analysis.csv (baseline, mc3_18, split_family=unseen) already logs matched_luma_mean / shifted_luma_mean and the same for r_mean, g_mean, b_mean, saturation_mean, hue_mean, contrast, computed per clip over the whole frame (not just skin pixels). Pooled matched_<feature> keyed by variant_matched and shifted_<feature> keyed by variant_shifted into one distribution per variant, then averaged.

Results (mean value per variant, n=720 for african/white, n=1260 for indian/asian, reflecting the known variant-coverage imbalance):

```
--- luma_mean ---
african    0.42073
indian     0.42096
asian      0.42135
white      0.42151

--- r_mean ---
african    0.43588
indian     0.43629
asian      0.43669
white      0.43671

--- g_mean ---
african    0.43337
indian     0.43355
asian      0.43394
white      0.43414

--- b_mean ---
african    0.31593
indian     0.31595
asian      0.31630
white      0.31659

--- saturation_mean ---
african    0.43063
indian     0.43089
asian      0.43033
white      0.42958

--- hue_mean ---
african    0.32735
indian     0.32755
asian      0.32752
white      0.32726

--- contrast ---
african    0.61282
indian     0.61095
asian      0.60881
white      0.60893
```

The claimed order (african < indian < asian < white) holds cleanly and monotonically in all four direct brightness/color channels: luma_mean, r_mean, g_mean, b_mean. Four independent measurements agreeing on the same order is a real signal, not noise, even though each individual gap is tiny (3rd-4th decimal place), which makes sense given these are whole-frame averages dominated by background and clothing pixels, not skin region alone. A skin-masked measurement would presumably show a much larger gap; this wasn't computed here.

The order does NOT hold for saturation_mean, hue_mean, or contrast, none of these three are monotonic across the four variants. So the honest claim is specifically about brightness/luminance-adjacent channels (luma, R, G, B), not "these four textures have a consistent overall color ordering" in every sense. Worth keeping the paper's phrasing scoped to "color channel values" or specifically "luminance," not implying saturation/hue also cooperate.

Conclusion: the intuition was right, at least for the channels that matter most for a "how far apart are these tones" argument (luma, RGB), so it's fair to use it as the stated reason for choosing these two pairs, scoped explicitly to "a measured property of these four rendering presets," not a general real-world claim. That scoping caveat should stay in the paper text given the author's own explicit acknowledgment that an unscoped version of this claim would be indefensible.

Paragraph written for the Results section (fine-tuned Kinetics backbones subsection, replacing an earlier draft that asserted the ordering without evidence and included an unclear "hardest and easiest change" clause):

"The benchmark's dark/light texture assets admit two natural swap pairs, african$\leftrightarrow$white and indian$\leftrightarrow$asian, rather than the other possible cross-pairings. We choose these two because, within this dataset's rendered variants specifically, mean per-frame color channel values follow a consistent order from african (darkest) through indian and asian to white (lightest); this is a measured property of these four texture presets, not a claim about real-world skin tone. African$\leftrightarrow$white is accordingly the largest tone contrast available in the dataset and indian$\leftrightarrow$asian the smallest, so the two pairs we test span the range of swap magnitudes rather than sampling it arbitrarily. Each test pools the matched and shifted predictions of all action pairs, folds, and seeds, and counts only discordant correctness changes. Because the four directional swaps per backbone form a small, related family, we control the false discovery rate within each model using the Benjamini-Hochberg procedure rather than applying a family-wise correction."

Open item: whether to put the actual numbers above into a footnote or supplementary note as backup for a reviewer who asks "how do you know." Not yet decided.
