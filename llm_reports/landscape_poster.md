# Landscape ECCV poster (140 x 100 cm)

Notes behind `poster/landscape_poster.tex` and
`scripts/make_landscape_poster_figures.py`. The portrait A3 poster
(`poster/poster.tex`) and the first landscape draft
(`poster/landscape_design_poster.tex`) are left as they were.

Last assembled: 2026-08-25 (round 6: equal-contribution note added to the
title band, probe chart made taller to close column 3's dead space — see
"Round 6" below. Round 5 covered the heatmap, radar, AI Act box and section 5).

## Why the figures had to be regenerated

The portrait poster's figures were sized for an A3 column. Dropped into a 32 cm
landscape column they printed their tick labels and cell values at 13-15 pt,
and the wide ones (the paired-drop heatmap at 2.4:1, the probe bars at 1.6:1)
filled only the top third of their column, which is what left the bottom ~35 %
of the first landscape draft empty.

Two changes fix both problems at once:

* **Aspect.** Poster columns are tall and narrow, so the figures are drawn
  tall and narrow. The heatmap is transposed (10 action pairs down, 7 models
  across) instead of being squeezed sideways; the probe bar chart is given a
  taller canvas so its 14 bars are thick rather than hairlines; the radar's
  legend moves underneath the circle instead of beside it.
* **Font scale.** See "Making every figure print at the same size" below.

`bbox_inches="tight"` trims a little off each canvas, so the delivered aspect
ratios differ from the `--*_aspect` arguments; every number below is measured
from the saved PDF, not requested.

## Layout budget

140 x 100 cm board, 2 cm text margins (136 cm of text width). The body columns are
**not** equal: `0.195 / 0.265 / 0.245 / 0.235` of `\linewidth`, separated by three
`\hfill` gaps of 0.02. Section 2's heatmap needed the width, section 3's probe
bars needed a little more room for the `F1_matched` labels, and section 1's
schematic did not. Those fractions are duplicated in `COL_FRAC` in
`make_landscape_poster_figures.py`, which draws each figure at its own slot width
-- **change one and you must change the other.** The top row splits 0.71 / 0.27,
trading a little hero width for a readable motivation box.

| Band | Height |
|---|---|
| title band (SMPL halves flanking the title) | 13.9 cm |
| `\aftertitlegap` | 0.9 cm |
| hero + "Why audit for this?" row | ~22 cm |
| `\rowgap` | 2.2 cm |
| four body columns | ~50 cm |
| footer (TU Delft logo, centred repo link) | from ~91 cm |

Section 5 spans columns 2-3 via a zero-width `\makebox` inside column 2: its
height still counts towards that column, and column 3 is sized to end above it.

Two title-band traps, both of which cost a compile to find:

* A `\raisebox{-0.5\height}` image, or a bare `tabular` cell, hangs below the
  baseline and deepens the band by several centimetres -- which pushes every body
  column down. `minipage[c]` centres the box on the baseline predictably; use it.
* Narrowing the title's own column wraps the title to two lines, which deepens the
  band again. The title needs ~110 cm.

## Figure inventory

Written to `out/poster_landscape/`, copied to `poster/figures/`.

| File | Slot | Aspect (w/h) | Printed height |
|---|---|---|---|
| `hero_tone_pairs_landscape.pdf` | 96.6 cm | 6.32 | 15.3 cm |
| `swap_design_schematic.pdf` | 26.5 cm | 0.96 | 27.7 cm |
| `pair_heatmap_paired_flip_rate_landscape.pdf` | 36.0 cm | 1.10 | ~39.6 cm |
| `probe_drop_by_model_landscape.pdf` | 33.3 cm | 1.05 | ~34.6 cm |
| `augmentation_radar_landscape.pdf` | 32.0 cm | 0.70 | ~22.4 cm |
| `smpl_title_{left,right}.png` | 12 cm tall | -- | title band |

Round 5: the heatmap was drawn squat (7 models wide, 10 pairs tall, cell
aspect ~1.4:1) with three-decimal cell text, which read as cluttered at poster
distance. `heatmap_aspect` went 0.709 -> 1.10 (near-square cells), cell text
dropped to two decimals, and thin white gridlines were added between cells
(`ax.grid(which="minor", ...)` on a half-integer minor tick grid) since nothing
had separated adjacent numbers before. The extra height uses exactly the
whitespace that used to sit between section 5's box and the footer link.
`probe_left_margin` went 0.30 -> 0.42 and the probe column widened 0.235 ->
0.245 so the `F1_matched=` label in front of the longest bar (CLIP) clears the
y-axis instead of nearly touching it. `radar_aspect` went 0.601 -> 0.70 and
`radar_tick_pad` went 18 -> 26 so spoke labels like "strong jitter +
grayscale" clear the outer circle; `radar_rect` opened up to
`[0.02, 0.20, 0.96, 0.70]` to use the added height for the circle itself.
`heatmap_aspect` was then eased back to 1.05 (from 1.10) -- 1.10 left the
heatmap's caption crowding section 5's title bar with almost no gap.

Section 5's heading moved from a `\section*` above its tcolorbox into the
box's own `title=` field (matching the Takeaways box), in `TUDNavy`
(`#0C2340`, TU Delft's brand navy) rather than a flat `black!70` -- the latter
read as an arbitrary shade next to the poster's blue/cyan/orange palette.
Aligning the itemize (left half) with the table (right half) inside that box
took several failed attempts, all of them from guessing rather than measuring.
The actual mechanism:

**Both cells are `[t]` minipages, and a `[t]` minipage is a `\vtop` whose
reference point is the baseline of its first *box*. If the vertical list
starts with *glue* instead, the reference point becomes the minipage's top
edge and the whole content hangs below it.** That is a discontinuity, not a
smooth knob:

* Table with no leading `\vspace` -> starts with the tabular box -> reference
  is the header row's baseline, and `\toprule` overhangs *above* the minipage,
  which shoves the shared line down.
* Table with any leading `\vspace` -> starts with glue -> reference is the top
  edge.
* The itemize *always* starts with glue (its own `\topsep`), so it is always in
  the second regime.

Interpolating across that discontinuity is what made three successive guesses
miss. Inside the glue-first regime the relation is linear and solvable; the
measured constants (see "Measuring the layout" below) are

    bullet1_top = navy_bar_bottom + 0.469 cm + v_itemize
    header_top  = navy_bar_bottom + 0.893 cm + v_table

Setting these equal with `v_table = 0` is *geometrically impossible*: it puts
`\toprule` about 0.5 cm above the title bar's lower edge, i.e. inside it. So
the two sides meet in the middle -- `v_itemize = 0.87 cm`, `v_table = 0.45 cm`
-- which aligns the first bullet with the header to within 0.05 cm while
leaving `\toprule` 0.89 cm of clearance below the title bar. Change either
value and you must re-derive the other.

The earlier `\vspace{-1\baselineskip}` on the itemize is the failure mode to
avoid: negative glue keeps the reference at the top edge but drags the first
bullet up *into* the title bar. (That overlap was real, not a QuickLook
thumbnail-cache artifact -- ruled out by re-rendering the PDF under a fresh
filename.)

Section 1's Dataset/Audit subset/Evaluation lines became a two-column
`tabular{@{}l p{0.62\linewidth}@{}}` (label, then clip count first followed by
the breakdown) instead of three `\textbf{Label:} ...` paragraphs -- the plain
paragraphs wrapped unpredictably at this column width.

The gap between section 2's caption and section 5's box was `\vspace{0.5cm}`,
which measured only **0.61 cm** on the board -- much too tight, and invisible
in the `.tex` because column 3's copy of the same gap measures 6.40 cm (that
column's content simply ends higher). It is now `\vspace{2.1cm}` -> 2.19 cm
measured, with 3.31 cm still left between the box and the footer link.

Section 2's caption lost its second sentence, "Most cells are within a clip or
two of zero." It was quietly false: each cell sums eight tone-swap directions
of n=108, so **n = 864 per cell**, and even a cell displaying 0.01 is 8 flipped
clips, while I3D-flow / squat vs tie is 60. Rewriting it as "a percentage point
or two" was accurate but unwanted; dropping the sentence outright also bought
back the vertical space the box needed.

Section 5's third bullet ("HMDB51: p = 8x10^-8 per clip, 0.49 per performer")
was removed rather than replaced: it read as foregrounding the naive
per-clip result that the pseudo-replication correction overturns, which is
the opposite of what the two remaining bullets ("six pretrained models, two
real-video datasets" / "no significant light- vs. other-tone gap") are there
to say.

`poster/figures/` also needs `EU_AI_ACT.png`, `tud-logo.pdf` and
`Politie-logo.png`, which are not generated here.

## Making every figure print at the same size

`POSTER_PT` holds the target point sizes **as measured on the printed board**
(tick 30, cell 27, label 34, title 40, legend 30). Each figure is drawn at a
native canvas whose width in inches equals its slot width in cm/2.54, so
`width=\linewidth` scales it by 1.0.

The catch: `bbox_inches="tight"` trims a different amount off each canvas, and it
is the *delivered* width that LaTeX scales. The radar is cropped to ~0.68 of its
canvas by its spoke labels, so before this was accounted for its text printed a
third larger than the heatmap's -- which is exactly the "section 2 and 3 look
tiny" complaint. `CROP` holds the measured delivered/native ratios and `_save()`
prints the current value on every run, so after any layout change: run, read the
printed ratio, update `CROP`, run again. It converges in two passes. Current
values (round 5, after the heatmap/probe/radar aspect changes above): heatmap
0.981, probe 0.992, radar 0.678, schematic 0.988 -- barely moved from the
previous round, since `bbox_inches="tight"` mostly trims a fixed margin rather
than a fraction that scales with the aspect.

### Hero and schematic: locating the actor

Both crop around the actor rather than showing the full 16:9 render, which is
what makes the skin tone itself readable from poster distance. The actor is
located by **differencing the two tone renders**, not by background
subtraction: the two clips are byte-identical except for the skin texture, so
their difference is exactly the visible skin. Background subtraction was tried
first and fails here — the Konzerthaus pavement is wet, so the actor's
reflection lands in the mask and pushes the box ~150 px wide of the body,
which is why the earlier hero showed a small figure lost in a wide plaza.

Frames are hand-picked from a contact sheet: cartwheel 2 / 22 / 40 for the hero
(wind-up, peak inverted extension, recovery, all within one repetition — the
same choice documented in `poster_reference_stats.md`), and cartwheel 22 +
lunge 88 for the schematic, lunge 88 being the only part of that clip where the
pose reads as a lunge rather than as standing.

The schematic's two rows share **one** crop window computed from the union of
both actions' actor boxes. A per-action crop would shift the building between
the two panels and read as a background change, which is the opposite of the
figure's point.

## Data sources

Nothing here recomputes a result; every figure reads an artefact that already
exists.

* **Heatmap** — `out/skin_tone_probe_v7_cv/skin_tone_raw_accuracy_by_direction_testonly.csv`,
  summed over the four swap directions per (action pair, model).
  `(b-c)/n` is computed as `(correct_matched - correct_shifted) / n`: a clip
  right under both tones cancels out of both `b` and `c`, so the marginal
  counts in that CSV carry the same difference as the paired ones. Verified
  cell-for-cell against the v7 `_poster` heatmap the first landscape draft used
  (e.g. I3D_flow / squat_vs_tie = 0.069, tie_vs_squat = -0.072). Note that the
  **v6** run's CSV gives different numbers — it is not the poster's source.
* **Probe bars** — `out/linear_probes/_probe_summary_cv.json`, with the same
  `hiera` / `hiera_large` handling as `plot_probe_ssm.py` (drop `hiera`, show
  `hiera_large` relabelled as `hiera`).
* **Radar** — `out/poster_landscape/radar_scan_deltas.json`, the per-(model,
  condition) delta cache written by
  `plot_augmentation_radar.py --out_dir out/poster_landscape --out_name radar_scan`
  over the five v7 roots (`_cv`, `_cjweak_cv`, `_cjstrong_cv`,
  `_cjstronggray_cv`, `_planckian_cv`). Regenerating it costs a few minutes of
  scanning ~2,700 per-seed JSONs on the network mount, so the cache is kept.

## Claims stated on the poster

Numbers all trace to `poster_reference_stats.md`:

* Section 5's "HMDB51: p = 8x10^-8 per clip, 0.49 per performer" — the
  pseudo-replication correction: 145 clips resolve to 23 source videos, and the
  gap that looks overwhelming per clip is not significant once repeats of one
  performer are treated as one unit.
* The real-video table (HMDB51 0.67/0.61 p=0.49, Kinetics-400 0.95/0.95 p=0.53)
  are the *corrected* numbers in both cases — source-video grouping for HMDB51,
  the sibling-class family metric for Kinetics.
* A "4 of 6 backbones differ significantly, at most 1.6 pp" line was on an
  earlier draft and has been **removed**: that is Table 2's cluster-bootstrap
  result, and the heatmap it sat under is not evidence for it. If it goes back
  on, it needs the per-model forest plot
  (`skin_tone_model_cluster_significance_unseen.pdf`) beside it.
* The I3D-flow row of the heatmap carries the *largest* individual cells
  (+0.069 and -0.072 on the two directions of one action pair) while its
  aggregate is 0.28 pp at q = 0.539. Anything the poster says about the flow
  control has to be consistent with both facts — "sits at zero" is not.

## Round 6: title-band note and column 3's dead space

**Equal-contribution mark.** Two authors carried a bare `\textdagger` with
nothing explaining it. It cannot be fixed with `\footnote`: line ~149 does
`\let\footnote=\endnote` and no `\theendnotes` is ever emitted, so a footnote
compiles clean and prints *nothing*. The note therefore lives as a third line
inside `\institute`.

That is free, and the reason is worth recording: **the title band's depth is set
by the 12 cm SMPL images, not by the text.** The band is
`0.010\paperheight + max(12 cm, text) + 0.009\paperheight = 13.9 cm`, and the
text block runs only ~8.5 cm, so there is ~3.5 cm of headroom for extra lines
before the band deepens and pushes every body column down. Measured effect of
the added line: band bottom 13.87 -> 13.96 cm (2 px), everything below shifted
by the same 0.09 cm, section 5's internal alignment unchanged.

**Column 3's dead space.** Column 3's content ended at 77.7 cm against section
5's block at 85.8 cm -- a 7.98 cm hole, versus 2.19 cm under column 2. It is
structural: section 5 is anchored in column 2's flow, and column 2's heatmap is
much taller than column 3's bar chart, so column 3 simply runs out of content.
Fixed by drawing the probe chart taller (`probe_aspect` 0.932 -> 1.05, printed
height 30.7 -> 34.6 cm), which closes the gap to 4.01 cm. Deliberately *not*
closed all the way: some whitespace there is wanted.

`probe_aspect` and `probe_bar_height` are coupled. Bar thickness is
`probe_bar_height x plot_height / 14`, so raising the aspect alone fattens every
bar. `probe_bar_height` went 0.74 -> 0.64 in the same step, which holds each bar
at its previous ~1.4 cm and spends the added height on space *between* bars
instead. Change one and you must re-check the other.

## Poster specification (ECCV 2026 workshops)

The board for ECCV 2026 workshops is **140 x 100 cm**, which this poster fills
exactly; the rendered page ratio measures 1.4006 against a target of 1.4000.

Two caveats:

* The **main conference** spec is different -- landscape **180 x 90 cm** max --
  so this poster is 10 cm too tall for a main-conference board and would need
  reworking if it ever moved there.
* The PFATCV workshop page itself publishes no size. The 140 x 100 figure comes
  from a sibling ECCV 2026 workshop (AI4VA, "Board size - 140 x 100 cm"). ECCV
  2026 states it does not handle poster uploads for workshop papers at all, so
  each workshop sets its own logistics -- worth confirming with the PFATCV
  organisers before printing.
* The design has **zero bleed margin**: the title bar is full-bleed and the SMPL
  images sit 1.5 cm from the edge, so a framed board or an off-centre trim would
  clip into them. Printing at ~138 x 98 cm centred is cheap insurance.

## Measuring the layout

Every number in this file that is described as "measured" comes from
rasterising the compiled PDF and scanning it, not from reading the `.tex`:

    qlmanage -t -s 3000 -o . landscape_poster.pdf     # -> landscape_poster.pdf.png

at 3000 px wide for a 140 cm board, i.e. 21.43 px/cm, so one pixel is 0.47 mm
-- fine enough to settle any of the alignment questions above. Then, with
numpy on the greyscale array: the navy title bar is rows where >80 % of pixels
are <60; body text is <150 (this deliberately excludes the `gray!5` box fill at
~242 and the `black!25` frame at ~191); a booktabs rule versus a row of text is
`mean(dark) > 0.9` versus `0 < mean(dark) < 0.9` across a narrow x-slice.

Two traps worth remembering, both of which produced confidently wrong numbers
before being caught:

* Scanning a whole column for its lowest ink finds the *footer*, not the
  column -- the centred repository link spans the full page width underneath
  every column. Always bound the search above the footer.
* Detecting rules with a threshold on the *full* region width rather than the
  table's own width reported `\toprule` as sitting **below** its own header
  row. When a measurement is internally impossible, stop and look at a crop of
  the render before acting on it.

## Checking the layout without LaTeX

There is no TeX installation on this machine. The column budget above was
verified with a throwaway scale mock-up that composites the real figure PNGs at
the widths the `.tex` gives them and blocks out each text run at its estimated
line count (it assumes beamerposter `normalsize` ~ 30 pt at `scale=1.55`). It
is accurate to about a centimetre per text block — good enough to catch an
overrun, not a substitute for compiling.
