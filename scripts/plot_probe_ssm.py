"""Linear-probe + SSM corroboration figures.

Reads:
  $PROBE_SUMMARY_PATH (default out/linear_probes/_probe_summary.json)
  out/bias_analysis/ssm_<METRIC>_<model>.csv        (per-clip d_skin, d_action, r)
Writes (prefix overridable via $PROBE_OUT_PREFIX, default out/linear_probes/_probe):
  {PROBE_OUT_PREFIX}_drop_by_model.{pdf,png}
  {PROBE_OUT_PREFIX}_vs_ssm.{pdf,png}
  {PROBE_OUT_PREFIX}_ssm_by_pair.{pdf,png}

Bar charts use Tab10 colours, no hatches.  The scatter plot adds distinct
marker shapes as a second cue for colorblind accessibility.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Override to point at a different probe summary (e.g. the CV run) and write
# figures under a different name, without touching the fixed-split defaults.
PROBE_SUMMARY_PATH = Path(os.environ.get("PROBE_SUMMARY_PATH", "out/linear_probes/_probe_summary.json"))
OUT_PREFIX = os.environ.get("PROBE_OUT_PREFIX", "out/linear_probes/_probe")

# Font sizes scaled to each figure's own native width, matching the pt-per-
# inch ratio used in benchmarks/skin_tone/summarize_skin_tone_significance.py
# (reference: 10.4in wide, title=12.5, label=10.5, tick=10.0, annotation=8.7)
# so text reads at the same apparent size across figures once each is scaled
# to a shared column width in the paper.
_FONT_RATIO_TITLE = 12.5 / 10.4
_FONT_RATIO_LABEL = 10.5 / 10.4
_FONT_RATIO_TICK = 10.0 / 10.4
_FONT_RATIO_ANNOTATION = 8.7 / 10.4


def font_sizes(width_in: float) -> dict[str, float]:
    return {
        "title": _FONT_RATIO_TITLE * width_in,
        "label": _FONT_RATIO_LABEL * width_in,
        "tick": _FONT_RATIO_TICK * width_in,
        "annotation": _FONT_RATIO_ANNOTATION * width_in,
    }


# Poster variants keep the paper figure's data and layout but pin every font
# larger relative to the canvas, so text stays legible once the figure is
# scaled into a poster column and read from a distance.
POSTER_FONT_SCALE = 1.5

# Which ssm_<METRIC>_<model>.csv files to read. "rsa" (1 - correlation between
# SSM off-diagonals) is scale-invariant and preferred over "frobenius", which
# is dominated by a few clips with unusually large SSM magnitude (see
# out/bias_analysis/ssm_frobenius_dinov2.csv vs ssm_rsa_dinov2.csv comparison).
# Figure text stays plain ("SSM"); the RSA definition belongs in the caption.
METRIC = "rsa"

# ── taxonomy ──────────────────────────────────────────────────────────────────
FAMILY = {
    "clip":        "language",
    "siglip":      "language",
    "tc_clip":     "language+K400",  # CLIP ViT-B/16 adapted for video via K400 training
    "dinov2":      "img-ssl",          # DINOv2: pure SSL, no labels
    "dinov3":      "img-ssl",          # DINOv3: pure SSL, no labels
    "hiera":       "img-ssl",          # Hiera-Base: pure SSL, no supervised stage (verified: no classifier head)
    "hiera_large": "img-ssl",          # Hiera-Large: same family as hiera, larger backbone
    "eva02":       "img-ssl+ImgNet",   # MIM SSL + supervised IN22k fine-tune
    "vjepa2":      "video-ssl",
    "r3d_18":      "K400",
    "mc3_18":      "K400",
    "r2plus1d_18": "K400",
    "mvit_v2_s":   "K400",
    "s3d":         "K400",
    "swin3d_s":    "K400",
}

# Tab10 colours, one per family.  Model colours below are chosen from the same
# palette so the same model reads consistently across all figures.
FAM_COLOR = {
    "language":     "#d62728",   # Tab10 red
    "language+K400":"#8c564b",   # Tab10 brown — CLIP adapted for video
    "img-ssl":      "#1f77b4",   # Tab10 blue
    "img-ssl+ImgNet":"#9467bd",  # Tab10 purple
    "video-ssl":    "#ff7f0e",   # Tab10 orange
    "K400":         "#2ca02c",   # Tab10 green
}
FAM_MARKER = {
    "language":     "o",
    "language+K400":"P",         # filled plus — distinct from all others
    "img-ssl":      "s",
    "img-ssl+ImgNet":"^",
    "video-ssl":    "D",
    "K400":         "v",
}

SSM_VALID = {"clip", "siglip", "tc_clip", "eva02", "dinov2", "dinov3", "vjepa2"}

# ── per-pair figure config ────────────────────────────────────────────────────
ACTION_PAIRS_LIST   = [("squat", "tie"), ("clap", "celebrate"), ("dribble", "golf"),
                       ("lunge", "cartwheel"), ("yawn", "fish")]
PAIR_LABELS_ORDERED = ["squat / tie", "clap / celebrate", "dribble / golf",
                        "lunge / cartwheel", "yawn / fish"]

# Models ordered lowest → highest average SSM ratio.
# Colours share the family hue so the same model reads consistently across figures:
#   clip/siglip  → red family  (language)
#   dinov2/dinov3 → blue family (img-ssl)
#   eva02        → purple      (img-ssl+ImgNet)
#   vjepa2       → orange      (video-ssl)
MODEL_ORDER = ["vjepa2", "dinov2",  "eva02", "dinov3", "siglip", "tc_clip", "clip"]
MODEL_COLOR = {
    "dinov2":  "#17becf",   # Tab10 cyan   — blue family (img-ssl)
    "vjepa2":  "#ff7f0e",   # Tab10 orange — video-ssl family
    "eva02":   "#9467bd",   # Tab10 purple — img-ssl+cls family
    "dinov3":  "#1f77b4",   # Tab10 blue   — img-ssl family
    "tc_clip": "#8c564b",   # Tab10 brown  — language+K400 family
    "siglip":  "#e377c2",   # Tab10 pink   — red/language family
    "clip":    "#d62728",   # Tab10 red    — language family
}

# ── load probe results ────────────────────────────────────────────────────────
probe = json.loads(PROBE_SUMMARY_PATH.read_text())

# matched F1 below this means the probe never learned the task, so its drop
# (robust-looking or not) is not evidence of anything — flagged in red below.
MATCHED_FLOOR = 0.75

rows = []
for m, r in probe.items():
    if m == "hiera":
        continue  # superseded by hiera_large, shown below relabeled as "hiera"
    display_name = "hiera" if m == "hiera_large" else m
    mm = r["eval_matched_unseen_ids"]
    ms = r["eval_shifted_unseen_ids"]
    csv = Path(f"out/bias_analysis/ssm_{METRIC}_{m}.csv")
    ssm_r = np.nan
    if csv.exists() and m in SSM_VALID:
        d = pd.read_csv(csv)
        # ratio of means (NOT mean of per-row ratios, which explodes when d_action~0)
        ssm_r = d["d_skin"].mean() / d["d_action"].mean()
    # Run-level 95% CI on the unseen drop, if build_probe_summary.py stored it
    # (present for the CV summary, absent for the old fixed-split one).
    ci_lo = r.get("drop_unseen_ci_low", np.nan)
    ci_hi = r.get("drop_unseen_ci_high", np.nan)
    rows.append(dict(model=display_name, family=FAMILY[m], matched=mm, shifted=ms,
                     drop=mm - ms, ssm_r=ssm_r, reliable=mm >= MATCHED_FLOOR,
                     drop_ci_lo=ci_lo, drop_ci_hi=ci_hi))
# bias = -(shifted − matched): positive = robust, negative = skin-tone sensitive.
# Sort descending so the least-biased model (vjepa2) is at the top.
df = pd.DataFrame(rows).sort_values("drop", ascending=True).reset_index(drop=True)

# ── Figure 1: probe drop by model ────────────────────────────────────────────
def draw_drop_by_model(width_in, height_in, fonts, left_margin, out_base, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    bias = -df["drop"]   # positive = robust (only vjepa2), negative = skin-tone sensitive
    for i, row in df.iterrows():
        ax.barh(
            i, -row["drop"],
            color=FAM_COLOR[row.family],
            edgecolor="white", linewidth=0.5,
        )
        # Run-level 95% CI as a capless hairline whisker at the bar tip (plotted in
        # shifted-minus-matched space, so the CI on the drop flips sign). Dark
        # neutral so it stays visible both over the colored bar and on white.
        lo, hi = row.get("drop_ci_lo", np.nan), row.get("drop_ci_hi", np.nan)
        if lo == lo and hi == hi:
            drop = row["drop"]
            ax.errorbar(
                -drop, i,
                xerr=[[hi - drop], [drop - lo]],
                fmt="none", ecolor="#2b2b2b", elinewidth=1.1,
                capsize=0, alpha=0.55, zorder=3,
            )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.model, fontsize=fonts["tick"])
    ax.set_ylabel("Model", fontsize=fonts["label"], fontweight="bold")
    ax.tick_params(axis="x", labelsize=fonts["tick"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlim(bias.min() - left_margin, bias.max() + 0.25)
    ax.set_xlabel("shifted $-$ matched  ($\\Delta$F1)", fontsize=fonts["label"])
    ax.set_title("Effect of a skin-tone shift on linear-probe accuracy", fontsize=fonts["title"])
    for i, row in df.iterrows():
        # Place the label just past the outer (more negative) end of the whisker so
        # it never overlaps the CI; fall back to the bar tip if no CI is present.
        hi = row.get("drop_ci_hi", np.nan)
        outer = -(hi if hi == hi else row["drop"])
        ax.text(outer - 0.008, i, f"matched F1={row.matched:.2f}", va="center", ha="right",
                fontsize=fonts["annotation"], color="#555")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COLOR[f]) for f in FAM_COLOR]
    ax.legend(handles, list(FAM_COLOR.keys()), fontsize=fonts["tick"], loc=legend_loc)
    plt.tight_layout()
    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.png", dpi=150)
    plt.close(fig)


fig1_w = 11.5
f1 = font_sizes(fig1_w)
draw_drop_by_model(fig1_w, 5.8, f1, 0.22, f"{OUT_PREFIX}_drop_by_model")
# Taller canvas and a wider left margin give the enlarged tick labels and
# "matched F1=" annotations room. The enlarged legend no longer fits bottom-right
# without covering the longest bars, so it moves to the empty upper-left wedge
# left by sorting the bars ascending.
draw_drop_by_model(
    fig1_w, 7.0,
    {k: v * POSTER_FONT_SCALE for k, v in f1.items()},
    0.30, f"{OUT_PREFIX}_drop_by_model_poster",
    legend_loc="upper left",
)

# ── Figure 2: probe drop vs SSM ratio ────────────────────────────────────────
# Font sizes are pinned to the 8.6in ratio (matching the other figures) but
# the canvas itself is drawn smaller at the same aspect ratio -- that's what
# makes the fixed-point-size text and markers occupy more of the figure.
fig2_w = 8.6
f2 = font_sizes(fig2_w)
f2["label"] *= 1.2  # axis labels emphasized beyond the base ratio, per request
fig, ax = plt.subplots(figsize=(6.3, 4.1))
sub = df.dropna(subset=["ssm_r"])
dr  = sub["drop"].values
r   = np.corrcoef(sub.ssm_r.values, dr)[0, 1]
b, a = np.polyfit(sub.ssm_r.values, dr, 1)
xs = np.linspace(sub.ssm_r.min(), sub.ssm_r.max(), 50)
ax.plot(xs, a + b * xs, "--", color="grey", lw=1, zorder=1)
ax.margins(x=0.06)
for _, row in sub.iterrows():
    # Hairline vertical CI: capless, same hue as the marker, low alpha, drawn
    # under the marker so it recedes rather than competing (drop axis only --
    # the SSM ratio on x is training-free and has no seed/fold spread).
    lo, hi = row.get("drop_ci_lo", np.nan), row.get("drop_ci_hi", np.nan)
    if lo == lo and hi == hi:
        ax.errorbar(
            row.ssm_r, row["drop"],
            yerr=[[row["drop"] - lo], [hi - row["drop"]]],
            fmt="none", ecolor=FAM_COLOR[row.family], elinewidth=1.3,
            capsize=0, alpha=0.5, zorder=2,
        )
    ax.scatter(
        row.ssm_r, row["drop"],
        s=130,
        color=FAM_COLOR[row.family],
        marker=FAM_MARKER[row.family],
        edgecolor="black", linewidth=1.2, zorder=3,
    )
    ax.annotate(row.model, (row.ssm_r, row["drop"]), fontsize=f2["annotation"],
                xytext=(6, 3), textcoords="offset points")
ax.set_title(f"SSM ratio correlates with linear-probe drop\nPearson r = {r:.2f}", fontsize=f2["title"])
ax.set_xlabel(
    "SSM ratio  $d_\\mathrm{skin}$ / $d_\\mathrm{action}$"
    "\n(higher = more sensitive to skin tone)",
    fontsize=f2["label"],
)
ax.set_ylabel("linear-probe skin-tone drop", fontsize=f2["label"])
ax.tick_params(axis="both", labelsize=f2["tick"])
fams = [f for f in FAM_COLOR if f in set(sub.family)]
handles = [
    plt.Line2D([], [], marker=FAM_MARKER[f], color="w",
               markerfacecolor=FAM_COLOR[f], markeredgecolor="k",
               markersize=9, label=f)
    for f in fams
]
ax.legend(handles=handles, fontsize=f2["tick"], loc="upper left")
plt.tight_layout()
fig.savefig(f"{OUT_PREFIX}_vs_ssm.pdf")
fig.savefig(f"{OUT_PREFIX}_vs_ssm.png", dpi=150)

# ── Figure 3: per-pair SSM ratio, grouped bars (one bar per model per pair) ──
# Compute ratio of means within each (action pair, model) subset.
pair_model_ratios: dict[str, list[float]] = {}
for m in SSM_VALID:
    csv_path = Path(f"out/bias_analysis/ssm_{METRIC}_{m}.csv")
    if not csv_path.exists():
        continue
    d = pd.read_csv(csv_path)
    per_pair = []
    for a, b in ACTION_PAIRS_LIST:
        subset = d[d["action"].isin([a, b])]
        per_pair.append(
            float(subset["d_skin"].mean() / subset["d_action"].mean())
            if len(subset) > 0 else np.nan
        )
    pair_model_ratios[m] = per_pair

models_ordered = [m for m in MODEL_ORDER if m in pair_model_ratios]
n_models  = len(models_ordered)
n_pairs   = len(PAIR_LABELS_ORDERED)
bar_width = 0.13
offsets   = np.arange(n_models) * bar_width - (n_models - 1) * bar_width / 2
x_pairs   = np.arange(n_pairs)

fig3_w = 13
f3 = font_sizes(fig3_w)
fig3, ax3 = plt.subplots(figsize=(fig3_w, 5.5))
for mi, m in enumerate(models_ordered):
    vals = pair_model_ratios[m]
    ax3.bar(
        x_pairs + offsets[mi], vals,
        width=bar_width,
        label=f"{m}  ({FAMILY[m]})",
        # color=MODEL_COLOR[m],
        edgecolor="white", linewidth=0.5,
    )

ax3.set_xticks(x_pairs)
ax3.set_xticklabels(PAIR_LABELS_ORDERED, fontsize=f3["tick"])
ax3.tick_params(axis="y", labelsize=f3["tick"])
ax3.set_ylabel(
    "SSM ratio  $d_\\mathrm{skin}$ / $d_\\mathrm{action}$\n"
    "(higher = more sensitive to skin tone)",
    fontsize=f3["label"],
)
ax3.set_title("SSM skin-tone sensitivity per action pair", fontsize=f3["title"], weight="bold")
ax3.legend(title="model", fontsize=f3["tick"], title_fontsize=f3["label"], ncol=1,
           bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax3.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout()
fig3.savefig(f"{OUT_PREFIX}_ssm_by_pair.pdf")
fig3.savefig(f"{OUT_PREFIX}_ssm_by_pair.png", dpi=150)

print(df.to_string(index=False))
print("\nsaved figures to out/linear_probes/")
