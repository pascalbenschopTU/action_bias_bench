"""Linear-probe + SSM corroboration figures.

Reads:
  out/linear_probes/_probe_summary.json            (probe matched/shifted f1, unseen)
  out/bias_analysis/ssm_<METRIC>_<model>.csv        (per-clip d_skin, d_action, r)
Writes:
  out/linear_probes/_probe_drop_by_model.{pdf,png}
  out/linear_probes/_probe_vs_ssm.{pdf,png}
  out/linear_probes/_probe_ssm_by_pair.{pdf,png}

Bar charts use Tab10 colours, no hatches.  The scatter plot adds distinct
marker shapes as a second cue for colorblind accessibility.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "hiera":       "img-ssl+ImgNet",   # MAE SSL + supervised IN1K fine-tune
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
probe = json.loads(Path("out/linear_probes/_probe_summary.json").read_text())

# matched F1 below this means the probe never learned the task, so its drop
# (robust-looking or not) is not evidence of anything — flagged in red below.
MATCHED_FLOOR = 0.75

rows = []
for m, r in probe.items():
    mm = r["eval_matched_unseen_ids"]
    ms = r["eval_shifted_unseen_ids"]
    csv = Path(f"out/bias_analysis/ssm_{METRIC}_{m}.csv")
    ssm_r = np.nan
    if csv.exists() and m in SSM_VALID:
        d = pd.read_csv(csv)
        # ratio of means (NOT mean of per-row ratios, which explodes when d_action~0)
        ssm_r = d["d_skin"].mean() / d["d_action"].mean()
    rows.append(dict(model=m, family=FAMILY[m], matched=mm, shifted=ms,
                     drop=mm - ms, ssm_r=ssm_r, reliable=mm >= MATCHED_FLOOR))
# bias = -(shifted − matched): positive = robust, negative = skin-tone sensitive.
# Sort descending so the least-biased model (vjepa2) is at the top.
df = pd.DataFrame(rows).sort_values("drop", ascending=True).reset_index(drop=True)

# ── Figure 1: probe drop by model ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bias = -df["drop"]   # positive = robust (only vjepa2), negative = skin-tone sensitive
for i, row in df.iterrows():
    ax.barh(
        i, -row["drop"],
        color=FAM_COLOR[row.family],
        edgecolor="white", linewidth=0.5,
    )
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df.model)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
ax.set_xlim(bias.min() - 0.15, bias.max() + 0.2)
ax.set_xlabel("shifted $-$ matched  ($\\Delta$F1)")
ax.set_title("Effect of a skin-tone shift on linear-probe accuracy")
for i, row in df.iterrows():
    val = -row["drop"]
    xpos = val + 0.005 if val >= 0 else val - 0.005
    ha = "left" if val >= 0 else "right"
    label_color = "#555"
    ax.text(xpos, i, f"matched F1={row.matched:.2f}", va="center", ha=ha,
            fontsize=7, color=label_color)
handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COLOR[f]) for f in FAM_COLOR]
ax.legend(handles, list(FAM_COLOR.keys()), fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig("out/linear_probes/_probe_drop_by_model.pdf")
fig.savefig("out/linear_probes/_probe_drop_by_model.png", dpi=150)

# ── Figure 2: probe drop vs SSM ratio ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
sub = df.dropna(subset=["ssm_r"])
dr  = sub["drop"].values
r   = np.corrcoef(sub.ssm_r.values, dr)[0, 1]
b, a = np.polyfit(sub.ssm_r.values, dr, 1)
xs = np.linspace(sub.ssm_r.min(), sub.ssm_r.max(), 50)
ax.plot(xs, a + b * xs, "--", color="grey", lw=1, zorder=1)
ax.margins(x=0.06)
for _, row in sub.iterrows():
    ax.scatter(
        row.ssm_r, row["drop"],
        s=130,
        color=FAM_COLOR[row.family],
        marker=FAM_MARKER[row.family],
        edgecolor="black", linewidth=1.2, zorder=3,
    )
    ax.annotate(row.model, (row.ssm_r, row["drop"]), fontsize=9,
                xytext=(6, 3), textcoords="offset points")
ax.set_title(f"SSM corroborates the linear probe\nPearson r = {r:.2f}")
ax.set_xlabel(
    "SSM ratio  $d_\\mathrm{skin}$ / $d_\\mathrm{action}$"
    "\n(higher = more sensitive to skin tone)",
    fontsize=10,
)
ax.set_ylabel("linear-probe skin-tone drop", fontsize=10)
fams = [f for f in FAM_COLOR if f in set(sub.family)]
handles = [
    plt.Line2D([], [], marker=FAM_MARKER[f], color="w",
               markerfacecolor=FAM_COLOR[f], markeredgecolor="k",
               markersize=9, label=f)
    for f in fams
]
ax.legend(handles=handles, fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig("out/linear_probes/_probe_vs_ssm.pdf")
fig.savefig("out/linear_probes/_probe_vs_ssm.png", dpi=150)

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

fig3, ax3 = plt.subplots(figsize=(13, 5.5))
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
ax3.set_xticklabels(PAIR_LABELS_ORDERED, fontsize=10)
ax3.set_ylabel(
    "SSM ratio  $d_\\mathrm{skin}$ / $d_\\mathrm{action}$\n"
    "(higher = more sensitive to skin tone)",
    fontsize=10,
)
ax3.set_title("SSM skin-tone sensitivity per action pair", fontsize=13, weight="bold")
ax3.legend(title="model", fontsize=8, title_fontsize=9, ncol=1,
           bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax3.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout()
fig3.savefig("out/linear_probes/_probe_ssm_by_pair.pdf")
fig3.savefig("out/linear_probes/_probe_ssm_by_pair.png", dpi=150)

print(df.to_string(index=False))
print("\nsaved figures to out/linear_probes/")
