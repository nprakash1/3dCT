#!/usr/bin/env python3
"""Generate docs/results_f1_table.png: per-class F1 (worse/stable/improved) plus
accuracy & macro-F1 for the zero-shot baseline vs the trained temporal module."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- actual results ---
classes = ["worsened", "stable", "improved"]
zs = {"worsened": 0.429, "stable": 0.339, "improved": 0.288, "acc": 0.365, "macroF1": 0.352}
tr = {"worsened": 0.406, "stable": 0.361, "improved": 0.428, "acc": 0.401, "macroF1": 0.398}
CHANCE = 0.333

ZS_C = "#9ecae1"   # zero-shot (blue)
TR_C = "#fdae6b"   # trained (orange)

fig, (axb, axt) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

# ---------- left: grouped bar chart ----------
labels = classes + ["macro-F1", "accuracy"]
zs_vals = [zs["worsened"], zs["stable"], zs["improved"], zs["macroF1"], zs["acc"]]
tr_vals = [tr["worsened"], tr["stable"], tr["improved"], tr["macroF1"], tr["acc"]]
x = np.arange(len(labels))
w = 0.38

b1 = axb.bar(x - w/2, zs_vals, w, label="Zero-shot baseline", color=ZS_C, edgecolor="#333", linewidth=0.8)
b2 = axb.bar(x + w/2, tr_vals, w, label="Trained temporal module", color=TR_C, edgecolor="#333", linewidth=0.8)
axb.axhline(CHANCE, ls="--", lw=1.2, color="#888")
axb.text(len(labels)-0.5, CHANCE+0.008, "chance 0.33", ha="right", fontsize=8, color="#666")

for bars in (b1, b2):
    for r in bars:
        axb.text(r.get_x()+r.get_width()/2, r.get_height()+0.008, f"{r.get_height():.3f}",
                 ha="center", va="bottom", fontsize=8)

axb.set_xticks(x)
axb.set_xticklabels(labels, fontsize=9)
axb.set_ylim(0, 0.52)
axb.set_ylabel("F1 / score", fontsize=10)
axb.set_title("Per-class F1 and overall scores", fontsize=12, fontweight="bold")
axb.axvline(2.5, color="#ccc", lw=1)  # divider between per-class and overall
axb.legend(fontsize=9, loc="upper left")
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

# ---------- right: table ----------
axt.axis("off")
rows = ["F1 worsened", "F1 stable", "F1 improved", "macro-F1", "accuracy"]
cell = [
    [f"{zs['worsened']:.3f}", f"{tr['worsened']:.3f}"],
    [f"{zs['stable']:.3f}",   f"{tr['stable']:.3f}"],
    [f"{zs['improved']:.3f}", f"{tr['improved']:.3f}"],
    [f"{zs['macroF1']:.3f}",  f"{tr['macroF1']:.3f}"],
    [f"{zs['acc']:.3f}",      f"{tr['acc']:.3f}"],
]
tbl = axt.table(cellText=cell, rowLabels=rows,
                colLabels=["Zero-shot", "Temporal"],
                cellLoc="center", rowLoc="right", loc="center",
                colColours=[ZS_C, TR_C])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 1.7)
# bold the winning cell per row
best_col = [1, 1, 1, 1, 1]  # trained wins every row except worsened
best_col[0] = 0             # zero-shot wins on worsened
for r in range(len(rows)):
    tbl[(r+1, best_col[r])].set_text_props(fontweight="bold")
    tbl[(r+1, best_col[r])].set_facecolor("#eaf7ea")
axt.set_title("Summary table", fontsize=12, fontweight="bold", y=0.86)

fig.suptitle("Interval-change classification: Zero-shot vs Trained temporal module  (CT-RATE, 3-class)",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.02,
         "Trained module wins overall (macro-F1 0.398 vs 0.352) — mainly by recovering 'improved' (0.288 -> 0.428); "
         "still near the 0.33 floor, i.e. direction not solved.",
         ha="center", fontsize=8.5, style="italic", color="#555")

os.makedirs("docs", exist_ok=True)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
out = "docs/results_f1_table.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
print("saved", out)
