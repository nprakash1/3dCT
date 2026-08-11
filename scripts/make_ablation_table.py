#!/usr/bin/env python3
"""Render the loss-ablation results as a clean table image, with a legend that
explains the three config knobs (mag / con / tgt) and every metric column.

Output: docs/ablation_table.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- data: (label, macroF1, F1_worse, F1_stable, F1_improved) as (mean, std) ----
rows = [
    ("mag=off | con=off | tgt=-",        (0.536, 0.014), (0.506, 0.009), (0.459, 0.027), (0.644, 0.018)),
    ("mag=on  | con=off | tgt=-",        (0.527, 0.010), (0.478, 0.027), (0.469, 0.025), (0.635, 0.015)),
    ("mag=on  | con=on  | tgt=dynamic",  (0.491, 0.018), (0.465, 0.034), (0.422, 0.019), (0.587, 0.014)),
    ("mag=off | con=on  | tgt=dynamic",  (0.496, 0.005), (0.515, 0.008), (0.389, 0.019), (0.583, 0.029)),
    ("mag=off | con=on  | tgt=evidence", (0.528, 0.008), (0.508, 0.021), (0.471, 0.033), (0.605, 0.046)),
    ("mag=on  | con=on  | tgt=evidence", (0.531, 0.008), (0.524, 0.022), (0.444, 0.017), (0.625, 0.019)),
]
col_headers = ["macro-F1", "F1 worse", "F1 stable", "F1 improved"]
FLOOR = 0.168

def fmt(p):
    return f"{p[0]:.3f} \u00b1 {p[1]:.3f}"

cell_text, means = [], []
for r in rows:
    means.append([v[0] for v in r[1:]])
    cell_text.append([r[0]] + [fmt(v) for v in r[1:]])

best_row_for_col = [max(range(len(rows)), key=lambda i: means[i][c]) for c in range(4)]
best_macro_row = best_row_for_col[0]

HEADER_BG, HEADER_FG = "#34557a", "white"
BEST_CELL, BEST_ROW, ROW_ALT = "#d7f0d7", "#fff4e3", "#f6f8fb"

fig = plt.figure(figsize=(13, 7.6))
ax = fig.add_axes([0.04, 0.46, 0.92, 0.42])   # table occupies upper portion
ax.axis("off")

col_labels = ["config  (mag | con | tgt)"] + col_headers
tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.0)

widths = [0.34, 0.165, 0.165, 0.165, 0.165]
for (r, c), cell in tbl.get_celld().items():
    cell.set_width(widths[c])
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_height(0.16)
        continue
    ri = r - 1
    if ri == best_macro_row:
        cell.set_facecolor(BEST_ROW)
    elif ri % 2 == 1:
        cell.set_facecolor(ROW_ALT)
    else:
        cell.set_facecolor("white")
    if c == 0:
        cell.set_text_props(ha="left", fontfamily="monospace", fontsize=10)
        cell.PAD = 0.03

for c in range(1, 5):
    ri = best_row_for_col[c - 1]
    cell = tbl[(ri + 1, c)]
    cell.set_facecolor(BEST_CELL)
    cell.set_text_props(fontweight="bold")

ax.set_title("Loss ablation \u2014 magnitude \u00d7 contrastive \u00d7 target   (CT-RATE, 3-class interval change)",
             fontsize=13.5, fontweight="bold", pad=14)

# ================= LEGEND / DEFINITIONS =================
LX, RX = 0.06, 0.55
mono = {"family": "monospace"}

fig.text(LX, 0.375, "Config knobs  (each row = one training run)",
         fontsize=11, fontweight="bold", color="#34557a")
fig.text(LX, 0.330, "mag", fontsize=10.5, fontweight="bold", **mono)
fig.text(LX + 0.055, 0.330,
         "magnitude head \u2014 auxiliary BCE(change vs stable) on \u2016d\u2016.   on / off", fontsize=9.6)
fig.text(LX, 0.288, "con", fontsize=10.5, fontweight="bold", **mono)
fig.text(LX + 0.055, 0.288,
         "contrastive term \u2014 InfoNCE aligning d with a report sentence.   on / off", fontsize=9.6)
fig.text(LX, 0.246, "tgt", fontsize=10.5, fontweight="bold", **mono)
fig.text(LX + 0.055, 0.246, "contrastive target sentence (only when con=on):", fontsize=9.6)
fig.text(LX + 0.075, 0.212, "dynamic  = report's comparison / change sentences", fontsize=9.2, color="#444")
fig.text(LX + 0.075, 0.182, "evidence = MedGemma-quoted line from the CURRENT report", fontsize=9.2, color="#444")
fig.text(LX + 0.075, 0.152, "\u2013           = not applicable (con=off)", fontsize=9.2, color="#444")

fig.text(RX, 0.375, "Metrics  (test set; higher is better)",
         fontsize=11, fontweight="bold", color="#34557a")
fig.text(RX, 0.330, "macro-F1", fontsize=10, fontweight="bold", **mono)
fig.text(RX + 0.12, 0.330, "unweighted mean of the 3 per-class F1 scores", fontsize=9.6)
fig.text(RX, 0.288, "F1 worse", fontsize=10, fontweight="bold", **mono)
fig.text(RX + 0.12, 0.288, "per-class F1 for the 'worsened' class", fontsize=9.6)
fig.text(RX, 0.246, "F1 stable", fontsize=10, fontweight="bold", **mono)
fig.text(RX + 0.12, 0.246, "per-class F1 for the 'stable' class", fontsize=9.6)
fig.text(RX, 0.204, "F1 improved", fontsize=10, fontweight="bold", **mono)
fig.text(RX + 0.12, 0.204, "per-class F1 for the 'improved' class", fontsize=9.6)
fig.text(RX, 0.150, f"floor = {FLOOR:.3f}", fontsize=10, fontweight="bold", **mono)
fig.text(RX + 0.12, 0.150, "trivial always-predict-'stable' macro-F1", fontsize=9.6)

fig.text(0.5, 0.075,
         "Cells: mean \u00b1 std over 3 seeds \u00b7 epochs=120.   "
         "Green = best in that metric column;  orange row = best macro-F1 (plain CE).",
         ha="center", fontsize=9, color="#555")
fig.text(0.5, 0.035,
         "Takeaway: plain CE is best (0.536); adding contrastive+dynamic hurts most \u2014 "
         "it collapses 'stable' (0.459 \u2192 0.389).",
         ha="center", fontsize=8.8, style="italic", color="#666")

os.makedirs("docs", exist_ok=True)
out = "docs/ablation_table.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
print("saved", out)
