#!/usr/bin/env python3
"""Render the loss-ablation results as a clean table image.

config (magnitude / contrastive / target)  x  {macroF1, F1_worse, F1_stable, F1_improved}
mean +/- std over 3 seeds.  Highlights: best cell per metric column (bold + green),
and the best macro-F1 row (baseline) shaded.

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

# ---- build cell text ----
def fmt(pair):
    return f"{pair[0]:.3f} \u00b1 {pair[1]:.3f}"

cell_text = []
means = []  # per row, list of 4 means for highlighting
for r in rows:
    label = r[0]
    vals = r[1:]
    means.append([v[0] for v in vals])
    cell_text.append([label] + [fmt(v) for v in vals])

# best (max) mean per metric column -> row index
best_row_for_col = [max(range(len(rows)), key=lambda i: means[i][c]) for c in range(4)]
best_macro_row = best_row_for_col[0]  # baseline

# ---- palette ----
HEADER_BG = "#34557a"
HEADER_FG = "white"
BEST_CELL = "#d7f0d7"
BEST_ROW  = "#fff4e3"
ROW_ALT   = "#f6f8fb"

fig, ax = plt.subplots(figsize=(12.5, 4.6))
ax.axis("off")

ncols = 5
col_labels = ["config"] + col_headers
tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.0)

# column widths: wide config col, even metric cols
widths = [0.34, 0.165, 0.165, 0.165, 0.165]
for (r, c), cell in tbl.get_celld().items():
    cell.set_width(widths[c])
    cell.set_edgecolor("#cccccc")
    # header row
    if r == 0:
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_height(0.16)
        continue
    # body
    ri = r - 1
    # zebra + best-macro row shading
    if ri == best_macro_row:
        cell.set_facecolor(BEST_ROW)
    elif ri % 2 == 1:
        cell.set_facecolor(ROW_ALT)
    else:
        cell.set_facecolor("white")
    # left-align the config label
    if c == 0:
        cell.set_text_props(ha="left", fontfamily="monospace", fontsize=10)
        cell.PAD = 0.03

# highlight the best cell in each metric column (bold + green)
for c in range(1, 5):
    ri = best_row_for_col[c - 1]
    cell = tbl[(ri + 1, c)]
    cell.set_facecolor(BEST_CELL)
    cell.set_text_props(fontweight="bold")

# ---- titles / captions ----
ax.set_title("Loss ablation \u2014 magnitude \u00d7 contrastive \u00d7 target  (CT-RATE 3-class, test macro-F1)",
             fontsize=13, fontweight="bold", pad=16)
fig.text(0.5, 0.085,
         f"mean \u00b1 std over 3 seeds \u00b7 epochs=120 \u00b7 always-stable floor macro-F1 = {FLOOR:.3f}",
         ha="center", fontsize=9.5, color="#555")
fig.text(0.5, 0.03,
         "Best per column shaded green; best macro-F1 row (plain CE) shaded orange.  "
         "Adding contrastive+dynamic hurts most \u2014 it collapses 'stable' (0.459 \u2192 0.389).",
         ha="center", fontsize=8.8, style="italic", color="#666")

os.makedirs("docs", exist_ok=True)
out = "docs/ablation_table.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
print("saved", out)
