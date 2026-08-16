#!/usr/bin/env python3
"""Current end-to-end pipeline PNG (train_ctclip_temporal_colab.ipynb).
Output: docs/current_pipeline.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZEN, FROZEN_E = "#cfe8ff", "#3b82c4"
TRAIN, TRAIN_E = "#ffe0b3", "#e08a1e"
TEXT, TEXT_E = "#e6f5e6", "#4a9b4a"
LOSS, LOSS_E = "#ffd6d6", "#c94b4b"
GREY, GREY_E = "#eeeeee", "#999999"
DATA, DATA_E = "#f3e8ff", "#7c3aed"
INNER = "#fff4e3"

fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 17)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)


def arrow(p1, p2, color="#333", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
        connectionstyle="arc3,rad=%s" % rad, zorder=4))


def section(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor="#ccc", facecolor="#fafafa",
        linestyle="--", zorder=1))
    ax.text(x + 0.12, y + h - 0.25, title, fontsize=10.5, fontweight="bold",
            color="#444", va="top", zorder=2)


ax.text(8.5, 10.65, "Current Pipeline — CT Temporal Progression",
        ha="center", fontsize=16, fontweight="bold")
ax.text(8.5, 10.3,
        "train_ctclip_temporal_colab.ipynb  |  frozen CT-CLIP + Difference Transformer",
        ha="center", fontsize=10.5, color="#555")

# 1 data
section(0.2, 7.85, 16.6, 2.2, "1. Data and labels")
box(0.4, 8.1, 2.4, 1.5, "CT-RATE pairs\nprior + current CT", DATA, DATA_E, fs=9)
box(3.0, 8.1, 2.6, 1.5, "MedGemma labels\nfinding x direction\nexplicit tier (v3)", DATA, DATA_E, fs=9)
box(5.8, 8.1, 2.5, 1.5, "Hub splits\ntrain_* train/tune\nvalid_* final test", GREY, GREY_E, fs=9)
box(8.5, 8.1, 2.8, 1.5, "Row: (pair, finding, y)\ny = worse/stable/impr", TEXT, TEXT_E, fs=9)
box(11.5, 8.1, 2.5, 1.5, "Optional text\ndynamic sentences\nor evidence", TEXT, TEXT_E, fs=9)
box(14.2, 8.1, 2.3, 1.5, "18 findings\nCT-RATE set", GREY, GREY_E, fs=9)
for a, b in [(2.8, 3.0), (5.6, 5.8), (8.3, 8.5), (11.3, 11.5)]:
    arrow((a, 8.85), (b, 8.85))

# 2 frozen
section(0.2, 5.15, 16.6, 2.5, "2. Frozen CT-CLIP (cache once; never trained)")
box(0.4, 5.4, 2.2, 1.85, "Prior volume\n\nCurrent volume", GREY, GREY_E, fs=9)
box(2.85, 6.35, 2.7, 0.9, "CT-CLIP image\nCTViT (frozen)", FROZEN, FROZEN_E, fs=9, bold=True)
box(2.85, 5.45, 2.7, 0.75, "shared weights", FROZEN, FROZEN_E, fs=8.5)
box(5.8, 6.45, 1.8, 0.75, "v_prior\n512-d", "#fff", FROZEN_E, fs=9)
box(5.8, 5.5, 1.8, 0.75, "v_current\n512-d", "#fff", FROZEN_E, fs=9)
box(7.9, 5.4, 3.0, 1.85, "CT-CLIP text\nCXR-BERT (frozen)\n\nprompts per finding\nworse / stable / impr", FROZEN, FROZEN_E, fs=8.8)
box(11.2, 5.7, 2.5, 1.25, "PROTO[finding]\n3 x 512 cached", TEXT, TEXT_E, fs=9, bold=True)
box(13.95, 5.7, 2.55, 1.25, "Text emb cache\ndynamic or evidence", TEXT, TEXT_E, fs=9)
arrow((2.6, 6.7), (2.85, 6.7))
arrow((2.6, 5.9), (2.85, 5.8))
arrow((5.55, 6.8), (5.8, 6.8))
arrow((5.55, 5.85), (5.8, 5.85))
arrow((10.9, 6.3), (11.2, 6.3))

# 3 module
section(0.2, 2.45, 10.4, 2.5, "3. Trainable Difference Transformer only")
box(0.4, 2.7, 1.9, 1.85, "IN\nv_prior\nv_current\n(no finding)", GREY, GREY_E, fs=9)
ax.add_patch(FancyBboxPatch(
    (2.5, 2.7), 5.1, 1.85, boxstyle="round,pad=0.02,rounding_size=0.12",
    linewidth=2, edgecolor=TRAIN_E, facecolor=TRAIN, zorder=3))
ax.text(5.05, 4.3, "DifferenceTransformer", ha="center", fontsize=10.5,
        fontweight="bold", zorder=5)
box(2.7, 3.7, 4.7, 0.4, "role emb + e_diff query token", INNER, TRAIN_E, fs=8)
box(2.7, 3.2, 4.7, 0.4, "2-layer Transformer [e_diff, t_p, t_c]", INNER, TRAIN_E, fs=8)
box(2.7, 2.8, 4.7, 0.32, "head -> d (512-d)  [opt mag]", INNER, TRAIN_E, fs=8)
box(7.85, 3.05, 2.5, 1.2, "d  (512-d)\nONE vector / pair\nnot finding-specific",
    "#fff", TRAIN_E, fs=8.8, bold=True)
arrow((2.3, 3.6), (2.5, 3.6))
arrow((7.6, 3.6), (7.85, 3.65))
ax.text(5.05, 2.55, "ANTISYM=False default; finding not input to g()",
        ha="center", fontsize=8, style="italic", color="#8a5a12")

# 4 losses
section(10.8, 2.45, 6.0, 2.5, "4. Losses and scores")
box(11.05, 3.95, 5.5, 0.7, "CE: cos(d, PROTO[finding]) * exp(logit_scale)", LOSS, LOSS_E, fs=8.5)
box(11.05, 3.2, 5.5, 0.6, "+ opt l_mag * BCE(mag, change vs stable)", LOSS, LOSS_E, fs=8.5)
box(11.05, 2.65, 5.5, 0.45, "+ opt l_con * InfoNCE(d, text) pair-level", LOSS, LOSS_E, fs=8.5)
arrow((10.35, 3.7), (11.05, 4.25))

# 5 eval
section(0.2, 0.35, 16.6, 1.9, "5. Train loop and evaluation")
box(0.4, 0.55, 3.0, 1.35, "Batches: randperm\nrows; AdamW module", TRAIN, TRAIN_E, fs=9)
box(3.6, 0.55, 3.0, 1.35, "Early stop tune\nmacro-F1 (train_*)", GREY, GREY_E, fs=9)
box(6.8, 0.55, 3.2, 1.35, "Final test once\nvalid_* macro/per-class F1", TEXT, TEXT_E, fs=9)
box(10.2, 0.55, 3.2, 1.35, "Inference: d=g(vp,vc)\nargmax vs prototypes\n(no report)", FROZEN, FROZEN_E, fs=9)
box(13.6, 0.55, 2.9, 1.35, "Gap: same d for all\nfindings on a pair", LOSS, LOSS_E, fs=9)
arrow((3.4, 1.2), (3.6, 1.2))
arrow((6.6, 1.2), (6.8, 1.2))
arrow((10.0, 1.2), (10.2, 1.2))

ly = 0.08


def leg(x, fc, ec, label):
    ax.add_patch(FancyBboxPatch(
        (x, ly), 0.28, 0.18, boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + 0.36, ly + 0.09, label, fontsize=8, va="center")


leg(0.4, DATA, DATA_E, "data")
leg(2.0, FROZEN, FROZEN_E, "frozen")
leg(3.8, TRAIN, TRAIN_E, "trainable")
leg(5.6, TEXT, TEXT_E, "text/proto")
leg(7.6, LOSS, LOSS_E, "loss/gap")
leg(9.4, GREY, GREY_E, "split")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "docs", "current_pipeline.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)

