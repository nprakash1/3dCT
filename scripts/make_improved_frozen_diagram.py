#!/usr/bin/env python3
"""
Render the "Improved Frozen Model" architecture (CT-CLIP Temporal Difference
Transformer) exactly as implemented in notebooks/train_ctclip_temporal_colab.ipynb.

Output: docs/improved_frozen_model.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette ----
FROZEN = "#cfe8ff"   # frozen (light blue)
FROZEN_E = "#3b82c4"
TRAIN = "#ffe0b3"    # trainable (orange)
TRAIN_E = "#e08a1e"
TEXT = "#e6f5e6"     # text / prototypes (green)
TEXT_E = "#4a9b4a"
LOSS = "#ffd6d6"     # losses (red)
LOSS_E = "#c94b4b"
GREY = "#eeeeee"
GREY_E = "#999999"

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=11, bold=False, style="round"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size=0.10" if style == "round"
                       else "square,pad=0.02",
                       linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)
    return (x + w / 2, y + h / 2)


def arrow(p1, p2, color="#333333", lw=1.8, style="-|>", rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        lw=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def snow(x, y):
    ax.text(x, y, "\u2744", fontsize=15, color=FROZEN_E, ha="center", va="center", zorder=6)


def fire(x, y):
    # small orange dot marker (emoji glyphs are unavailable in the default font)
    ax.plot([x], [y], marker="o", markersize=9, color=TRAIN_E, zorder=6)



# ---- title ----
ax.text(7.5, 8.65, "Improved Frozen Model \u2014 CT-CLIP Temporal Difference Transformer",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(7.5, 8.25, "frozen chest-native CT foundation model  +  a small trainable difference module",
        ha="center", va="center", fontsize=10.5, color="#555555")

# ============ INPUTS ============
box(0.3, 6.7, 2.1, 0.9, "Prior CT\nvolume", GREY, GREY_E, fs=10)
box(0.3, 4.6, 2.1, 0.9, "Current CT\nvolume", GREY, GREY_E, fs=10)

# ============ FROZEN CT-CLIP IMAGE ENCODER (shared) ============
box(2.9, 6.6, 2.6, 1.05, "CT-CLIP image enc.\n(CTViT, frozen)", FROZEN, FROZEN_E, fs=10)
snow(5.35, 7.5)
box(2.9, 4.5, 2.6, 1.05, "CT-CLIP image enc.\n(CTViT, frozen)", FROZEN, FROZEN_E, fs=10)
snow(5.35, 5.4)
ax.text(4.2, 3.95, "shared weights", ha="center", fontsize=8.5, style="italic", color="#777")

arrow((2.4, 7.15), (2.9, 7.12))
arrow((2.4, 5.05), (2.9, 5.02))

# pooled vectors
box(5.9, 6.75, 1.7, 0.75, "v_prior\n(512-d)", "#ffffff", FROZEN_E, fs=9.5)
box(5.9, 4.65, 1.7, 0.75, "v_current\n(512-d)", "#ffffff", FROZEN_E, fs=9.5)
arrow((5.5, 7.12), (5.9, 7.12))
arrow((5.5, 5.02), (5.9, 5.02))

# ============ TRAINABLE DIFFERENCE TRANSFORMER ============
tx, ty, tw, th = 8.1, 4.2, 3.5, 3.4
p = FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0.02,rounding_size=0.12",
                   linewidth=2.2, edgecolor=TRAIN_E, facecolor=TRAIN)
ax.add_patch(p)
fire(tx + 0.32, ty + th - 0.28)
ax.text(tx + tw / 2, ty + th - 0.30, "Difference Transformer  (trainable)",
        ha="center", fontsize=10.5, fontweight="bold")

# internal tokens
box(tx + 0.35, ty + 2.25, tw - 0.7, 0.55,
    "input proj  W  +  role emb  (prior / current)", "#fff4e3", TRAIN_E, fs=8.8)
box(tx + 0.35, ty + 1.55, tw - 0.7, 0.55,
    "tokens: [ e_diff (query) , t_prior , t_current ]", "#fff4e3", TRAIN_E, fs=8.8)
box(tx + 0.35, ty + 0.85, tw - 0.7, 0.55,
    "tiny Transformer encoder  (2 layers)", "#fff4e3", TRAIN_E, fs=8.8)
box(tx + 0.35, ty + 0.18, tw - 0.7, 0.5,
    "read out e_diff \u2192 Linear \u2192 512-d", "#fff4e3", TRAIN_E, fs=8.8)

arrow((7.6, 7.12), (tx + tw / 2, ty + 2.82), rad=-0.15)
arrow((7.6, 5.02), (tx + tw / 2, ty + 2.82), rad=0.12)

# antisymmetry note
ax.text(tx + tw / 2, ty - 0.32,
        "antisym:  d = g(c,p) \u2212 g(p,c)   \u2192  stable = 0 vector",
        ha="center", fontsize=8.6, style="italic", color="#8a5a12")

# ============ difference embedding d ============
box(12.05, 5.55, 1.7, 0.85, "diff emb  d\n(512-d)", "#ffffff", TRAIN_E, fs=10, bold=True)
arrow((tx + tw, ty + 1.7), (12.05, 5.97))

# magnitude head (optional)
box(12.05, 4.35, 1.7, 0.75, "||change||\nmag head (opt.)", "#fff4e3", TRAIN_E, fs=8.6)
arrow((tx + tw, ty + 0.9), (12.05, 4.72), rad=0.1)

# ============ FROZEN TEXT TOWER -> PROTOTYPES ============
box(8.4, 1.05, 3.0, 1.05, "CT-CLIP text enc.\n(CXR-BERT, frozen)", FROZEN, FROZEN_E, fs=9.5)
snow(11.2, 1.95)
box(5.0, 1.15, 3.0, 0.85,
    "prompt bank per finding\n{worsened / stable / improved}", TEXT, TEXT_E, fs=8.6)
arrow((8.0, 1.57), (8.4, 1.57))
box(11.9, 1.1, 1.85, 0.95, "finding\nprototypes\n(3 \u00d7 512)", TEXT, TEXT_E, fs=8.8, bold=True)
arrow((11.4, 1.57), (11.9, 1.55))

# ============ COSINE + LOSS ============
box(11.7, 3.15, 2.4, 0.8, "cosine(d, protos)\n\u00d7 logit_scale", "#ffffff", "#555", fs=9)
arrow((12.9, 5.55), (12.9, 3.95))                       # d down
arrow((12.85, 2.05), (12.9, 3.15))                      # protos up
box(11.9, 2.05, 2.0, 0.62, "3-way logits", GREY, GREY_E, fs=9)  # (visual anchor)

# main loss
box(9.55, 3.2, 1.9, 0.7, "weighted CE\n(w / s / i)", LOSS, LOSS_E, fs=9)
arrow((11.7, 3.55), (11.45, 3.55))

# contrastive aux (optional)
box(9.35, 5.7, 2.4, 0.62, "InfoNCE(d, evidence)\ncontrastive (opt.)", LOSS, LOSS_E, fs=8.4)
arrow((12.05, 5.97), (11.75, 6.0))

# ============ LEGEND ============
lx, ly = 0.3, 1.15
box(lx, ly + 0.75, 0.42, 0.32, "", FROZEN, FROZEN_E); ax.text(lx + 0.6, ly + 0.91, "frozen (CT-CLIP)", fontsize=9, va="center")
box(lx, ly + 0.30, 0.42, 0.32, "", TRAIN, TRAIN_E); ax.text(lx + 0.6, ly + 0.46, "trainable module", fontsize=9, va="center")
box(lx, ly - 0.15, 0.42, 0.32, "", TEXT, TEXT_E); ax.text(lx + 0.6, ly + 0.01, "text prototypes", fontsize=9, va="center")
box(lx, ly - 0.60, 0.42, 0.32, "", LOSS, LOSS_E); ax.text(lx + 0.6, ly - 0.44, "loss", fontsize=9, va="center")
ax.text(lx, ly + 1.35, "Only the orange module is trained (~few M params).",
        fontsize=8.6, style="italic", color="#555")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "improved_frozen_model.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
