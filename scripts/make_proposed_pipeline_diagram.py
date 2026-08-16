#!/usr/bin/env python3
"""Proposed pipeline: finding-conditioned Difference Transformer + masked SupCon.
Layout uses wide gaps so boxes/text do not overlap.
Output: docs/proposed_pipeline.png
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
INNER = "#fff8ee"
NEW, NEW_E = "#fef3c7", "#d97706"

# Extra-tall canvas; each section has a dedicated title band (~0.45)
W, H = 18.5, 15.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=9.5, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
    ))
    ax.text(
        x + w / 2.0, y + h / 2.0, text,
        ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal", zorder=5,
        linespacing=1.28,
    )


def arrow(p1, p2, color="#333333", lw=1.55, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color,
        connectionstyle="arc3,rad={0}".format(rad), zorder=4,
        shrinkA=3, shrinkB=3,
    ))


def section(x, y, w, h, title):
    """Title sits in top ~0.42 of band; keep content boxes below y+h-0.55."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.15, edgecolor="#c5c5c5", facecolor="#fbfbfb",
        linestyle="--", zorder=1,
    ))
    ax.text(
        x + 0.2, y + h - 0.22, title,
        fontsize=11, fontweight="bold", color="#333333",
        ha="left", va="center", zorder=2,
    )


# ----- Title -----
ax.text(
    W / 2.0, 15.05,
    "Proposed Pipeline — Finding-Conditioned CT Temporal Progression",
    ha="center", va="center", fontsize=16.5, fontweight="bold",
)
ax.text(
    W / 2.0, 14.6,
    "d = g(v_prior, v_current, finding)   ·   CE + optional magnitude + masked same-finding SupCon",
    ha="center", va="center", fontsize=11, color="#555555",
)

# =====================================================================
# 1. Data  section y=12.15 h=2.15  title band -> boxes start y=12.35
# =====================================================================
section(0.3, 12.15, 17.9, 2.15, "1. Data and labels  (same sources; prefer report_explicit)")
box(0.5, 12.35, 2.55, 1.45, "CT-RATE pairs\nprior + current CT", DATA, DATA_E, fs=9.2)
box(3.3, 12.35, 2.85, 1.45, "Silver labels\nfinding x direction\nreport_explicit preferred", DATA, DATA_E, fs=9.0)
box(6.4, 12.35, 2.65, 1.45, "Hub splits\ntrain_* -> train / tune\nvalid_* -> final test", GREY, GREY_E, fs=9.0)
box(9.3, 12.35, 3.15, 1.45, "Row: (pair, finding, y)\ny in {worsened, stable, improved}", TEXT, TEXT_E, fs=9.0)
box(12.7, 12.35, 2.55, 1.45, "Contrastive text\ntemplates (v1) or\nreal temporal sentence", NEW, NEW_E, fs=8.8)
box(15.5, 12.35, 2.45, 1.45, "18 findings\nCT-RATE set", GREY, GREY_E, fs=9.0)
for x0, x1 in [(3.05, 3.3), (6.15, 6.4), (9.05, 9.3), (12.45, 12.7), (15.25, 15.5)]:
    arrow((x0, 13.075), (x1, 13.075))


# =====================================================================
# 2. Frozen  section y=9.2 h=2.65  boxes below y=11.3
# =====================================================================
section(0.3, 9.2, 17.9, 2.65, "2. Frozen CT-CLIP  (cache once; never trained)")
box(0.5, 9.5, 2.2, 1.95, "Prior volume\n\nCurrent volume", GREY, GREY_E, fs=9.2)
box(3.0, 10.55, 2.55, 0.9, "CT-CLIP image\nCTViT (frozen)", FROZEN, FROZEN_E, fs=9.2, bold=True)
box(3.0, 9.55, 2.55, 0.8, "shared frozen weights", FROZEN, FROZEN_E, fs=8.5)
box(5.85, 10.6, 1.75, 0.8, "v_prior\n512-d", "#ffffff", FROZEN_E, fs=9.2)
box(5.85, 9.55, 1.75, 0.8, "v_current\n512-d", "#ffffff", FROZEN_E, fs=9.2)
box(7.9, 9.5, 3.2, 1.95, "CT-CLIP text\nCXR-BERT (frozen)\n\nprompt bank / finding\nworse · stable · improved", FROZEN, FROZEN_E, fs=8.8)
box(11.4, 9.85, 2.7, 1.35, "PROTO[finding]\n3 x 512 cached", TEXT, TEXT_E, fs=9.5, bold=True)
box(14.4, 9.85, 3.5, 1.35, "Sentence emb t\n(template or real)\ncached, frozen", NEW, NEW_E, fs=9.0)
arrow((2.7, 10.9), (3.0, 10.9))
arrow((2.7, 10.0), (3.0, 9.95))
arrow((5.55, 11.0), (5.85, 11.0))
arrow((5.55, 9.95), (5.85, 9.95))
arrow((11.1, 10.5), (11.4, 10.5))
arrow((14.1, 10.5), (14.4, 10.5))
ax.text(13.0, 9.4, "finding name can also be embedded once for conditioning",
        ha="center", fontsize=7.8, style="italic", color="#666666")

# =====================================================================
# 3. Trainable  section y=5.15 h=3.7
# =====================================================================
section(0.3, 5.15, 11.15, 3.7, "3. Trainable module  —  finding-conditioned Difference Transformer")
box(0.5, 6.7, 2.2, 1.65, "INPUTS\n\nv_prior\nv_current\nfinding id / name", NEW, NEW_E, fs=9.0, bold=True, lw=2.0)
box(0.5, 5.5, 2.2, 0.95, "e_f = embed(finding)\n(learned or frozen text)", NEW, NEW_E, fs=8.3)

ax.add_patch(FancyBboxPatch(
    (3.05, 5.5), 5.4, 2.85,
    boxstyle="round,pad=0.02,rounding_size=0.14",
    linewidth=2.2, edgecolor=TRAIN_E, facecolor=TRAIN, zorder=3,
))
ax.text(5.75, 8.05, "DifferenceTransformer  (TRAINABLE)",
        ha="center", va="center", fontsize=11, fontweight="bold", zorder=5)
box(3.25, 7.3, 5.0, 0.5, "e_diff <- e_diff + e_f   (or finding as 4th token)", INNER, TRAIN_E, fs=8.5)
box(3.25, 6.65, 5.0, 0.5, "tokens: [ e_diff(f) , t_prior , t_current ]", INNER, TRAIN_E, fs=8.5)
box(3.25, 6.0, 5.0, 0.5, "2-layer Transformer encoder (self-attn)", INNER, TRAIN_E, fs=8.5)
ax.text(5.75, 5.7, "readout e_diff -> head -> d_f (512-d)   ·   optional mag head",
        ha="center", va="center", fontsize=8.3, zorder=5, color="#5c3d0a")

box(8.75, 6.2, 2.45, 1.85, "d_f  (512-d)\n\nchange emb for\nTHIS finding\non this pair",
    "#ffffff", TRAIN_E, fs=9.5, bold=True, lw=2.0)
arrow((2.7, 7.5), (3.05, 7.5))
arrow((2.7, 5.95), (3.25, 6.25), rad=0.12)
arrow((8.45, 7.15), (8.75, 7.15))
ax.text(5.75, 5.28, "Key change vs current code: finding enters g(.) so one pair -> many d_f",
        ha="center", fontsize=8.0, style="italic", color="#9a3412")


# =====================================================================
# 4. Losses  section y=5.15 h=3.7
# =====================================================================
section(11.7, 5.15, 6.5, 3.7, "4. Losses  (independently weighted)")
box(11.95, 7.7, 6.0, 0.75, "L_CE   cosine(d_f , PROTO[finding]) x exp(logit_scale)", LOSS, LOSS_E, fs=8.6)
box(11.95, 6.7, 6.0, 0.75, "L_mag  optional BCE(mag, change vs stable)", LOSS, LOSS_E, fs=8.6)
box(11.95, 5.5, 6.0, 0.95,
    "L_con  masked SupCon (same finding)\n+ same dir  ·  - other dir  ·  ignore other findings\ntau_con separate from CE logit_scale",
    NEW, NEW_E, fs=8.3, lw=2.0)
arrow((11.2, 7.15), (11.95, 8.05), rad=-0.08)
ax.text(14.95, 5.35, "L = λ_ce L_CE + λ_mag L_mag + λ_con L_con",
        ha="center", fontsize=9.0, fontweight="bold", color="#7f1d1d")

# =====================================================================
# 5. Batching / eval  section y=1.7 h=3.1
# =====================================================================
section(0.3, 1.7, 17.9, 3.1, "5. Batching, train loop, and evaluation")
box(0.5, 2.05, 3.2, 2.2,
    "Contrastive-aware batches\n~K findings x\nworse / stable / improved\n(so negatives exist)",
    NEW, NEW_E, fs=9.0, lw=2.0)
box(4.0, 2.05, 3.15, 2.2,
    "Train\nAdamW on module only\nearly-stop on tune\nmacro-F1 (train_*)",
    TRAIN, TRAIN_E, fs=9.2)
box(7.45, 2.05, 3.25, 2.2,
    "Final test once\nHub valid_* only\nmacro-F1 + per-class\n(explicit-only primary)",
    TEXT, TEXT_E, fs=9.0)
box(11.0, 2.05, 3.35, 2.2,
    "Inference\ninput: vp, vc, finding\nd_f = g(vp, vc, f)\nargmax vs PROTO[f]\n(no report text)",
    FROZEN, FROZEN_E, fs=8.8)
box(14.65, 2.05, 3.25, 2.2,
    "Ablations\nCE | +mag | +SupCon\ntemplates vs real text\n+/- finding conditioning",
    GREY, GREY_E, fs=8.8)
arrow((3.7, 3.15), (4.0, 3.15))
arrow((7.15, 3.15), (7.45, 3.15))
arrow((10.7, 3.15), (11.0, 3.15))
arrow((14.35, 3.15), (14.65, 3.15))

# ----- Legend -----
ly = 0.7


def leg(x, fc, ec, label):
    ax.add_patch(FancyBboxPatch(
        (x, ly), 0.34, 0.28,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3,
    ))
    ax.text(x + 0.48, ly + 0.14, label, fontsize=9.0, va="center", ha="left")


leg(0.5, DATA, DATA_E, "data / labels")
leg(3.15, FROZEN, FROZEN_E, "frozen CT-CLIP")
leg(5.95, TRAIN, TRAIN_E, "trainable")
leg(8.4, TEXT, TEXT_E, "text / prototypes")
leg(11.1, NEW, NEW_E, "new vs current pipeline")
leg(14.5, LOSS, LOSS_E, "loss")
leg(16.2, GREY, GREY_E, "split / ablations")

ax.text(
    W / 2.0, 0.28,
    "Compared to current_pipeline.png: finding enters the Difference Transformer; SupCon is same-finding masked (not instance InfoNCE).",
    ha="center", fontsize=8.5, color="#555555", style="italic",
)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "docs", "proposed_pipeline.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
