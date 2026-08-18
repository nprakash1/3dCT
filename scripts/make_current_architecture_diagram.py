#!/usr/bin/env python3
"""Architecture-only diagram of current proposed model -> docs/current_architecture.png"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FROZEN, FROZEN_E = "#cfe8ff", "#3b82c4"
TRAIN, TRAIN_E = "#ffe0b3", "#e08a1e"
TEXT, TEXT_E = "#e6f5e6", "#4a9b4a"
LOSS, LOSS_E = "#ffd6d6", "#c94b4b"
GREY, GREY_E = "#f0f0f0", "#888888"
OUT, OUT_E = "#f3e8ff", "#7c3aed"
WHITE, INNER = "#ffffff", "#fff8ee"

W, H = 15.2, 9.6
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, fc, ec, fs=10, bold=False, lw=1.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        zorder=5,
        linespacing=1.25,
    )


def arrow(p1, p2, color="#333333", lw=1.55, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=lw,
            color=color,
            connectionstyle="arc3,rad={0}".format(rad),
            zorder=4,
            shrinkA=2,
            shrinkB=2,
        )
    )


def snow(x, y):
    ax.text(x, y, "\u2744", fontsize=12, color=FROZEN_E, ha="center", va="center", zorder=8)


# Title
ax.text(
    W / 2,
    9.25,
    "Current Architecture — Finding-Conditioned CT Temporal Progression",
    ha="center",
    va="center",
    fontsize=15.5,
    fontweight="bold",
)
ax.text(
    W / 2,
    8.88,
    r"$d_f = g(v_{prior},\, v_{current},\, f)$"
    "     frozen CT-CLIP  +  trainable Difference Transformer",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#555555",
)

# Legend
for i, (lab, fc, ec) in enumerate(
    [
        ("Frozen", FROZEN, FROZEN_E),
        ("Trainable", TRAIN, TRAIN_E),
        ("Text", TEXT, TEXT_E),
        ("Loss", LOSS, LOSS_E),
    ]
):
    box(0.3 + i * 1.95, 8.4, 0.3, 0.24, "", fc, ec, fs=6)
    ax.text(0.68 + i * 1.95, 8.52, lab, va="center", fontsize=9, color="#444444")

# Inputs
box(0.3, 7.0, 1.65, 0.8, "Prior CT\nvolume", GREY, GREY_E, fs=9.5)
box(0.3, 5.7, 1.65, 0.8, "Current CT\nvolume", GREY, GREY_E, fs=9.5)
box(0.3, 4.4, 1.65, 0.8, "Finding f\n(id / name)", GREY, GREY_E, fs=9.5)

# Frozen image encoders
box(2.3, 6.9, 2.25, 1.0, "CT-CLIP image\nencoder (CTViT)", FROZEN, FROZEN_E, fs=9.5)
snow(4.35, 7.75)
box(2.3, 5.6, 2.25, 1.0, "CT-CLIP image\nencoder (shared)", FROZEN, FROZEN_E, fs=9.5)
snow(4.35, 6.45)
ax.text(
    3.4,
    6.72,
    "shared frozen weights",
    ha="center",
    fontsize=7.5,
    style="italic",
    color="#666666",
)
arrow((1.95, 7.4), (2.3, 7.4))
arrow((1.95, 6.1), (2.3, 6.1))

box(4.95, 7.05, 1.4, 0.7, "v_prior\n512-d", WHITE, FROZEN_E, fs=9.5, bold=True)
box(4.95, 5.75, 1.4, 0.7, "v_current\n512-d", WHITE, FROZEN_E, fs=9.5, bold=True)
arrow((4.55, 7.4), (4.95, 7.4))
arrow((4.55, 6.1), (4.95, 6.1))

box(2.3, 4.4, 2.25, 0.8, "Finding emb e_f\n(learned)", TRAIN, TRAIN_E, fs=9)
arrow((1.95, 4.8), (2.3, 4.8))

# Difference Transformer shell
TX, TY, TW, TH = 6.75, 3.85, 3.7, 4.4
ax.add_patch(
    FancyBboxPatch(
        (TX, TY),
        TW,
        TH,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=2.4,
        edgecolor=TRAIN_E,
        facecolor=TRAIN,
        zorder=2,
    )
)
ax.plot([TX + 0.22], [TY + TH - 0.24], marker="o", markersize=8, color=TRAIN_E, zorder=6)
ax.text(
    TX + TW / 2 + 0.08,
    TY + TH - 0.24,
    "Difference Transformer  g(.)",
    ha="center",
    fontsize=10.5,
    fontweight="bold",
    zorder=6,
)



ix, iw = TX + 0.22, TW - 0.44
box(ix, TY + 3.35, iw, 0.48, "Linear W + role emb (prior / current)", INNER, TRAIN_E, fs=8)
box(ix, TY + 2.65, iw, 0.48, "tokens: [ e_diff+e_f , t_p , t_c ]", INNER, TRAIN_E, fs=8)
box(ix, TY + 1.95, iw, 0.48, "Transformer encoder (2 layers)", INNER, TRAIN_E, fs=8.5)
box(ix, TY + 1.25, iw, 0.48, "h_diff -> Linear -> 512-d", INNER, TRAIN_E, fs=8.5)
box(ix, TY + 0.45, iw, 0.48, "optional mag head -> scalar m", INNER, TRAIN_E, fs=8.5)

arrow((6.35, 7.4), (TX, TY + 3.9), rad=-0.05)
arrow((6.35, 6.1), (TX, TY + 3.45), rad=0.04)
arrow((4.55, 4.8), (TX, TY + 2.8), rad=0.08)

box(10.85, 7.0, 1.65, 0.8, "d_f\n512-d", OUT, OUT_E, fs=11, bold=True)
box(10.85, 5.75, 1.65, 0.7, "m\nmag logit", OUT, OUT_E, fs=10)
arrow((TX + TW, TY + 3.25), (10.85, 7.4))
arrow((TX + TW, TY + 1.05), (10.85, 6.1), rad=0.03)

# Frozen text
box(12.85, 5.45, 2.1, 2.45, "", FROZEN, FROZEN_E, fs=8, lw=1.6)
snow(14.65, 7.65)
ax.text(
    13.9,
    7.65,
    "Frozen text",
    ha="center",
    fontsize=9.5,
    fontweight="bold",
    color=FROZEN_E,
)
box(13.0, 6.75, 1.8, 0.6, "CT-CLIP\ntext encoder", WHITE, FROZEN_E, fs=8.5)
box(13.0, 5.95, 1.8, 0.55, "PROTO[f]\n3 x 512", TEXT, TEXT_E, fs=8.5, bold=True)
box(13.0, 5.55, 1.8, 0.28, "temporal t", TEXT, TEXT_E, fs=7.5)

# Losses
box(10.85, 3.45, 4.1, 1.7, "", LOSS, LOSS_E, fs=8, lw=1.6)
ax.text(
    12.9,
    4.95,
    "Losses (on/off ablations)",
    ha="center",
    fontsize=10,
    fontweight="bold",
    color=LOSS_E,
)
box(11.0, 4.4, 3.8, 0.38, "L_CE    cos(d_f, PROTO[f]) x logit_scale", WHITE, LOSS_E, fs=8)
box(11.0, 3.95, 3.8, 0.38, "L_mag   BCE(m, change vs stable)", WHITE, LOSS_E, fs=8)
box(11.0, 3.55, 3.8, 0.32, "L_con   masked SupCon  d_f <-> t  (same finding)", WHITE, LOSS_E, fs=7.8)
arrow((11.65, 7.0), (11.65, 5.15), color=LOSS_E, lw=1.2)
arrow((11.65, 5.75), (11.65, 5.15), color=LOSS_E, lw=1.0)
arrow((13.9, 5.45), (13.9, 5.15), color=LOSS_E, lw=1.0)

# Inference
box(0.3, 1.05, 10.2, 1.95, "", "#fafafa", GREY_E, fs=8, lw=1.3)
ax.text(
    5.4,
    2.75,
    "Inference readout (no report text)",
    ha="center",
    fontsize=10.5,
    fontweight="bold",
)
box(0.5, 1.25, 2.2, 1.15, "v_prior, v_current\nfinding f", WHITE, GREY_E, fs=9.5)
box(3.15, 1.25, 2.4, 1.15, "d_f = g(v_p, v_c, f)", TRAIN, TRAIN_E, fs=10, bold=True)
box(6.0, 1.25, 4.2, 1.15, "y_hat = argmax  cos( d_f , PROTO[f] )", TEXT, TEXT_E, fs=10, bold=True)
arrow((2.7, 1.8), (3.15, 1.8))
arrow((5.55, 1.8), (6.0, 1.8))

ax.text(
    W / 2,
    0.5,
    "SupCon: + same finding & direction   |   - same finding, other direction   |   other findings ignored",
    ha="center",
    fontsize=8.5,
    color="#555555",
)
ax.text(
    W / 2,
    0.18,
    "tau_con independent of CE logit_scale     |     CT-CLIP image & text towers frozen",
    ha="center",
    fontsize=8.5,
    color="#555555",
)

out = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "docs", "current_architecture.png")
)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.12)
print("wrote", out)

