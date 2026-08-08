#!/usr/bin/env python3
"""
make_model_v1_diagram.py

Visual of MODEL v1 -- the "Global Temporal Transformer" (GTT) from
scripts/09_global_temporal_model.py, plus a results table comparing it to the
zero-shot baselines (step 08).

Pipeline: cached 512-d v_prior/v_current (FROZEN MERLIN) ->
          [e_diff, prior+role, current+role] tokens -> tiny Transformer ->
          e_diff readout -> proj head -> v_d -> cosine vs 3 frozen text
          prototypes -> weighted CE.  Only the orange module is TRAINED.

Numbers are the real output of
  python scripts/09_global_temporal_model.py     (GTT_d512_L2_plain, 5-fold CV)

Output: docs/model_v1_global.png
Usage:  python scripts/make_model_v1_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZEN = "#cfe3ff"   # frozen MERLIN pieces
TRAIN = "#ffe0b3"    # trainable module (orange)
TEXT_C = "#d9f2d9"   # text side
PRED = "#efe0ff"     # prediction
IN = "#eeeeee"       # inputs
BEST = "#fff2b3"     # highlighted row
REF = "#e8e8e8"      # reference row

FIG_W, FIG_H = 15, 17


def box(ax, x, y, w, h, text, color, fs=10, bold=False, ec="#333", lw=1.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.5", linewidth=lw,
                 edgecolor=ec, facecolor=color, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, color="#333", rad=0.0, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                 connectionstyle=f"arc3,rad={rad}", linewidth=lw, color=color,
                 zorder=1, linestyle=ls))


def label(ax, x, y, text, fs=8.5, color="#333", italic=False, bg=False):
    bbox = dict(facecolor="white", edgecolor="none", pad=1.0) if bg else None
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style="italic" if italic else "normal", zorder=5, bbox=bbox)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 98.5, "Model v1:  Global Temporal Transformer  (trained; frozen MERLIN)",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(50, 96.2, "CT interval-change on DLT lesion pairs  -  learned difference over two global 512-d vectors",
            ha="center", fontsize=10.5, style="italic", color="#555")

    # ---------------- IMAGE SIDE (frozen, cached) ----------------
    box(ax, 17, 91, 20, 4, "PRIOR CT\nvolume", IN, fs=9)
    box(ax, 39, 91, 20, 4, "CURRENT CT\nvolume", IN, fs=9)
    box(ax, 17, 84, 20, 4.6, "MERLIN image enc\ni3D ResNet-152\n(FROZEN, cached)", FROZEN, fs=8.3)
    box(ax, 39, 84, 20, 4.6, "MERLIN image enc\ni3D ResNet-152\n(FROZEN, cached)", FROZEN, fs=8.3)
    arrow(ax, (17, 89), (17, 86.3))
    arrow(ax, (39, 89), (39, 86.3))
    label(ax, 17, 79.7, "v_prior (512-d)", fs=8.5, bg=True)
    label(ax, 39, 79.7, "v_current (512-d)", fs=8.5, bg=True)

    # ---------------- TRAINABLE MODULE (orange) ----------------
    ax.add_patch(FancyBboxPatch((6, 55.5), 52, 21.5,
                 boxstyle="round,pad=0.3,rounding_size=0.6", linewidth=1.6,
                 edgecolor="#e08a00", facecolor="#fff7ec", zorder=0))
    ax.text(32, 75.4, "TRAINABLE temporal module  (~few M params; the ONLY trained part)",
            ha="center", fontsize=9, fontweight="bold", color="#b3690a", zorder=3)

    # token row
    box(ax, 15, 70.5, 15, 4.4, "e_diff\n(learnable query)", TRAIN, fs=7.8)
    box(ax, 32, 70.5, 15, 4.4, "W·v_prior\n+ role_prior", TRAIN, fs=7.8)
    box(ax, 49, 70.5, 15, 4.4, "W·v_current\n+ role_current", TRAIN, fs=7.8)
    arrow(ax, (17, 81.7), (30, 72.7), rad=-0.12)   # prior -> prior token
    arrow(ax, (39, 81.7), (49, 72.7), rad=0.10)    # current -> current token

    box(ax, 32, 64, 40, 4.4, "tiny Transformer encoder  (2 layers, self-attention)", TRAIN, fs=9)
    arrow(ax, (15, 68.3), (24, 66.2), rad=0.0)
    arrow(ax, (32, 68.3), (32, 66.2), rad=0.0)
    arrow(ax, (49, 68.3), (40, 66.2), rad=0.0)

    box(ax, 32, 58, 40, 4.0, "e_diff readout  ->  proj head (d_model -> 512)", TRAIN, fs=9)
    arrow(ax, (32, 61.8), (32, 60.0))
    label(ax, 66, 58, "v_d (512-d,\nback in text space)", fs=8, italic=True, bg=True)
    arrow(ax, (52, 58), (60, 51.7), rad=-0.1)

    # ---------------- TEXT SIDE ----------------
    box(ax, 83, 91, 30, 4, "Class prompts:  improved / stable / worse\n(5 paraphrases each)", TEXT_C, fs=8)
    box(ax, 83, 84, 30, 4.6, "MERLIN text enc  -  Clinical-Longformer  (FROZEN)", TEXT_C, fs=8.2)
    arrow(ax, (83, 89), (83, 86.3))
    label(ax, 83, 79.7, "3 frozen text prototypes t_k (512-d)", fs=8.3, bg=True)
    arrow(ax, (83, 81.7), (83, 51.7), rad=0.0, ls="--", color="#2e8b57")
    label(ax, 92, 66, "cosine\ntargets", fs=8, color="#2e8b57", italic=True, bg=True)

    # ---------------- PREDICTION ----------------
    box(ax, 50, 49.5, 60, 4.4,
        "logits = logit_scale · cos(v_d, t_k)   ->   weighted cross-entropy",
        PRED, fs=9.5, bold=True)
    arrow(ax, (50, 47.3), (50, 45.0))
    box(ax, 50, 43, 46, 4.0,
        "prediction  in  { improved ,  stable ,  worse }", PRED, fs=10, bold=True)

    # ================= RESULTS TABLE =================
    ax.text(50, 37.5, "Results  (patient-level 5-fold CV; macro-F1 / accuracy / per-class F1)",
            ha="center", fontsize=12, fontweight="bold")

    cols = ["Method", "macro-F1", "Accuracy", "F1 improved", "F1 stable", "F1 worse"]
    colx = [7, 40, 53, 67, 80, 92]
    rows = [
        ("always-stable (ref)",        "0.305", "0.844", "0.000", "0.915", "0.000", REF),
        ("zero-shot (A) mag gate [08]", "0.315", "0.483", "0.113", "0.645", "0.188", "white"),
        ("zero-shot (C) ensemble [08]", "0.234", "0.289", "0.101", "0.393", "0.206", "white"),
        ("v1 GTT plain (trained)",     "0.321", "0.718", "0.039", "0.837", "0.087", BEST),
        ("v1 GTT + antisymmetry",      "0.251", "0.357", "0.110", "0.509", "0.135", "#ffe8cc"),
    ]

    x0, x1 = 4, 98
    y_hdr = 35
    row_h = 3.0

    ax.add_patch(FancyBboxPatch((x0, y_hdr - row_h / 2), x1 - x0, row_h,
                 boxstyle="square,pad=0", linewidth=0, facecolor="#d9d9d9", zorder=1))
    for j, c in enumerate(cols):
        ha = "left" if j == 0 else "center"
        xx = colx[0] if j == 0 else colx[j]
        ax.text(xx, y_hdr, c, ha=ha, va="center", fontsize=9, fontweight="bold",
                zorder=3)

    for i, r in enumerate(rows):
        yy = y_hdr - (i + 1) * row_h
        rowcolor = r[6]
        if rowcolor != "white":
            ax.add_patch(FancyBboxPatch((x0, yy - row_h / 2), x1 - x0, row_h,
                         boxstyle="square,pad=0", linewidth=0,
                         facecolor=rowcolor, zorder=1))
        ax.text(colx[0], yy, r[0], ha="left", va="center", fontsize=8.5, zorder=3)
        for j in range(1, 6):
            bold = (j == 1 and r[0].startswith("v1"))
            ax.text(colx[j], yy, r[j], ha="center", va="center",
                    fontsize=8.5, fontweight="bold" if bold else "normal", zorder=3)

    ax.plot([x0, x1], [y_hdr + row_h / 2] * 2, color="#333", lw=1.0, zorder=4)
    ax.plot([x0, x1], [y_hdr - row_h / 2] * 2, color="#333", lw=1.0, zorder=4)
    ax.plot([x0, x1], [y_hdr - len(rows) * row_h - row_h / 2] * 2,
            color="#333", lw=1.0, zorder=4)

    # ================= TAKEAWAY =================
    ty = 11.5
    ax.add_patch(FancyBboxPatch((5, ty - 8.5), 90, 15.5,
                 boxstyle="round,pad=0.3,rounding_size=0.6", linewidth=1.4,
                 edgecolor="#c0392b", facecolor="#ffecec", zorder=1))
    ax.text(50, ty + 5.2, "Takeaway", ha="center", fontsize=11,
            fontweight="bold", color="#c0392b")
    lines = [
        "- Plain v1: learning a difference operator on GLOBAL vectors barely beats the floor (macro-F1 0.321 vs 0.305); it mostly predicts 'stable'.",
        "- Antisymmetry (no-change = 0 vector) flips the story: improved/worse RECALL jump 4-7x (F1 0.039->0.110, 0.087->0.135) ...",
        "  ... but precision stays ~0.07-0.08, so it over-calls change and overall macro-F1 drops to 0.251.",
        "- Reading: DIRECTION is learnable (antisymmetry unlocks recall), but global pooling CAPS PRECISION (signal too weak to fire selectively).",
        "=> motivates PATCH-WISE cross-exam model (keep MERLIN's layer4 grid) + magnitude head to gate 'stable' by how-much, not direction.",
    ]
    for k, ln in enumerate(lines):
        ax.text(8, ty + 3.0 - k * 2.15, ln, ha="left", va="center", fontsize=7.9,
                color="#333")


    ax.text(50, 2.5, "Dataset: DLT (DeepLesion) mini -- 532 prior->current lesion pairs, "
            "243 volumes, 84.4% stable, depth=32.  Source: scripts/09_global_temporal_model.py",
            ha="center", fontsize=7.6, style="italic", color="#666")

    out = os.path.join("docs", "model_v1_global.png")
    os.makedirs("docs", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
