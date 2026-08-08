#!/usr/bin/env python3
"""
make_arch_diagram.py

Render the TempA-VLP-inspired architecture to a PNG using matplotlib
(no graphviz / mermaid-cli needed).

Design note
-----------
This is the MINIMAL-PARAMETER variant: MERLIN's image + text encoders are
FROZEN and only the change-aware (dynamic) path is trained.  Because every
node in a "static" InfoNCE(v_current, static-sentences) term would be frozen,
that term has NO trainable parameters in its graph and therefore trains
nothing -- so the static pipeline is removed entirely.  Only L_dynamic remains.

Trainable weights: cross-exam encoder + the dynamic image projection (Proj_img).
Everything else (both encoders, the text projection) is frozen / reused.

Output: docs/architecture_tempavlp.png
Usage:  python scripts/make_arch_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette ----
FROZEN = "#cfe3ff"   # light blue   = frozen MERLIN
TRAIN = "#ffe0b3"    # light orange = trainable
TEXT_C = "#d9f2d9"   # light green  = text / report
EMB = "#efe0ff"      # light purple = embeddings / shared space
LOSS = "#ffd6d6"     # light red    = losses
IN = "#eeeeee"       # inputs

FIG_W, FIG_H = 14, 19

# ---- column x-positions (single centered dynamic pipeline) ----
X_PRIOR = 32
X_CUR = 68
X_MID = 50


def box(ax, x, y, w, h, text, color, fs=11, bold=False, ec="#333333", lw=1.4):
    """x,y = center; w,h = size."""
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        linewidth=lw, edgecolor=ec, facecolor=color, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, color="#333333", style="-|>", rad=0.0, lw=1.6, ls="-"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=15,
        connectionstyle=f"arc3,rad={rad}", linewidth=lw,
        color=color, zorder=1, linestyle=ls,
    )
    ax.add_patch(a)


def label(ax, x, y, text, fs=9, color="#333", italic=False, bg=False):
    bbox = dict(facecolor="white", edgecolor="none", pad=1.0) if bg else None
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style="italic" if italic else "normal", zorder=5, bbox=bbox)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------- TITLE ----------------
    ax.text(50, 98.5, "TempA-VLP-Inspired Architecture (MERLIN / 3D CT)",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(50, 96, "single change-aware (dynamic) contrastive path",
            ha="center", va="center", fontsize=11, style="italic", color="#555")

    # ================= IMAGE SIDE (top) =================
    box(ax, X_PRIOR, 90, 24, 4.2, "PRIOR CT volume\n(224x224xD)", IN, fs=10)
    box(ax, X_CUR, 90, 24, 4.2, "CURRENT CT volume\n(224x224xD)", IN, fs=10)

    box(ax, X_PRIOR, 82, 24, 5.2,
        "MERLIN image encoder\ni3D ResNet-152 (FROZEN)", FROZEN, fs=9.5)
    box(ax, X_CUR, 82, 24, 5.2,
        "MERLIN image encoder\ni3D ResNet-152 (FROZEN)", FROZEN, fs=9.5)
    arrow(ax, (X_PRIOR, 87.9), (X_PRIOR, 84.6))
    arrow(ax, (X_CUR, 87.9), (X_CUR, 84.6))
    # shared-weights link
    arrow(ax, (X_PRIOR + 12, 82), (X_CUR - 12, 82), style="<|-|>",
          color="#3366cc", lw=1.1, ls="--")
    label(ax, 50, 83.4, "shared weights", fs=8.5, color="#3366cc", italic=True,
          bg=True)

    # token labels
    label(ax, X_PRIOR, 78.2, "layer4 tokens P'  (L x 2048)", fs=8.5, bg=True)
    label(ax, X_CUR, 78.2, "layer4 tokens P  (L x 2048)", fs=8.5, bg=True)

    # cross-exam encoder
    box(ax, X_MID, 70.5, 44, 6.2,
        "CROSS-EXAM ENCODER  (TRAINABLE)\n3-layer Transformer\n"
        "input: [ P + S + Xt ;  P' + S + Xt-1 ;  CLS ]", TRAIN, fs=10, bold=True)
    arrow(ax, (X_PRIOR, 77.0), (X_MID - 14, 73.7), rad=-0.08)
    arrow(ax, (X_CUR, 77.0), (X_MID + 14, 73.7), rad=0.08)
    label(ax, 74, 76.2, "+ positional S /\ntemporal Xt", fs=7.5, italic=True,
          bg=True)

    # antisymmetric diff
    box(ax, X_MID, 61.5, 38, 5.6,
        "ANTISYMMETRIC DIFF  (our upgrade)\n"
        "d = g(c,p) - g(p,c)   ->   no change = 0", TRAIN, fs=9.5)
    arrow(ax, (X_MID, 67.4), (X_MID, 64.3))
    label(ax, X_MID + 5.5, 65.8, "H_cls", fs=8, italic=True)

    # image projection (trainable)
    box(ax, X_MID, 54, 16, 4.4, "Proj_img\n(TRAINABLE)", TRAIN, fs=8)
    arrow(ax, (X_MID, 58.7), (X_MID, 56.2))

    # ---- backprop path of L_dynamic into the trainable image path ----
    bp = FancyArrowPatch((84, 49.2), (71.5, 69.5), arrowstyle="-|>",
                         mutation_scale=15, connectionstyle="arc3,rad=-0.30",
                         linewidth=1.9, color="#c0392b", ls=(0, (5, 3)),
                         zorder=4)
    ax.add_patch(bp)
    label(ax, 88, 60, "backprop of\nL_dynamic\n(updates cross-exam\n+ Proj_img)",
          fs=7.5, color="#c0392b", italic=True, bg=True)
    label(ax, 84, 74.5, "grad stops at\nfrozen MERLIN", fs=7, color="#c0392b",
          italic=True, bg=True)

    # ================= SHARED SPACE =================
    sx0, sw = 20, 60
    space = FancyBboxPatch((sx0, 44.5), sw, 4.6,
                           boxstyle="round,pad=0.02,rounding_size=0.6",
                           linewidth=2, edgecolor="#7a3fbf", facecolor=EMB,
                           zorder=2)
    ax.add_patch(space)
    ax.text(50, 46.8, "SHARED  EMBEDDING  SPACE", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#5a2ea6", zorder=3)

    # image embedding enters from top
    arrow(ax, (X_MID, 52.0), (X_MID, 49.2))
    label(ax, X_MID - 6, 50.7, "vd", fs=9, color="#5a2ea6", bg=True)

    # text embedding enters from bottom
    arrow(ax, (X_MID, 39.2), (X_MID, 44.4))
    label(ax, X_MID - 6, 42.0, "td", fs=9, color="#5a2ea6", bg=True)
    label(ax, X_MID + 12, 43.2, "change-alignment\ngame", fs=7.5,
          color="#c0392b", italic=True, bg=True)

    # ================= TEXT SIDE (bottom) =================
    box(ax, X_MID, 37, 16, 4.4, "Proj_txt\n(MERLIN, frozen)", FROZEN, fs=7.5)

    box(ax, X_MID, 30, 54, 5.0,
        "MERLIN text encoder  -  Clinical-Longformer  (FROZEN)", FROZEN, fs=10.5)
    arrow(ax, (X_MID, 32.5), (X_MID, 34.9))

    box(ax, X_MID, 23, 30, 4.6, '"CHANGE" sentences\n(dynamic / comparison)',
        TEXT_C, fs=9.5)
    arrow(ax, (X_MID, 25.3), (X_MID, 27.5))

    box(ax, X_MID, 15.5, 42, 4.6,
        "LLM report splitter\nkeep CHANGE (dynamic) sentences", TEXT_C, fs=9.5,
        bold=True)
    arrow(ax, (X_MID, 17.8), (X_MID, 20.7))

    box(ax, X_MID, 8.5, 30, 4.2, "CURRENT report\n(free text)", IN, fs=10)
    arrow(ax, (X_MID, 10.6), (X_MID, 13.2))

    # ================= OBJECTIVE PANEL (left) =================
    ox, oy, ow, oh = 12, 58, 22, 9.5
    panel = FancyBboxPatch((ox - ow / 2, oy - oh / 2), ow, oh,
                           boxstyle="round,pad=0.03,rounding_size=0.6",
                           linewidth=1.6, edgecolor="#c0392b", facecolor=LOSS,
                           zorder=2)
    ax.add_patch(panel)
    ax.text(ox, oy + oh / 2 - 1.5, "OBJECTIVE", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#c0392b", zorder=3)
    ax.text(ox, oy + 0.3, r"$L_{dynamic}=$", ha="center", va="center",
            fontsize=10, zorder=3)
    ax.text(ox, oy - 2.0, r"$\mathrm{InfoNCE}(v_d,\,t_d)$", ha="center",
            va="center", fontsize=10, zorder=3)

    ax.text(ox, 50.0,
            "Trainable:\ncross-exam encoder\n+ Proj_img\n(all else frozen)",
            ha="center", va="top", fontsize=8, color="#333", zorder=3)

    # ================= LEGEND (top-left) =================
    legend = [
        (FROZEN, "Frozen (MERLIN)"),
        (TRAIN, "Trainable"),
        (TEXT_C, "Text / report"),
        (EMB, "Embeddings / space"),
        (LOSS, "Loss"),
        (IN, "Inputs"),
    ]
    lx, ly = 2.5, 90
    ax.text(lx, ly + 3.0, "Legend", ha="left", va="center", fontsize=9.5,
            fontweight="bold")
    for i, (c, lab) in enumerate(legend):
        yy = ly - i * 3.0
        p = FancyBboxPatch((lx, yy - 0.9), 2.0, 1.8,
                           boxstyle="round,pad=0.02", linewidth=1,
                           edgecolor="#333", facecolor=c)
        ax.add_patch(p)
        ax.text(lx + 2.8, yy, lab, ha="left", va="center", fontsize=8.5)

    out = os.path.join("docs", "architecture_tempavlp.png")
    os.makedirs("docs", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
