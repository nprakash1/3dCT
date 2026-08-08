#!/usr/bin/env python3
"""
make_prez_diagrams.py

Generates the three presentation diagrams into docs/:
  A) diagram_zeroshot.png    - naive zero-shot: d = v_cur - v_pri, cosine vs prompts (all frozen)
  B) diagram_transformer.png - trained temporal module -> v_d, cosine vs finding prompts, CE loss
  C) diagram_newarch.png     - TempA-VLP static+dynamic (cross-exam + antisymmetric diff, 2 InfoNCE)

Color code (matches scripts/make_model_v1_diagram.py):
  grey/blue = FROZEN MERLIN,  orange = TRAINABLE,  green = text side,  purple = prediction/loss.

Usage: python scripts/make_prez_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZEN = "#cfe3ff"   # frozen MERLIN
TRAIN  = "#ffe0b3"   # trainable (orange)
TEXT_C = "#d9f2d9"   # text side
PRED   = "#efe0ff"   # prediction / loss
IN     = "#eeeeee"   # inputs
CALL   = "#fff2b3"   # callout


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


def new_ax(w=13, h=8.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------- A: zero-shot
def diagram_zeroshot():
    fig, ax = new_ax()
    ax.text(50, 67, "Zero-shot baseline  -  difference of image embeddings, cosine to prompts",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(50, 63.5, "0 trainable parameters  -  the naive floor",
            ha="center", fontsize=10.5, style="italic", color="#b3690a")

    # image side
    box(ax, 16, 57, 18, 4.2, "PRIOR CT\nvolume", IN, fs=9)
    box(ax, 38, 57, 18, 4.2, "CURRENT CT\nvolume", IN, fs=9)
    box(ax, 16, 49, 18, 5.0, "MERLIN image enc\n(FROZEN, shared)", FROZEN, fs=8.3)
    box(ax, 38, 49, 18, 5.0, "MERLIN image enc\n(FROZEN, shared)", FROZEN, fs=8.3)
    arrow(ax, (16, 54.9), (16, 51.6))
    arrow(ax, (38, 54.9), (38, 51.6))
    label(ax, 16, 44.5, "v_prior (512-d)", bg=True)
    label(ax, 38, 44.5, "v_current (512-d)", bg=True)

    # difference
    box(ax, 27, 38, 26, 4.6, "d = v_current  -  v_prior", CALL, fs=10.5, bold=True)
    arrow(ax, (16, 42.6), (22, 40.3), rad=0.1)
    arrow(ax, (38, 42.6), (32, 40.3), rad=-0.1)

    # text side
    box(ax, 80, 57, 30, 4.6, "3 prompts per finding:\n\"...worsened / stable / improved\"", TEXT_C, fs=8.2)
    box(ax, 80, 49, 30, 5.0, "MERLIN text enc  -  Clinical-Longformer  (FROZEN)", TEXT_C, fs=8)
    arrow(ax, (80, 54.7), (80, 51.6))
    label(ax, 80, 44.5, "3 frozen text prototypes t_k", bg=True)

    # cosine + prediction
    box(ax, 50, 30, 46, 4.6, "cosine( d , t_k )   ->   argmax", PRED, fs=11, bold=True)
    arrow(ax, (27, 35.7), (44, 32.3), rad=0.05)
    arrow(ax, (80, 42.3), (58, 32.3), rad=-0.05, ls="--", color="#2e8b57")
    label(ax, 71, 37, "cosine targets", color="#2e8b57", italic=True, bg=True)
    box(ax, 50, 22, 40, 4.4, "prediction  {worsened, stable, improved}", PRED, fs=10.5, bold=True)
    arrow(ax, (50, 27.7), (50, 24.2))

    ax.text(50, 12, "Result:  macro-F1 0.352  |  accuracy 0.365   (chance 0.33; always-worsened 0.21)",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")
    ax.text(50, 8.5, "Behavior: almost never predicts 'improved' - collapses toward worsened/stable.",
            ha="center", fontsize=9, style="italic", color="#666")
    save(fig, "diagram_zeroshot.png")


# ---------------------------------------------------------- B: transformer/CE
def diagram_transformer():
    fig, ax = new_ax()
    ax.text(50, 67, "Trained temporal module  -  learned difference, cosine to finding prompts",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(50, 63.5, "only the orange module is trained (~few-M params); encoders + prompts frozen",
            ha="center", fontsize=10.5, style="italic", color="#b3690a")
    ax.text(50, 61.0, "operates on two GLOBAL 512-d vectors (v_prior, v_current)  -  no patches",
            ha="center", fontsize=9, style="italic", color="#666")

    box(ax, 16, 57, 18, 4.2, "PRIOR CT\nvolume", IN, fs=9)
    box(ax, 38, 57, 18, 4.2, "CURRENT CT\nvolume", IN, fs=9)
    box(ax, 16, 50, 18, 4.6, "MERLIN image enc\n(FROZEN, shared)", FROZEN, fs=8.3)
    box(ax, 38, 50, 18, 4.6, "MERLIN image enc\n(FROZEN, shared)", FROZEN, fs=8.3)
    arrow(ax, (16, 54.9), (16, 52.3))
    arrow(ax, (38, 54.9), (38, 52.3))
    label(ax, 16, 46.3, "v_prior", bg=True)
    label(ax, 38, 46.3, "v_current", bg=True)

    # trainable module
    ax.add_patch(FancyBboxPatch((5, 27), 52, 17,
                 boxstyle="round,pad=0.3,rounding_size=0.6", linewidth=1.6,
                 edgecolor="#e08a00", facecolor="#fff7ec", zorder=0))
    ax.text(31, 42.4, "TRAINABLE temporal module", ha="center", fontsize=9,
            fontweight="bold", color="#b3690a", zorder=3)
    box(ax, 16, 38.5, 15, 4.2, "e_diff\n(learnable query)", TRAIN, fs=7.8)
    box(ax, 40, 38.5, 22, 4.2, "v_prior, v_current\n(global 512-d) + role", TRAIN, fs=7.8)
    arrow(ax, (16, 44.0), (16, 40.6))
    arrow(ax, (38, 44.0), (40, 40.6))
    box(ax, 31, 33, 44, 4.2, "2-layer Transformer over 3 tokens  ->  e_diff readout  ->  proj head", TRAIN, fs=8.1)
    arrow(ax, (16, 36.4), (22, 35.1))
    arrow(ax, (40, 36.4), (38, 35.1))
    box(ax, 31, 29, 44, 3.6, "learnable logit_scale (temperature)", TRAIN, fs=8.3)
    label(ax, 66, 33, "v_d (512-d)", italic=True, bg=True)
    arrow(ax, (53, 33), (62, 24.5), rad=-0.1)

    # text side
    box(ax, 82, 57, 30, 4.6, "finding-specific 3 prompts\n(worsened/stable/improved)", TEXT_C, fs=8)
    box(ax, 82, 50, 30, 4.6, "MERLIN text enc (FROZEN)", TEXT_C, fs=8.2)
    arrow(ax, (82, 54.7), (82, 52.3))
    arrow(ax, (82, 47.7), (72, 24.5), rad=0.15, ls="--", color="#2e8b57")
    label(ax, 88, 37, "cosine\ntargets", color="#2e8b57", italic=True, bg=True)

    box(ax, 50, 22, 60, 4.6,
        "logits = logit_scale . cos(v_d, t_k)   ->   weighted cross-entropy",
        PRED, fs=10, bold=True)
    box(ax, 50, 15, 44, 4.2, "prediction  {worsened, stable, improved}", PRED, fs=10.5, bold=True)
    arrow(ax, (50, 19.7), (50, 17.1))

    ax.text(50, 8, "Result:  macro-F1 0.398  |  accuracy 0.401   (best of three)",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")
    ax.text(50, 4.8, "Behavior: predicts all 3 classes, but over-calls 'improved' (46 worsened -> improved). Wins by rebalancing, not by solving direction.",
            ha="center", fontsize=8.4, style="italic", color="#666")
    save(fig, "diagram_transformer.png")


# ------------------------------------------------------------- C: new arch
def diagram_newarch():
    fig, ax = new_ax(w=14, h=9.5)
    ax.set_xlim(0, 105)   # extra right margin so the "now/ts" cluster isn't clipped
    ax.text(50, 67.5, "New architecture  -  TempA-VLP static + dynamic (cross-exam + antisymmetric diff)",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(50, 64.3, "grey = FROZEN MERLIN   |   orange = TRAINABLE   |   dashed = optional LoRA",
            ha="center", fontsize=9.5, style="italic", color="#555")
    ax.text(50, 62.0, "GLOBAL-EMBEDDING version: dynamic path runs on pooled v_prior/v_current  "
            "(patch/token version is the v2 upgrade for location)",
            ha="center", fontsize=8, style="italic", color="#777")

    # images
    box(ax, 13, 58, 16, 4.0, "PRIOR CT", IN, fs=9)
    box(ax, 31, 58, 16, 4.0, "CURRENT CT", IN, fs=9)
    box(ax, 22, 51, 34, 4.6, "MERLIN image enc  (FROZEN, shared)", FROZEN, fs=8.5)
    arrow(ax, (13, 56.0), (18, 53.3), rad=0.1)
    arrow(ax, (31, 56.0), (26, 53.3), rad=-0.1)
    label(ax, 13, 46.5, "v_prior (global 512-d)", fs=7.8, bg=True)
    label(ax, 31, 46.5, "v_current (global 512-d)", fs=7.8, bg=True)
    label(ax, 47, 51, "v_current\n(global, to static)", fs=7.3, italic=True, bg=True)

    # trainable dynamic path
    ax.add_patch(FancyBboxPatch((4, 30), 44, 14.5,
                 boxstyle="round,pad=0.3,rounding_size=0.6", linewidth=1.6,
                 edgecolor="#e08a00", facecolor="#fff7ec", zorder=0))
    ax.text(26, 43.0, "DYNAMIC path (TRAINABLE)", ha="center", fontsize=8.6,
            fontweight="bold", color="#b3690a", zorder=3)
    box(ax, 22, 39, 34, 4.2, "temporal Transformer\n(v_prior, v_current + roles, CLS)", TRAIN, fs=8)
    arrow(ax, (13, 45.2), (16, 41.1), rad=0.0)
    arrow(ax, (31, 45.2), (28, 41.1), rad=0.0)
    box(ax, 22, 33.5, 34, 3.8, "antisymmetric diff  d = g(c,p) - g(p,c)", TRAIN, fs=8.3)
    arrow(ax, (22, 36.9), (22, 35.4))
    box(ax, 12, 26, 14, 3.6, "Proj_img", TRAIN, fs=8.5)
    arrow(ax, (18, 31.6), (13, 27.8), rad=0.1)
    label(ax, 12, 22.5, "vd (dynamic)", fs=8, italic=True, bg=True)

    # static path
    box(ax, 40, 26, 14, 3.6, "Proj_img", TRAIN, fs=8.5)
    arrow(ax, (46, 48.7), (40, 27.8), rad=0.25)
    label(ax, 40, 22.5, "vs (static)", fs=8, italic=True, bg=True)

    # text side
    box(ax, 82, 58, 30, 4.0, "CURRENT report (free text)", IN, fs=8.5)
    box(ax, 82, 51, 30, 4.4, "LLM splitter:  static  vs  dynamic", TEXT_C, fs=8.3)
    arrow(ax, (82, 56.0), (82, 53.2))
    box(ax, 70, 44, 20, 4.4, "\"change\" sentences", TEXT_C, fs=7.8)
    box(ax, 93, 44, 18, 4.4, "\"now\" sentences", TEXT_C, fs=7.8)
    arrow(ax, (77, 48.8), (72, 46.2), rad=0.0)
    arrow(ax, (87, 48.8), (92, 46.2), rad=0.0)
    box(ax, 82, 37.5, 40, 4.4, "MERLIN text enc  (FROZEN + optional LoRA)", TEXT_C, fs=8, ec="#2e8b57", lw=1.6)
    arrow(ax, (70, 41.8), (75, 39.7), rad=0.0)
    arrow(ax, (93, 41.8), (89, 39.7), rad=0.0)
    box(ax, 70, 31, 18, 3.6, "Proj_txt", TRAIN, fs=8.5)
    box(ax, 93, 31, 18, 3.6, "Proj_txt", TRAIN, fs=8.5)
    arrow(ax, (78, 35.3), (72, 32.8), rad=0.0)
    arrow(ax, (86, 35.3), (91, 32.8), rad=0.0)
    label(ax, 70, 27.6, "td (dynamic)", fs=8, italic=True, bg=True)
    label(ax, 93, 27.6, "ts (static)", fs=8, italic=True, bg=True)

    # shared space + losses
    ax.add_patch(FancyBboxPatch((6, 16), 88, 4.2,
                 boxstyle="round,pad=0.02,rounding_size=0.4", linewidth=1.4,
                 edgecolor="#333", facecolor="#f2f2f2", zorder=1))
    ax.text(50, 18.1, "SHARED  EMBEDDING  SPACE", ha="center", fontsize=10,
            fontweight="bold", zorder=3)
    arrow(ax, (12, 20.7), (12, 20.2))
    arrow(ax, (40, 20.7), (40, 20.2))
    arrow(ax, (70, 25.8), (70, 20.2))
    arrow(ax, (93, 25.8), (93, 20.2))

    box(ax, 27, 10.5, 40, 4.0, "L_dynamic = InfoNCE( vd , td )", PRED, fs=9.5, bold=True)
    box(ax, 74, 10.5, 40, 4.0, "L_static = InfoNCE( vs , ts )", PRED, fs=9.5, bold=True)
    arrow(ax, (16, 15.9), (22, 12.6), rad=0.1)
    arrow(ax, (70, 15.9), (66, 12.6), rad=0.0)
    arrow(ax, (40, 15.9), (48, 12.6), rad=-0.1)
    arrow(ax, (93, 15.9), (86, 12.6), rad=0.1)

    ax.text(50, 4.5,
            "L_total = a . L_dynamic + b . L_static     "
            "|  dynamic path learns DIRECTION (antisymmetry) on global vectors;  static path anchors DISEASE identity",
            ha="center", fontsize=8.6, fontweight="bold", color="#333")
    save(fig, "diagram_newarch.png")


def main():
    diagram_zeroshot()
    diagram_transformer()
    diagram_newarch()
    print("done: 3 diagrams in docs/")


if __name__ == "__main__":
    main()
