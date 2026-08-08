#!/usr/bin/env python3
"""
make_baseline_diagram.py

Visual of the INITIAL zero-shot baseline (no training) for CT interval-change,
i.e. the experiment in scripts/08_prompt_variants.py, plus a results table.

Pipeline: prior & current volumes -> FROZEN MERLIN -> 512-d each ->
          d = v_current - v_prior -> three decision rules (A/B/C) ->
          predict {improved, stable, worse}.

Numbers are the real output of `python scripts/08_prompt_variants.py --depth 32`
on the DLT mini set (532 prior->current lesion pairs, 243 volumes, 84.4% stable).

Output: docs/baseline_results.png
Usage:  python scripts/make_baseline_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZEN = "#cfe3ff"
CALC = "#ffe0b3"
TEXT_C = "#d9f2d9"
PRED = "#efe0ff"
IN = "#eeeeee"
BEST = "#fff2b3"
REF = "#e8e8e8"

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

    ax.text(50, 98.5, "Initial Zero-Shot Baseline  (frozen MERLIN, no training)",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(50, 96.2, "CT interval-change on DLT lesion pairs  -  two prompt-variant decision rules",
            ha="center", fontsize=10.5, style="italic", color="#555")


    # ---------------- IMAGE SIDE ----------------
    box(ax, 17, 91, 20, 4, "PRIOR CT\nvolume", IN, fs=9)
    box(ax, 39, 91, 20, 4, "CURRENT CT\nvolume", IN, fs=9)
    box(ax, 17, 84, 20, 4.6, "MERLIN image enc\ni3D ResNet-152\n(FROZEN)", FROZEN, fs=8.5)
    box(ax, 39, 84, 20, 4.6, "MERLIN image enc\ni3D ResNet-152\n(FROZEN)", FROZEN, fs=8.5)
    arrow(ax, (17, 89), (17, 86.3))
    arrow(ax, (39, 89), (39, 86.3))
    label(ax, 17, 79.6, "v_prior (512-d)", fs=8.5, bg=True)
    label(ax, 39, 79.6, "v_current (512-d)", fs=8.5, bg=True)

    box(ax, 28, 73, 30, 4.4, "d = v_current - v_prior   (512-d)", CALC, fs=9.5, bold=True)
    arrow(ax, (17, 81.7), (24, 75.2), rad=-0.1)
    arrow(ax, (39, 81.7), (32, 75.2), rad=0.1)

    # ---------------- TEXT SIDE ----------------
    box(ax, 78, 91, 30, 4, "Class prompts:  improved / stable / worse", TEXT_C, fs=8.5)
    box(ax, 78, 84, 30, 4.6, "MERLIN text enc  -  Clinical-Longformer  (FROZEN)", TEXT_C, fs=8.5)
    arrow(ax, (78, 89), (78, 86.3))
    label(ax, 78, 79.6, "class text vectors (512-d)", fs=8.5, bg=True)
    arrow(ax, (78, 81.7), (78, 66.5), rad=0.0, ls="--", color="#2e8b57")
    label(ax, 88, 73, "cosine\ntargets", fs=8, color="#2e8b57", italic=True, bg=True)

    # ---------------- THREE DECISION RULES ----------------
    ax.text(50, 66.5, "two zero-shot decision rules", ha="center",
            fontsize=10, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0))



    box(ax, 30, 59, 34, 8.5,
        "(A) magnitude gate\n||d|| < tau  ->  stable\nelse cosine(d, {worse,\nimproved}) -> argmax",
        CALC, fs=8.6)
    box(ax, 70, 59, 34, 8.5,
        "(C) 3-way cosine\nprompt ENSEMBLE\n(5 paraphrases/class)\nargmax cosine",
        CALC, fs=8.6)
    arrow(ax, (28, 70.8), (30, 63.3), rad=-0.04)
    arrow(ax, (34, 71.5), (70, 63.3), rad=0.06)

    box(ax, 50, 50, 46, 4.2,
        "prediction  in  { improved ,  stable ,  worse }", PRED, fs=10, bold=True)
    arrow(ax, (30, 54.75), (42, 52.1), rad=0.0)
    arrow(ax, (70, 54.75), (58, 52.1), rad=0.0)


    # ================= RESULTS TABLE =================
    ax.text(50, 44.5, "Results  (macro-F1 / accuracy / per-class F1)",
            ha="center", fontsize=12, fontweight="bold")

    cols = ["Method", "macro-F1", "Accuracy", "F1 improved", "F1 stable", "F1 worse"]
    colx = [7, 40, 53, 67, 80, 92]      # left edge for Method, centers for rest
    rows = [
        ("always-stable (ref)", "0.305", "0.844", "0.000", "0.915", "0.000", REF),
        ("(A) magnitude gate",  "0.315", "0.483", "0.113", "0.645", "0.188", BEST),
        ("(C) prompt ensemble", "0.234", "0.289", "0.101", "0.393", "0.206", "white"),

    ]

    x0, x1 = 4, 98
    y_hdr = 41
    row_h = 3.6
    # header background
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
        ax.text(colx[0], yy, r[0], ha="left", va="center", fontsize=8.7, zorder=3)
        for j in range(1, 6):
            bold = (j == 1 and r[0].startswith("(A)"))
            ax.text(colx[j], yy, r[j], ha="center", va="center",
                    fontsize=8.7, fontweight="bold" if bold else "normal", zorder=3)

    # table border lines
    ax.plot([x0, x1], [y_hdr + row_h / 2] * 2, color="#333", lw=1.0, zorder=4)
    ax.plot([x0, x1], [y_hdr - row_h / 2] * 2, color="#333", lw=1.0, zorder=4)
    ax.plot([x0, x1], [y_hdr - (len(rows)) * row_h - row_h / 2] * 2,
            color="#333", lw=1.0, zorder=4)

    # ================= TAKEAWAY =================
    ty = 15

    ax.add_patch(FancyBboxPatch((5, ty - 8.5), 90, 15,
                 boxstyle="round,pad=0.3,rounding_size=0.6", linewidth=1.4,
                 edgecolor="#c0392b", facecolor="#ffecec", zorder=1))
    ax.text(50, ty + 4.7, "Takeaway", ha="center", fontsize=11,
            fontweight="bold", color="#c0392b")
    lines = [
        "- Best zero-shot macro-F1 = 0.315 (rule A) is essentially the always-'stable' floor (0.305).",
        "- Adding a 'stable' DIRECTION prompt (rule C) HURTS (0.234): cosine cannot represent 'no change'",

        "  -- a ~0 change vector has no direction, so stable scatters into worse/improved.",
        "- MERLIN's global 512-d embedding difference does NOT cleanly encode interval change.",
        "=> motivates a trainable cross-exam module: antisymmetric difference (stable = 0) + magnitude head.",
    ]
    for k, ln in enumerate(lines):
        ax.text(8, ty + 2.2 - k * 2.4, ln, ha="left", va="center", fontsize=8.6,
                color="#333")

    ax.text(50, 3, "Dataset: DLT (DeepLesion) mini -- 532 prior->current lesion pairs, "
            "243 volumes, 84.4% stable, depth=32.  Source: scripts/08_prompt_variants.py",
            ha="center", fontsize=7.6, style="italic", color="#666")

    out = os.path.join("docs", "baseline_results.png")
    os.makedirs("docs", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
