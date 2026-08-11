#!/usr/bin/env python3
"""
Render the CURRENT model architecture (CT-CLIP Temporal Difference Transformer)
as actually configured: ANTISYM=False, MAGNITUDE=True, USE_CONTRASTIVE=True.

Difference from make_improved_frozen_diagram.py:
  - NO antisymmetry module: a single forward pass d = g(current, prior); direction
    is *learned* via prior/current role embeddings (there is no structural zero for
    "stable").
  - magnitude head and contrastive term are shown as ON (this run's config), not optional.

Layout is spaced so no text/arrows overlap.
Output: docs/current_model.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette ----
FROZEN, FROZEN_E = "#cfe8ff", "#3b82c4"     # frozen (blue)
TRAIN,  TRAIN_E  = "#ffe0b3", "#e08a1e"     # trainable (orange)
TEXT,   TEXT_E   = "#e6f5e6", "#4a9b4a"     # text / prototypes (green)
LOSS,   LOSS_E   = "#ffd6d6", "#c94b4b"     # losses (red)
GREY,   GREY_E   = "#eeeeee", "#999999"
INNER            = "#fff4e3"                  # inner sub-box fill

fig, ax = plt.subplots(figsize=(16.5, 10))
ax.set_xlim(0, 16.5)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.6, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)


def arrow(p1, p2, color="#333333", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=4))


def snow(x, y):
    ax.text(x, y, "\u2744", fontsize=14, color=FROZEN_E, ha="center", va="center", zorder=6)


# ================= TITLE =================
ax.text(8.25, 9.6, "Current Model \u2014 CT-CLIP Temporal Difference Transformer",
        ha="center", va="center", fontsize=16, fontweight="bold")
ax.text(8.25, 9.2, "config:  ANTISYM = False   \u00b7   MAGNITUDE = True   \u00b7   CONTRASTIVE = True   "
                   "(single forward pass, no antisymmetry)",
        ha="center", va="center", fontsize=10.5, color="#555555")

# ================= TOP FLOW: inputs -> frozen enc -> pooled -> diff module =================
# --- inputs ---
box(0.3, 7.30, 1.9, 0.85, "Prior CT\nvolume", GREY, GREY_E, fs=10)
box(0.3, 5.55, 1.9, 0.85, "Current CT\nvolume", GREY, GREY_E, fs=10)

# --- frozen image encoders (shared) ---
box(2.8, 7.20, 2.5, 1.05, "CT-CLIP image enc.\n(CTViT)", FROZEN, FROZEN_E, fs=10)
snow(5.10, 8.05)
box(2.8, 5.45, 2.5, 1.05, "CT-CLIP image enc.\n(CTViT)", FROZEN, FROZEN_E, fs=10)
snow(5.10, 6.30)
ax.text(4.05, 6.85, "shared frozen weights", ha="center", fontsize=8.5,
        style="italic", color="#777")
arrow((2.2, 7.72), (2.8, 7.72))
arrow((2.2, 5.97), (2.8, 5.97))

# --- pooled vectors (cached) ---
box(5.8, 7.35, 1.6, 0.75, "v_prior\n(512-d)", "#ffffff", FROZEN_E, fs=9.5)
box(5.8, 5.55, 1.6, 0.75, "v_current\n(512-d)", "#ffffff", FROZEN_E, fs=9.5)
ax.text(6.6, 8.32, "cached features", ha="center", fontsize=8, style="italic", color="#777")
arrow((5.3, 7.72), (5.8, 7.72))
arrow((5.3, 5.97), (5.8, 5.97))

# --- trainable Difference Transformer (big box) ---
TX, TY, TW, TH = 8.0, 4.55, 3.9, 3.7
ax.add_patch(FancyBboxPatch((TX, TY), TW, TH,
             boxstyle="round,pad=0.02,rounding_size=0.12",
             linewidth=2.4, edgecolor=TRAIN_E, facecolor=TRAIN))
# title bar centered (no marker dot to avoid overlap)
ax.text(TX + TW / 2, TY + TH - 0.28, "Difference Transformer (trainable)",
        ha="center", fontsize=10, fontweight="bold")

ix, iw = TX + 0.30, TW - 0.60
box(ix, TY + 2.68, iw, 0.52, "input proj  W  +  role emb (prior / current)", INNER, TRAIN_E, fs=8.6)
box(ix, TY + 1.95, iw, 0.52, "tokens:  [ e_diff ,  t_prior ,  t_current ]", INNER, TRAIN_E, fs=8.6)
box(ix, TY + 1.22, iw, 0.52, "self-attn Transformer encoder  (2 layers)", INNER, TRAIN_E, fs=8.6)
box(ix, TY + 0.42, iw, 0.55, "read out e_diff token  \u2192  heads", INNER, TRAIN_E, fs=8.6)

arrow((7.4, 7.72), (TX, TY + TH - 0.75), rad=-0.12)
arrow((7.4, 5.92), (TX, TY + TH - 0.75), rad=0.10)

# NO-antisymmetry note (well below the box, on its own line)
ax.text(TX + TW / 2, TY - 0.40,
        "single pass:  d = g(current, prior)   \u00b7   direction learned via role emb",
        ha="center", fontsize=8.8, style="italic", color="#8a5a12")
ax.text(TX + TW / 2, TY - 0.72,
        "(no antisymmetry \u2014 no structural zero for \u201cstable\u201d)",
        ha="center", fontsize=8.8, style="italic", color="#8a5a12")

# ================= RIGHT: diff emb d (center), aux heads offset further right =================
# diff emb d  (CE column center x = 13.0)
box(12.05, 6.95, 1.9, 0.9, "diff emb  d\n(512-d)", "#ffffff", TRAIN_E, fs=10, bold=True)
arrow((TX + TW, TY + 2.3), (12.05, 7.40))

# --- CE column straight down under d (x = 13.0) ---
box(11.70, 3.55, 2.6, 0.85, "cosine(d, protos)\n\u00d7 logit_scale", "#ffffff", "#555555", fs=9)
arrow((13.00, 6.95), (13.00, 4.40))                     # d -> cosine (clear of aux heads)
box(12.15, 2.55, 1.7, 0.65, "3-way logits", GREY, GREY_E, fs=9)
arrow((13.00, 3.55), (13.00, 3.20))
box(12.00, 1.55, 2.0, 0.70, "weighted CE\n(w / s / i)", LOSS, LOSS_E, fs=9)
arrow((13.00, 2.55), (13.00, 2.25))

# --- aux heads offset to the right so nothing overlaps the CE column ---
# contrastive (upper-right)
box(14.30, 7.55, 2.05, 0.95, "InfoNCE(d, evidence)\ncontrastive \u2014 ON", LOSS, LOSS_E, fs=8.2)
arrow((13.95, 7.55), (14.30, 8.00), rad=-0.10)          # d -> contrastive
# magnitude head (lower-right)
box(14.30, 5.80, 2.05, 0.95, "||change|| head \u2014 ON\nBCE(change vs stable)\n(aux, not a gate)", INNER, TRAIN_E, fs=7.8)
arrow((13.95, 7.10), (14.30, 6.45), rad=0.10)           # d -> magnitude

# ================= TEXT PATH (bottom-left) -> prototypes -> cosine =================
box(0.6, 1.45, 3.0, 0.90, "prompt bank / finding\n{worsened, stable, improved}", TEXT, TEXT_E, fs=8.6)
box(4.05, 1.40, 2.7, 1.00, "CT-CLIP text enc.\n(CXR-BERT)", FROZEN, FROZEN_E, fs=9.5)
snow(6.55, 2.20)
box(7.25, 1.45, 2.3, 0.90, "finding prototypes\n(3 \u00d7 512)", TEXT, TEXT_E, fs=9, bold=True)
arrow((3.6, 1.90), (4.05, 1.90))
arrow((6.75, 1.90), (7.25, 1.90))
arrow((9.55, 1.95), (11.70, 3.70), rad=-0.12)           # prototypes -> cosine (left edge)

# ================= LOSS SUMMARY (own line, bottom) =================
ax.text(8.25, 0.55, "L  =  weighted CE   +   0.5 \u00b7 BCE(magnitude)   +   0.5 \u00b7 InfoNCE(d, evidence)",
        ha="center", fontsize=9.8, color="#333", fontweight="bold")

# ================= LEGEND (very bottom row) =================
ly = 0.12
def legend(x, fc, ec, label):
    ax.add_patch(FancyBboxPatch((x, ly), 0.38, 0.26, boxstyle="round,pad=0.01,rounding_size=0.06",
                 linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + 0.50, ly + 0.13, label, fontsize=9, va="center")
legend(0.6, FROZEN, FROZEN_E, "frozen (CT-CLIP)")
legend(4.2, TRAIN, TRAIN_E, "trainable module")
legend(7.8, TEXT, TEXT_E, "text prototypes")
legend(11.0, LOSS, LOSS_E, "loss")

plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "current_model.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
print("wrote docs/current_model.png")
