# Speaker Notes — Proposed Architecture (TempA-VLP-inspired, MERLIN / 3D CT)

> Companion to `docs/architecture_tempavlp.png`. Timing target: ~5–7 minutes.
> **bold** = say aloud, *italics* = optional depth / Q&A.

---

## 0. One-line framing (say first)
"The baseline showed a frozen model's raw embedding difference can't read interval
change. So we **keep the foundation model frozen** and add **one small, trainable
module** that learns the *change* between two scans and aligns it with the
*comparison language* in the report. That's the entire idea on this slide."

---

## 1. What we keep frozen (and why it matters)
Point at the blue boxes.
- **MERLIN's image encoder (i3D ResNet-152)** and **text encoder (Clinical
  Longformer)** stay **frozen** — no weights change.
- Both CTs (prior + current) go through the **same shared-weight** image encoder.
- *Why frozen: (1) cost — we train a few-million-parameter module, not a
  billion-parameter backbone; (2) reproducibility; (3) it isolates the
  contribution of our module — any gain is provably from the change module, not
  from re-training the encoder. We can cache MERLIN's features once and reuse them.*

## 2. What's new / trainable (the orange boxes)
Only **two** things learn:
1. **Cross-exam encoder** — a small (3-layer) Transformer that takes the token
   grids of the prior and current scans **together** and lets them attend to each
   other, so the current scan can "look up" its counterpart region in the prior.
   *We add positional encodings (where in the volume) and a temporal encoding
   (how far apart in time) to each token.*
2. **The dynamic image projection `Proj_img`** — maps the change representation
   into MERLIN's shared vision–language space.
Everything else — both encoders, the text projection — is frozen/reused.

## 3. The key architectural idea — the antisymmetric difference
This is the slide's intellectual core; slow down here.
- Instead of a naive subtraction of pooled vectors (the baseline), we compute the
  change as **`d = g(c, p) − g(p, c)`**, where `g` is the cross-exam encoder,
  `c` = current, `p` = prior.
- **Antisymmetric** means: if you swap the two scans (pretend the current is the
  prior), the change vector **flips sign** — `d(p, c) = −d(c, p)`.
- Two payoffs:
  - **"No change" is exactly the zero vector, by construction.** This directly
    fixes the baseline's fatal flaw — "stable" now has a well-defined home (the
    origin), instead of being an undefined direction.
  - **Direction encodes worse vs improved automatically** — worsening and
    improving are opposite signs of the *same* axis, not two unrelated
    directions the model has to learn separately. *(This also mirrors the
    real-world involution: new↔resolved, worse↔improved, stable↔stable.)*

## 4. The contrastive objective — how it learns
Point to the shared embedding space and the bottom text path.
- We push the change embedding `vd` and the report's **change sentences** `td`
  into the **same shared space** and align them with **InfoNCE** — the
  "change-alignment game": the correct (image-change, change-text) pair should be
  closer than mismatched pairs.
- **`L_dynamic = InfoNCE(vd, td)`** is the *only* loss.
- The text side is entirely **frozen**: the change sentences are encoded by
  MERLIN's Clinical Longformer and its projection — we align our new image-change
  vector *to MERLIN's existing text space*, we don't move the text.

## 5. Where the report's "change sentences" come from (the LLM splitter)
Bottom of the diagram.
- A radiology report mixes **static description** ("a 3 cm mass in the RLL") with
  **comparison / change language** ("increased from 2 cm," "new since prior").
- An **LLM splitter** pulls out only the **dynamic (comparison) sentences** —
  those are the supervision signal for the change embedding.
- *Why this matters: if you align the change vector to the WHOLE report, the model
  cheats — most of the report is static description of the current scan, so it
  learns `d ≈ v_current` and ignores the prior entirely. Filtering to change
  sentences forces the module to actually model change.*

## 6. How training flows — backprop (the dashed red arrow)
Trace the red dashed arrow upward.
- The gradient of `L_dynamic` flows **back through `Proj_img`, through the
  antisymmetric difference, into the cross-exam encoder** — and **stops** at
  MERLIN's frozen (cached) tokens. Nothing past that boundary updates.
- So the only things that learn are exactly the two orange boxes. *This is why
  it's cheap and stable to train.*

## 7. Why there's no "static" branch (pre-empt the obvious question)
- *Earlier drafts had a parallel `L_static` term aligning `v_current` to the
  static sentences. But in the fully-frozen design, every part of that path is
  frozen — so it has no trainable parameters and its gradient is zero. It would
  literally train nothing, so we removed it.*
- *If we ever wanted a static anchor to do real work, we'd have to unfreeze a
  shared projection head; we're choosing not to, to keep the module minimal.*

## 8. How this fixes the baseline — the payoff (say explicitly)
- Baseline problem 1: "stable" undefined under cosine → **fixed** by the
  antisymmetric zero.
- Baseline problem 2: worse/improved not separable → **fixed** by learning a
  signed change axis via cross-attention instead of subtracting pooled vectors.
- Baseline problem 3: no supervision for *change* specifically → **fixed** by
  contrasting against the report's change sentences only.

---

## Anticipated Q&A
- **"Why not just fine-tune MERLIN end-to-end?"** Cost + we want to prove the
  module is what helps; we ablate frozen vs LoRA vs full fine-tune later.
- **"Does the cross-exam encoder need registered scans?"** Cross-attention does
  soft alignment, but we still pre-register prior→current so it spends capacity on
  pathology, not table-position offsets — that's an ablation.
- **"Why antisymmetric instead of just concatenating [c; p]?"** Concatenation
  gives no guarantee that 'stable' maps to a fixed point or that reversing time
  flips the sign; antisymmetry bakes both in as architecture, not something the
  model has to discover.
- **"Is InfoNCE enough with small batches on 3D volumes?"** We cache frozen
  features, so we can use very large effective batches cheaply — InfoNCE likes
  that.
- **"What supervises 'how much' it changed, not just direction?"** Optional
  add-on: a magnitude head on `‖d‖` (calibrated to RECIST change), which is where
  the DeepLesion diameters come back in as ground truth.
- **"What if reports have little comparison language?"** Real risk on CT; the LLM
  splitter measures the fraction of dynamic sentences — if it's too low we lean on
  a report-rich corpus (CT-RATE / institutional).
