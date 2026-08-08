# Dataset — DeepLesion (via Deep Lesion Tracker pairs)

## What it is
- **DeepLesion (NIH Clinical Center):** large-scale CT dataset mined from PACS — ~32k annotated lesions across ~10k studies / ~4.4k patients, each with a **RECIST bookmark** (long- + short-axis diameter).
- **Deep Lesion Tracker (DLT):** provides the **longitudinal linkage** — matched **prior → current** lesion pairs of the *same* lesion across two timepoints. This is what makes DeepLesion usable for *temporal* progression.
- We use it as our **quantitative interval-change testbed** for the frozen-MERLIN baseline.

## Subset actually used ("DLT mini")
- **532 prior→current lesion pairs** across **243 CT volumes**.
- Each pair carries: source/target NIfTI, voxel spacing, lesion center/box, and **long/short-axis RECIST diameters** at both timepoints.

## How labels are assigned (RECIST 1.1, single-lesion, on long-axis)
- **Worse (progression):** ≥ **+20%** increase **AND** ≥ **+5 mm** absolute (the +5 mm floor blocks sub-pixel noise).
- **Improved (regression):** ≥ **−30%** decrease.
- **Stable:** everything in between.
- Result: strongly **stable-dominated → 84.4% "stable"** (a realistic, hard class-imbalance).

## Preprocessing (matched to MERLIN)
- Resample ~**1.5 mm in-plane / 3 mm slice**, HU window **−1000:1000 → 0:1**, center-crop **224×224×160**.
- Baseline ran at reduced **depth = 32** for speed.

## Why this dataset (strengths)
- **Physically measurable ground-truth change** (RECIST diameters) → objective progression labels; CXR temporal work fundamentally can't do this.
- **Free longitudinal pairs** of the same lesion — no manual pairing needed.
- Directly probes whether MERLIN's embeddings encode fine **interval change**.

## Limitations / caveats (be upfront on the slide)
- **No paired free-text reports** → can only evaluate the *image-side* change signal here; the text-alignment (dynamic-sentence) objective needs a report-bearing corpus (e.g., CT-RATE / institutional).
- **Lesion-centric**, not whole-exam / whole-report reasoning.
- **Severe class imbalance** (stable-heavy) → macro-F1, not accuracy, is the honest metric.
- **Domain gap:** MERLIN pretrained on abdominal CT; DeepLesion spans chest/abdomen/other.
