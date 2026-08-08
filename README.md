# 3dCT — Temporal Progression on CT (MERLIN baseline)

Adapting CheXTemporal-style temporal-progression reasoning from chest X-ray to
**3D CT**, on top of a **frozen [MERLIN](https://github.com/StanfordMIMI/Merlin)**
vision–language foundation model.

**Long-term idea:** keep MERLIN's image & text encoders frozen; add a small
*difference module* that compares a patient's **prior** and **current** CT and
aligns the resulting change embedding with the *comparison* sentences in the
current report.

Progress so far: **Step 1** (MERLIN running; embed + cache scans) and a
**Step 2** naive baseline (`v_current − v_prior` cosine-matched to progression
prompts). The trainable difference module is Step 3.


---

## Quick start

```bash
# 1) Create the environment and install MERLIN + helpers
bash setup_env.sh
source .venv/bin/activate

# 2) Prove MERLIN is alive (downloads a demo CT scan, runs inference)
python scripts/01_run_demo.py

# 3) Extract + cache the pre-pool feature grid and pooled embedding
python scripts/02_inspect_embeddings.py
```

## What each script does

| Script | Purpose |
|---|---|
| `scripts/01_run_demo.py` | Loads pretrained MERLIN, runs the built-in demo CT scan, prints image/text embedding shapes. Success = no crash. |
| `scripts/02_inspect_embeddings.py` | Hooks the image encoder to capture the pre-pool feature grid (paper: `2048 × 10 × 7 × 7`), prints the 512-d pooled embedding, and saves both to `cache/embeddings/`. |
| `scripts/merlin_utils.py` | Reusable `MerlinEmbedder` (frozen): `embed_images()` and `embed_texts()` → 512-d vectors. |
| `scripts/03_baseline_progression.py` | **Step 2 baseline:** `d = v_current − v_prior`, cosine-matched to progression prompts. Defaults to demo-as-both (a "stable" plumbing test); pass `--prior/--current` for real pairs. |


## Key facts about the MERLIN backbone (verified from the paper)

- **Image encoder:** I3D ResNet-152. Pre-pool feature grid = **2048 × 10 × 7 × 7** (≈490 tokens).
- **Text encoder:** Clinical Longformer, 4096-token context.
- **Joint embedding dim:** 512. Losses: InfoNCE(report findings) + BCE(EHR phenotypes).
- **Volume preprocessing:** 1.5 mm in-plane / 3 mm slice; HU −1000:1000 → 0:1; center-crop 224×224×160.

## Roadmap

- **Step 1 (this repo):** MERLIN up & running; embed + cache scans. ✅
- **Step 2 — baseline:** measure change with `v_current − v_prior` and cosine-match
  to progression prompts ("worse" / "improved" / "stable"). Expected to fail on "stable".
- **Step 3 — improvement:** trainable difference module with an antisymmetric
  construction (`d(c,p) = g(c,p) − g(p,c)`), a magnitude head for "stable", and
  static/dynamic contrastive alignment to report sentences. MERLIN stays frozen.

## Layout

```
3dCT/
├── README.md
├── requirements.txt
├── setup_env.sh
├── scripts/
│   ├── 01_run_demo.py
│   ├── 02_inspect_embeddings.py
│   ├── merlin_utils.py
│   └── 03_baseline_progression.py
├── data/     # downloaded demo scan (gitignored)

└── cache/    # cached embeddings + feature grids (gitignored)
```
