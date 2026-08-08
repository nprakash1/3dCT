# CT-RATE Temporal Pipeline

Adapting the temporal-progression project from the small DeepLesion-Tracker set
(532 synthetic-labeled pairs) to **CT-RATE**, a public chest-CT dataset with real
paired reports. This doc records the feasibility findings and the labeling pipeline.

## Why CT-RATE (feasibility findings)

Probed metadata-only (report + DICOM-header CSVs, a few MB — **no** 21.3 TB download):

| Question | Result |
|---|---|
| Longitudinal prior→current pairs | **4,385** (dated) from 2,807 patients — ~8× the old 532 |
| Real study dates? | **Yes** — `StudyDate` 100% populated (overturns the "no dates" assumption) |
| Comparison language in current reports | **93.6%** (clean Findings+Impressions only) |
| Interval spread (`delta_days`) | 545 @1-7d · 953 @8-30d · 859 @31-90d · 641 @91-180d · 706 @181-365d · 681 @>365d |
| Other useful fields | `PatientAge` (100%), full geometry (`ImagePositionPatient`, `XYSpacing`, `ZSpacing`, orientation), scanner (`Manufacturer`, `ConvolutionKernel`) |

The reports are an oncology/follow-up-heavy cohort and contain the full 5-class
change vocabulary naturally (new / worse / stable / improved / resolved), e.g.
"newly emerged", "increased", "decreased partially", "significant regression",
"not detected in the current examination" (resolved), "stable".

## Scripts (run in order)

| Script | Purpose | Compute |
|---|---|---|
| `scripts/10_ctrate_probe.py` | Count longitudinal pairs + comparison-language fraction | laptop, ~MB download |
| `scripts/11_ctrate_inspect.py` | Dump report/metadata columns; confirm date/age/geometry fields | laptop |
| `scripts/12_ctrate_enrich.py` | Build **date-ordered** clean pair manifest (Δt, scanner-match, Findings+Impressions) | laptop |
| `scripts/13_medgemma_label.py` | MedGemma-27B silver labels (5-class + static/dynamic split) | **cloud GPU** |
| `scripts/15_aggregate_labels.py` | Turn label JSONL → training tables + QC (class balance, static/dynamic corpora) | laptop |
| `scripts/14_ctrate_download_pairs.py` | Selectively download only the volumes for a chosen pair subset | laptop, ~GB |


Outputs (gitignored under `data/ctrate/`):
- `ctrate_pairs_enriched.csv` — the manifest fed to the labeler
- `labels/medgemma_labels.jsonl` — silver labels (one JSON per pair)

## Auth (one-time)

CT-RATE and MedGemma are **gated**. Accept both licenses on HF, then:
```bash
hf auth login          # paste a Read token; verify with `hf auth whoami`
```
Scripts read the cached token automatically (or set `HF_TOKEN`).

## Labeling: laptop → GPU handoff

**1. Validate prompts on the laptop (no model load):**
```bash
python scripts/13_medgemma_label.py --dry-run --limit 3
```

**2. On the cloud A100/H100** (needs `torch`, `transformers>=4.50`, `accelerate`,
and accepted access to `google/medgemma-27b-text-it`):
```bash
# pilot 50 first, then INSPECT before scaling
python scripts/13_medgemma_label.py --limit 50
#   -> check data/ctrate/labels/medgemma_labels.jsonl: parse_ok rate,
#      label sanity, static/dynamic split quality; tighten prompt if needed
python scripts/13_medgemma_label.py            # full run (resumable, skips done pairs)
```

### Label schema (per pair, CheXTemporal-aligned)
```json
{
  "overall": "improved | stable | worse | mixed",
  "overall_confidence": 0.0,
  "findings": [{"finding": "...", "change": "new|worse|stable|improved|resolved", "evidence": "..."}],
  "dynamic_sentences": ["...changes vs prior..."],
  "static_sentences": ["...current-state-only..."]
}
```
`dynamic_sentences` / `static_sentences` are the decomposition that later prevents
the difference-embedding→report contrastive loss from collapsing to `d ≈ v_current`.

## Next steps after labeling

1. **QC the silver labels**: parse-ok rate, class balance, spot-check vs manual read on ~50.
2. **Selective volume download**: pull only the ~600 volumes for a 100–300 pair
   prototype (~85–250 GB), run frozen MERLIN, cache 512-d embeddings, delete volumes.
3. **Train** the joint-attention / difference module on cached embeddings (as in
   `scripts/09_global_temporal_model.py`), now with real dates → Fourier Δt embedding,
   and the static/dynamic two-path contrastive objective.
4. **Gold set**: radiologist-annotate 500–1,500 pair×finding examples for eval.

## CT-CLIP re-embedding (chest-native encoder swap)

To remove the abdomen→chest domain gap, we re-encode the same **600-pair subset
(≈1,114 unique volumes)** with **CT-CLIP** (the CT-RATE-native model) instead of
MERLIN. Everything downstream (difference module, two-path InfoNCE, eval) consumes
the cached 512-d vectors unchanged — only the source of the cache changes.

**CT-CLIP preprocessing (DIFFERENT from MERLIN — must match exactly):**

| | CT-CLIP | (MERLIN was) |
|---|---|---|
| HU clip | **−1000 → +200** | −1000 → +1000 |
| Normalize | **[−1, 1]** | [0, 1] |
| Spacing (x/y/z) | **0.75 / 0.75 / 1.5 mm** | 1.5 / 3 mm |
| Target shape | **480 × 480 × 240** | 224 × 224 × 160 |
| Image enc | CTViT → flatten(294912) → 512 | inflated ResNet-152 → 512 |
| Text enc | CXR-BERT (768 → 512) | Clinical-Longformer → 512 |

**Files**
- `scripts/ctclip_utils.py` — `CTCLIPEmbedder` (frozen CT-CLIP) + CT-CLIP preprocessing
  + streaming volume download. Mirrors `merlin_utils.MerlinEmbedder`.
- `notebooks/ctclip_features_colab.ipynb` — GPU orchestration on Colab.

**Run (Colab, GPU ≥24 GB):** open the notebook and run top-to-bottom. It:
1. clones CT-CLIP, installs `transformer_maskgit` + `CT_CLIP`;
2. HF-logs-in, downloads `CT-CLIP_v2.pt` (cached to Drive);
3. runs the **CXR-BERT separability spike** (is the chest text encoder less
   direction-blind than Longformer? — a free result);
4. **streams each volume → CT-CLIP encode → saves 512-d to Drive → deletes volume.**

**Why this is safe on tiny disk / flaky sessions:**
- Embeddings write **directly to Google Drive, one file per volume, as computed**
  (continuous auto-save) → a session timeout loses at most the *one* volume in flight.
- Loop **skips already-cached** volumes → re-run to **resume** where it stopped.
- Volumes are streamed one-at-a-time and deleted → **peak disk < 1 GB**; final cache
  is only ~MB. No GCP needed (raw volumes were deleted last run; must re-encode anyway).

## Caveats

- Chest / non-contrast only → domain gap vs MERLIN's abdominal pretraining
  (the CT-CLIP swap above directly addresses this).
- `StudyDate` ordering is reliable; scan-letter ties broken by date.
- Silver labels are LLM-generated — treat as noisy; validate on a gold subset.
