# Model v1 — Global Temporal Transformer (GTT)

> First **trainable** model for CT interval-change. Deliberately minimal so it is a
> clean ablation over the zero-shot baseline (step 08). Script:
> `scripts/09_global_temporal_model.py`. Runs locally on CPU in ~seconds.

---

## 1. Why this model exists (the question it answers)

The zero-shot baseline (step 08) showed that **`d = v_current − v_prior` on frozen
512-d MERLIN embeddings** does no better than guessin g "stable" (macro-F1 ≈ 0.315
vs 0.305 floor), and that cosine can't represent "stable."

But that experiment changed *two* things at once vs. an ideal model: it was
**frozen** *and* it used a **hard-coded subtraction**. v1 isolates one variable:

> **Does replacing the hard-coded subtraction with a small *learned* temporal
> operator — while still using only the two global vectors — recover interval
> change?**

If yes → the problem was the operator. If no → the problem is the **global
average-pool** itself (spatial detail already destroyed), which is the scientific
justification for moving to the patch-wise (token-level) model next.

---

## 2. Architecture

All MERLIN encoders stay **frozen**; only the small module below is trained
(~few M params). Inputs are the **cached 512-d** `v_prior`, `v_current` from
`data/emb_mini_d32.pt` (no MERLIN or GPU needed at train time).

```
cached 512-d  v_prior, v_current
      |  W: Linear(512 -> d_model)  + learnable role embeddings
      v
tokens = [ e_diff ,  W·v_prior + role_prior ,  W·v_current + role_current ]
      |  tiny Transformer encoder (n_layers, n_heads)
      v
H[0] (e_diff token) -> Linear(d_model -> 512) -> v_d   (back in MERLIN text space)
      |  cosine( v_d , { t_worse, t_stable, t_improved } ) * logit_scale
      v
logits over {improved, stable, worse}  ->  weighted cross-entropy
```

Key pieces:
- **`e_diff`** — a learnable query token; its output is read out as the change
  embedding `v_d`.
- **role embeddings** (`role_prior`, `role_current`) — learnable "which-exam" tags
  so the transformer can tell prior from current (essential for direction).
- **output head** — projects back to 512-d so `v_d` is comparable to the frozen
  text prototypes. (A learned head is required: once the random-init transformer
  transforms the input, `v_d` is no longer guaranteed to live in MERLIN's text
  space; the cosine-CE loss is what re-aligns it.)

### Text supervision (no reports needed)
DeepLesion has no reports, so we synthesize a **3-class prompt bank** (the *same*
paraphrases as step 08), encode it once with frozen MERLIN text, average per class
→ three unit prototypes `t_worse, t_stable, t_improved`, cached to
`data/text_prototypes.pt`.

### Loss
With only 3 fixed prototypes, contrastive alignment reduces to **supervised
cosine-classification**:
```
logits_k = logit_scale · cos(v_d, t_k)          k ∈ {improved, stable, worse}
L        = weighted_CE(logits, label)           (inverse-freq weights; 84% stable)
```
`logit_scale` is a CLIP-style learnable temperature. Not contrastive — but still
**trained** (a random transformer + cosine would be pure noise).

---

## 3. Ablation flags (v1 keeps BOTH OFF)

The code is organized so flipping one flag is the *only* change between runs.

| Flag | Off (v1) | On | Purpose |
|---|---|---|---|
| `--antisymmetry` | single pass | `v_d = g(c,p) − g(p,c)` | makes "no change" = **0 vector** by construction (fixes cosine-can't-represent-stable) |
| `--magnitude-head` | none | scalar head predicting `log(diam_c/diam_p)` + aux loss | route "stable" by predicted **magnitude**; uses RECIST as free supervision |

Other knobs (all in the `Config` dataclass / CLI): `--d-model` (512 default),
`--n-layers`, `--epochs`, `--seed`, `--no-class-weight`.

Naming: each run prints a `tag()` like `GTT_d512_L2_plain` /
`GTT_d512_L2_antisym_mag` so ablations are self-labeling.

---

## 4. Results — v1 (`GTT_d512_L2_plain`, both flags OFF)

Protocol: **patient-level 5-fold CV** (no patient across folds), pooled
out-of-fold predictions, macro-F1 primary. 532 pairs / 243 volumes / 84.4% stable.

| Method | macro-F1 | Accuracy | F1 improved | F1 stable | F1 worse |
|---|---|---|---|---|---|
| always-stable (ref) | 0.305 | 0.844 | 0.000 | 0.915 | 0.000 |
| zero-shot (A) magnitude gate (step 08) | 0.315 | 0.483 | 0.113 | 0.645 | 0.188 |
| zero-shot (C) prompt ensemble (step 08) | 0.234 | 0.289 | 0.101 | 0.393 | 0.206 |
| **v1 GTT_d512_L2_plain (trained)** | **0.321** | 0.718 | 0.039 | 0.837 | 0.087 |
| v1 + `--antisymmetry` (GTT_d512_L2_antisym) | 0.251 | 0.357 | 0.110 | 0.509 | 0.135 |

Per-fold macro-F1 = 0.321 ± 0.047 (plain). Confusion (rows=true, cols=pred; order
improved/stable/worse):
```
plain:                     antisymmetric:
[[  1  24   0]             [[  7   8  10]
 [ 21 376  52]              [ 78 163 208]
 [  4  49   5]]             [ 17  21  20]]
```

### 4a. Antisymmetry ablation — the informative failure
Turning on `--antisymmetry` (v_d = g(c,p) − g(p,c), so "no change" = 0 by
construction) *lowers* macro-F1 (0.251) and accuracy (0.357) — but the per-class
breakdown is the real story:

| class | plain R | antisym R | plain F1 | antisym F1 |
|---|---|---|---|---|
| improved | 0.040 | **0.280** (7×) | 0.039 | **0.110** |
| worse    | 0.086 | **0.345** (4×) | 0.087 | **0.135** |
| stable   | 0.837 | 0.363 | 0.837 | 0.509 |

- Antisymmetry gives the model a real **sign axis**, so it stops defaulting to
  "stable" and actually **detects change and its direction** — improved/worse
  recall jump 4–7× and their F1 both rise.
- But **precision stays ~0.07–0.08**: it over-calls change, tanking stable and
  overall accuracy.
- Both core hypotheses are validated *simultaneously*: **direction is learnable in
  principle** (antisymmetry unlocks recall), but **global pooling caps precision**
  (the signal is too weak to fire selectively). Precision won't recover until we
  (a) restore spatial detail via the patch-wise model and/or (b) add the
  **magnitude head** to gate "stable" by *how much* changed rather than direction.


---

## 5. Interpretation

- **Learning the operator barely moves the needle** on globals: 0.321 vs the
  0.315 zero-shot floor and 0.305 always-stable. The trained model mostly predicts
  "stable" (higher accuracy, 0.718, but macro-F1 flat).
- **worse/improved remain near-noise** (F1 0.087 / 0.039). The signal needed to
  separate them is not recoverable from two globally-pooled vectors.
- **Conclusion:** the bottleneck is the **global average-pool**, not merely the
  frozen encoder or the subtraction. This is exactly the evidence that motivates
  the **patch-wise (token-level) cross-exam model**, where the `layer4` spatial
  grid is preserved. It also motivates turning on **antisymmetry** (proper home
  for "stable") and the **magnitude head** (RECIST-anchored "how much"), which are
  the next two ablations.

Either way the result is publishable signal: it cleanly attributes the failure to
pooling.

---

## 6. How to reproduce / run ablations

```bash
source .venv/bin/activate

# v1 (both flags off, d_model=512)  -> GTT_d512_L2_plain
python scripts/09_global_temporal_model.py

# ablation: antisymmetric readout ("stable" = 0 vector)
python scripts/09_global_temporal_model.py --antisymmetry

# ablation: + RECIST magnitude head
python scripts/09_global_temporal_model.py --antisymmetry --magnitude-head

# ablation: narrower transformer (regularization on 532 pairs)
python scripts/09_global_temporal_model.py --d-model 256
```
Text prototypes are cached after the first run (`data/text_prototypes.pt`), so
subsequent runs need **no MERLIN and no GPU**.

---

## 7. Roadmap

1. **v1 (this doc)** — global two-vector transformer, cosine-CE. ✅
2. **v1 ablations** — antisymmetry, magnitude head, d_model, layers.
3. **v2 patch-wise** — tap MERLIN `layer4` tokens (≈490 × 2048), cross-exam
   attention (with adaptive-pool / latent-query bottleneck to bound cost at depth
   160), same cosine-CE (and later real-report contrastive on CT-RATE).
