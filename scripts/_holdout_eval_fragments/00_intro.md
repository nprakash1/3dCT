# Holdout-eval sentence ablation — CT Temporal Progression

**Question:** If CE is trained on one set of class sentences, does performance hold when **eval** picks the nearest of 3 classes using **different sentences never used in CE training**?

## Eval paradigm (this notebook only)

- For each finding × class store **5 held-out paraphrase sentences** (disjoint from train CE templates).
- At **every eval example**, randomly sample **one sentence per class**:
  `y_hat = argmax_c cos(d_f, emb(s_c))`
- **Train CE** still uses fixed `PROTO_TRAIN[f]` from a separate train template bank.
- Mag + optional cross-modal SupCon match the proposed pipeline.
- Ablation: **CE × Mag × SupCon** under this eval paradigm only.

| Text role | Source |
|-----------|--------|
| Train CE | `TRAIN_TEMPLATES` → frozen `PROTO_TRAIN[f]` (3×512) |
| Eval readout | `EVAL_BANK[f][c]` = 5 strings; sample 1/class per example |
| SupCon (train) | per-example evidence/dynamic (**not** used at test) |

Protocol: Hub `train_*` train+tune; Hub `valid_*` one-shot test; CT-CLIP frozen.
