# Speaker Notes — Initial Zero-Shot Baseline (frozen MERLIN)

> Companion to `docs/baseline_results.png`. Timing target: ~4–5 minutes.
> Read/paraphrase; **bold** = say it out loud, *italics* = optional depth / Q&A.

---

## 0. One-line framing (say first)
"Before we build anything, we ask the cheapest possible question: **does a frozen
CT foundation model — MERLIN — already 'know' how a lesion changed between two
scans, with no training at all?** This baseline is the honest floor everything
else has to beat."

---

## 1. The setup / what we're testing
- We take MERLIN **completely frozen** — no fine-tuning, no new weights.
- For a patient we have a **prior** CT and a **current** CT of the same lesion.
- We embed each into MERLIN's **512-dimensional** space → `v_prior`, `v_current`.
- The natural "change" signal is just the **difference vector**: `d = v_current − v_prior`.
- Question: **does the direction / size of `d` tell us whether the lesion got
  worse, stayed stable, or improved?**
- *Why this is the right first experiment: if the answer were "yes," we'd be
  nearly done for free. It almost certainly isn't — and understanding exactly
  HOW it fails is what designs the real model.*

## 2. Data (quick recap — don't dwell)
- **DeepLesion** lesions linked by **Deep Lesion Tracker** into **prior→current pairs**.
- **532 pairs across 243 volumes.**
- Labels from **RECIST 1.1 on the long-axis diameter**: worse = ≥+20% & ≥+5 mm;
  improved = ≥−30%; stable = in between.
- **Crucial fact for reading the results: 84.4% of pairs are "stable."** This
  imbalance is realistic (most follow-ups don't change dramatically) but it means
  **accuracy is a trap** — a model that screams "stable" every time already
  scores 84% accuracy. So we judge on **macro-F1** (averages the three classes
  equally), not accuracy.

## 3. Method — two zero-shot decision rules
Point at the two orange boxes in the figure.
- **Rule A — magnitude gate:** first look at the *size* of the change vector,
  `‖d‖`. If it's below a threshold τ → call it **"stable"** (nothing moved).
  Only if it's above τ do we look at *direction*: cosine-compare `d` to a
  "worse" text prompt vs an "improved" text prompt, and take the closer one.
- **Rule C — pure cosine, prompt ensemble:** skip the gate; compare `d` by
  cosine similarity against text prompts for all three classes (improved /
  stable / worse), averaging 5 paraphrases per class to be fair, and take the
  argmax.
- *τ is set two ways — an unsupervised median-`‖d‖` heuristic and an "oracle" τ
  that maximizes macro-F1 (a generous upper bound). Even the oracle barely moves
  the needle.*
- The text prompts are encoded by **MERLIN's own frozen text encoder**, so this
  is a genuine zero-shot vision–language test, not a trained classifier.

## 4. Results — walk the table
Reference numbers (macro-F1 / accuracy):
- **Always-"stable" baseline:** macro-F1 **0.305**, accuracy **0.844**.
  (Accuracy looks great, macro-F1 is terrible — it gets 0 on both worse and
  improved. This is the number to beat.)
- **Rule A (magnitude gate):** macro-F1 **0.315**, accuracy 0.483.
- **Rule C (prompt ensemble):** macro-F1 **0.234** — *below* the trivial floor.

Say plainly: **"Our best zero-shot result, 0.315, is essentially tied with the
do-nothing 'always stable' floor of 0.305."**

## 5. The key finding (this is the slide's whole point)
- **Cosine similarity structurally cannot represent 'stable.'** "Stable" means
  the change vector is ~**zero**. A zero vector has **no direction** — so cosine
  similarity is undefined/meaningless for it. That's why adding an explicit
  "stable" *direction* prompt (Rule C) actively **hurts**: the near-zero stable
  cases get randomly flung toward "worse" or "improved."
- The magnitude gate (Rule A) helps a little precisely because it handles
  "stable" by *size*, not direction — but it still can't cleanly separate worse
  vs improved.
- Bottom line: **MERLIN's global 512-d embedding difference does not cleanly
  encode interval change.** *That's not a failure of the experiment — it's the
  finding.*
- *Optional nuance / pre-empt a critique: there's also a domain mismatch —
  MERLIN was pretrained on whole-abdomen 224×224×160 volumes, whereas DeepLesion
  gives thin lesion slabs run at reduced depth. So some of the weakness is
  off-domain input, not just the difference-vector idea. Both point the same way:
  we need a trained, change-aware module.*

## 6. So what — the bridge to our method
"This baseline hands us two concrete design requirements, not just a low number:"
1. **Represent 'no change' as a true zero** — motivates an **antisymmetric
   difference** `d(c,p) = g(c,p) − g(p,c)`, where 'stable' is exactly the origin.
2. **Judge magnitude separately from direction** — motivates an explicit
   **magnitude head** to gate 'stable,' instead of leaning on cosine.
"Everything in the next slides is built to fix exactly what this baseline
exposed."

---

## Anticipated Q&A
- **"Isn't 0.315 vs 0.305 just noise?"** Yes — and that's the point: a frozen
  foundation model gives us essentially *nothing* over guessing on this task,
  which is why a trained module is justified.
- **"Why not just fine-tune MERLIN?"** We deliberately keep it frozen (cost,
  reproducibility, and to isolate the value of the difference module); the plan
  ablates frozen vs LoRA vs full fine-tune later.
- **"Why RECIST labels instead of report labels here?"** DeepLesion has
  measurable diameters but no paired reports, so it's our *quantitative* testbed;
  the text-alignment objective moves to a report-bearing corpus (e.g., CT-RATE).
- **"Could better prompts fix Rule C?"** No — the failure is structural (zero
  vector has no direction), not a prompt-wording problem; we already ensembled
  paraphrases.
- **"Why macro-F1?"** Because 84.4% of the data is one class; accuracy rewards a
  constant "stable" predictor and hides all the interesting behavior.
