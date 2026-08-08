#!/usr/bin/env python3
"""
13_medgemma_label.py

STEP 2: MedGemma-27B silver-labeler for CT-RATE prior->current report pairs.

Replicates the CheXTemporal labeling protocol on CT:
  - per-finding 5-class progression: {new, worse, stable, improved, resolved}
  - an overall study-level change label: {improved, stable, worse, mixed}
  - static vs dynamic sentence split of the CURRENT report
Outputs one JSON record per pair to a JSONL file (resumable).

Design notes
------------
* Local `transformers` inference, meant to run on a cloud A100/H100.
  Default model: google/medgemma-27b-text-it  (gated -- accept license + be logged in).
* --dry-run builds and prints prompts WITHOUT loading the model, so you can validate
  the pipeline on a laptop before renting a GPU.
* --limit N labels only the first N pairs (pilot on ~50 first!).
* Resumable: skips pairs whose curr_volume is already in the output JSONL.

Usage
-----
  # laptop: validate prompts, no GPU
  python scripts/13_medgemma_label.py --dry-run --limit 3

  # GPU box: pilot 50, then inspect, then scale
  python scripts/13_medgemma_label.py --limit 50
  python scripts/13_medgemma_label.py            # full run (resumes)

Requires (GPU box): torch, transformers>=4.50, accelerate, and HF access to the model.
"""
import argparse
import csv
import json
import os
import re
import sys

MANIFEST = "data/ctrate/ctrate_pairs_enriched.csv"
OUT_DIR = "data/ctrate/labels"
OUT_JSONL = os.path.join(OUT_DIR, "medgemma_labels.jsonl")
DEFAULT_MODEL = "google/medgemma-27b-text-it"

SYSTEM_PROMPT = (
    "You are an expert thoracic radiologist comparing two chest CT studies of the "
    "same patient: a PRIOR study and a CURRENT study. Assess how findings changed "
    "from prior to current. Base your judgment ONLY on the report text provided. "
    "Respond with a SINGLE JSON object and nothing else."
)

# The schema we ask the model to fill. Kept explicit so outputs are parseable.
SCHEMA_INSTRUCTIONS = """\
Return JSON with EXACTLY these keys:
{
  "overall": one of ["improved","stable","worse","mixed"],
  "overall_confidence": a number in [0,1],
  "findings": [
    {
      "finding": short name of the finding/lesion (e.g. "right upper lobe nodule"),
      "change": one of ["new","worse","stable","improved","resolved"],
      "evidence": a short quote from the CURRENT report supporting this
    }
  ],
  "dynamic_sentences": [sentences from the CURRENT report that express change/comparison to prior],
  "static_sentences": [sentences from the CURRENT report that describe the current state only, with no comparison]
}
Rules:
- "new" = present now, absent on prior. "resolved" = present on prior, absent now.
- "worse"/"improved" = present in both but larger/more vs smaller/less.
- "stable" = present in both, unchanged.
- If the report gives no comparative information for a finding, put it in static_sentences
  and do not invent a change label for it.
- Every sentence of the CURRENT report should appear in exactly one of
  dynamic_sentences or static_sentences.
- Output ONLY the JSON object, no prose, no markdown fences.
"""


def build_user_prompt(row):
    prior = (row.get("prior_findings", "") + " " + row.get("prior_impression", "")).strip()
    curr = (row.get("curr_findings", "") + " " + row.get("curr_impression", "")).strip()
    clinical = row.get("curr_clinical", "").strip()
    gap = row.get("delta_days", "")
    return (
        f"CLINICAL INDICATION (current): {clinical or 'Not given.'}\n"
        f"INTERVAL between studies: {gap} days\n\n"
        f"PRIOR STUDY REPORT:\n{prior or '(no prior report text)'}\n\n"
        f"CURRENT STUDY REPORT:\n{curr or '(no current report text)'}\n\n"
        f"{SCHEMA_INSTRUCTIONS}"
    )


def extract_json(text):
    """Pull the first balanced {...} JSON object out of a model response."""
    # strip code fences if present
    text = re.sub(r"```(json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["curr_volume"])
                except Exception:
                    pass
    return done


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {model_id} (this needs a GPU + accepted license)...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


def generate(tok, model, system, user, max_new_tokens=1024):
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true",
                    help="build/print prompts without loading the model")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} pairs to consider from {args.manifest}")

    if args.dry_run:
        for row in rows[:3]:
            print("\n" + "=" * 72)
            print(f"PAIR {row['prior_volume']} -> {row['curr_volume']}  "
                  f"(delta={row['delta_days']}d)")
            print("=" * 72)
            print("SYSTEM:\n" + SYSTEM_PROMPT)
            print("\nUSER:\n" + build_user_prompt(row))
        print(f"\n[dry-run] built prompts for {min(3, len(rows))} pairs; model NOT loaded.")
        return

    done = load_done(args.out)
    if done:
        print(f"resuming: {len(done)} pairs already labeled, skipping those")

    tok, model = load_model(args.model)

    n_ok, n_bad = 0, 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            cv = row["curr_volume"]
            if cv in done:
                continue
            user = build_user_prompt(row)
            raw = generate(tok, model, SYSTEM_PROMPT, user, args.max_new_tokens)
            parsed = extract_json(raw)
            rec = {
                "patient": row["patient"],
                "prior_volume": row["prior_volume"],
                "curr_volume": cv,
                "delta_days": row["delta_days"],
                "label": parsed,
                "parse_ok": parsed is not None,
            }
            if parsed is None:
                rec["raw"] = raw[:2000]  # keep for debugging failed parses
                n_bad += 1
            else:
                n_ok += 1
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)}  ok={n_ok} parse_fail={n_bad}")

    print(f"\ndone. parsed_ok={n_ok}  parse_fail={n_bad}  -> {args.out}")
    if n_bad:
        print("  (inspect the 'raw' field on failed rows and tighten the prompt)")


if __name__ == "__main__":
    main()
