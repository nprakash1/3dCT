#!/usr/bin/env python3
"""
17_aggregate_labels.py

PHASE A: turn the raw MedGemma label dump + the v2 manifest into a clean, trainable
dataset with patient-level splits and a zero-shot prompt bank.

Inputs
------
--labels   medgemma_labels.jsonl   (default: 'medgemma_labels (2).jsonl')
data/ctrate/ctrate_pairs_enriched_v2.csv   (report text + structured columns)

Outputs (data/ctrate/)
----------------------
labels_final.jsonl     one row per pair: findings[], dynamic/static sentences,
                       full prior/current report text, delta_days, split
labels_findings.csv    FLAT one row per (pair, finding): the table you train/eval on
splits.json            {patient_id: "train"|"val"|"test"}   (no patient leakage)
prompt_bank.json       18 findings x 5 change-classes -> text prompts for zero-shot

Split: patient-level 80/10/10 (a patient's pairs never cross splits).
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)

OUT_DIR = "data/ctrate"
MANIFEST = os.path.join(OUT_DIR, "ctrate_pairs_enriched_v2.csv")
CLASSES = ["new", "worse", "stable", "improved", "resolved"]

FINDINGS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# Natural-language templates per change class (used for zero-shot cosine matching).
PROMPT_TEMPLATES = {
    "new":      "{f} is new compared to the prior study",
    "worse":    "{f} has worsened compared to the prior study",
    "stable":   "{f} is unchanged compared to the prior study",
    "improved": "{f} has improved compared to the prior study",
    "resolved": "{f} has resolved compared to the prior study",
}


def build_prompt_bank():
    bank = {}
    for f in FINDINGS:
        fl = f[0].lower() + f[1:]
        bank[f] = {c: PROMPT_TEMPLATES[c].format(f=fl) for c in CLASSES}
    return bank


def patient_splits(patients, seed=0, val=0.1, test=0.1):
    pats = sorted(set(patients))
    random.Random(seed).shuffle(pats)
    n = len(pats)
    n_test = int(n * test)
    n_val = int(n * val)
    split = {}
    for p in pats[:n_test]:
        split[p] = "test"
    for p in pats[n_test:n_test + n_val]:
        split[p] = "val"
    for p in pats[n_test + n_val:]:
        split[p] = "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="medgemma_labels (2).jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.labels):
        raise SystemExit(f"missing labels file: {args.labels}")
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"missing {MANIFEST} -- run scripts/16 first")

    # report text keyed by (prior_volume, curr_volume)
    man = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            man[(r["prior_volume"], r["curr_volume"])] = r
    print(f"manifest rows: {len(man):,}")

    recs = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    print(f"label records: {len(recs):,}")

    split = patient_splits([r["patient"] for r in recs], seed=args.seed)
    json.dump(split, open(os.path.join(OUT_DIR, "splits.json"), "w"), indent=0)

    bank = build_prompt_bank()
    json.dump(bank, open(os.path.join(OUT_DIR, "prompt_bank.json"), "w"), indent=2)

    final_path = os.path.join(OUT_DIR, "labels_final.jsonl")
    flat_path = os.path.join(OUT_DIR, "labels_findings.csv")
    ffo = open(final_path, "w", encoding="utf-8")
    flat = open(flat_path, "w", newline="", encoding="utf-8")
    fw = csv.writer(flat)
    fw.writerow(["patient", "split", "prior_volume", "curr_volume", "delta_days",
                 "finding", "change", "source", "evidence"])

    per_split = Counter()
    class_by_split = defaultdict(Counter)
    n_findings = 0
    for r in recs:
        key = (r["prior_volume"], r["curr_volume"])
        m = man.get(key, {})
        sp = split[r["patient"]]
        per_split[sp] += 1
        row = {
            "patient": r["patient"], "split": sp,
            "prior_volume": r["prior_volume"], "curr_volume": r["curr_volume"],
            "delta_days": r["delta_days"],
            "findings": r["findings"],
            "dynamic_sentences": r.get("dynamic_sentences", []),
            "static_sentences": r.get("static_sentences", []),
            "curr_findings": m.get("curr_findings", ""),
            "curr_impression": m.get("curr_impression", ""),
            "prior_findings": m.get("prior_findings", ""),
            "prior_impression": m.get("prior_impression", ""),
            "curr_clinical": m.get("curr_clinical", ""),
            "severity_parse_ok": r.get("severity_parse_ok", True),
        }
        ffo.write(json.dumps(row) + "\n")
        for fd in r["findings"]:
            n_findings += 1
            class_by_split[sp][fd["change"]] += 1
            fw.writerow([r["patient"], sp, r["prior_volume"], r["curr_volume"],
                         r["delta_days"], fd["finding"], fd["change"],
                         fd.get("source", ""), fd.get("evidence", "")])
    ffo.close()
    flat.close()

    print(f"\npairs per split: {dict(per_split)}")
    print(f"total (pair,finding) rows: {n_findings:,}")
    print("\nclass balance per split:")
    for sp in ["train", "val", "test"]:
        c = class_by_split[sp]
        tot = sum(c.values()) or 1
        pct = {k: f"{100*c[k]/tot:.0f}%" for k in CLASSES}
        print(f"  {sp:>5} (n={tot:,}): " + "  ".join(f"{k}={c[k]}({pct[k]})" for k in CLASSES))

    print("\nwrote:")
    for p in [final_path, flat_path,
              os.path.join(OUT_DIR, "splits.json"),
              os.path.join(OUT_DIR, "prompt_bank.json")]:
        print("  ", p)
    print("\nexample prompt bank entry (Pleural effusion):")
    print("  ", json.dumps(bank["Pleural effusion"], indent=2).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
