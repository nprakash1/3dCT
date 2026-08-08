#!/usr/bin/env python3
"""
15_aggregate_labels.py

Turn MedGemma's raw JSONL (from 13_medgemma_label.py) into training-ready tables and
report label quality. Run this the moment the GPU labeling produces some output.

Produces
--------
data/ctrate/labels/labels_aggregated.csv
    one row per pair: patient, volumes, delta_days, overall(4-class),
    overall_3class (mixed->worse), overall_confidence, n_findings, parse_ok
data/ctrate/labels/dynamic_sentences.csv   (curr_volume, sentence)   <- for L_dyn
data/ctrate/labels/static_sentences.csv    (curr_volume, sentence)   <- for L_static
data/ctrate/labels/findings_long.csv       (curr_volume, finding, change, evidence)

Prints: parse-ok rate, overall class balance, per-finding change balance,
static/dynamic sentence counts. No GPU, no downloads.
"""
import argparse
import csv
import json
import os
from collections import Counter

IN_JSONL = "data/ctrate/labels/medgemma_labels.jsonl"
OUT_DIR = "data/ctrate/labels"

VALID_OVERALL = {"improved", "stable", "worse", "mixed"}
VALID_CHANGE = {"new", "worse", "stable", "improved", "resolved"}


def to_3class(overall):
    # collapse for the existing 3-class v1 model; "mixed" counts as worse
    # (any progression present). Adjust here if you prefer mixed->separate.
    return {"improved": "improved", "stable": "stable",
            "worse": "worse", "mixed": "worse"}.get(overall, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN_JSONL)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        print(f"!! no label file at {args.inp} -- run 13_medgemma_label.py first")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    recs = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    n = len(recs)
    parse_ok = sum(1 for r in recs if r.get("parse_ok"))
    print(f"records: {n}   parse_ok: {parse_ok} ({100*parse_ok/max(n,1):.1f}%)")

    agg_rows, dyn_rows, stat_rows, find_rows = [], [], [], []
    overall_bal, change_bal = Counter(), Counter()
    invalid_overall = 0

    for r in recs:
        lbl = r.get("label") or {}
        overall = (lbl.get("overall") or "").strip().lower()
        if overall not in VALID_OVERALL:
            invalid_overall += 1
            overall = ""
        overall_bal[overall or "MISSING"] += 1
        cv = r.get("curr_volume", "")

        findings = lbl.get("findings") or []
        for fd in findings:
            ch = (fd.get("change") or "").strip().lower()
            if ch in VALID_CHANGE:
                change_bal[ch] += 1
            find_rows.append({
                "curr_volume": cv,
                "finding": fd.get("finding", ""),
                "change": ch,
                "evidence": fd.get("evidence", ""),
            })

        for s in (lbl.get("dynamic_sentences") or []):
            if isinstance(s, str) and s.strip():
                dyn_rows.append({"curr_volume": cv, "sentence": s.strip()})
        for s in (lbl.get("static_sentences") or []):
            if isinstance(s, str) and s.strip():
                stat_rows.append({"curr_volume": cv, "sentence": s.strip()})

        agg_rows.append({
            "patient": r.get("patient", ""),
            "prior_volume": r.get("prior_volume", ""),
            "curr_volume": cv,
            "delta_days": r.get("delta_days", ""),
            "overall": overall,
            "overall_3class": to_3class(overall),
            "overall_confidence": (lbl.get("overall_confidence")
                                   if isinstance(lbl.get("overall_confidence"), (int, float))
                                   else ""),
            "n_findings": len(findings),
            "parse_ok": int(bool(r.get("parse_ok"))),
        })

    # ---- write tables ----
    def dump(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path}  ({len(rows)} rows)")

    print("\nwriting tables:")
    dump(os.path.join(args.out_dir, "labels_aggregated.csv"), agg_rows,
         ["patient", "prior_volume", "curr_volume", "delta_days", "overall",
          "overall_3class", "overall_confidence", "n_findings", "parse_ok"])
    dump(os.path.join(args.out_dir, "dynamic_sentences.csv"), dyn_rows,
         ["curr_volume", "sentence"])
    dump(os.path.join(args.out_dir, "static_sentences.csv"), stat_rows,
         ["curr_volume", "sentence"])
    dump(os.path.join(args.out_dir, "findings_long.csv"), find_rows,
         ["curr_volume", "finding", "change", "evidence"])

    # ---- report ----
    print("\noverall class balance (study-level, 4-class):")
    for k, v in overall_bal.most_common():
        print(f"    {k:>8}: {v:5d}  ({100*v/max(n,1):.1f}%)")
    if invalid_overall:
        print(f"  ({invalid_overall} had an invalid/missing 'overall' value)")

    print("\nper-finding change balance (5-class):")
    tot = sum(change_bal.values())
    for k in ["new", "worse", "stable", "improved", "resolved"]:
        v = change_bal.get(k, 0)
        print(f"    {k:>8}: {v:5d}  ({100*v/max(tot,1):.1f}%)")

    print(f"\nsentence corpora: dynamic={len(dyn_rows)}  static={len(stat_rows)}")
    print("  -> dynamic_sentences.csv feeds L_dyn (align to difference embedding)")
    print("  -> static_sentences.csv  feeds L_static (align to current embedding)")


if __name__ == "__main__":
    main()
