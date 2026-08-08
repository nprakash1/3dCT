#!/usr/bin/env python3
"""
20_select_gold_subset.py

Pick a GOLD, 3-CLASS-BALANCED subset of prior->current pairs from the v3 MedGemma
labels, so you cache a few hundred GB from CT-RATE instead of the whole ~2.9 TB.

What "gold" and "3-class" mean here (decided from the v3 audit):
  * GOLD   = only findings whose direction was stated in the report text
             (tier == "explicit"). These are the ~45%-more-reliable labels; the
             "inferred" tier came from the noisy presence-diff and is excluded.
  * 3-CLASS= balance on DIRECTION {worsened, stable, improved}, NOT the 5-class
             change. Globally the gold set is already near-balanced
             (7007 / 6540 / 8675), so a balanced subset is easily achievable.

Why not balance per (finding, direction)? Because some findings physically never
improve -- e.g. arterial/coronary calcification and fibrosis have ~0 "improved"
gold labels. Chasing those empty cells would starve the subset. So we balance on
the GLOBAL direction distribution and use (finding, direction) only as a rarity
weight over CELLS THAT ACTUALLY OCCUR (empty cells never enter the greedy).

Splitting: v3 has no split field, so we assign a deterministic PATIENT-level
70/15/15 split by hashing the patient id (md5). No patient leaks across splits.

Input : medgemma_labels_v3.jsonl   (fields: patient, prior_volume, curr_volume,
                                     delta_days, findings[{finding,change,
                                     direction,tier,...}])
Output: data/ctrate/subset_pairs.csv
        columns: patient, prior_volume, curr_volume, delta_days, split
        -> feed straight to the Phase-B caching notebook (cell 8).

Usage
-----
  python scripts/20_select_gold_subset.py --target 600
  python scripts/20_select_gold_subset.py --target 600 --gb-per-vol 0.42
"""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

IN = "medgemma_labels_v3.jsonl"
OUT_DIR = "data/ctrate"
OUT = os.path.join(OUT_DIR, "subset_pairs.csv")

DIRS = ["worsened", "stable", "improved"]
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}


def patient_split(patient: str) -> str:
    """Deterministic 70/15/15 patient-level split via md5 hash -> [0,1)."""
    h = hashlib.md5(patient.encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < SPLIT_FRAC["train"]:
        return "train"
    if x < SPLIT_FRAC["train"] + SPLIT_FRAC["val"]:
        return "val"
    return "test"


def greedy_balanced(pairs, target):
    """Rarity-weighted greedy set selection.

    Each pair carries `cells` = set of (finding, direction) gold cells and
    `dirs` = Counter of direction -> count within the pair. The score rewards
    (a) covering under-represented DIRECTIONS (primary) and (b) covering
    under-represented (finding,direction) cells (secondary, diversity).
    """
    selected = []
    dir_count = Counter()   # direction -> selected so far (primary balance axis)
    cell_count = Counter()  # (finding,direction) -> selected (diversity)
    remaining = list(range(len(pairs)))

    while len(selected) < target and remaining:
        best_i, best_score = None, -1.0
        for idx in remaining:
            p = pairs[idx]
            # primary: flatten the global direction distribution
            dir_score = sum(p["dirs"][d] / (1.0 + dir_count[d]) for d in DIRS)
            # secondary: reward rare (finding,direction) cells (diversity)
            cell_score = sum(1.0 / (1.0 + cell_count[c]) for c in p["cells"])
            score = dir_score + 0.25 * cell_score
            # normalize by sqrt(#cells) so label-dense pairs don't dominate
            score /= (len(p["cells"]) ** 0.5 or 1)
            if score > best_score:
                best_score, best_i = score, idx
        selected.append(pairs[best_i])
        for d in DIRS:
            dir_count[d] += pairs[best_i]["dirs"][d]
        for c in pairs[best_i]["cells"]:
            cell_count[c] += 1
        remaining.remove(best_i)
    return selected, dir_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600, help="total pairs to select")
    ap.add_argument("--gb-per-vol", type=float, default=0.42)
    ap.add_argument("--in", dest="inp", default=IN)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"missing {args.inp}")
    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]

    # build gold pair records, bucketed by patient-level split
    by_split = defaultdict(list)
    n_gold_pairs = 0
    for r in recs:
        gold = [fd for fd in r["findings"]
                if fd.get("tier") == "explicit" and fd.get("direction") in DIRS]
        if not gold:
            continue  # skip pairs with no gold directional label
        n_gold_pairs += 1
        cells = {(fd["finding"], fd["direction"]) for fd in gold}
        dirs = Counter(fd["direction"] for fd in gold)
        sp = patient_split(r["patient"])
        by_split[sp].append({
            "patient": r["patient"],
            "prior_volume": r["prior_volume"],
            "curr_volume": r["curr_volume"],
            "delta_days": r.get("delta_days", ""),
            "split": sp,
            "cells": cells,
            "dirs": dirs,
            "gold": gold,
        })

    chosen = []
    for sp, frac in SPLIT_FRAC.items():
        pool = by_split.get(sp, [])
        tgt = min(len(pool), int(round(args.target * frac)))
        sel, _ = greedy_balanced(pool, tgt)
        chosen.extend(sel)
        print(f"{sp}: gold_pool={len(pool)} selected={len(sel)}")

    # dedupe volumes to estimate real download size
    vols = []
    for c in chosen:
        vols += [c["prior_volume"], c["curr_volume"]]
    uniq = list(dict.fromkeys(vols))
    gb = len(uniq) * args.gb_per_vol

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "prior_volume", "curr_volume", "delta_days", "split"])
        for c in chosen:
            w.writerow([c["patient"], c["prior_volume"], c["curr_volume"],
                        c["delta_days"], c["split"]])

    # ---- report ----
    print(f"\ngold pairs available: {n_gold_pairs} / {len(recs)} total")
    print(f"SELECTED {len(chosen)} pairs -> {len(uniq)} unique volumes")
    print(f"ESTIMATED DOWNLOAD: ~{gb:.0f} GB (@ {args.gb_per_vol} GB/volume)")

    # direction balance: full-gold vs subset (label-level)
    full_d, sub_d = Counter(), Counter()
    for r in recs:
        for fd in r["findings"]:
            if fd.get("tier") == "explicit" and fd.get("direction") in DIRS:
                full_d[fd["direction"]] += 1
    for c in chosen:
        for fd in c["gold"]:
            sub_d[fd["direction"]] += 1

    def pct(cc):
        t = sum(cc.values()) or 1
        return "  ".join(f"{k}={cc[k]}({100 * cc[k] // t}%)" for k in DIRS)
    print("\ngold direction balance (label-level):")
    print("  FULL  :", pct(full_d))
    print("  SUBSET:", pct(sub_d), " <- flatter = better")

    # per-finding direction support in the subset
    pf = defaultdict(Counter)
    for c in chosen:
        for fd in c["gold"]:
            pf[fd["finding"]][fd["direction"]] += 1
    print("\nper-finding direction support in subset:")
    print(f"  {'finding':<32}{'wors':>6}{'stab':>6}{'impr':>6}")
    for fnd in sorted(pf, key=lambda x: -sum(pf[x].values())):
        c = pf[fnd]
        print(f"  {fnd:<32}{c['worsened']:>6}{c['stable']:>6}{c['improved']:>6}")

    print(f"\nwrote -> {OUT}")
    print("next: upload this CSV to the Phase-B caching notebook (cell 8).")


if __name__ == "__main__":
    main()
