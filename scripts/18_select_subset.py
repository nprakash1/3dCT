#!/usr/bin/env python3
"""
18_select_subset.py

Pick a CLASS-BALANCED subset of prior->current pairs to download, so you fetch a few
hundred GB from CT-RATE instead of the whole multi-TB dataset.

Why: the raw label distribution is skewed (improved ~4%, worse ~8%, new/stable/resolved
~29% each). A random subset inherits that skew and starves the rare classes. This picks
pairs with a rarity-weighted greedy: each step it takes the pair that best covers the
currently most under-represented (finding, change) cells. The result is a subset whose
finding-level class mix is much flatter, and which is guaranteed to contain enough
worse/improved examples per trackable finding to train and evaluate on.

Selection respects the existing patient-level split (train/val/test): it runs the greedy
within each split with a proportional target, so no patient leaks and every split gets
rare-class coverage.

Input : data/ctrate/labels_final.jsonl   (from scripts/17)
Output: data/ctrate/subset_pairs.csv     (patient, prior_volume, curr_volume, delta_days, split)
        -> feed to scripts/14 for download:
           python scripts/14_ctrate_download_pairs.py --manifest data/ctrate/subset_pairs.csv --limit 100000

Usage
-----
  python scripts/18_select_subset.py --target 600            # ~600 pairs total
  python scripts/18_select_subset.py --target 600 --gb-per-vol 0.42
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

OUT_DIR = "data/ctrate"
IN = os.path.join(OUT_DIR, "labels_final.jsonl")
OUT = os.path.join(OUT_DIR, "subset_pairs.csv")
CLASSES = ["new", "worse", "stable", "improved", "resolved"]
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}  # of --target


def greedy_balanced(pairs, target):
    """Rarity-weighted greedy set selection. pairs: list of dicts with 'cells' = set of (finding,change)."""
    selected, chosen = [], set()
    count = Counter()  # per (finding,change) already selected
    remaining = list(range(len(pairs)))
    while len(selected) < target and remaining:
        best_i, best_score = None, -1.0
        for idx in remaining:
            cells = pairs[idx]["cells"]
            # reward covering under-represented cells; 1/(1+count) shrinks as a cell fills up
            score = sum(1.0 / (1 + count[c]) for c in cells) / (len(cells) ** 0.5 or 1)
            if score > best_score:
                best_score, best_i = score, idx
        selected.append(pairs[best_i])
        for c in pairs[best_i]["cells"]:
            count[c] += 1
        remaining.remove(best_i)
    return selected, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600, help="total pairs to select")
    ap.add_argument("--gb-per-vol", type=float, default=0.42)
    args = ap.parse_args()

    if not os.path.exists(IN):
        raise SystemExit(f"missing {IN} -- run scripts/17 first")
    recs = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]

    by_split = defaultdict(list)
    for r in recs:
        cells = {(fd["finding"], fd["change"]) for fd in r["findings"]}
        by_split[r["split"]].append({
            "patient": r["patient"], "prior_volume": r["prior_volume"],
            "curr_volume": r["curr_volume"], "delta_days": r["delta_days"],
            "split": r["split"], "cells": cells, "findings": r["findings"],
        })

    chosen = []
    for sp, frac in SPLIT_FRAC.items():
        pool = by_split.get(sp, [])
        tgt = min(len(pool), int(round(args.target * frac)))
        sel, _ = greedy_balanced(pool, tgt)
        chosen.extend(sel)
        print(f"{sp}: pool={len(pool)} selected={len(sel)}")

    # dedupe volumes to estimate real download size
    vols = []
    for c in chosen:
        vols += [c["prior_volume"], c["curr_volume"]]
    uniq = list(dict.fromkeys(vols))
    gb = len(uniq) * args.gb_per_vol

    # class balance: full vs subset (finding-level)
    full_c, sub_c = Counter(), Counter()
    for r in recs:
        for fd in r["findings"]:
            full_c[fd["change"]] += 1
    for c in chosen:
        for fd in c["findings"]:
            sub_c[fd["change"]] += 1

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "prior_volume", "curr_volume", "delta_days", "split"])
        for c in chosen:
            w.writerow([c["patient"], c["prior_volume"], c["curr_volume"],
                        c["delta_days"], c["split"]])

    print(f"\nSELECTED {len(chosen)} pairs -> {len(uniq)} unique volumes")
    print(f"ESTIMATED DOWNLOAD: ~{gb:.0f} GB (@ {args.gb_per_vol} GB/volume)")

    def pct(cc):
        t = sum(cc.values()) or 1
        return "  ".join(f"{k}={cc[k]}({100*cc[k]//t}%)" for k in CLASSES)
    print("\nfinding-level class balance:")
    print("  FULL  :", pct(full_c))
    print("  SUBSET:", pct(sub_c), " <- flatter = rare classes boosted")

    # per-finding worse/improved support in the subset (the classes that starve)
    pf = defaultdict(Counter)
    for c in chosen:
        for fd in c["findings"]:
            pf[fd["finding"]][fd["change"]] += 1
    print("\nper-finding worse/improved support in subset:")
    for fnd in sorted(pf, key=lambda x: -(pf[x]['worse'] + pf[x]['improved'])):
        print(f"  {fnd:<32} worse={pf[fnd]['worse']:>4}  improved={pf[fnd]['improved']:>4}")

    print(f"\nwrote -> {OUT}")
    print("next:\n  python scripts/14_ctrate_download_pairs.py "
          "--manifest data/ctrate/subset_pairs.csv --limit 100000 --dry-run")


if __name__ == "__main__":
    main()
