#!/usr/bin/env python3
"""Reproduce the deterministic 600-pair greedy subset from medgemma_labels_v3.jsonl
and report pairs + unique volumes. Mirrors scripts/18_select_subset.py logic."""
import json
from collections import Counter

SRC = "medgemma_labels_v3.jsonl"
recs = []
for line in open(SRC):
    s = line.strip()
    if not s:
        continue
    d = json.loads(s)
    if not d.get("parse_ok", True):
        continue
    findings = d.get("findings", [])
    cells = set()
    for fd in findings:
        if isinstance(fd, dict) and "finding" in fd and "change" in fd:
            cells.add((fd["finding"], fd["change"]))
    if not cells:
        continue
    recs.append({"patient": d["patient"], "pv": d["prior_volume"],
                 "cv": d["curr_volume"], "cells": cells})

print(f"eligible pairs (parse_ok + has findings): {len(recs)}")

def greedy(pairs, target):
    selected = []
    count = Counter()
    remaining = list(range(len(pairs)))
    while len(selected) < target and remaining:
        best_i, best = None, -1.0
        for idx in remaining:
            cells = pairs[idx]["cells"]
            score = sum(1.0/(1+count[c]) for c in cells) / (len(cells)**0.5 or 1)
            if score > best:
                best, best_i = score, idx
        selected.append(pairs[best_i])
        for c in pairs[best_i]["cells"]:
            count[c] += 1
        remaining.remove(best_i)
    return selected

for target in (600,):
    sel = greedy(recs, target)
    vols = set()
    pats = set()
    for c in sel:
        vols.add(c["pv"]); vols.add(c["cv"]); pats.add(c["patient"])
    print(f"\n--target {target}:")
    print(f"  PAIRS selected : {len(sel)}")
    print(f"  UNIQUE VOLUMES : {len(vols)}")
    print(f"  patients       : {len(pats)}")
    print(f"  vols/pair ratio: {len(vols)/len(sel):.2f}")
