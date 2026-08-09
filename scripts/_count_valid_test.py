#!/usr/bin/env python3
"""How many pairs/examples if TEST = CT-RATE valid_* pairs only (encoder-unseen)?
Counts over the full labeled pool (medgemma_labels_v3.jsonl), not just the 600 subset.
Writes results to /tmp/valid_test_counts.txt so they survive shell interruptions.
"""
import json, collections

C = {"worsened", "stable", "improved"}
rows = [json.loads(l) for l in open("medgemma_labels_v3.jsonl") if l.strip()]

def prefix(v):
    v = str(v)
    for p in ("train", "valid", "test"):
        if v.startswith(p):
            return p
    return "other"

pair_pref = collections.Counter()
valid_pairs = valid_examples = usable_valid_pairs = 0
train_pairs = train_examples = usable_train_pairs = 0
valid_by_dir = collections.Counter()
train_by_dir = collections.Counter()

for r in rows:
    if not r.get("parse_ok", True):
        continue
    pf = prefix(r.get("curr_volume"))
    pair_pref[pf] += 1
    expl = [fd for fd in r.get("findings", [])
            if fd.get("tier") == "explicit" and fd.get("direction") in C]
    if pf == "valid":
        valid_pairs += 1
        if expl:
            usable_valid_pairs += 1
        for fd in expl:
            valid_examples += 1
            valid_by_dir[fd["direction"]] += 1
    elif pf == "train":
        train_pairs += 1
        if expl:
            usable_train_pairs += 1
        for fd in expl:
            train_examples += 1
            train_by_dir[fd["direction"]] += 1


out = []
out.append("=== FULL labeled pool (medgemma_labels_v3.jsonl), parse_ok ===")
out.append(f"pairs by CT-RATE source split (curr_volume prefix): {dict(pair_pref)}")
out.append("")
out.append("=== If TEST = CT-RATE valid_* pairs only (encoder-UNSEEN) ===")
out.append(f"valid_* pairs total           : {valid_pairs}")
out.append(f"valid_* pairs with >=1 explicit finding (usable): {usable_valid_pairs}")
out.append(f"valid_* (pair,finding) EXAMPLES: {valid_examples}")
out.append(f"  by direction: {dict(valid_by_dir)}")
out.append("")
out.append("=== If TRAIN(+val) = CT-RATE train_* pairs (encoder-SEEN) ===")
out.append(f"train_* pairs total           : {train_pairs}")
out.append(f"train_* pairs usable (>=1 explicit): {usable_train_pairs}")
out.append(f"train_* (pair,finding) EXAMPLES: {train_examples}")
out.append(f"  by direction: {dict(train_by_dir)}")

txt = "\n".join(out)
open("/tmp/valid_test_counts.txt", "w").write(txt + "\n")
print(txt)
