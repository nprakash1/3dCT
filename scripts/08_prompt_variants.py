#!/usr/bin/env python3
"""
08_prompt_variants.py

Compare three ZERO-SHOT (no-training) baselines for CT interval-change on the
DLT lesion pairs, all reusing the cached MERLIN image embeddings from step 06:

  (A) magnitude gate  : ||d|| < tau -> "stable", else cosine(d, {worse,improved})
  (B) 3-way prompts   : argmax_c cosine(d, prompt_c) over {worse,stable,improved}
                        (a single prompt per class; no magnitude special-casing)
  (C) prompt ensemble : same as B, but each class vector is the average of
                        several paraphrase prompts (a fairer, less noisy floor)

Only the text prompts are (re)embedded here; the 512-d image embeddings are
loaded from data/emb_mini_d32.pt, so this is fast.

Usage:
    python scripts/08_prompt_variants.py --depth 32
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merlin_utils import MerlinEmbedder  # noqa: E402

warnings.filterwarnings("ignore")

CLASSES = ["improved", "stable", "worse"]

# ---- (B) single prompt per class ----
SINGLE_PROMPTS = {
    "worse":    ["the lesion has increased in size and worsened compared to the prior study"],
    "stable":   ["the lesion is unchanged and stable compared to the prior study"],
    "improved": ["the lesion has decreased in size and improved compared to the prior study"],
}

# ---- (C) small paraphrase ensemble per class ----
ENSEMBLE_PROMPTS = {
    "worse": [
        "the lesion has increased in size and worsened compared to the prior study",
        "the finding is larger and has progressed since the previous exam",
        "interval enlargement of the lesion, consistent with disease progression",
        "the mass has grown compared to the prior study",
        "increased size of the lesion indicating worsening",
    ],
    "stable": [
        "the lesion is unchanged and stable compared to the prior study",
        "no significant interval change in the lesion since the previous exam",
        "the finding is stable with no change in size",
        "the lesion appears similar to the prior study",
        "no measurable change in the lesion compared to prior",
    ],
    "improved": [
        "the lesion has decreased in size and improved compared to the prior study",
        "the finding is smaller and has regressed since the previous exam",
        "interval decrease in the lesion, consistent with treatment response",
        "the mass has shrunk compared to the prior study",
        "decreased size of the lesion indicating improvement",
    ],
}


def load_pairs(manifest):
    with open(manifest) as f:
        return list(csv.DictReader(f))


def class_vectors(embedder, prompt_bank):
    """Return {class: unit 512-d vector} = normalized mean of prompt embeddings."""
    vecs = {}
    for c in CLASSES:
        embs = embedder.embed_texts(prompt_bank[c], normalize=True)  # (k,512)
        mean = embs.mean(dim=0)
        vecs[c] = F.normalize(mean, dim=-1)
    return vecs


def confusion(y_true, y_pred):
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def metrics(M):
    per, f1s = {}, []
    for i, c in enumerate(CLASSES):
        tp = M[i, i]; fp = M[:, i].sum() - tp; fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = (prec, rec, f1); f1s.append(f1)
    return np.trace(M) / M.sum(), float(np.mean(f1s)), per


def report(name, y_true, y_pred):
    M = confusion(y_true, y_pred)
    acc, mf1, per = metrics(M)
    print(f"\n===== {name} =====")
    print(f"  accuracy = {acc:.3f}   macro-F1 = {mf1:.3f}")
    for c in CLASSES:
        p, r, f1 = per[c]
        print(f"  {c:9s} P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("  confusion (rows=true, cols=pred)", CLASSES)
    print(M)
    return acc, mf1


def predict_cosine(pairs, emb, cvecs):
    yt, yp = [], []
    for r in pairs:
        s, t = r["source_nii"], r["target_nii"]
        if s not in emb or t not in emb:
            continue
        d = F.normalize(emb[t] - emb[s], dim=-1)
        sims = {c: float(d @ cvecs[c]) for c in CLASSES}
        yp.append(max(sims, key=sims.get))
        yt.append(r["label"])
    return yt, yp


def predict_magnitude(pairs, emb, dir_vecs, tau):
    yt, yp = [], []
    for r in pairs:
        s, t = r["source_nii"], r["target_nii"]
        if s not in emb or t not in emb:
            continue
        d = emb[t] - emb[s]
        if d.norm().item() < tau:
            pred = "stable"
        else:
            du = F.normalize(d, dim=-1)
            pred = "worse" if float(du @ dir_vecs["worse"]) >= float(du @ dir_vecs["improved"]) else "improved"
        yp.append(pred); yt.append(r["label"])
    return yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dlt/manifest_mini.csv")
    ap.add_argument("--depth", type=int, default=32)
    args = ap.parse_args()

    pairs = load_pairs(args.manifest)
    cache_path = f"data/emb_mini_d{args.depth}.pt"
    if not os.path.exists(cache_path):
        print(f"!! missing {cache_path}; run 06 first to build image embeddings.")
        return
    emb = torch.load(cache_path, weights_only=False)
    print(f"loaded {len(emb)} cached image embeddings from {cache_path}")

    print("==> Loading frozen MERLIN (for text prompts)...")
    embedder = MerlinEmbedder(device="cpu")

    single = class_vectors(embedder, SINGLE_PROMPTS)
    ens = class_vectors(embedder, ENSEMBLE_PROMPTS)

    # trivial reference
    yt_ref = [r["label"] for r in pairs
              if r["source_nii"] in emb and r["target_nii"] in emb]
    acc_ref = sum(1 for y in yt_ref if y == "stable") / len(yt_ref)
    stable_f1 = 2 * acc_ref * 1.0 / (acc_ref + 1.0)  # P=acc_ref, R=1
    print(f"\n[reference] always-'stable': accuracy={acc_ref:.3f}  "
          f"macro-F1={stable_f1/3:.3f}")

    # ---- (A) magnitude gate ----
    dir_vecs = {"worse": single["worse"], "improved": single["improved"]}
    all_dn = np.array([(emb[r["target_nii"]] - emb[r["source_nii"]]).norm().item()
                       for r in pairs
                       if r["source_nii"] in emb and r["target_nii"] in emb])
    tau = float(np.median(all_dn))
    yt, yp = predict_magnitude(pairs, emb, dir_vecs, tau)
    a_acc, a_f1 = report(f"(A) magnitude gate  (tau=median={tau:.3f})", yt, yp)

    # ---- (B) 3-way single prompt ----
    yt, yp = predict_cosine(pairs, emb, single)
    b_acc, b_f1 = report("(B) 3-way cosine, single prompt (incl. stable)", yt, yp)

    # ---- (C) 3-way ensemble ----
    yt, yp = predict_cosine(pairs, emb, ens)
    c_acc, c_f1 = report("(C) 3-way cosine, prompt ENSEMBLE (5 paraphrases/class)", yt, yp)

    print("\n================ SUMMARY (macro-F1 / accuracy) ================")
    print(f"  always-stable : {stable_f1/3:.3f} / {acc_ref:.3f}")
    print(f"  (A) magnitude : {a_f1:.3f} / {a_acc:.3f}")
    print(f"  (B) single    : {b_f1:.3f} / {b_acc:.3f}")
    print(f"  (C) ensemble  : {c_f1:.3f} / {c_acc:.3f}")


if __name__ == "__main__":
    main()
