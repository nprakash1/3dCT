#!/usr/bin/env python3
"""
06_baseline_on_pairs.py

The zero-shot "floor" baseline for CT temporal progression, evaluated on the
DLT lesion pairs (RECIST labels: worse / stable / improved).

Pipeline (NO training):
  1. Build a MERLIN-style transform at a chosen depth (Option B, default 32),
     so lesion sub-volumes are mostly real tissue rather than zero-padding.
  2. Embed every unique NIfTI with FROZEN MERLIN -> 512-d (cached to disk).
  3. For each pair:  d = v_current - v_prior.
  4. Classify:
       - magnitude gate: if ||d|| < tau  -> predict "stable"
                         (cosine direction of a ~0 vector is meaningless --
                          this is the structural reason cosine can't do stable)
       - else: cosine(d, prompt_worse) vs cosine(d, prompt_improved) -> argmax
  5. Report accuracy, macro-F1, per-class recall, confusion matrix, and the
     ||d|| distribution per class -- against the trivial "always stable" ref.

We report tau three ways: an unsupervised median-||d|| heuristic AND an
"oracle" tau chosen to maximize macro-F1 (an upper bound for this simple rule).

Usage:
    python scripts/06_baseline_on_pairs.py --manifest data/dlt/manifest_mini.csv \
        --nifti-dir data/nifti_mini --depth 32 --device cpu
"""
import argparse
import ast
import csv
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, ToTensord,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merlin_utils import MerlinEmbedder  # noqa: E402

warnings.filterwarnings("ignore")

CLASSES = ["improved", "stable", "worse"]

# Direction prompts (stable is handled by the magnitude gate, not direction).
DIR_PROMPTS = {
    "worse": "the lesion has increased in size and worsened compared to the prior study",
    "improved": "the lesion has decreased in size and improved compared to the prior study",
}


def build_transforms(depth: int):
    """MERLIN's exact pipeline, but with a configurable out-of-plane depth."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000,
                             b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, depth]),
        CenterSpatialCropd(roi_size=[224, 224, depth], keys=["image"]),
        ToTensord(keys=["image"]),
    ])


@torch.no_grad()
def embed_volumes(names, nifti_dir, embedder, depth, cache_path, device):
    """Return {name: 512-d normalized tensor}, using a disk cache."""
    cache = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        print(f"    loaded {len(cache)} cached embeddings from {cache_path}")
    todo = [n for n in names if n not in cache]
    if todo:
        tf = build_transforms(depth)
        for i, name in enumerate(todo, 1):
            path = os.path.join(nifti_dir, name)
            if not os.path.exists(path):
                continue
            img = tf({"image": path})["image"]        # (1, 224, 224, depth)
            x = img.unsqueeze(0).to(device)           # (1, 1, 224, 224, depth)
            feats, _ehr = embedder.arch.encode_image(x)
            v = feats.detach().float().cpu().reshape(-1)[:512]
            cache[name] = F.normalize(v, dim=-1)
            if i % 25 == 0 or i == len(todo):
                print(f"    embedded {i}/{len(todo)}")
        torch.save(cache, cache_path)
        print(f"    cached {len(cache)} embeddings -> {cache_path}")
    return cache


def load_pairs(manifest):
    rows = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def confusion(y_true, y_pred):
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def metrics_from_confusion(M):
    per = {}
    f1s = []
    for i, c in enumerate(CLASSES):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = (prec, rec, f1)
        f1s.append(f1)
    acc = np.trace(M) / M.sum()
    return acc, float(np.mean(f1s)), per


def evaluate(pairs, emb, dir_vecs, tau):
    y_true, y_pred = [], []
    for r in pairs:
        s, t = r["source_nii"], r["target_nii"]
        if s not in emb or t not in emb:
            continue
        d = emb[t] - emb[s]
        dn = d.norm().item()
        if dn < tau:
            pred = "stable"
        else:
            du = F.normalize(d, dim=-1)
            sw = float(du @ dir_vecs["worse"])
            si = float(du @ dir_vecs["improved"])
            pred = "worse" if sw >= si else "improved"
        y_true.append(r["label"])
        y_pred.append(pred)
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dlt/manifest_mini.csv")
    ap.add_argument("--nifti-dir", default="data/nifti_mini")
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="limit #pairs (debug)")
    args = ap.parse_args()

    pairs = load_pairs(args.manifest)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Pairs: {len(pairs)}")

    names = sorted({r["source_nii"] for r in pairs} |
                   {r["target_nii"] for r in pairs})
    print(f"Unique volumes: {len(names)}")

    print("==> Loading frozen MERLIN...")
    embedder = MerlinEmbedder(device=args.device)
    print(f"    device: {embedder.device}")

    cache_path = os.path.join(os.path.dirname(args.nifti_dir),
                              f"emb_mini_d{args.depth}.pt")
    print("==> Embedding volumes (cached)...")
    emb = embed_volumes(names, args.nifti_dir, embedder, args.depth,
                        cache_path, embedder.device)

    print("==> Embedding direction prompts...")
    dir_vecs = {}
    for k, prompt in DIR_PROMPTS.items():
        tv = embedder.embed_texts([prompt], normalize=True)[0]
        dir_vecs[k] = tv

    # ---- change-vector magnitude diagnostics --------------------------------
    dn_by_class = {c: [] for c in CLASSES}
    for r in pairs:
        s, t = r["source_nii"], r["target_nii"]
        if s in emb and t in emb:
            dn_by_class[r["label"]].append((emb[t] - emb[s]).norm().item())
    print("\n||d|| = ||v_current - v_prior||  by true class:")
    all_dn = []
    for c in CLASSES:
        v = dn_by_class[c]
        all_dn += v
        if v:
            print(f"  {c:9s} n={len(v):4d}  mean={np.mean(v):.4f}  "
                  f"median={np.median(v):.4f}")
    all_dn = np.array(all_dn)

    # ---- reference: always-stable ------------------------------------------
    yt_ref = [r["label"] for r in pairs
              if r["source_nii"] in emb and r["target_nii"] in emb]
    acc_ref = sum(1 for y in yt_ref if y == "stable") / len(yt_ref)
    print(f"\n[reference] always-'stable'  accuracy = {acc_ref:.3f}  "
          f"(macro-F1 = {(2*0*0/1):.3f} -> only 'stable' class predicted)")

    # ---- baseline @ unsupervised tau (median ||d||) -------------------------
    tau_med = float(np.median(all_dn))
    yt, yp = evaluate(pairs, emb, dir_vecs, tau_med)
    M = confusion(yt, yp)
    acc, mf1, per = metrics_from_confusion(M)
    print(f"\n===== Baseline (magnitude gate tau=median={tau_med:.4f}) =====")
    print(f"  accuracy = {acc:.3f}   macro-F1 = {mf1:.3f}")
    for c in CLASSES:
        p, rc, f1 = per[c]
        print(f"  {c:9s} P={p:.3f} R={rc:.3f} F1={f1:.3f}")
    print("  confusion (rows=true, cols=pred) order", CLASSES)
    print(M)

    # ---- oracle tau (upper bound for this simple rule) ----------------------
    best = (-1, None, None)
    for q in np.linspace(0.05, 0.95, 19):
        tau = float(np.quantile(all_dn, q))
        yt2, yp2 = evaluate(pairs, emb, dir_vecs, tau)
        _, mf1_2, _ = metrics_from_confusion(confusion(yt2, yp2))
        if mf1_2 > best[0]:
            best = (mf1_2, tau, q)
    print(f"\n[oracle tau] best macro-F1 = {best[0]:.3f} "
          f"at tau={best[1]:.4f} (quantile {best[2]:.2f})")

    print("\n==> Interpretation: if macro-F1 is barely above the always-stable "
          "floor, MERLIN's global embedding difference does NOT cleanly encode "
          "interval change -- which is exactly what the Step 3 trainable "
          "difference module + magnitude head are designed to fix.")


if __name__ == "__main__":
    main()
