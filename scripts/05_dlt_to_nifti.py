#!/usr/bin/env python3
"""
05_dlt_to_nifti.py

Convert DeepLesion PNG slice stacks (as referenced by the DLT manifest) into
lesion-centered NIfTI sub-volumes that MERLIN can ingest.

Each DLT nii name encodes everything we need:
    000019_01_01_015-027.nii.gz
    ^patient ^study ^series ^slice-range (inclusive)
The matching PNGs live in:
    <images_root>/000019_01_01/015.png ... 027.png   (16-bit, HU = pixel - 32768)

We stack those slices into a (H, W, D) volume, convert to HU, and write a NIfTI
with a diagonal affine built from the CT spacing (from the manifest). MERLIN's
own transforms later handle RAS reorientation, resampling to (1.5,1.5,3),
intensity scaling, and padding/cropping -- so here we only need correct spacing
and a consistent build.

Can be used as a CLI (convert everything in a manifest) or imported
(`ensure_nifti(...)`) by the baseline script.

Usage:
    python scripts/05_dlt_to_nifti.py \
        --manifest data/dlt/manifest_mini.csv \
        --images-root data/archive/minideeplesion \
        --out-dir data/nifti_mini
"""
import argparse
import ast
import csv
import os
import re

import numpy as np
import nibabel as nib
from PIL import Image

HU_OFFSET = 32768  # DeepLesion 16-bit PNGs store (HU + 32768)

NAME_RE = re.compile(r"^(\d{6})_(\d{2})_(\d{2})_(\d+)-(\d+)$")


def parse_nii_name(nii_name: str):
    """'000019_01_01_015-027.nii.gz' -> (study_folder, start, end)."""
    base = nii_name.split(".nii")[0]
    m = NAME_RE.match(base)
    if not m:
        raise ValueError(f"Unexpected nii name: {nii_name}")
    pid, study, series, s0, s1 = m.groups()
    study_folder = f"{pid}_{study}_{series}"
    return study_folder, int(s0), int(s1)


def _spacing_from_manifest(manifest_path: str):
    """Map each nii name -> [sx, sy, sz] using whichever row references it."""
    spacing = {}
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            for role in ("source", "target"):
                name = r[f"{role}_nii"]
                sp = r.get(f"{role}_spacing", "")
                if name and name not in spacing and sp:
                    try:
                        spacing[name] = [float(x) for x in ast.literal_eval(sp)]
                    except Exception:
                        pass
    return spacing


def build_volume(study_folder: str, s0: int, s1: int, images_root: str):
    """Read PNG slices s0..s1 (inclusive) -> (H, W, D) float32 HU volume."""
    folder = os.path.join(images_root, study_folder)
    slices = []
    for idx in range(s0, s1 + 1):
        fp = os.path.join(folder, f"{idx:03d}.png")
        if not os.path.exists(fp):
            continue  # tolerate an occasional missing slice
        arr = np.asarray(Image.open(fp)).astype(np.float32) - HU_OFFSET
        slices.append(arr)
    if not slices:
        raise FileNotFoundError(f"No slices found for {study_folder} [{s0}-{s1}]")
    vol = np.stack(slices, axis=-1)  # (H, W, D)
    return vol


def ensure_nifti(nii_name: str, images_root: str, out_dir: str,
                 spacing=None, overwrite: bool = False) -> str:
    """Create <out_dir>/<nii_name> if missing; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, nii_name)
    if os.path.exists(out_path) and not overwrite:
        return out_path
    study_folder, s0, s1 = parse_nii_name(nii_name)
    vol = build_volume(study_folder, s0, s1, images_root)
    sx, sy, sz = (spacing or [1.0, 1.0, 1.0])
    affine = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(vol, affine), out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dlt/manifest_mini.csv")
    ap.add_argument("--images-root", default="data/archive/minideeplesion")
    ap.add_argument("--out-dir", default="data/nifti_mini")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spacing = _spacing_from_manifest(args.manifest)

    names = set()
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            names.add(r["source_nii"])
            names.add(r["target_nii"])
    names = sorted(names)
    print(f"Unique sub-volumes to build: {len(names)}")

    ok = miss = 0
    for i, name in enumerate(names, 1):
        try:
            ensure_nifti(name, args.images_root, args.out_dir,
                         spacing=spacing.get(name), overwrite=args.overwrite)
            ok += 1
        except FileNotFoundError:
            miss += 1
        if i % 50 == 0:
            print(f"  {i}/{len(names)} built (ok={ok}, missing={miss})")
    print(f"\nDone. built/exists={ok}, missing={miss} -> {args.out_dir}")


if __name__ == "__main__":
    main()
