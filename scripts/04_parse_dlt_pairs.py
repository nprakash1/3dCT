#!/usr/bin/env python3
"""
04_parse_dlt_pairs.py

Parse the Deep Lesion Tracker (DLT) JSON annotations into a flat manifest of
prior->current lesion pairs, and assign an interval-change label to each pair
using a RECIST-style rule on the long-axis diameter.

DLT JSON structure (per pair record, keyed by an id like "0001"):
    source / target                : NIfTI filenames (patient_study_series_startslice-endslice.nii.gz)
    source spacing / target spacing: [x_mm, y_mm, z_mm]
    source recist diameter         : [long_axis_mm, short_axis_mm]
    target recist diameter         : [long_axis_mm, short_axis_mm]
    source/target center, box, ... : geometry for lesion-centered cropping

RECIST 1.1 (single-lesion adaptation, on the long-axis diameter):
    worse    (progression) : increase >= +20%  AND  absolute increase >= 5 mm
    improved (regression)  : decrease >= 30%
    stable                 : everything in between
This is the standard single-target-lesion reading of RECIST; the +5 mm floor
prevents tiny sub-pixel "increases" from being called progression.

Usage:
    python scripts/04_parse_dlt_pairs.py --split train
    python scripts/04_parse_dlt_pairs.py --split all --out data/dlt/manifest_all.csv
"""
import argparse
import csv
import json
import os
from collections import Counter

# RECIST thresholds (fractions of the prior long-axis diameter)
PROGRESSION_FRAC = 0.20     # >= +20% -> worse
PROGRESSION_ABS_MM = 5.0    # and >= +5 mm absolute
RESPONSE_FRAC = 0.30        # >= -30% -> improved


def recist_label(prior_long_mm: float, curr_long_mm: float):
    """Return (label, pct_change) from prior/current long-axis diameters (mm)."""
    if prior_long_mm <= 0:
        return "unknown", None
    delta = curr_long_mm - prior_long_mm
    pct = delta / prior_long_mm
    if pct >= PROGRESSION_FRAC and delta >= PROGRESSION_ABS_MM:
        return "worse", pct
    if pct <= -RESPONSE_FRAC:
        return "improved", pct
    return "stable", pct


def parse_split(json_path: str):
    """Yield one flat dict per lesion pair in a DLT json file."""
    with open(json_path) as f:
        data = json.load(f)
    for pair_id, rec in data.items():
        # long-axis is index 0 of the [long, short] recist diameter list
        try:
            prior_long = float(rec["source recist diameter"][0])
            curr_long = float(rec["target recist diameter"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        label, pct = recist_label(prior_long, curr_long)
        yield {
            "pair_id": pair_id,
            "source_nii": rec.get("source", ""),
            "target_nii": rec.get("target", ""),
            "prior_long_mm": round(prior_long, 3),
            "curr_long_mm": round(curr_long, 3),
            "pct_change": None if pct is None else round(pct, 4),
            "label": label,
            "source_spacing": rec.get("source spacing", ""),
            "target_spacing": rec.get("target spacing", ""),
            "source_center": rec.get("source center", ""),
            "target_center": rec.get("target center", ""),
            "source_recist_slice": rec.get("source recist slice", ""),
            "target_recist_slice": rec.get("target recist slice", ""),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "valid", "test", "DLTMix", "all"],
                    help="Which DLT split to parse (or 'all' to concatenate).")
    ap.add_argument("--dlt-dir", default="data/dlt",
                    help="Directory containing the DLT *.json files.")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: data/dlt/manifest_<split>.csv).")
    args = ap.parse_args()

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    out_path = args.out or os.path.join(args.dlt_dir, f"manifest_{args.split}.csv")

    rows = []
    for sp in splits:
        jp = os.path.join(args.dlt_dir, f"{sp}.json")
        if not os.path.exists(jp):
            print(f"[warn] missing {jp}, skipping")
            continue
        split_rows = list(parse_split(jp))
        for r in split_rows:
            r["split"] = sp
        rows.extend(split_rows)
        print(f"[{sp}] parsed {len(split_rows)} pairs")

    if not rows:
        print("No pairs parsed. Check --dlt-dir.")
        return

    fieldnames = ["split", "pair_id", "source_nii", "target_nii",
                  "prior_long_mm", "curr_long_mm", "pct_change", "label",
                  "source_spacing", "target_spacing",
                  "source_center", "target_center",
                  "source_recist_slice", "target_recist_slice"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    dist = Counter(r["label"] for r in rows)
    total = len(rows)
    print(f"\nWrote {total} pairs -> {out_path}")
    print("Label distribution (RECIST on long-axis):")
    for lab in ["worse", "stable", "improved", "unknown"]:
        n = dist.get(lab, 0)
        print(f"  {lab:9s}: {n:5d}  ({100.0*n/total:5.1f}%)")


if __name__ == "__main__":
    main()
