#!/usr/bin/env python3
"""
11_ctrate_inspect.py

Inspect exactly WHAT fields CT-RATE exposes, and specifically answer:
  - What columns are in the radiology-report CSVs?
  - What columns are in the DICOM-header metadata CSVs?
  - Do DATES / TIMES / AGE fields exist (StudyDate, AcquisitionDate, PatientAge...)?
    If so, are they real values or blanked/anonymized?

Downloads only small CSVs (report + metadata). No volumes.
"""
import csv
import os
from collections import Counter

REPO_ID = "ibrahimhamamci/CT-RATE"
OUT_DIR = "data/ctrate/_reports"

REPORT_CSVS = [
    "dataset/radiology_text_reports/train_reports.csv",
    "dataset/radiology_text_reports/validation_reports.csv",
]
META_CANDIDATES = [
    "dataset/metadata/train_metadata.csv",
    "dataset/metadata/validation_metadata.csv",
    "dataset/metadata/train_fixed_metadata.csv",
    "dataset/metadata/valid_metadata.csv",
    "dataset/train_metadata.csv",
    "dataset/validation_metadata.csv",
    "metadata/train_metadata.csv",
    "metadata/validation_metadata.csv",
]

# DICOM-ish temporal / age tags we care about
DATE_HINTS = ["date", "time", "age", "birth", "acqui", "study", "series", "content"]


def dl(path):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(REPO_ID, path, repo_type="dataset",
                               token=os.environ.get("HF_TOKEN"), local_dir=OUT_DIR)
    except Exception as e:
        print(f"  (skip {path}: {type(e).__name__})")
        return None


def peek(path, label, max_rows=3):
    print("\n" + "=" * 72)
    print(f"{label}: {path}")
    print("=" * 72)
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        print(f"{len(header)} columns:")
        for h in header:
            print("   -", h)

        # flag date/time/age-ish columns
        date_cols = [h for h in header if any(k in h.lower() for k in DATE_HINTS)]
        print("\nDATE/TIME/AGE-like columns:", date_cols or "NONE")

        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i + 1 >= 5000:   # enough to judge if a column is populated
                break

    # for each flagged column, show fill-rate + a few sample values
    for c in date_cols:
        vals = [(r.get(c) or "").strip() for r in rows]
        nonblank = [v for v in vals if v and v.lower() not in ("nan", "none", "0")]
        fill = 100 * len(nonblank) / max(len(vals), 1)
        sample = list(dict.fromkeys(nonblank))[:6]
        print(f"\n  [{c}] fill={fill:.0f}%  samples={sample}")

    # show one truncated sample row for context
    if rows:
        print("\nsample row (first, truncated):")
        for k, v in list(rows[0].items())[:12]:
            sv = (v or "")[:60].replace("\n", " ")
            print(f"   {k}: {sv}")
    return header


def main():
    print("Downloading small CSVs (reports + metadata)...")
    # reports (already cached from probe, re-resolves instantly)
    for p in REPORT_CSVS:
        fp = dl(p)
        if fp:
            peek(fp, "REPORT CSV")

    print("\n\nSearching for DICOM-header metadata CSVs...")
    found_meta = False
    for p in META_CANDIDATES:
        fp = dl(p)
        if fp:
            found_meta = True
            peek(fp, "METADATA CSV")
    if not found_meta:
        print("!! No metadata CSV found at candidate paths. "
              "List the repo's dataset/metadata/ folder on HF to get exact names.")


if __name__ == "__main__":
    main()
