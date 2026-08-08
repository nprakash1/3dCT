#!/usr/bin/env python3
"""
10_ctrate_probe.py

CT-RATE FEASIBILITY PROBE  --  metadata only, NO volume download (a few MB).

Goal: before committing to CT-RATE's 21.3 TB, answer the two questions that decide
whether it can support a temporal / interval-change project:

  Q1. How many longitudinal PAIRS are actually available?
      (patients with >=2 distinct studies -> prior->current pairs)
  Q2. What fraction of reports contain COMPARISON language?
      ("increased", "decreased", "interval", "compared to prior", "stable", ...)
      -> if this is low, the "align difference embedding to dynamic report text"
         premise is weak and CT-RATE alone may not suffice.

It downloads ONLY the radiology-report CSVs from the gated HF dataset
(ibrahimhamamci/CT-RATE), parses volume names of the form
    split_patientID_scanID_reconstructionID   (e.g. train_1_a_1.nii.gz)
groups by patient, builds a candidate prior->current pair manifest (using the
scanID letter a<b<c... as a time proxy, one reconstruction per study), and
reports the two numbers above.

Requires: `pip install huggingface_hub`, an accepted CT-RATE license, and either
`huggingface-cli login` or an HF_TOKEN env var.

Usage:
    python scripts/10_ctrate_probe.py
    HF_TOKEN=hf_xxx python scripts/10_ctrate_probe.py
    python scripts/10_ctrate_probe.py --max-print 40
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

REPO_ID = "ibrahimhamamci/CT-RATE"
OUT_DIR = "data/ctrate"

# --- comparison / temporal language (Q2). Word-boundary regex, case-insensitive. ---
COMPARISON_TERMS = [
    r"increas", r"decreas", r"enlarg", r"larger", r"smaller", r"shrunk", r"shrank",
    r"interval", r"compar", r"prior", r"previous", r"unchanged", r"stable",
    r"progress", r"regress", r"worsen", r"improv", r"new(ly)?\b", r"resolv",
    r"reduction", r"reduced", r"growth", r"grew", r"since", r"redemonstrat",
]
COMPARISON_RE = re.compile("|".join(COMPARISON_TERMS), re.IGNORECASE)


# Known small-file paths in the CT-RATE repo (confirmed). We DO NOT call
# list_repo_files() -- recursively walking CT-RATE's 50k+ file tree hangs for
# minutes. Downloading these directly is instant.
REPORT_CSVS = [
    "dataset/radiology_text_reports/train_reports.csv",
    "dataset/radiology_text_reports/validation_reports.csv",
]
# Best-effort: volumes flagged as non-chest, to exclude from pairing.
NO_CHEST_CANDIDATES = [
    "dataset/radiology_text_reports/no_chest_train.txt",
    "dataset/radiology_text_reports/no_chest_valid.txt",
    "dataset/no_chest_train.txt",
    "dataset/no_chest_valid.txt",
    "no_chest_train.txt",
    "no_chest_valid.txt",
]


def find_report_csvs():
    """Return the known report CSV paths directly (no slow repo listing)."""
    return REPORT_CSVS


def download_optional(paths):
    """Try to download each path; silently skip ones that don't exist."""
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    got = []
    for p in paths:
        try:
            fp = hf_hub_download(REPO_ID, p, repo_type="dataset", token=token,
                                 local_dir=os.path.join(OUT_DIR, "_reports"))
            got.append(fp)
        except Exception:
            pass
    return got


def load_no_chest(paths):
    """Return a set of non-chest volume basenames from downloaded no_chest lists."""
    excluded = set()
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    v = line.strip()
                    if v:
                        excluded.add(v)
        except Exception:
            pass
    return excluded



def download(paths):
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")  # or rely on cached login
    local = []
    for p in paths:
        print(f"  downloading {p} ...")
        fp = hf_hub_download(REPO_ID, p, repo_type="dataset", token=token,
                             local_dir=os.path.join(OUT_DIR, "_reports"))
        local.append(fp)
    return local


def parse_volume_name(vol):
    """'train_1_a_1.nii.gz' -> (split, patientID, scanID, recon). None if malformed."""
    base = vol.replace(".nii.gz", "").replace(".nii", "")
    parts = base.split("_")
    if len(parts) < 4:
        return None
    split, pid, scan, recon = parts[0], parts[1], parts[2], parts[3]
    return split, pid, scan, recon


def text_columns(header):
    """Pick report-text columns: anything ending in _EN or mentioning findings/impression."""
    cols = []
    for h in header:
        hl = h.lower()
        if hl.endswith("_en") or "finding" in hl or "impression" in hl or "clinical" in hl:
            cols.append(h)
    return cols


def load_reports(csv_paths):
    """Return {volume_name: report_text} merged across the given CSVs."""
    reports = {}
    vol_col_candidates = ["VolumeName", "volume_name", "Volume", "name"]
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            vcol = next((c for c in vol_col_candidates if c in header), header[0])
            tcols = text_columns(header) or [c for c in header if c != vcol]
            for row in reader:
                vol = (row.get(vcol) or "").strip()
                if not vol:
                    continue
                txt = " ".join((row.get(c) or "") for c in tcols).strip()
                reports[vol] = txt
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-print", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("CT-RATE feasibility probe  (metadata only; no volume download)")
    print("=" * 72)

    try:
        report_paths = find_report_csvs()
        print("report CSVs found in repo:")
        for p in report_paths:
            print("  -", p)
        local = download(report_paths)
    except Exception as e:
        print("\n!! Could not access CT-RATE report CSVs.")
        print("   Make sure you (1) accepted the license on the HF page and")
        print("   (2) are logged in (`huggingface-cli login`) or set HF_TOKEN.")
        print("   Underlying error:", repr(e))
        return

    reports = load_reports(local)
    print(f"\nloaded {len(reports):,} reports total")

    # ---- best-effort: exclude non-chest volumes ----
    no_chest_files = download_optional(NO_CHEST_CANDIDATES)
    excluded = load_no_chest(no_chest_files)
    if excluded:
        # normalize: no_chest lists may or may not include the .nii.gz suffix
        excl_norm = {e.replace(".nii.gz", "").replace(".nii", "") for e in excluded}
        before = len(reports)
        reports = {v: t for v, t in reports.items()
                   if v.replace(".nii.gz", "").replace(".nii", "") not in excl_norm}
        print(f"excluded {before - len(reports):,} non-chest volumes "
              f"({len(excluded):,} listed) -> {len(reports):,} chest reports")
    else:
        print("no_chest lists not found at candidate paths; proceeding without exclusion")

    # ---- group volumes by patient ----
    # patient key = split + patientID (ids may repeat across splits)
    studies = defaultdict(dict)   # patient -> {scanID: representative volume}
    malformed = 0
    for vol in reports:
        parsed = parse_volume_name(vol)
        if not parsed:
            malformed += 1
            continue
        split, pid, scan, recon = parsed

        patient = f"{split}_{pid}"
        # keep the lexicographically-smallest reconstruction as the study's representative
        cur = studies[patient].get(scan)
        if cur is None or vol < cur:
            studies[patient][scan] = vol

    n_patients = len(studies)
    multi = {p: s for p, s in studies.items() if len(s) >= 2}
    print(f"unique patients: {n_patients:,}")
    print(f"patients with >=2 distinct studies (scanIDs): {len(multi):,}  "
          f"({100*len(multi)/max(n_patients,1):.1f}%)")
    if malformed:
        print(f"(skipped {malformed} volume names that didn't parse)")

    # ---- build consecutive prior->current pairs (time proxy = scanID letter) ----
    pairs = []
    for patient, scanmap in multi.items():
        scans = sorted(scanmap.keys())           # 'a' < 'b' < 'c' ...
        for i in range(len(scans) - 1):
            prior_vol = scanmap[scans[i]]
            curr_vol = scanmap[scans[i + 1]]
            pt = reports.get(prior_vol, "")
            ct = reports.get(curr_vol, "")
            pairs.append({
                "patient": patient,
                "prior_scan": scans[i],
                "curr_scan": scans[i + 1],
                "prior_volume": prior_vol,
                "curr_volume": curr_vol,
                "prior_has_comparison": int(bool(COMPARISON_RE.search(pt))),
                "curr_has_comparison": int(bool(COMPARISON_RE.search(ct))),
                "curr_findings": ct[:400].replace("\n", " "),
            })

    # ---- Q2: comparison-language fractions ----
    all_frac = sum(1 for t in reports.values() if COMPARISON_RE.search(t)) / max(len(reports), 1)
    curr_frac = (sum(p["curr_has_comparison"] for p in pairs) / len(pairs)) if pairs else 0.0

    print("\n" + "-" * 72)
    print("RESULTS")
    print("-" * 72)
    print(f"Q1  longitudinal prior->current pairs available : {len(pairs):,}")
    print(f"    (from {len(multi):,} patients with multiple studies)")
    print(f"Q2  comparison language in ALL reports          : {100*all_frac:.1f}%")
    print(f"    comparison language in CURRENT-of-pair reports: {100*curr_frac:.1f}%")
    print("    (higher = the 'align difference to dynamic text' premise is viable)")

    # ---- write manifest + summary ----
    manifest = os.path.join(OUT_DIR, "ctrate_pairs_probe.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) if pairs else
                           ["patient", "prior_volume", "curr_volume"])
        w.writeheader()
        for p in pairs:
            w.writerow(p)
    print(f"\nwrote candidate pair manifest -> {manifest}  ({len(pairs):,} pairs)")

    summary = os.path.join(OUT_DIR, "ctrate_probe_summary.txt")
    with open(summary, "w") as f:
        f.write("CT-RATE feasibility probe\n")
        f.write(f"reports: {len(reports)}\n")
        f.write(f"patients: {n_patients}\n")
        f.write(f"patients_with_multi_studies: {len(multi)}\n")
        f.write(f"longitudinal_pairs: {len(pairs)}\n")
        f.write(f"comparison_frac_all: {all_frac:.4f}\n")
        f.write(f"comparison_frac_current_of_pair: {curr_frac:.4f}\n")
    print(f"wrote summary -> {summary}")

    # ---- storage back-of-envelope ----
    approx_gb_per_vol = 0.42  # 21.3TB / 50,188 files ~= 0.42 GB/file (1 reconstruction)
    vols_needed = len({p["prior_volume"] for p in pairs} | {p["curr_volume"] for p in pairs})
    print("\nstorage estimate (1 reconstruction per study):")
    print(f"  unique volumes to download for ALL pairs: {vols_needed:,}"
          f"  (~{vols_needed*approx_gb_per_vol/1024:.2f} TB)")
    print(f"  for a 300-pair prototype: ~{300*2*approx_gb_per_vol:.0f} GB")

    # ---- peek at a few current-of-pair reports ----
    print("\nsample current-of-pair reports (has_cmp | findings):")
    for p in pairs[: args.max_print]:
        flag = "CMP" if p["curr_has_comparison"] else "   "
        print(f"  [{flag}] {p['curr_volume']}: {p['curr_findings'][:120]}")


if __name__ == "__main__":
    main()
