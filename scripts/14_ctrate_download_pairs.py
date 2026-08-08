#!/usr/bin/env python3
"""
14_ctrate_download_pairs.py

Selective volume downloader: pull ONLY the NIfTI volumes needed for a chosen subset
of prior->current pairs (a prototype), not the whole 21.3 TB dataset.

Given the enriched manifest, it downloads each pair's prior_volume + curr_volume from
CT-RATE, trying the known nested folder layouts (v2 "fixed" first, then v1).

Usage
-----
  # see what WOULD be downloaded + size estimate, no download
  python scripts/14_ctrate_download_pairs.py --limit 100 --dry-run

  # actually download a 100-pair prototype (~85 GB) into data/ctrate/volumes/
  python scripts/14_ctrate_download_pairs.py --limit 100

Options let you bias toward informative pairs (e.g. longer intervals).
"""
import argparse
import csv
import os

REPO_ID = "ibrahimhamamci/CT-RATE"
MANIFEST = "data/ctrate/ctrate_pairs_enriched.csv"
VOL_DIR = "data/ctrate/volumes"


def remote_candidates(vol):
    """
    Build candidate HF paths for a volume like 'train_3_b_1.nii.gz'.
    CT-RATE nests as  <split_folder>/<patient>/<patient_scan>/<file>.
    We try v2 'fixed' folders first, then v1.
    """
    base = vol.replace(".nii.gz", "").replace(".nii", "")
    parts = base.split("_")
    split, pid, scan = parts[0], parts[1], parts[2]
    patient = f"{split}_{pid}"
    scan_folder = f"{split}_{pid}_{scan}"
    split_folders = ([f"{split}_fixed", split] if split in ("train", "valid")
                     else [split])
    cands = []
    for sf in split_folders:
        cands.append(f"dataset/{sf}/{patient}/{scan_folder}/{vol}")
    return cands


def download_volume(vol, token, out_dir=None):
    from huggingface_hub import hf_hub_download
    for path in remote_candidates(vol):
        try:
            return hf_hub_download(REPO_ID, path, repo_type="dataset",
                                   token=token, local_dir=(out_dir or VOL_DIR))
        except Exception:
            continue
    return None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--delta-min", type=int, default=0,
                    help="only pairs with delta_days >= this")
    ap.add_argument("--delta-max", type=int, default=10**9)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=VOL_DIR,
                    help="where volumes land: a big disk, or a gcsfuse-mounted "
                         "GCS bucket (e.g. /gcs/ctrate/volumes)")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN")


    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if args.delta_min <= int(r["delta_days"]) <= args.delta_max]
    rows = rows[: args.limit]

    vols = []
    for r in rows:
        vols += [r["prior_volume"], r["curr_volume"]]
    vols = list(dict.fromkeys(vols))  # dedupe, keep order

    print(f"{len(rows)} pairs -> {len(vols)} unique volumes")
    print(f"estimated size: ~{len(vols) * 0.42:.0f} GB (@~0.42 GB/volume)")

    if args.dry_run:
        for v in vols[:10]:
            print("  would fetch:", remote_candidates(v)[0], "(+ fallback)")
        print(f"[dry-run] {len(vols)} volumes NOT downloaded.")
        return

    ok, miss = 0, []
    for i, v in enumerate(vols):
        fp = download_volume(v, token, out_dir)

        if fp:
            ok += 1
        else:
            miss.append(v)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(vols)}  ok={ok} missing={len(miss)}")

    print(f"\ndone. downloaded={ok}  missing={len(miss)} -> {out_dir}")

    if miss:
        print("  missing (path layout may differ):", miss[:10])


if __name__ == "__main__":
    main()
