#!/usr/bin/env python3
"""
19_audit_labels.py

EVIDENCE AUDIT: quantify how trustworthy the silver labels are, using only data we
already have. Answers three questions that decide whether a re-label is worth it.

Q1. CONTRADICTIONS. The pipeline let the structured presence-diff win unchallenged:
    the LLM was only asked about findings marked present-in-BOTH, so a finding marked
    `new`/`resolved` (0->1 / 1->0) never got a chance to be overridden even if the
    current report explicitly says it increased/decreased. How often does the report
    actually contain a comparison sentence naming a finding we labeled new/resolved?
    -> measures how often the bug fires.

Q2. EXPLICIT vs DEFAULTED `stable`. MedGemma was told "stable = explicitly unchanged
    OR the report gives no comparison". So `stable` conflates two very different
    things. Using the saved `evidence` quote, split them.

Q3. MEASUREMENTS. Do the reports contain mm measurements we could mine for a
    quantitative change signal (objective, immune to the mention artifact)?

Input : data/ctrate/labels_final.jsonl
Output: console report + data/ctrate/label_audit.csv (per-label tier/confidence flags)
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

IN = "data/ctrate/labels_final.jsonl"
OUT = "data/ctrate/label_audit.csv"

# Words that signal a genuine prior-vs-current comparison was made by the radiologist.
CMP_WORDS = [
    "increase", "increased", "increasing", "decrease", "decreased", "decreasing",
    "larger", "smaller", "grew", "growth", "progress", "progressed", "progression",
    "regress", "regressed", "regression", "unchanged", "stable", "stability",
    "previous", "previously", "prior", "compared", "comparison", "interval",
    "new", "newly", "resolved", "resolution", "disappeared", "no longer",
    "persist", "persists", "persistent", "reduced", "enlarged", "expanded",
]
CMP_RE = re.compile(r"\b(" + "|".join(CMP_WORDS) + r")\b", re.I)

# direction-bearing words (which way did it go)
UP_RE = re.compile(r"\b(increase\w*|larger|grew|growth|progress\w*|enlarged|expanded|new|newly)\b", re.I)
DOWN_RE = re.compile(r"\b(decrease\w*|smaller|regress\w*|reduced|resolved|resolution|disappeared|no longer)\b", re.I)
SAME_RE = re.compile(r"\b(unchanged|stable|stability|persist\w*)\b", re.I)

# a measurement like "15 mm", "9mm", "1.5 cm"
MEAS_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mm|cm)\b", re.I)

# findings that are physiologically irreversible -> new/resolved/improved are suspect
IRREVERSIBLE = {
    "Emphysema", "Pulmonary fibrotic sequela", "Bronchiectasis",
    "Arterial wall calcification", "Coronary artery wall calcification",
}

# map finding name -> keywords to search for in report sentences
FIND_KEYS = {
    "Medical material": ["catheter", "stent", "tube", "clip", "port", "device", "material", "drain"],
    "Arterial wall calcification": ["calcific", "calcification", "atheroma", "aort"],
    "Cardiomegaly": ["cardiomegaly", "heart size", "heart contour", "cardiac size"],
    "Pericardial effusion": ["pericardial"],
    "Coronary artery wall calcification": ["coronary"],
    "Hiatal hernia": ["hernia", "hiatal"],
    "Lymphadenopathy": ["lymph", "node", "adenopathy"],
    "Emphysema": ["emphysema", "emphysematous"],
    "Atelectasis": ["atelecta"],
    "Lung nodule": ["nodule", "nodular"],
    "Lung opacity": ["opacity", "opacities", "ground-glass", "ground glass", "infiltrat"],
    "Pulmonary fibrotic sequela": ["fibro", "sequela", "scar"],
    "Pleural effusion": ["pleural effusion", "pleural fluid", "effusion"],
    "Mosaic attenuation pattern": ["mosaic", "attenuation pattern"],
    "Peribronchial thickening": ["peribronchial", "bronchial wall"],
    "Consolidation": ["consolidat"],
    "Bronchiectasis": ["bronchiecta"],
    "Interlobular septal thickening": ["septal", "interlobular"],
}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if len(s.strip()) > 3]


def finding_cmp_sentences(finding, curr_text):
    """Sentences in the CURRENT report that mention this finding AND contain comparison language."""
    keys = FIND_KEYS.get(finding, [finding.lower()])
    hits = []
    for s in sentences(curr_text):
        sl = s.lower()
        if any(k in sl for k in keys) and CMP_RE.search(s):
            hits.append(s)
    return hits


def direction_of(text):
    """Which way does this sentence point? up/down/same/None."""
    if SAME_RE.search(text):
        return "same"
    up, down = bool(UP_RE.search(text)), bool(DOWN_RE.search(text))
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


EXPECTED_DIR = {"new": "up", "worse": "up", "stable": "same",
                "improved": "down", "resolved": "down"}


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"missing {IN} -- run scripts/17 first")
    recs = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    print(f"pairs: {len(recs):,}\n")

    rows = []
    # Q1
    q1 = Counter()            # for structured labels: has contradicting / agreeing / no cmp sentence
    q1_by_finding = defaultdict(Counter)
    # Q2
    q2 = Counter()            # stable: explicit vs defaulted
    # Q3
    n_pairs_with_meas = 0
    n_pairs_both_meas = 0
    meas_sentences = 0

    for r in recs:
        curr = (r.get("curr_findings", "") + " " + r.get("curr_impression", "")).strip()
        prior = (r.get("prior_findings", "") + " " + r.get("prior_impression", "")).strip()

        cm = len(MEAS_RE.findall(curr))
        pm = len(MEAS_RE.findall(prior))
        meas_sentences += cm
        if cm:
            n_pairs_with_meas += 1
        if cm and pm:
            n_pairs_both_meas += 1

        for fd in r["findings"]:
            f, ch, src = fd["finding"], fd["change"], fd.get("source", "")
            ev = (fd.get("evidence", "") or "")
            cmp_hits = finding_cmp_sentences(f, curr)
            dirs = {direction_of(s) for s in cmp_hits} - {None}
            expected = EXPECTED_DIR[ch]

            status = "no_comparison_sentence"
            if cmp_hits:
                if expected in dirs:
                    status = "agrees"
                elif dirs:
                    status = "CONTRADICTS"
                else:
                    status = "cmp_sentence_unclear"

            # Q1: only meaningful for structured (new/resolved) labels
            if src == "structured":
                q1[status] += 1
                q1_by_finding[f][status] += 1

            # Q2: stable explicit vs defaulted
            if ch == "stable":
                if src == "llm_default":
                    q2["defaulted (LLM never rated)"] += 1
                elif SAME_RE.search(ev):
                    q2["explicit ('unchanged'/'stable' in evidence)"] += 1
                elif ev.strip():
                    q2["evidence present but no stable-word"] += 1
                else:
                    q2["no evidence quote"] += 1

            low_conf = (f in IRREVERSIBLE and ch in ("new", "resolved", "improved"))
            rows.append({
                "patient": r["patient"], "finding": f, "change": ch, "source": src,
                "direction": {"new": "worsened", "worse": "worsened", "stable": "stable",
                              "improved": "improved", "resolved": "improved"}[ch],
                "text_status": status,
                "n_cmp_sentences": len(cmp_hits),
                "label_confidence": "low" if low_conf else "high",
                "evidence": ev[:120],
            })

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot_struct = sum(q1.values())
    print("=" * 78)
    print("Q1. Do explicit report comparisons CONTRADICT the structured new/resolved labels?")
    print("=" * 78)
    for k in ["CONTRADICTS", "agrees", "cmp_sentence_unclear", "no_comparison_sentence"]:
        print(f"  {k:<26} {q1[k]:>7,}  ({100*q1[k]/max(tot_struct,1):.1f}%)")
    print(f"  {'TOTAL structured labels':<26} {tot_struct:>7,}")
    print(f"\n  -> the bug fires on {100*q1['CONTRADICTS']/max(tot_struct,1):.1f}% of structured labels")
    print("     (report explicitly states a direction opposite to the presence-diff)")

    print("\n  worst findings by contradiction rate:")
    ranked = sorted(q1_by_finding.items(),
                    key=lambda kv: -(kv[1]["CONTRADICTS"] / max(sum(kv[1].values()), 1)))
    for f, c in ranked[:8]:
        t = sum(c.values())
        print(f"    {f:<34} {c['CONTRADICTS']:>4}/{t:<5} ({100*c['CONTRADICTS']/max(t,1):.0f}%)")

    print("\n" + "=" * 78)
    print("Q2. Is `stable` explicitly stated, or just 'no comparison found'?")
    print("=" * 78)
    t2 = sum(q2.values())
    for k, v in q2.most_common():
        print(f"  {k:<44} {v:>7,}  ({100*v/max(t2,1):.1f}%)")
    print(f"  {'TOTAL stable labels':<44} {t2:>7,}")

    print("\n" + "=" * 78)
    print("Q3. Are mm/cm measurements available to mine for quantitative change?")
    print("=" * 78)
    print(f"  pairs whose CURRENT report has >=1 measurement : {n_pairs_with_meas:,} "
          f"({100*n_pairs_with_meas/len(recs):.1f}%)")
    print(f"  pairs where BOTH reports have measurements     : {n_pairs_both_meas:,} "
          f"({100*n_pairs_both_meas/len(recs):.1f}%)  <- minable for size ratios")
    print(f"  total measurement mentions (current reports)   : {meas_sentences:,}")

    lc = Counter(r["label_confidence"] for r in rows)
    print("\n" + "=" * 78)
    print("label_confidence (low = new/resolved/improved on irreversible findings)")
    print("=" * 78)
    for k, v in lc.most_common():
        print(f"  {k:<6} {v:>7,}  ({100*v/len(rows):.1f}%)")
    print(f"\nwrote per-label flags -> {OUT}")


if __name__ == "__main__":
    main()
