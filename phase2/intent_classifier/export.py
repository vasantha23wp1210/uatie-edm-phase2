#!/usr/bin/env python3
"""Task 1, step 1: export 400 records for manual intent labelling, then STOP.

200 records the Phase 1 keyword rules labelled (stratified over the 11 intent classes) and 200 records the rules
left Unclassified, sampled with seed 42 from data/processed/asrs_faiss_meta.csv (same rows as the FAISS index).
The export is blind: it does not show the rule label. Each record is assigned to gold dev or gold test (50/50 within
each stratum); the confidence threshold will be picked on dev only.

Run:  python phase2/intent_classifier/export.py --phase1 <path to uatie-edm-phase1>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root, sha256  # noqa: E402

REPO = phase1_root()
SEED = 42
N_RULE, N_UNK = 200, 200
MIN_CHARS = 40
CLASSES = ["ATC_Operational_Error", "Emergency_Event", "Runway_Incursion", "Ground_Incident", "Communication_Failure",
           "Airspace_Violation", "IMC_Spatial_Disorientation", "Weather_Deviation", "Crew_Resource_Management",
           "Maintenance_Error", "Procedural_Non_Compliance"]
UNK = "Unclassified"


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_csv(REPO / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["summary"] = meta["summary"].fillna("").astype(str)
    ok = meta[meta.summary.str.len() >= MIN_CHARS]
    assert set(meta.intent.dropna()) <= set(CLASSES) | {UNK}, set(meta.intent.dropna()) - set(CLASSES)

    per = {c: N_RULE // len(CLASSES) for c in CLASSES}
    for c in CLASSES[: N_RULE - sum(per.values())]:
        per[c] += 1
    picks = []
    for c in CLASSES + [UNK]:
        pool = ok[ok.intent == c].sort_values("faiss_idx")
        n = N_UNK if c == UNK else per[c]
        sel = pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))].copy()
        sel["stratum"] = c
        sel["gold_split"] = np.where(rng.permutation(n) < n // 2, "dev", "test")
        picks.append(sel)
    gold = pd.concat(picks, ignore_index=True)
    gold = gold.iloc[rng.permutation(len(gold))].reset_index(drop=True)
    gold["label_id"] = [f"I{i:03d}" for i in range(1, len(gold) + 1)]

    (TASK / "splits").mkdir(exist_ok=True)
    (TASK / "judgements").mkdir(exist_ok=True)
    key = gold[["label_id", "record_id", "faiss_idx", "source", "stratum", "gold_split"]]
    key.to_csv(TASK / "splits" / "gold_key.csv", index=False)
    blind = pd.DataFrame({"label_id": gold.label_id, "record_id": gold.record_id, "source": gold.source,
                          "summary": gold.summary, "your_intent": "", "notes": ""})
    blind.to_csv(TASK / "judgements" / "intent_to_label.csv", index=False)
    (TASK / "judgements" / "allowed_labels.txt").write_text("\n".join(CLASSES + [UNK]) + "\n", encoding="utf-8")

    R = {"task": "intent_classifier/export", "status": "STOPPED: waiting for judgements/intent_labelled.csv",
         "seed": SEED, "min_summary_chars": MIN_CHARS, "records_total": int(len(meta)),
         "records_eligible": int(len(ok)),
         "rule_unclassified_pct_all_records": round(float((meta.intent == UNK).mean() * 100), 2),
         "sample_per_stratum": gold.stratum.value_counts().reindex(CLASSES + [UNK]).astype(int).to_dict(),
         "gold_split": gold.gold_split.value_counts().astype(int).to_dict(),
         "phase1": phase1_info(REPO),
         "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                           [TASK / "splits" / "gold_key.csv", TASK / "judgements" / "intent_to_label.csv"]}}
    (TASK / "export_results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Task 1: Intent classifier (STOPPED, waiting for labels)", "",
         "Numbers come from `export_results.json`, written by `export.py`. No classifier has been trained yet.", "",
         "## Phase 1 starting point",
         f"- Keyword rules (`asrs_nlp_pipeline.classify_intent`, unchanged) leave {R['rule_unclassified_pct_all_records']}% of "
         f"{R['records_total']:,} records Unclassified (target: < 15%).", "",
         "## Gold sample",
         f"- {N_RULE} rule-labelled records stratified over the 11 classes and {N_UNK} rule-Unclassified records, seed {SEED}, "
         f"from {R['records_eligible']:,} records with a summary of at least {MIN_CHARS} characters.",
         f"- Gold dev/test split within each stratum: {R['gold_split']}. The confidence threshold is picked on gold dev only; "
         "accuracy is reported on gold test, and on all 400 where the plan asks for it.",
         "- These 400 records are excluded from training.", "",
         "| Stratum (rule label, hidden from you) | Records |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in R["sample_per_stratum"].items()]
    L += ["", "## What you need to do",
          "Fill `your_intent` in `judgements/intent_to_label.csv` with one value from `judgements/allowed_labels.txt` "
          "(use `Unclassified` when none of the 11 fits). Return it as `judgements/intent_labelled.csv`. "
          "Do not change other columns. The file does not show the rule label."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["records_eligible", "rule_unclassified_pct_all_records", "sample_per_stratum", "gold_split"]}, indent=1))


if __name__ == "__main__":
    main()
