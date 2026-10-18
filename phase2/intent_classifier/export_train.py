#!/usr/bin/env python3
"""Task 1, step 1b: export 2,000 training records for LLM ("scaled") intent labelling.

Sampled with seed 43 from records with a summary of at least 40 characters, excluding the 400 gold dev/test
records: 1,000 rule-labelled (stratified over the 11 classes) and 1,000 rule-Unclassified, as in export.py.
Blind (no rule label). Written as 4 batches of 500 to scaled/to_label_batch{1..4}.csv; return each as
scaled/labelled_batch{N}.csv with `your_intent` filled from judgements/allowed_labels.txt. These labels are
used for training only; dev/test gold is untouched.
Run:  python phase2/intent_classifier/export_train.py --phase1 <path to uatie-edm-phase1>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_root, sha256  # noqa: E402
from export import CLASSES, MIN_CHARS, UNK  # noqa: E402

SEED, N_RULE, N_UNK, BATCH = 43, 1000, 1000, 500
OUT = TASK / "scaled"


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_csv(phase1_root() / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["summary"] = meta["summary"].fillna("").astype(str)
    gold = pd.read_csv(TASK / "splits" / "gold_key.csv")
    ok = meta[(meta.summary.str.len() >= MIN_CHARS) & ~meta.faiss_idx.isin(gold.faiss_idx)]
    per = {c: N_RULE // len(CLASSES) for c in CLASSES}
    for c in CLASSES[: N_RULE - sum(per.values())]:
        per[c] += 1
    picks = []
    for c in CLASSES + [UNK]:
        pool = ok[ok.intent == c].sort_values("faiss_idx")
        n = min(N_UNK if c == UNK else per[c], len(pool))
        sel = pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))].copy()
        sel["stratum"] = c
        picks.append(sel)
    s = pd.concat(picks, ignore_index=True)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s["label_id"] = [f"T{i:04d}" for i in range(1, len(s) + 1)]
    assert not s.faiss_idx.isin(gold.faiss_idx).any() and s.record_id.is_unique
    OUT.mkdir(exist_ok=True)
    s[["label_id", "record_id", "faiss_idx", "stratum"]].to_csv(OUT / "train_key.csv", index=False)
    files = []
    for b in range(0, len(s), BATCH):
        p = OUT / f"to_label_batch{b // BATCH + 1}.csv"
        part = s.iloc[b:b + BATCH]
        pd.DataFrame({"label_id": part.label_id, "record_id": part.record_id, "source": part.source,
                      "summary": part.summary, "your_intent": "", "notes": ""}).to_csv(p, index=False)
        files.append(p)
    R = {"seed": SEED, "records": len(s), "batches": len(files), "excluded_gold": int(len(gold)),
         "per_stratum": s.stratum.value_counts().reindex(CLASSES + [UNK]).astype(int).to_dict(),
         "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files + [OUT / "train_key.csv"]}}
    (OUT / "export_results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["records", "batches", "per_stratum"]}))


if __name__ == "__main__":
    main()
