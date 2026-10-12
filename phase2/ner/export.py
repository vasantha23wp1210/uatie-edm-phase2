#!/usr/bin/env python3
"""Task 2, step 1: export 260 rule-pre-annotated ASRS sentences for manual correction, then STOP.

Sentences come from a seed-42 sample of ASRS summaries (asrs_faiss_meta.csv), split with the spaCy sentencizer.
200 sentences with at least one rule entity (round-robin over entity types so rare types are represented) and
60 random sentences with no rule entity (to measure what the rules miss). 200 train / 60 test is assigned now
(seeded, stratified by has-entity) and kept in splits/ner_split.csv.

Annotation format: entities written inline as [span text](LABEL). Removing the markup must give back `text` exactly.

Run:  python phase2/ner/export.py --phase1 <path to uatie-edm-phase1>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK))
from phase1 import phase1_info, phase1_root, sha256  # noqa: E402
from rules import LABELS, build_nlp  # noqa: E402

REPO = phase1_root()
SEED = 42
N_DOCS = 20000
N_ENT, N_NONE, N_TEST = 200, 60, 60
MIN_C, MAX_C = 30, 300


def markup(text, ents):
    out, last = [], 0
    for s, e, lab in ents:
        out += [text[last:s], f"[{text[s:e]}]({lab})"]
        last = e
    return "".join(out + [text[last:]])


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_csv(REPO / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False,
                       usecols=["faiss_idx", "source", "summary", "record_id"])
    asrs = meta[(meta.source == "ASRS") & meta.summary.notna()].sort_values("faiss_idx")
    docs = asrs.iloc[np.sort(rng.choice(len(asrs), N_DOCS, replace=False))]
    nlp = build_nlp()
    rows, seen = [], set()
    for rid, doc in zip(docs.record_id, nlp.pipe(docs.summary.astype(str), batch_size=500)):
        for s in doc.sents:
            t = s.text.strip()
            if not (MIN_C <= len(t) <= MAX_C) or t.lower() in seen:
                continue
            seen.add(t.lower())
            off = s.start_char + (len(s.text) - len(s.text.lstrip()))
            ents = [(e.start_char - off, e.end_char - off, e.label_) for e in s.ents]
            rows.append({"record_id": rid, "text": t, "ents": ents})
    pool = pd.DataFrame(rows)
    pool["types"] = pool.ents.map(lambda es: sorted({x[2] for x in es}))
    allsum = meta[meta.source == "ASRS"].summary.fillna("").astype(str).str.lower()
    deid = {"asrs_summaries": int(len(allsum)),
            "mention_word_callsign": int(allsum.str.contains(r"\bcall ?signs?\b", regex=True).sum()),
            "match_callsign_rule_regex": int(allsum.str.contains(r"\b(?:acr|cpr|sma|mlt|ltt|acft|plt|spi) [xyz]\d{0,4}\b", regex=True).sum())}
    support_pool = Counter(lab for es in pool.ents for *_, lab in es)

    with_ent = pool[pool.types.map(len) > 0]
    chosen, by_type = [], {lab: list(rng.permutation(with_ent.index[with_ent.types.map(lambda t, l=lab: l in t)])) for lab in LABELS}
    taken = set()
    while len(chosen) < N_ENT and any(by_type.values()):
        for lab in LABELS:
            while by_type[lab] and by_type[lab][0] in taken:
                by_type[lab].pop(0)
            if by_type[lab] and len(chosen) < N_ENT:
                i = by_type[lab].pop(0)
                taken.add(i)
                chosen.append(i)
    none = pool[pool.types.map(len) == 0]
    chosen += list(none.index[np.sort(rng.choice(len(none), N_NONE, replace=False))])
    sel = pool.loc[chosen].copy()
    sel["has_rule_entity"] = sel.types.map(len) > 0
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel["sent_id"] = [f"S{i:03d}" for i in range(1, len(sel) + 1)]
    test = set()
    for flag, n in ((True, round(N_TEST * N_ENT / (N_ENT + N_NONE))), (False, N_TEST - round(N_TEST * N_ENT / (N_ENT + N_NONE)))):
        ids = sel.sent_id[sel.has_rule_entity == flag].tolist()
        test |= set(rng.choice(ids, n, replace=False))
    sel["split"] = np.where(sel.sent_id.isin(test), "test", "train")

    for d in ("splits", "judgements"):
        (TASK / d).mkdir(exist_ok=True)
    sel["pre_annotated"] = [markup(t, e) for t, e in zip(sel.text, sel.ents)]
    sel[["sent_id", "record_id", "split", "has_rule_entity"]].to_csv(TASK / "splits" / "ner_split.csv", index=False)
    with open(TASK / "splits" / "rule_preannotation.jsonl", "w", encoding="utf-8") as f:
        for r in sel.itertuples():
            f.write(json.dumps({"sent_id": r.sent_id, "text": r.text, "entities": r.ents}) + "\n")
    pd.DataFrame({"sent_id": sel.sent_id, "text": sel.text, "pre_annotated": sel.pre_annotated,
                  "corrected": sel.pre_annotated, "notes": ""}).to_csv(TASK / "judgements" / "ner_to_correct.csv", index=False)

    sup_sel = Counter(lab for es in sel.ents for *_, lab in es)
    R = {"task": "ner/export", "status": "STOPPED: waiting for judgements/ner_corrected.csv", "seed": SEED,
         "spacy": spacy.__version__, "summaries_sampled": N_DOCS, "candidate_sentences": int(len(pool)),
         "rule_entity_support_in_candidates": {l: int(support_pool.get(l, 0)) for l in LABELS},
         "rule_entity_support_in_export": {l: int(sup_sel.get(l, 0)) for l in LABELS},
         "callsign_deidentification_check": deid,
         "export": {"sentences": int(len(sel)), "with_rule_entity": int(sel.has_rule_entity.sum()),
                    "train": int((sel.split == "train").sum()), "test": int((sel.split == "test").sum())},
         "phase1": phase1_info(REPO),
         "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                           [TASK / "splits" / "ner_split.csv", TASK / "judgements" / "ner_to_correct.csv"]}}
    (TASK / "export_results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Task 2: Named entity recognition (STOPPED, waiting for corrections)", "",
         "Numbers come from `export_results.json`, written by `export.py`. No model has been trained yet.", "",
         "## Export",
         f"- {N_DOCS:,} ASRS summaries sampled (seed {SEED}), split into {len(pool):,} candidate sentences ({MIN_C}–{MAX_C} chars, deduplicated).",
         f"- Exported {len(sel)}: {int(sel.has_rule_entity.sum())} with at least one rule entity (round-robin over types), "
         f"{int((~sel.has_rule_entity).sum())} with none. Split: {R['export']['train']} train / {R['export']['test']} test.",
         "- Rules: spaCy `EntityRuler` patterns in `rules.py`.", "",
         "## Rule-entity support (before your corrections)", "| Entity | Candidate sentences | Export |", "|---|---|---|"]
    L += [f"| {l} | {R['rule_entity_support_in_candidates'][l]} | {R['rule_entity_support_in_export'][l]} |" for l in LABELS]
    L += ["", "ASRS narratives are de-identified (airports often `ZZZ`, callsigns `ACR X`), so CALLSIGN and AIRPORT are expected "
          "to be rare or templated. Across all "
          f"{deid['asrs_summaries']:,} ASRS summaries, {deid['mention_word_callsign']} mention the word callsign/call sign but only "
          f"{deid['match_callsign_rule_regex']} contain a templated callsign (`acr x`-style); real callsigns are removed by de-identification. Support per entity will be reported after correction; any entity with fewer than 10 test "
          "mentions will be reported as too rare to evaluate.", "",
          "## What you need to do",
          "Edit the `corrected` column of `judgements/ner_to_correct.csv`: fix, add or remove `[span](LABEL)` markup using only "
          f"{', '.join(LABELS)}. Do not change any other text. Return it as `judgements/ner_corrected.csv`."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["candidate_sentences", "rule_entity_support_in_candidates", "rule_entity_support_in_export", "export"]}, indent=1))
    for t in sel.pre_annotated[sel.has_rule_entity].head(12):
        print("  ", t)


if __name__ == "__main__":
    main()
