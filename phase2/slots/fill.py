#!/usr/bin/env python3
"""Task 6: fill bracketed slots in matched CAA phraseology from NER entities. Never guesses.

For each of the 100 queries: pick the CAA phrase (TF-IDF, query vs Phraseology + Circumstances + Section_Title),
run the Task 2 spaCy NER model on the query text plus the top-1 incident summary from the dev-selected retrieval
system, and fill a slot only when its type is mapped AND exactly one distinct entity value of that type was found.
Otherwise the slot stays bracketed and is flagged (unmapped / no_entity / ambiguous).
Run:  python phase2/slots/fill.py [--phase1 PATH]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
P2 = ROOT / "phase2"
sys.path.insert(0, str(ROOT))
from phase1 import phase1_root, sha256  # noqa: E402

SEED, N_EXAMPLES = 42, 15
SLOT = re.compile(r"(\S+\s+)?\(([^()]*)\)")
SLOT_TYPES = {"level": ["FLIGHT_LEVEL", "ALTITUDE"], "altitude": ["ALTITUDE"], "frequency": ["FREQUENCY"],
              "aircraft call sign": ["CALLSIGN"], "aircraft type": ["AIRCRAFT_TYPE"],
              "aerodrome": ["AIRPORT"], "name of aerodrome": ["AIRPORT"], "airport": ["AIRPORT"]}
CONTEXT_TYPES = {("number", "RUNWAY"): ["RUNWAY"], ("three digits", "HEADING"): ["HEADING"]}


def is_slot(name):
    """Parentheses holding alternatives ("or LEFT") or a plural suffix ("S") are wording, not variables."""
    n = name.strip().lower()
    return bool(n) and n != "s" and not n.startswith("or ")


def slot_type(name, prev):
    name, prev = name.strip().lower(), (prev or "").strip().strip("[(").upper()
    return CONTEXT_TYPES.get((name, prev)) or SLOT_TYPES.get(name)


def fill(phrase, ents):
    slots, out, last = [], [], 0
    for m in SLOT.finditer(phrase):
        name = m.group(2)
        if not is_slot(name):
            continue
        types = slot_type(name, m.group(1))
        status, value = "unmapped", None
        if types:
            for t in types:
                vals = sorted({v for v, lab in ents if lab == t})
                if len(vals) == 1:
                    status, value = "filled", vals[0]
                    break
                if len(vals) > 1:
                    status = f"ambiguous ({t}: {len(vals)} values)"
                    break
            else:
                status = "no_entity"
        s, e = m.span(2)
        out.append(phrase[last:s - 1] + (value if value else f"({name})"))
        last = e + 1
        slots.append({"slot": name, "type": "|".join(types) if types else "", "status": status, "value": value})
    out.append(phrase[last:])
    return "".join(out), slots


def main():
    import spacy
    p1 = phase1_root()
    best = json.loads((P2 / "retrieval" / "eval" / "results.json").read_text())
    system = best.get("best_system_chosen_on_dev_ndcg10")
    assert system, "retrieval/eval/results.json has no dev-selected system"
    q = pd.concat([pd.read_csv(P2 / "gold_eval_rebuild" / "splits" / f"{s}_queries.csv").assign(split=s)
                   for s in ["dev", "test"]], ignore_index=True)
    runs = pd.read_csv(P2 / "retrieval" / "runs" / "runs_top10.csv")
    top1 = runs[(runs.system == system) & (runs["rank"] == 1)].set_index("query_id").incident_id
    meta = pd.read_csv(p1 / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["record_id", "summary"])
    summ = meta.drop_duplicates("record_id").set_index("record_id").summary.fillna("").astype(str)
    missing = set(top1) - set(summ.index)
    assert not missing, f"top-1 ids missing from asrs_faiss_meta.csv: {sorted(missing)[:5]}"
    cm = pd.read_csv(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv")
    doc = (cm.Phraseology.astype(str) + " " + cm.Circumstances.fillna("").astype(str) + " " +
           cm.Section_Title.fillna("").astype(str))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(list(doc) + list(q.query_text))
    sim = cosine_similarity(vec.transform(q.query_text), vec.transform(doc))
    nlp = spacy.load(P2 / "ner" / "out" / "model")

    rows = []
    for i, r in q.iterrows():
        j = int(sim[i].argmax())
        phrase = str(cm.Phraseology.iloc[j])
        text = r.query_text + "\n" + summ.get(top1.get(r.query_id, ""), "")
        ents = [(e.text, e.label_) for e in nlp(text).ents]
        filled, slots = fill(phrase, ents)
        st = Counter(s["status"].split(" ")[0] for s in slots)
        status = "no_slots" if not slots else ("complete" if st["filled"] == len(slots) else "incomplete")
        rows.append({"query_id": r.query_id, "split": r.split, "query_text": r.query_text,
                     "top1_incident": top1.get(r.query_id, ""), "phrase_section": cm.Section_Number.iloc[j],
                     "phrase_match_score": round(float(sim[i, j]), 4), "phraseology": phrase, "filled": filled,
                     "status": status, "n_slots": len(slots), "n_filled": st["filled"],
                     "entities": "; ".join(f"{t}|{l}" for t, l in ents), "slots": json.dumps(slots)})
    d = pd.DataFrame(rows)
    (TASK / "out").mkdir(exist_ok=True)
    d.to_csv(TASK / "out" / "slot_fills.csv", index=False)

    all_slots = [(m.group(2).strip().lower(), slot_type(m.group(2), m.group(1)))
                 for p in cm.Phraseology.astype(str) for m in SLOT.finditer(p) if is_slot(m.group(2))]
    per = Counter()
    for s in d.slots:
        for x in json.loads(s):
            per[(x["type"] or "unmapped", x["status"].split(" ")[0])] += 1
    R = {"retrieval_system": system, "queries": len(d), "status_counts": d.status.value_counts().to_dict(),
         "complete_pct_of_queries_with_slots": round(100 * (d.status == "complete").sum() /
                                                     max(1, (d.status != "no_slots").sum()), 2),
         "slot_occurrences_in_selected_phrases": int(d.n_slots.sum()), "slots_filled": int(d.n_filled.sum()),
         "slot_outcomes_by_type": {f"{t}:{s}": n for (t, s), n in sorted(per.items())},
         "crossmap_slot_occurrences": len(all_slots),
         "crossmap_slot_occurrences_with_mapped_type": sum(1 for _, t in all_slots if t),
         "crossmap_distinct_slot_names": len({n for n, _ in all_slots}),
         "inputs_sha256": {"phase2/phraseology/out/CAA_ICAO_crossmap_v2.csv":
                           sha256(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv"),
                           "phase2/retrieval/runs/runs_top10.csv": sha256(P2 / "retrieval" / "runs" / "runs_top10.csv")},
         "output_sha256": sha256(TASK / "out" / "slot_fills.csv")}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")

    ex = d[d.n_slots > 0].sample(min(N_EXAMPLES, int((d.n_slots > 0).sum())), random_state=SEED).sort_values("query_id")
    L = ["# Task 6: slot filling", "", "## Method", "",
         "- Phrase per query: TF-IDF (1-2 grams) over CAA Phraseology + Circumstances + Section_Title of "
         "`phase2/phraseology/out/CAA_ICAO_crossmap_v2.csv`; top-1 phrase.",
         f"- Entities: Task 2 spaCy model (`phase2/ner/out/model`) on the query text plus the top-1 incident summary "
         f"from `{system}`, the retrieval system chosen on dev in Task 3.",
         "- A slot is filled only if its type is mapped and exactly one distinct value of that type was found. "
         "Unmapped, missing and ambiguous slots stay bracketed and are flagged in `out/slot_fills.csv`.",
         "- Parentheses holding alternatives (`(or LEFT)`) or a plural suffix (`(S)`) are wording, not slots.",
         "- Slot-type map: " + "; ".join(f"`({k})` -> {'/'.join(v)}" for k, v in SLOT_TYPES.items()) +
         "; `RUNWAY (number)` -> RUNWAY; `HEADING (three digits)` -> HEADING. Every other slot name is unmapped.",
         "", "## Numbers", "", "| Status | Queries |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in R["status_counts"].items()]
    L += ["", f"Complete among queries whose phrase has slots: {R['complete_pct_of_queries_with_slots']}%. "
          f"Slots filled: {R['slots_filled']} of {R['slot_occurrences_in_selected_phrases']}.",
          "", f"Across the whole crossmap, {R['crossmap_slot_occurrences_with_mapped_type']} of "
          f"{R['crossmap_slot_occurrences']} slot occurrences ({R['crossmap_distinct_slot_names']} distinct names) "
          "have a mapped entity type; the rest (time, significant point, distance, reason, ...) have no NER label.",
          "", "| Slot type : outcome | Count |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in R["slot_outcomes_by_type"].items()]
    L += ["", "## Failures and limits", "",
          "- NER labels are provisional (LLM-judged (Devin), pending user review); so are these fills.",
          "- CALLSIGN slots can never fill from ASRS text: ASRS de-identifies callsigns (Task 2).",
          "- Phrase choice is a lexical TF-IDF match, not judged for correctness here.",
          "- Entities from the top-1 incident describe that incident, not necessarily the user's situation.",
          "", f"## {len(ex)} examples (seed {SEED})", "",
          "| Query | Phrase | Filled | Status |", "|---|---|---|---|"]
    L += [f"| {r.query_id} {r.query_text} | {r.phraseology} | {r.filled} | {r.status} |".replace("\n", " ")
          for r in ex.itertuples()]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in list(R)[:9]}, indent=1))


if __name__ == "__main__":
    main()
