#!/usr/bin/env python3
"""Task 6: fill bracketed slots in matched CAA phraseology from NER entities and fixed regexes. Never guesses.

For each of the 100 queries: pick the CAA phrase (TF-IDF, query vs Phraseology + Circumstances + Section_Title),
run the Task 2 spaCy NER model on the query text only (retrieved incidents describe other events),
and fill a slot only when its type is mapped AND exactly one distinct entity value of that type was found.
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
REGEX_SLOTS = {"time": r"\b(?:[01]\d|2[0-3])[0-5]\d ?(?:z|zulu|utc)\b",
               "distance": r"\b\d{1,3}(?:\.\d)? ?(?:nm|nautical miles?|dme)\b",
               "speed": r"\b\d{2,3} ?(?:kts?|knots)\b",
               "code": r"(?<=squawk )[0-7]{4}\b|(?<=code )[0-7]{4}\b",
               "ssr code": r"(?<=squawk )[0-7]{4}\b|(?<=code )[0-7]{4}\b",
               "direction": r"(?<=turn )(?:left|right)\b|(?<=turned )(?:left|right)\b"}


def is_slot(name):
    """Parentheses holding alternatives ("or LEFT") or a plural suffix ("S") are wording, not variables."""
    n = name.strip().lower()
    return bool(n) and n != "s" and not n.startswith("or ")


def slot_type(name, prev):
    name, prev = name.strip().lower(), (prev or "").strip().strip("[(").upper()
    return (CONTEXT_TYPES.get((name, prev)) or SLOT_TYPES.get(name)
            or ([f"REGEX:{name}"] if name in REGEX_SLOTS else None))


def regex_values(name, text):
    return sorted({m.group(0).upper() for m in re.finditer(REGEX_SLOTS[name], text.lower())})


def fill(phrase, ents, text=""):
    slots, out, last = [], [], 0
    for m in SLOT.finditer(phrase):
        name = m.group(2)
        if not is_slot(name):
            continue
        types = slot_type(name, m.group(1))
        status, value = "unmapped", None
        if types:
            for t in types:
                vals = (regex_values(t[6:], text) if t.startswith("REGEX:")
                        else sorted({v for v, lab in ents if lab == t}))
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
    system = best.get("primary_system")
    assert system, "retrieval/eval/results.json has no primary system"
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
        text = r.query_text
        ents = [(e.text, e.label_) for e in nlp(text).ents]
        filled, slots = fill(phrase, ents, text)
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
    in_query = sum(str(x["value"]).lower() in r.query_text.lower()
                   for r in d.itertuples() for x in json.loads(r.slots) if x["status"] == "filled")
    R = {"retrieval_system": system, "queries": len(d), "status_counts": d.status.value_counts().to_dict(),
         "complete_pct_of_queries_with_slots": round(100 * (d.status == "complete").sum() /
                                                     max(1, (d.status != "no_slots").sum()), 2),
         "slot_occurrences_in_selected_phrases": int(d.n_slots.sum()), "slots_filled": int(d.n_filled.sum()),
         "slots_filled_value_in_query_text": int(in_query),
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
         f"- Entities: Task 2 spaCy model (`phase2/ner/out/model`) on the query text only. The top-1 incident from "
         f"`{system}` (primary Task 3 system) is recorded but not used for values: it describes a different event.",
         "- A slot is filled only if its type is mapped and exactly one distinct value of that type was found. "
         "Unmapped, missing and ambiguous slots stay bracketed and are flagged in `out/slot_fills.csv`.",
         "- Parentheses holding alternatives (`(or LEFT)`) or a plural suffix (`(S)`) are wording, not slots.",
         "- Slot-type map: " + "; ".join(f"`({k})` -> {'/'.join(v)}" for k, v in SLOT_TYPES.items()) +
         "; `RUNWAY (number)` -> RUNWAY; `HEADING (three digits)` -> HEADING.",
         "- Slots with no NER label, filled by fixed regular expressions over the same text (exactly one distinct "
         "match required): " + ", ".join(f"`({k})`" for k in REGEX_SLOTS) + ". Every other slot name is unmapped.",
         "", "## Numbers", "", "| Status | Queries |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in R["status_counts"].items()]
    L += ["", f"Complete among queries whose phrase has slots: {R['complete_pct_of_queries_with_slots']}%. "
          f"Slots filled: {R['slots_filled']} of {R['slot_occurrences_in_selected_phrases']}, so "
          f"{R['slots_filled'] - R['slots_filled_value_in_query_text']} wrong fills. The rest stay bracketed: "
          f"{sum(v for k, v in R['slot_outcomes_by_type'].items() if k.endswith(':no_entity'))} have a mapped type but the "
          "query states no value of that type, and "
          f"{sum(v for k, v in R['slot_outcomes_by_type'].items() if k.endswith(':unmapped'))} have no mapped type. "
          "These gold queries carry almost no slot values, so they test only that nothing is invented; filling is "
          "measured on the slot test set (testset/REPORT.md). "
          f"{R['slots_filled_value_in_query_text']} of the {R['slots_filled']} filled values occur in the query text, "
          "(slots are filled from the query text only).",
          "", f"Across the whole crossmap, {R['crossmap_slot_occurrences_with_mapped_type']} of "
          f"{R['crossmap_slot_occurrences']} slot occurrences ({R['crossmap_distinct_slot_names']} distinct names) "
          "have a mapped type (NER entity or fixed regex); the rest (significant point, reason, ...) are unmapped and stay bracketed.",
          "", "| Slot type : outcome | Count |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in R["slot_outcomes_by_type"].items()]
    L += ["", "## Failures and limits", "",
          "- NER labels are provisional (LLM-judged (Claude), pending user review); so are these fills.",
          "- CALLSIGN slots can never fill from ASRS text: ASRS de-identifies callsigns (Task 2).",
          "- Phrase choice is a lexical TF-IDF match, not judged for correctness here.",
          "- Values are taken from the query text only, never from retrieved incidents (whose values describe a "
          "different event); short gold queries rarely state levels, frequencies or times, so most slots stay bracketed.",
          "", f"## {len(ex)} examples (seed {SEED})", "",
          "| Query | Phrase | Filled | Status |", "|---|---|---|---|"]
    L += [f"| {r.query_id} {r.query_text} | {r.phraseology} | {r.filled} | {r.status} |".replace("\n", " ")
          for r in ex.itertuples()]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in list(R)[:9]}, indent=1))


if __name__ == "__main__":
    main()
