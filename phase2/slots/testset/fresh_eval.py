#!/usr/bin/env python3
"""Scores the frozen ner_rules_roles filler on a fresh slot test set written by a second author, once.
Input CSV columns: query_text, callsign, flight_level, altitude, heading, runway, frequency, squawk, airport.
A cell is the value the query states for that type, or empty. "cleared FL240; reached FL260" marks roles: the
first part is the instructed value (gold); role words before the value are dropped. Per type: values of that
entity type (squawk: the fill.py code regex); one value -> filled; several -> roles.resolve; else bracketed.
Scoring and normalisation as evaluate.py. The filler files are checked against freeze.json first.
Run:  python phase2/slots/testset/fresh_eval.py --items PATH_TO_CSV
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P2 / "slots"))
sys.path.insert(0, str(P2.parent))
import fill as F  # noqa: E402
import roles  # noqa: E402
import rules  # noqa: E402
from evaluate import norm  # noqa: E402
from phase1 import sha256  # noqa: E402

TYPES = {"callsign": "CALLSIGN", "flight_level": "FLIGHT_LEVEL", "altitude": "ALTITUDE", "heading": "HEADING",
         "runway": "RUNWAY", "frequency": "FREQUENCY", "squawk": "REGEX:code", "airport": "AIRPORT"}
ROLE_WORDS = re.compile(r"^(?:cleared|instructed|assigned|told|given|expected|to|at)\s+", re.I)


def gold_of(cell):
    if pd.isna(cell) or not str(cell).strip():
        return None
    v = str(cell).split(";")[0].strip()
    while ROLE_WORDS.match(v):
        v = ROLE_WORDS.sub("", v, count=1)
    return v


def predict(t, ents, text):
    vals = (F.regex_values(t[6:], text) if t.startswith("REGEX:") else sorted({v for v, lab in ents if lab == t}))
    if len(vals) > 1:
        vals = roles.resolve(vals, text)
    return vals[0] if len(vals) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    fz = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    for f, h in fz["sha256"].items():
        assert sha256(P2.parent / f) == h, f"{f} changed after the freeze"
    import spacy
    nlp = spacy.load(P2 / "ner" / "out" / "model")
    d = pd.read_csv(a.items, dtype=str)
    rows, c = [], Counter()
    for k, r in d.iterrows():
        ents = rules.entities_v2(r.query_text, [(e.text, e.label_) for e in nlp(r.query_text).ents])
        ok = True
        for col, t in TYPES.items():
            g, p = norm(gold_of(r.get(col))), norm(predict(t, ents, r.query_text))
            s = "correct" if p and p == g else "wrong" if p else "missed" if g else "correct_empty"
            c[s] += 1
            c["gold_values"] += bool(g)
            ok &= s in ("correct", "correct_empty")
            rows.append({"row": k + 1, "query_text": r.query_text, "type": col, "gold": g, "pred": p, "status": s})
        c["items_fully_correct"] += ok
    (HERE / "out").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "out" / "fresh_fills.csv", index=False)
    R = {"items": len(d), "items_sha256": sha256(Path(a.items)), **dict(c),
         "precision": round(c["correct"] / (c["correct"] + c["wrong"]), 4) if c["correct"] + c["wrong"] else None,
         "recall": round(c["correct"] / c["gold_values"], 4) if c["gold_values"] else None, "freeze": fz}
    (HERE / "results_fresh.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in R.items() if k != "freeze"}, indent=1))


if __name__ == "__main__":
    main()
