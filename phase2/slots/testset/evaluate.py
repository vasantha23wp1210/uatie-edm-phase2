#!/usr/bin/env python3
"""Scores slot filling on the slot test set (items.csv, built by build_items.py) with phrase fixed per item.
Per slot: correct (filled = gold), wrong (filled != gold, or filled where gold is empty), missed (gold value,
left bracketed), correct_empty (no gold value, left bracketed). Normalisation is fixed: upper case, drop
non-alphanumerics and unit words (FEET, FT, NM, Z, ZULU, UTC, RUNWAY, RWY, HEADING, HDG), FLIGHT LEVEL -> FL.
Run:  python phase2/slots/testset/evaluate.py --method ner|ner_rules|ner_rules_roles --splits dev[,test] [--items CSV]
ner_rules_roles = ner_rules + runway/frequency patterns (rules.entities_v2) + roles.py for multi-value slots.
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
sys.path.insert(0, str(P2 / "slots"))
sys.path.insert(0, str(P2.parent))
import fill as F  # noqa: E402
from phase1 import sha256  # noqa: E402

UNITS = re.compile(r"\b(FEET|FT|NM|NAUTICAL MILES?|DME|ZULU|UTC|RUNWAY|RWY|HEADING|HDG)\b")


def norm(v):
    if v is None:
        return None
    v = str(v).upper().replace("FLIGHT LEVEL", "FL")
    v = UNITS.sub("", v)
    v = re.sub(r"(?<=\d)Z\b", "", v)
    return re.sub(r"[^A-Z0-9]", "", v) or None


def score(fills, gold):
    c = Counter()
    for f, g in zip(fills, gold):
        f, g = norm(f), norm(g)
        c["correct" if f and f == g else "wrong" if f else "missed" if g else "correct_empty"] += 1
        c["gold_values"] += bool(g)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ner")
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--items", default="items.csv")
    args = ap.parse_args()
    import spacy
    nlp = spacy.load(P2 / "ner" / "out" / "model")
    rules = None
    roles = None
    if args.method in ("ner_rules", "ner_rules_roles"):
        sys.path.insert(0, str(HERE))
        import rules  # noqa: E402
    if args.method == "ner_rules_roles":
        import roles  # noqa: E402
    d = pd.read_csv(HERE / args.items)
    d = d[d.split.isin(args.splits.split(","))]
    rows, tot = [], {}
    for r in d.itertuples():
        ents = [(e.text, e.label_) for e in nlp(r.query_text).ents]
        if rules:
            ents = (rules.entities_v2 if roles else rules.entities)(r.query_text, ents)
        filled, slots = (roles.fill(F, r.phraseology, ents, r.query_text) if roles
                         else F.fill(r.phraseology, ents, r.query_text))
        gold = json.loads(r.gold)
        c = score([s["value"] for s in slots], gold)
        tot.setdefault(r.split, Counter()).update(c)
        rows.append({"item_id": r.item_id, "split": r.split, "query_text": r.query_text, "phraseology": r.phraseology,
                     "filled": filled, "gold": r.gold, "values": json.dumps([s["value"] for s in slots]),
                     "statuses": json.dumps([s["status"] for s in slots]),
                     "entities": "; ".join(f"{t}|{lab}" for t, lab in ents), **c,
                     "item_correct": c["wrong"] == 0 and c["missed"] == 0})
    out = pd.DataFrame(rows)
    (HERE / "out").mkdir(exist_ok=True)
    out.to_csv(HERE / "out" / f"fills_{args.method}_{args.splits.replace(',', '_')}{'' if args.items == 'items.csv' else '_' + Path(args.items).stem}.csv", index=False)
    R = {}
    for s, c in tot.items():
        sub = out[out.split == s]
        R[s] = {"items": len(sub), "slots": sum(c.values()) - c["gold_values"], **dict(c),
                "precision": round(c["correct"] / (c["correct"] + c["wrong"]), 4) if c["correct"] + c["wrong"] else None,
                "recall": round(c["correct"] / c["gold_values"], 4) if c["gold_values"] else None,
                "items_fully_correct": int(sub.item_correct.sum())}
    R["method"], R["items"], R["items_sha256"] = args.method, args.items, sha256(HERE / args.items)
    print(json.dumps(R, indent=1))
    return R, out


if __name__ == "__main__":
    main()
