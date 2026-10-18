#!/usr/bin/env python3
"""Writes results.json and REPORT.md for the slot test set from the fills written by evaluate.py.
Order of runs: build_items.py; evaluate.py --method ner --splits dev,test (baseline, fixed before any change);
rules.py written from the dev errors only; evaluate.py --method ner_rules --splits dev; selection on dev;
evaluate.py --method ner_rules --splits test (once). Multi-value round (dev only): evaluate.py --method
ner_rules[_roles] --splits dev [--items dev_multi.csv]; then freeze.json; fresh_eval.py on a second-author set."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from phase1 import sha256  # noqa: E402

COLS = ["correct", "wrong", "missed", "correct_empty", "gold_values"]


def metrics(d):
    c = {k: int(d[k].fillna(0).sum()) if k in d else 0 for k in COLS}
    return {"items": len(d), **c, "items_fully_correct": int(d.item_correct.sum()),
            "precision": round(c["correct"] / (c["correct"] + c["wrong"]), 4) if c["correct"] + c["wrong"] else None,
            "recall": round(c["correct"] / c["gold_values"], 4) if c["gold_values"] else None}


def main():
    base = pd.read_csv(HERE / "out" / "fills_ner_dev_test.csv")
    rd = pd.read_csv(HERE / "out" / "fills_ner_rules_dev.csv")
    rt = pd.read_csv(HERE / "out" / "fills_ner_rules_test.csv")
    R = {"items_sha256": sha256(HERE / "items.csv"),
         "ner": {s: metrics(base[base.split == s]) for s in ["dev", "test"]},
         "ner_rules": {"dev": metrics(rd), "test": metrics(rt)},
         "selection": "ner_rules chosen on dev (higher recall, precision not lower); test scored once"}
    O = HERE / "out"
    if (O / "fills_ner_rules_roles_dev_dev_multi.csv").exists():
        R["roles_round_dev_only"] = {
            "dev_multi_sha256": sha256(HERE / "dev_multi.csv"),
            "ner_rules": {"dev": metrics(rd), "dev_multi": metrics(pd.read_csv(O / "fills_ner_rules_dev_dev_multi.csv"))},
            "ner_rules_roles": {"dev": metrics(pd.read_csv(O / "fills_ner_rules_roles_dev.csv")),
                                "dev_multi": metrics(pd.read_csv(O / "fills_ner_rules_roles_dev_dev_multi.csv"))},
            "freeze": json.loads((HERE / "freeze.json").read_text(encoding="utf-8")),
            "test": "not scored on the 40-item test set (its failures were seen); fresh second-author set pending"}
    (HERE / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    t = R["ner_rules"]["test"]
    L = ["# Slot filling on a slot test set (queries that state values)", "",
         "The 100 gold retrieval queries state almost no slot values (Task 6: 0 of 91 filled, 0 wrong fills; the slots stay bracketed), so they cannot "
         "measure slot filling. This set measures it directly.", "", "## Method", "",
         "- `items.csv`: 60 ATC-style queries **written by Devin** (not ASRS text), each paired with a fixed crossmap "
         "phrase and the gold value of every slot in it (empty = the query states none; the slot must stay bracketed). "
         "Includes queries with two values of one type. Split 20 dev / 40 test (seed 42), frozen before scoring "
         f"(SHA-256 `{R['items_sha256'][:16]}...`).",
         "- Phrase is fixed per item, so this isolates slot filling from phrase matching.",
         "- `ner`: the Task 6 filler unchanged (spaCy NER on the query + fixed regexes; fill only if exactly one value).",
         "- `ner_rules`: adds two deterministic patterns written from the 4 dev errors only (`rules.py`): flight levels "
         "(`FL350`, `flight level 90`) and callsigns (`UAL881`, `N123AB`, `Speedbird 123`); a pattern match replaces an "
         "overlapping NER entity.",
         "- Normalisation for comparison (fixed in `evaluate.py`): upper case, unit words and punctuation dropped.", "",
         "## Numbers", "", "| Method | Split | Slots with a value | Correct | Wrong fills | Missed | Correctly left empty | "
         "Precision | Recall | Items fully correct |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ["ner", "ner_rules"]:
        for s in ["dev", "test"]:
            x = R[m][s]
            L.append(f"| `{m}` | {s} | {x['gold_values']} | {x['correct']} | {x['wrong']} | {x['missed']} | "
                     f"{x['correct_empty']} | {x['precision']} | {x['recall']} | {x['items_fully_correct']} of {x['items']} |")
    miss = rt[~rt.item_correct]
    L += ["", f"Test, chosen method: {t['correct']} of {t['gold_values']} stated values filled correctly, "
          f"{t['wrong']} wrong fills, {t['items_fully_correct']} of {t['items']} items fully correct.", "",
          "## Failures and limits", "", "Test items still wrong:", ""]
    L += [f"- {r.item_id} `{r.query_text}`: gold {r.gold}, got {r.values} ({r.statuses})." for r in miss.itertuples()]
    L += ["", "- Two values of one type (e.g. cleared level and level reached) are left bracketed by design: the filler "
          "never guesses, so recall drops but no wrong value is filled.",
          "- The same author wrote the queries and the rules, and the set is small (40 test items); the numbers show the "
          "filler works when values are stated, not how often real queries state them.",
          "- Runway and some frequency forms are missed when the NER model does not tag them; no rule was added for "
          "them because no dev item failed on them.",
          "- The Task 6 numbers on the 100 gold queries are unchanged; `rules.py` is not used by `fill.py`."]
    if "roles_round_dev_only" in R:
        X = R["roles_round_dev_only"]
        L += ["", "## Multi-value round (dev only; frozen before a fresh test set)", "",
              "- `dev_multi.csv`: 12 new dev items written by Devin before the rules (10 with two or three values of one "
              "type, plus runway/frequency formats and one empty item), SHA-256 "
              f"`{X['dev_multi_sha256'][:16]}...`.",
              "- `ner_rules_roles`: `rules.entities_v2` adds runway designators (`27L`), VHF frequencies (`128.725`) "
              "and altitudes in feet; `roles.py` resolves a slot with several values by keeping values stated before a "
              "deviation cue (but, stopped at, dialled, instead of, ...) and not directly after \"from\", and fills only "
              "if exactly one remains. Assumption: the query states the instructed value before the deviation.",
              "- The altitude pattern was added after `roles` gave one wrong fill on dev item D02 (NER missed `4000 feet`).",
              "", "| Method | Items | Slots with a value | Correct | Wrong fills | Missed | Precision | Recall | Items fully correct |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in ["ner_rules", "ner_rules_roles"]:
            for k in ["dev", "dev_multi"]:
                x = X[m][k]
                L.append(f"| `{m}` | {k} | {x['gold_values']} | {x['correct']} | {x['wrong']} | {x['missed']} | "
                         f"{x['precision']} | {x['recall']} | {x['items_fully_correct']} of {x['items']} |")
        L += ["", "These are dev numbers from items written by the rule author, so they are optimistic by construction. "
              "The 40-item test set is not re-scored with these rules because its failures were already seen. The "
              "files are hashed in `freeze.json`; `fresh_eval.py` scores a second-author set once and refuses to run "
              "if any hashed file changed. No fresh test result exists yet."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({m: {s: (R[m][s]["precision"], R[m][s]["recall"]) for s in R[m]} for m in ["ner", "ner_rules"]}))


if __name__ == "__main__":
    main()
