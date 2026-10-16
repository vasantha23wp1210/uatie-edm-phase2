#!/usr/bin/env python3
"""Task 2, step 2: train a spaCy NER on the corrected markup and compare with the rule-only baseline.

Inputs:  judgements/ner_to_correct.csv (identity reference), judgements/ner_corrected.csv (filled `corrected`),
         splits/ner_split.csv (200 train / 60 test, fixed by export.py)
Writes:  out/model/, out/test_predictions.csv, results.json, REPORT.md
Scoring: exact span + label match, entity-level micro P/R/F1 on the 60 test sentences.
Run:  python phase2/ner/train.py
"""
import json
import platform
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(ROOT))
from phase1 import sha256  # noqa: E402
from rules import LABELS, build_nlp  # noqa: E402

OUT = TASK / "out"
SEED, EPOCHS, DROPOUT, MIN_SUPPORT = 42, 30, 0.2, 5
MARK = re.compile(r"\[([^\]]+)\]\(([A-Z_]+)\)")


def parse(markup):
    text, ents, pos = [], [], 0
    last = 0
    for m in MARK.finditer(markup):
        text.append(markup[last:m.start()])
        pos += m.start() - last
        ents.append((pos, pos + len(m.group(1)), m.group(2)))
        text.append(m.group(1))
        pos += len(m.group(1))
        last = m.end()
    text.append(markup[last:])
    return "".join(text), ents


def load():
    ref = pd.read_csv(TASK / "judgements" / "ner_to_correct.csv", dtype=str, keep_default_na=False)
    got = pd.read_csv(TASK / "judgements" / "ner_corrected.csv", dtype=str, keep_default_na=False)
    assert len(got) == len(ref) and list(got.sent_id) == list(ref.sent_id), "rows differ from the export"
    split = pd.read_csv(TASK / "splits" / "ner_split.csv").set_index("sent_id").split
    rows = []
    for r, g in zip(ref.itertuples(), got.itertuples()):
        t, ents = parse(g.corrected)
        assert t == r.text, f"{r.sent_id}: text changed"
        bad = [e for e in ents if e[2] not in LABELS]
        assert not bad, f"{r.sent_id}: unknown label {bad}"
        rows.append(dict(sent_id=r.sent_id, text=t, ents=ents, split=split[r.sent_id], notes=g.notes))
    return pd.DataFrame(rows)


def prf(gold, pred):
    out = {}
    for lab in LABELS + ["ALL"]:
        g = {e for e in gold if lab == "ALL" or e[3] == lab}
        p = {e for e in pred if lab == "ALL" or e[3] == lab}
        tp = len(g & p)
        P = tp / len(p) if p else 0.0
        R = tp / len(g) if g else 0.0
        out[lab] = {"precision": round(P, 4), "recall": round(R, 4),
                    "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0, "support": len(g), "predicted": len(p)}
    return out


def ents_of(df, col):
    return [(r.sent_id, s, e, l) for r in df.itertuples() for s, e, l in getattr(r, col)]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    fix_random_seed(SEED)
    df = load()
    train, test = df[df.split == "train"], df[df.split == "test"]

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for lab in LABELS:
        ner.add_label(lab)
    examples, misaligned = [], 0
    for r in train.itertuples():
        doc = nlp.make_doc(r.text)
        spans = []
        for s, e, l in r.ents:
            sp = doc.char_span(s, e, label=l, alignment_mode="expand")
            misaligned += int(sp is None or (sp.start_char, sp.end_char) != (s, e))
            if sp is not None:
                spans.append((sp.start_char, sp.end_char, l))
        examples.append(Example.from_dict(doc, {"entities": spans}))
    opt = nlp.initialize(lambda: examples)
    losses_per_epoch = []
    for ep in range(EPOCHS):
        random.Random(SEED + ep).shuffle(examples)
        losses = {}
        for b in minibatch(examples, size=8):
            nlp.update(b, sgd=opt, drop=DROPOUT, losses=losses)
        losses_per_epoch.append(round(float(losses["ner"]), 3))
    OUT.mkdir(exist_ok=True)
    nlp.to_disk(OUT / "model")

    rule = build_nlp()
    test = test.assign(pred_model=[[(e.start_char, e.end_char, e.label_) for e in nlp(t).ents] for t in test.text],
                       pred_rules=[[(e.start_char, e.end_char, e.label_) for e in rule(t).ents] for t in test.text])
    gold = ents_of(test, "ents")
    R = {"task": "ner", "seed": SEED, "epochs": EPOCHS, "dropout": DROPOUT, "spacy": spacy.__version__,
         "label_provenance": df.notes.value_counts().to_dict(),
         "sentences": {"train": int(len(train)), "test": int(len(test))},
         "train_entity_counts": pd.Series([l for es in train.ents for *_, l in es]).value_counts()
         .reindex(LABELS, fill_value=0).astype(int).to_dict(),
         "test_entity_counts": pd.Series([l for es in test.ents for *_, l in es]).value_counts()
         .reindex(LABELS, fill_value=0).astype(int).to_dict(),
         "train_spans_not_token_aligned": misaligned, "epoch_loss": losses_per_epoch,
         "rules_baseline": prf(gold, ents_of(test, "pred_rules")),
         "spacy_ner": prf(gold, ents_of(test, "pred_model"))}
    R["too_rare_to_evaluate_test_support_lt_5"] = [l for l in LABELS if R["test_entity_counts"][l] < MIN_SUPPORT]
    fmt = lambda t, es: "; ".join(f"{t[s:e]}|{l}" for s, e, l in es)
    pd.DataFrame({"sent_id": test.sent_id, "text": test.text,
                  "gold": [fmt(t, es) for t, es in zip(test.text, test.ents)],
                  "rules": [fmt(t, es) for t, es in zip(test.text, test.pred_rules)],
                  "model": [fmt(t, es) for t, es in zip(test.text, test.pred_model)]}).to_csv(OUT / "test_predictions.csv", index=False)
    R["env"] = {"python": platform.python_version()}
    R["inputs_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                          [TASK / "judgements" / "ner_corrected.csv", TASK / "splits" / "ner_split.csv"]}
    R["output_sha256"] = {"phase2/ner/out/test_predictions.csv": sha256(OUT / "test_predictions.csv")}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({k: R[k]["ALL"] for k in ["rules_baseline", "spacy_ner"]}, indent=2))


def write_report(R):
    L = ["# Task 2: Named entity recognition", "", "All numbers come from `results.json`, written by `train.py`.", "",
         "## Label provenance (read first)", "",
         f"- Corrected markup: {R['label_provenance']}. While it is `pending user review`, the scores below measure "
         "agreement with provisional LLM corrections, not with human gold.", "",
         f"## Test (n = {R['sentences']['test']} sentences, exact span + label)", "",
         "| Entity | test support | rules P | rules R | rules F1 | spaCy P | spaCy R | spaCy F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for lab in LABELS + ["ALL"]:
        a, b = R["rules_baseline"][lab], R["spacy_ner"][lab]
        L.append(f"| {lab if lab != 'ALL' else '**micro (all)**'} | {a['support']} | {a['precision']:.4f} | "
                 f"{a['recall']:.4f} | {a['f1']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} |")
    L += ["", f"Train entity counts: {R['train_entity_counts']}.", "",
          f"Too rare to evaluate (test support < {MIN_SUPPORT}): {', '.join(R['too_rare_to_evaluate_test_support_lt_5'])}.",
          "CALLSIGN in particular cannot be evaluated: ASRS de-identifies callsigns, and none of the 260 sentences "
          "contains one.", "",
          f"- spaCy blank `en` + `ner`, {R['epochs']} epochs, dropout {R['dropout']}, seed {R['seed']}, CPU. "
          f"{R['train_spans_not_token_aligned']} training spans were not token-aligned and were expanded to token "
          "boundaries.", "- Per-sentence gold/rules/model entities: `out/test_predictions.csv`."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
