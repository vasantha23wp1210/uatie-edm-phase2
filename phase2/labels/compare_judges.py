"""Agreement between the two provisional LLM label sets: Claude (judgements/*.csv, used by every task) and Devin's
superseded set (judgements/*_devin.csv). Neither set is human-reviewed. Writes results.json and REPORT.md here.
Run: python phase2/labels/compare_judges.py"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ENT = re.compile(r"\[([^\]]+)\]\(([A-Z_]+)\)")


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def pair(task, name, key):
    c, d = read(P2 / task / "judgements" / f"{name}.csv"), read(P2 / task / "judgements" / f"{name}_devin.csv")
    assert list(c[key]) == list(d[key]), f"{name}: row order differs"
    return c, d


def kappa(a, b, **kw):
    return round(float(cohen_kappa_score(a, b, **kw)), 4)


def retrieval():
    c, d = pair("retrieval", "judged", "pair_id")
    a, b = d.relevance.astype(int), c.relevance.astype(int)
    return {"pairs": len(c), "exact_agreement": round(float((a == b).mean()), 4), "cohen_kappa": kappa(a, b),
            "linear_weighted_kappa": kappa(a, b, weights="linear"),
            "binary_rel_ge1_kappa": kappa(a >= 1, b >= 1), "binary_rel2_kappa": kappa(a == 2, b == 2),
            "disagreements_by_2_points": int(((a - b).abs() == 2).sum()),
            "confusion_rows_devin_cols_claude_0_1_2": confusion_matrix(a, b, labels=[0, 1, 2]).tolist(),
            "claude_label_counts": {str(k): int(v) for k, v in b.value_counts().sort_index().items()}}


def intent():
    c, d = pair("intent_classifier", "intent_labelled", "label_id")
    split = read(P2 / "intent_classifier" / "splits" / "gold_key.csv").set_index("label_id").gold_split
    out = {"rows": len(c), "exact_agreement": round(float((c.your_intent == d.your_intent).mean()), 4),
           "cohen_kappa": kappa(d.your_intent, c.your_intent)}
    for s in ["dev", "test"]:
        m = c.label_id.map(split).eq(s).values
        out[s] = {"rows": int(m.sum()), "exact_agreement": round(float((c.your_intent[m] == d.your_intent[m]).mean()), 4),
                  "cohen_kappa": kappa(d.your_intent[m], c.your_intent[m])}
    dis = Counter(zip(d.your_intent[c.your_intent != d.your_intent], c.your_intent[c.your_intent != d.your_intent]))
    out["top_disagreements_devin_to_claude"] = [{"devin": x, "claude": y, "rows": n} for (x, y), n in dis.most_common(5)]
    out["claude_label_counts"] = c.your_intent.value_counts().to_dict()
    out["claude_ambiguity_notes"] = len(read(P2 / "intent_classifier" / "judgements" / "intent_ambiguity_notes.csv"))
    return out


def spans(markup):
    out, pos, plain = set(), 0, ""
    for m in ENT.finditer(markup):
        plain += markup[pos:m.start()]
        out.add((len(plain), len(plain) + len(m.group(1)), m.group(2)))
        plain += m.group(1)
        pos = m.end()
    return out


def ner():
    c, d = pair("ner", "ner_corrected", "sent_id")
    test = set(read(P2 / "ner" / "splits" / "ner_split.csv").query("split == 'test'").sent_id)
    out = {}
    for name, m in [("all", [True] * len(c)), ("test", c.sent_id.isin(test).tolist())]:
        sc = [spans(x) for x, k in zip(c.corrected, m) if k]
        sd = [spans(x) for x, k in zip(d.corrected, m) if k]
        both = sum(len(x & y) for x, y in zip(sc, sd))
        nc, nd = sum(map(len, sc)), sum(map(len, sd))
        out[name] = {"sentences": len(sc), "entities_claude": nc, "entities_devin": nd,
                     "sentences_with_different_markup": sum(x != y for x, y in zip(sc, sd)),
                     "entity_agreement_f1": round(2 * both / (nc + nd), 4) if nc + nd else None}
    return out


def main():
    r = {"note": "Both label sets are LLM-generated and pending user review; agreement is not accuracy.",
         "retrieval": retrieval(), "intent": intent(), "ner": ner()}
    (HERE / "results.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    t, i, n = r["retrieval"], r["intent"], r["ner"]
    L = ["# Provisional label sets: Claude vs Devin", "",
         "Every Phase 2 task now uses the Claude label sets (`judgements/*.csv`, method in `LABELLING_PROMPT.md`).",
         "Devin's earlier provisional sets are kept as `judgements/*_devin.csv`. Both are LLM-generated and **pending user",
         "review**, so the agreement below says how consistent two LLM judges are, not how accurate either is.", "",
         "| File | Rows | Agreement | Cohen's κ |", "|---|---:|---:|---:|",
         f"| Retrieval relevance (0/1/2) | {t['pairs']} | {t['exact_agreement']:.1%} | {t['cohen_kappa']} |",
         f"| Intent (all) | {i['rows']} | {i['exact_agreement']:.1%} | {i['cohen_kappa']} |",
         f"| Intent (dev) | {i['dev']['rows']} | {i['dev']['exact_agreement']:.1%} | {i['dev']['cohen_kappa']} |",
         f"| Intent (test) | {i['test']['rows']} | {i['test']['exact_agreement']:.1%} | {i['test']['cohen_kappa']} |", "",
         f"- Retrieval: linear-weighted κ {t['linear_weighted_kappa']}, binary rel ≥ 1 κ {t['binary_rel_ge1_kappa']}, "
         f"binary rel 2 κ {t['binary_rel2_kappa']}; {t['disagreements_by_2_points']} pairs differ by 2 points. "
         f"Confusion (rows Devin, columns Claude, 0/1/2): {t['confusion_rows_devin_cols_claude_0_1_2']}.",
         "- Intent, largest disagreements (Devin → Claude): " + "; ".join(
             f"{x['devin']} → {x['claude']} ({x['rows']})" for x in i["top_disagreements_devin_to_claude"]) + ".",
         f"- Intent: Claude left notes on {i['claude_ambiguity_notes']} ambiguous rows "
         "(`phase2/intent_classifier/judgements/intent_ambiguity_notes.csv`).",
         f"- NER, entity-level agreement F1 (exact span and label): all {n['all']['sentences']} sentences "
         f"{n['all']['entity_agreement_f1']} ({n['all']['entities_claude']} Claude vs {n['all']['entities_devin']} Devin "
         f"entities, {n['all']['sentences_with_different_markup']} sentences differ); test {n['test']['sentences']} sentences "
         f"{n['test']['entity_agreement_f1']}.",
         "- Earlier calibration against the user's own labels (see `phase2/gold_eval_rebuild/`): Devin κ 0.0619 on 362 dev pairs; "
         "Claude κ 0.34 on the user's 100-pair test spot-check. This is why the Claude sets were adopted.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({k: r[k] for k in ["retrieval", "ner"]}, indent=1)[:1500])


if __name__ == "__main__":
    main()
