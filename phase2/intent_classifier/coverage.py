#!/usr/bin/env python3
"""Coverage-accuracy curve of the dev-chosen intent model (distilbert_scaled), using train.py's own functions,
seed and data. Retrains it once (CPU, deterministic), checks that its thresholded gold predictions equal the
scored ones in out/gold_predictions.csv, and saves gold-row probabilities. For each threshold: coverage (share not
Unclassified) and accuracy on the covered rows, dev and test. The operating point stays the dev choice; the test
curve is reported, not used to choose. Writes out/gold_proba_distilbert_scaled.csv, results_coverage.json,
REPORT_coverage.md.
Run:  python phase2/intent_classifier/coverage.py --phase1 PATH
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("p2_intent_train", TASK / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    tm.set_seeds()
    gold = tm.load_gold()
    meta = pd.read_csv(tm.REPO / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False,
                       usecols=["faiss_idx", "record_id", "summary", "intent"])
    meta["summary"] = meta.summary.fillna("").astype(str)
    scaled, _ = tm.load_scaled(meta, gold)
    cls = [c for c in tm.LABELS if c in set(scaled.intent)]
    model, tok, _ = tm.bert_train(scaled.summary, scaled.intent, cls)
    P = tm.bert_predict(model, tok, meta.summary.to_numpy()[gold.faiss_idx.to_numpy()].tolist())
    R0 = json.loads((TASK / "results.json").read_text(encoding="utf-8"))
    thr0 = R0["models"]["distilbert_scaled"]["threshold_chosen_on_dev"]
    scored = pd.read_csv(TASK / "out" / "gold_predictions.csv", dtype=str, keep_default_na=False).set_index("label_id")
    pred0, conf = tm.apply_thr(P, cls, thr0)
    same = int((pred0 == scored.loc[gold.label_id, "pred_distilbert_scaled"].to_numpy()).sum())
    pd.DataFrame(P, columns=cls).assign(label_id=gold.label_id.values, gold_split=gold.gold_split.values,
                                        your_intent=gold.your_intent.values).to_csv(
        TASK / "out" / "gold_proba_distilbert_scaled.csv", index=False)
    y, sp = gold.your_intent.to_numpy(), gold.gold_split.to_numpy()
    curve = []
    for t in tm.THRESHOLDS:
        p, _ = tm.apply_thr(P, cls, t)
        row = {"threshold": t}
        for s in ["dev", "test"]:
            m = sp == s
            cov = p[m] != tm.UNK
            row[s] = {"coverage": round(float(cov.mean()), 4), "accuracy_all": round(float((p[m] == y[m]).mean()), 4),
                      "accuracy_covered": round(float((p[m][cov] == y[m][cov]).mean()), 4) if cov.any() else None}
        curve.append(row)
    R = {"model": "distilbert_scaled", "dev_threshold": thr0, "reproduced_gold_predictions": f"{same}/{len(gold)}",
         "curve": curve}
    (TASK / "results_coverage.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Intent: coverage vs accuracy (distilbert_scaled)", "",
         f"Retrained with train.py's functions and seed; {same} of {len(gold)} gold predictions at the dev threshold "
         f"({thr0}) match the scored run. Coverage = share of rows not predicted Unclassified; accuracy on covered "
         "rows ignores the rest. The dev threshold is unchanged; the test columns are for reporting only.", "",
         "| threshold | dev coverage | dev acc (covered) | dev acc (all) | test coverage | test acc (covered) | test acc (all) |",
         "|---|---|---|---|---|---|---|"]
    for r in curve:
        d, t = r["dev"], r["test"]
        mark = " (dev choice)" if r["threshold"] == thr0 else ""
        L.append(f"| {r['threshold']}{mark} | {d['coverage']} | {d['accuracy_covered']} | {d['accuracy_all']} | "
                 f"{t['coverage']} | {t['accuracy_covered']} | {t['accuracy_all']} |")
    (TASK / "REPORT_coverage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
