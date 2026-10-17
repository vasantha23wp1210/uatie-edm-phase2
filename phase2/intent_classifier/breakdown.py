#!/usr/bin/env python3
"""Intent breakdown of the already scored predictions (no training, no threshold change):
accuracy with 95% Wilson CI on all / non-borderline / borderline gold rows, per split, for every candidate,
and the most frequent confusions of the dev-chosen model. Borderline = the gold notes contain "Borderline"
(Claude's section 5 relabel). Reads out/gold_predictions.csv and results.json; writes results_breakdown.json and
REPORT_breakdown.md.
Run:  python phase2/intent_classifier/breakdown.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def acc(d, col):
    k = int((d[col] == d.your_intent).sum())
    return {"n": len(d), "correct": k, "accuracy": round(k / len(d), 4) if len(d) else None, "ci95": wilson(k, len(d))}


def main():
    g = pd.read_csv(TASK / "out" / "gold_predictions.csv", dtype=str, keep_default_na=False)
    R0 = json.loads((TASK / "results.json").read_text(encoding="utf-8"))
    best = R0["model_chosen_on_dev_macro_f1"]
    g["borderline"] = g.notes.str.contains("Borderline")
    models = [c[5:] for c in g.columns if c.startswith("pred_")]
    R = {"model_chosen_on_dev": best, "borderline_rows": {s: int(d.borderline.sum()) for s, d in g.groupby("gold_split")},
         "accuracy": {}}
    for m in models:
        R["accuracy"][m] = {s: {"all": acc(d, f"pred_{m}"), "non_borderline": acc(d[~d.borderline], f"pred_{m}"),
                                "borderline": acc(d[d.borderline], f"pred_{m}")}
                            for s, d in g.groupby("gold_split")}
    t = g[g.gold_split == "test"]
    wrong = t[t[f"pred_{best}"] != t.your_intent]
    R["test_confusions_chosen_model"] = [{"gold": a, "pred": b, "rows": n} for (a, b), n in
                                         Counter(zip(wrong.your_intent, wrong[f"pred_{best}"])).most_common(10)]
    R["test_errors_chosen_model_predicted_unclassified"] = int((wrong[f"pred_{best}"] == "Unclassified").sum())
    (TASK / "results_breakdown.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Intent: breakdown by borderline status, with 95% intervals", "",
         "From `out/gold_predictions.csv` (already scored; nothing retrained, no threshold changed). Borderline rows are "
         "those Claude marked borderline in the section 5 relabel. Intervals: Wilson 95%.", "",
         "| Model | split | all | non-borderline | borderline |", "|---|---|---|---|---|"]
    for m in models:
        for s in ["dev", "test"]:
            x = R["accuracy"][m][s]
            f = lambda v: f"{v['accuracy']} {v['ci95']} (n={v['n']})"
            L.append(f"| `{m}`{' (chosen)' if m == best else ''} | {s} | {f(x['all'])} | {f(x['non_borderline'])} | "
                     f"{f(x['borderline'])} |")
    L += ["", f"Most frequent test confusions of `{best}` (gold -> predicted):", ""]
    L += [f"- {c['gold']} -> {c['pred']}: {c['rows']}" for c in R["test_confusions_chosen_model"]]
    L += ["", f"{R['test_errors_chosen_model_predicted_unclassified']} of {len(wrong)} test errors are predictions of "
          "Unclassified (below the dev threshold or the class itself).", "", "## Limits", "",
          "- Borderline flags come from the same LLM that wrote the gold labels; there is no human agreement figure, so "
          "no measured ceiling for accuracy.",
          "- Splitting by borderline status was not used to choose anything."]
    (TASK / "REPORT_breakdown.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[5:]))


if __name__ == "__main__":
    main()
