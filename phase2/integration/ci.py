#!/usr/bin/env python3
"""95% intervals for the headline Phase 2 numbers, from files already written (no new runs).
Proportions (intent accuracy, judged unsupported claims, slot precision/recall): Wilson interval.
Retrieval P@5 / MAP@10 (test, rel >= 1): percentile bootstrap over queries, 10,000 resamples, seed 42.
Writes ci.json and CI.md.
Run:  python phase2/integration/ci.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SEED, N_BOOT = 42, 10000


def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def prop(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    R = {"method": "Wilson for proportions; bootstrap over queries for retrieval (10,000, seed 42)"}
    b = load(P2 / "intent_classifier" / "results_breakdown.json")
    m = b["model_chosen_on_dev"]
    R["intent_test_accuracy"] = {k: {"model": k, **{x: b["accuracy"][k]["test"]["all"][x]
                                                     for x in ["correct", "n", "accuracy", "ci95"]}}
                                 for k in [m, "logreg_minilm_scaled", "logreg_minilm"]}
    R["hallucination_llm_judge_3b"] = {}
    for name, d in [("ground llama3.2:3b test", None), ("ground llama3.1:8b test", "out_llama3.1_8b_test")]:
        if d is None:
            ans = {json.loads(x)["query_id"]: json.loads(x) for x in
                   (P2 / "generation" / "out" / "answers.jsonl").read_text(encoding="utf-8").splitlines()}
            J = [json.loads(x) for x in (P2 / "generation" / "out" / "judgements.jsonl").read_text(encoding="utf-8").splitlines()]
            J = [j for j in J if ans[j["query_id"]]["split"] == "test"]
        else:
            J = [json.loads(x) for x in (P2 / "generation" / d / "judgements.jsonl").read_text(encoding="utf-8").splitlines()]
        cl = [c for j in J for c in j["claims"]]
        u = sum(not (c["supported"] is True or str(c["supported"]).lower() == "true") for c in cl)
        R["hallucination_llm_judge_3b"][name] = prop(u, len(cl))
    for name, f in [("cockpit (all 100 queries)", P2 / "edge" / "out" / "results.json")]:
        h = load(f)["hallucination_filtered"]
        R["hallucination_llm_judge_3b"][name] = prop(h["unsupported"], h["claims"])
    for m_, v in load(P2 / "generation" / "compare" / "results.json")["models"].items():
        R["hallucination_llm_judge_3b"][f"dev free {m_}"] = prop(v["unsupported"], v["claims"])
    for k, v in load(P2 / "generation" / "compare_anchored" / "results.json")["candidates"].items():
        if k.startswith("anchored"):
            R["hallucination_llm_judge_3b"][f"dev {k}"] = prop(v["unsupported"], v["claims"])
    s = load(P2 / "slots" / "testset" / "results.json")["ner_rules"]["test"]
    R["slots_testset_test"] = {"precision": prop(s["correct"], s["correct"] + s["wrong"]),
                               "recall": prop(s["correct"], s["gold_values"]),
                               "items_fully_correct": prop(s["items_fully_correct"], s["items"])}
    q = pd.read_csv(P2 / "retrieval" / "eval" / "per_query.csv")
    q = q[(q.split == "test") & (q.thr == 1)]
    rng = np.random.default_rng(SEED)
    R["retrieval_test_rel_ge1"] = {}
    for sys_, d in q.groupby("system"):
        d = d.sort_values("query_id")
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        R["retrieval_test_rel_ge1"][sys_] = {
            met: {"mean": round(float(d[met].mean()), 4), "queries": len(d),
                  "ci95": [round(float(x), 4) for x in np.percentile(d[met].to_numpy()[idx].mean(1), [2.5, 97.5])]}
            for met in ["p5", "map10"]}
    (HERE / "ci.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# 95% intervals for headline numbers", "", f"{R['method']}. From `ci.py`; no new runs.", "",
         "| Quantity | value | 95% CI | n |", "|---|---|---|---|"]
    for k, v in R["intent_test_accuracy"].items():
        L.append(f"| Intent test accuracy, `{k}` | {v['accuracy']} | {v['ci95']} | {v['n']} |")
    for k, v in R["hallucination_llm_judge_3b"].items():
        L.append(f"| Unsupported claims (3B judge), {k} | {v['rate']} | {v['ci95']} | {v['n']} claims |")
    for k, v in R["slots_testset_test"].items():
        L.append(f"| Slot test set (test), {k} | {v['rate']} | {v['ci95']} | {v['n']} |")
    for k, v in R["retrieval_test_rel_ge1"].items():
        for met, x in v.items():
            L.append(f"| Retrieval test {met}, `{k}` | {x['mean']} | {x['ci95']} | {x['queries']} queries |")
    L += ["", "## Reading", "",
          "- Overlapping intervals mean the data do not separate the two numbers; e.g. the 3B and 8B test hallucination "
          "rates overlap almost completely.",
          "- Dev has 20 queries: a one-claim or one-query difference on dev is within noise, which is why dev choices "
          "(retrieval `faiss`, generator `llama3.1:8b`) did not carry over to test.",
          "- Claim intervals treat claims as independent; claims from one answer are correlated, so the true intervals "
          "are wider.", "- Intervals do not include label error (provisional LLM labels) or judge error."]
    (HERE / "CI.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[4:-6]))


if __name__ == "__main__":
    main()
