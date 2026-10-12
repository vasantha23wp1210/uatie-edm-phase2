#!/usr/bin/env python3
"""Task 3, step 1: reproduce the Phase 1 HySOR baseline (paper Table 5) before any Phase 2 retrieval work.

Fresh rerun of the Phase 1 engine (rag_query_engine.hybrid_search: alpha 0.65, recency 1990-current year)
for all 100 gold queries, scored with the published protocol (bm25_baseline.compute_metrics, relevance >= 1,
labels = evaluation/gold_relevance.csv as committed). The stored Phase 1 ranking
(evaluation/retrieval_results_enriched.csv) is scored the same way for comparison.

Run:  python phase2/retrieval/baseline.py --phase1 <path to uatie-edm-phase1>
"""
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root, sha256  # noqa: E402

REPO = phase1_root()
OUT = TASK / "baseline"
SEED = 42
FETCH_K = 20            # Phase 1 eval called hybrid_search(k=20) (build_retrieval_results_v2.TOP_N)
K_VALUES = [1, 3, 5, 10]
METRICS = ["P@1", "P@3", "P@5", "P@10", "nDCG@5", "nDCG@10", "MRR", "MAP"]
PAPER = {"P@1": 0.896, "P@3": 0.909, "P@5": 0.891, "P@10": 0.888, "nDCG@5": 0.899, "nDCG@10": 0.901,
         "MRR": 0.934, "MAP": 0.896}
TARGETS = ["P@5", "MRR", "MAP", "nDCG@10"]
TOL = 0.01
STORED = REPO / "evaluation" / "retrieval_results_enriched.csv"


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def score(bm25_mod, lists, gold):
    rows = []
    for qid, ids in lists.items():
        rel = gold.get(qid, set())
        m = bm25_mod.compute_metrics(ids, rel, len(rel), K_VALUES)
        rows.append({"query_id": qid, "n_relevant": len(rel),
                     **{k: m[k] for k in METRICS if k in m}, "MRR": m["RR"], "MAP": m["AP"]})
    return pd.DataFrame(rows)


def main():
    set_seeds()
    os.chdir(REPO)
    rag = load_module("rag_query_engine", REPO / "src" / "rag" / "rag_query_engine.py")
    bm25_mod = load_module("bm25_baseline", REPO / "src" / "rag" / "bm25_baseline.py")
    rag.initialize_resources(verbose=False)
    if not rag.SEMANTIC_OK:
        raise RuntimeError("Phase 1 engine failed to load FAISS/embedding model")

    q = pd.read_csv(REPO / "evaluation" / "gold_queries.csv")
    rel = pd.read_csv(REPO / "evaluation" / "gold_relevance.csv")
    rel = rel[rel.relevance >= 1]
    gold = rel.groupby("query_id").incident_id.apply(lambda s: set(s.astype(str))).to_dict()

    fresh = {r.query_id: [str(x["incident_id"]) for x in rag.hybrid_search(str(r.query_text), k=FETCH_K)]
             for r in q.itertuples()}
    stored_df = pd.read_csv(STORED)
    stored = {qid: [str(i) for i in bm25_mod.get_hybrid_results(stored_df, qid, FETCH_K)] for qid in q.query_id}

    OUT.mkdir(exist_ok=True)
    pd.DataFrame([(k, r, i) for k, v in fresh.items() for r, i in enumerate(v, 1)],
                 columns=["query_id", "rank", "incident_id"]).to_csv(OUT / "hysor_fresh_top20.csv", index=False)

    res = {}
    for name, lists in [("fresh_rerun", fresh), ("stored_phase1_ranking", stored)]:
        pq = score(bm25_mod, lists, gold)
        pq.to_csv(OUT / f"per_query_{name}.csv", index=False, float_format="%.6f")
        ev = pq[pq.n_relevant > 0]
        agg = {m: round(float(ev[m].mean()), 4) for m in METRICS}
        res[name] = {"queries_evaluated": int(len(ev)), "queries_total": int(len(pq)),
                     "queries_skipped_no_relevant_label": sorted(pq[pq.n_relevant == 0].query_id),
                     "metrics": agg,
                     "diff_vs_paper": {m: round(agg[m] - PAPER[m], 4) for m in METRICS},
                     "within_tol_targets": {m: abs(agg[m] - PAPER[m]) <= TOL for m in TARGETS}}
    same = sum(fresh[k] == stored[k] for k in fresh)
    R = {"task": "retrieval/baseline", "seed": SEED, "fetch_k": FETCH_K, "tolerance": TOL, "paper": PAPER,
         "engine": {"SIM_WEIGHT": rag.SIM_WEIGHT, "RECENCY_WEIGHT": rag.RECENCY_WEIGHT,
                    "RECENCY_MIN_YEAR": rag.RECENCY_MIN_YEAR, "RECENCY_MAX_YEAR": rag.RECENCY_MAX_YEAR,
                    "TOP_K_default": rag.TOP_K, "embed_model": rag.EMBED_MODEL},
         "fresh_vs_stored_identical_top20_lists": same,
         **res,
         "reproduced": all(res["fresh_rerun"]["within_tol_targets"].values()),
         "env": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
         "phase1": phase1_info(REPO), "phase1_extra_sha256": {"evaluation/retrieval_results_enriched.csv": sha256(STORED)},
         "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(OUT.glob("*.csv"))}}
    (OUT / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({k: R[k] for k in ["reproduced", "fresh_vs_stored_identical_top20_lists"]} |
                     {"fresh": R["fresh_rerun"]["metrics"], "n": R["fresh_rerun"]["queries_evaluated"]}, indent=2))


def write_report(R):
    f, s = R["fresh_rerun"], R["stored_phase1_ranking"]
    L = ["# Task 3 step 1: HySOR baseline reproduction", "",
         "All numbers come from `baseline/results.json`, written by `baseline.py` in the same run.", "",
         "## Method",
         f"- Fresh rerun of Phase 1 `rag_query_engine.hybrid_search(k={R['fetch_k']})` for all {f['queries_total']} gold queries: "
         f"score = {R['engine']['SIM_WEIGHT']}·cosine + {R['engine']['RECENCY_WEIGHT']}·recency, recency range "
         f"{R['engine']['RECENCY_MIN_YEAR']}–{R['engine']['RECENCY_MAX_YEAR']}, model `{R['engine']['embed_model']}`.",
         "- Scored with the published protocol, Phase 1 `bm25_baseline.compute_metrics` unchanged: relevance ≥ 1, "
         "AP over the top-20 list divided by all relevant labels of the query, binary nDCG, labels = `evaluation/gold_relevance.csv` as committed "
         "(Q001–Q020 manual, Q021–Q100 MiniLM cosine ≥ 0.52 auto-labels).",
         f"- The stored Phase 1 ranking (`evaluation/retrieval_results_enriched.csv`) is scored the same way. "
         f"Fresh and stored top-20 lists are identical on {R['fresh_vs_stored_identical_top20_lists']} / {f['queries_total']} queries.",
         f"- Target: P@5, MRR, MAP, nDCG@10 within ±{R['tolerance']} of the paper.", "",
         "## Result", "| Metric | Paper | Fresh rerun | Δ | Stored ranking | Δ |", "|---|---|---|---|---|---|"]
    for m in METRICS:
        L.append(f"| {m} | {R['paper'][m]:.3f} | {f['metrics'][m]:.4f} | {f['diff_vs_paper'][m]:+.4f} | "
                 f"{s['metrics'][m]:.4f} | {s['diff_vs_paper'][m]:+.4f} |")
    L += ["", f"**Reproduced within ±{R['tolerance']}: {'yes' if R['reproduced'] else 'NO'}** "
          f"({', '.join(f'{k} {v}' for k, v in f['within_tol_targets'].items())}).", "",
          "## Differences from the paper's description (not hidden by the match)",
          f"- Queries evaluated: {f['queries_evaluated']} of {f['queries_total']}, not 100. `compute_metrics` returns NaN for queries with no "
          f"relevance ≥ 1 label and the mean skips them: {', '.join(f['queries_skipped_no_relevant_label'])}.",
          f"- The paper says all systems return top-5; the published numbers come from top-{R['fetch_k']} lists (P@10 and MAP need more than 5).",
          "- Q021–Q100 labels are MiniLM cosine auto-labels from the same model as the FAISS index, so this baseline is circular by construction. "
          "Task 3 evaluates on the graded judgements from `phase2/gold_eval_rebuild` instead.",
          f"- `RECENCY_MAX_YEAR = datetime.now().year` ({R['engine']['RECENCY_MAX_YEAR']} in this run): the ranking depends on the run year."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
