#!/usr/bin/env python3
"""Task 3: run the six retrieval systems on all 100 gold queries and export every top-10 pair that has no
judgement yet. Metrics are NOT computed here (unjudged pairs must be judged first; see REPORT.md).

Systems (top-10 kept for evaluation):
  1 bm25        Phase 1 bm25_baseline.bm25_retrieve over asrs_faiss_meta.summary
  2 faiss       raw FAISS inner product, Phase 1 index + all-MiniLM-L6-v2
  3 hybrid      Phase 1 rag_query_engine.hybrid_search(k=20)
  4 rrf         hysor_rrf.rrf_fuse of systems 1 and 2 at depth 50, k = 60
  5 ce          union of bm25 top-50 and faiss top-50 re-ranked by cross-encoder/ms-marco-MiniLM-L-6-v2
  6 ce_recency  system 5 with the Phase 1 composite: 0.65 * sigmoid(CE logit) + 0.35 * recency (weights untuned)

Run:  python phase2/retrieval/run.py --phase1 <path to uatie-edm-phase1>
"""
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root, sha256  # noqa: E402

REPO = phase1_root()
GOLD = ROOT / "phase2" / "gold_eval_rebuild"
SEED = 42
K = 10
DEPTH = 50
HYBRID_FETCH_K = 20
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
SYSTEMS = ["bm25", "faiss", "hybrid", "rrf", "ce", "ce_recency"]


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def judged_pairs():
    test = pd.read_csv(GOLD / "judgements" / "test_judged.csv")
    dev = pd.read_csv(GOLD / "splits" / "qrels_dev_phase1_manual.csv")
    return set(zip(test.query_id, test.incident_id.astype(str))) | set(zip(dev.query_id, dev.incident_id.astype(str)))


def main():
    set_seeds()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    rag = load_module("rag_query_engine", REPO / "src" / "rag" / "rag_query_engine.py")
    bm25_mod = load_module("bm25_baseline", REPO / "src" / "rag" / "bm25_baseline.py")
    rrf = load_module("hysor_rrf", REPO / "src" / "hysor_rrf.py")
    rag.initialize_resources(verbose=False)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_MODEL, device="cpu", revision=CE_REVISION)

    meta = pd.read_csv(REPO / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["incident_id"] = meta["incident_id"].astype(str)
    text = meta.set_index("incident_id")["summary"].astype(str)
    pos = {i: p for p, i in enumerate(meta.incident_id)}
    bm25, _ = bm25_mod.build_bm25(meta, "summary")
    q = pd.read_csv(REPO / "evaluation" / "gold_queries.csv")

    def year_month(inc):
        ym = rag._date_lookup.get(inc) or rag._date_lookup.get(str(pos.get(inc, -1)))
        return ym if ym else (None, None)

    runs, lat, depth50 = [], [], []
    ce.predict([("warm up", "warm up")])
    rag.hybrid_search("warm up", k=HYBRID_FETCH_K)
    for n, r in enumerate(q.itertuples(), 1):
        qid, qt = r.query_id, str(r.query_text)
        t0 = time.perf_counter()
        b = [(str(x["incident_id"]), float(next(v for k, v in x.items() if "score" in k)))
             for x in bm25_mod.bm25_retrieve(bm25, qt, meta, "incident_id", DEPTH)]
        t1 = time.perf_counter()
        sims, idx = rag.index.search(rag.embed_query(qt), DEPTH)
        d = [(str(meta.incident_id.iat[i]), float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0]
        t2 = time.perf_counter()
        h = [str(x["incident_id"]) for x in rag.hybrid_search(qt, k=HYBRID_FETCH_K)][:K]
        t3 = time.perf_counter()
        f = [str(i) for i in rrf.rrf_fuse(b, d, k=K, rrf_k=rrf.RRF_K)]
        t4 = time.perf_counter()
        union = list(dict.fromkeys([i for i, _ in b] + [i for i, _ in d]))
        logits = ce.predict([(qt, text[i]) for i in union], batch_size=32, show_progress_bar=False)
        order = sorted(range(len(union)), key=lambda j: (-float(logits[j]), j))
        c = [union[j] for j in order]
        t5 = time.perf_counter()
        comp = [rag.compute_composite_score(1 / (1 + np.exp(-float(logits[j]))), *year_month(union[j]))
                for j in range(len(union))]
        cr = [union[j] for j in sorted(range(len(union)), key=lambda j: (-comp[j], j))]
        t6 = time.perf_counter()
        lists = {"bm25": [i for i, _ in b][:K], "faiss": [i for i, _ in d][:K], "hybrid": h, "rrf": f,
                 "ce": c[:K], "ce_recency": cr[:K]}
        for s, ids in lists.items():
            runs += [(qid, s, k, i) for k, i in enumerate(ids, 1)]
        ms = lambda a, z: round((z - a) * 1000, 2)
        lat.append({"query_id": qid, "bm25": ms(t0, t1), "faiss": ms(t1, t2), "hybrid": ms(t2, t3),
                    "rrf": ms(t0, t2) + ms(t3, t4), "ce": ms(t0, t2) + ms(t4, t5),
                    "ce_recency": ms(t0, t2) + ms(t4, t6), "union_size": len(union)})
        print(f"[{n}/{len(q)}] {qid} union={len(union)}", flush=True)

    out = TASK / "runs"
    out.mkdir(exist_ok=True)
    runs = pd.DataFrame(runs, columns=["query_id", "system", "rank", "incident_id"])
    runs.to_csv(out / "runs_top10.csv", index=False)
    lat = pd.DataFrame(lat)
    lat.to_csv(out / "latency_ms.csv", index=False)

    pool = pd.read_csv(GOLD / "pool" / "system_runs_top10.csv", dtype={"incident_id": str})
    same_as_pool = {s: int(sum(runs[(runs.system == s) & (runs.query_id == x)].incident_id.tolist()
                               == pool[(pool.system == s) & (pool.query_id == x)].sort_values("rank").incident_id.tolist()
                               for x in q.query_id)) for s in ["bm25", "faiss", "hybrid"]}

    J = judged_pairs()
    runs["judged"] = [(a, b) in J for a, b in zip(runs.query_id, runs.incident_id)]
    cov = runs.groupby("system").judged.mean().round(4).reindex(SYSTEMS).to_dict()
    un = runs[~runs.judged]
    prov = (un.groupby(["query_id", "incident_id"])
              .agg(systems=("system", lambda s: ";".join(sorted(set(s)))), best_rank=("rank", "min")).reset_index())
    jd = TASK / "judgements"
    jd.mkdir(exist_ok=True)
    prov.to_csv(jd / "unjudged_provenance.csv", index=False)
    rng = np.random.default_rng(SEED)
    qinfo = q.set_index("query_id")
    parts = [prov[prov.query_id == x].sort_values("incident_id") for x in sorted(prov.query_id.unique())]
    parts = [p.iloc[rng.permutation(len(p))] for p in parts]
    df = pd.concat(parts, ignore_index=True)
    blind = pd.DataFrame({"pair_id": [f"R{i:05d}" for i in range(1, len(df) + 1)], "query_id": df.query_id.values,
                          "query_text": qinfo.loc[df.query_id, "query_text"].values,
                          "information_need": qinfo.loc[df.query_id, "relevance_notes"].values,
                          "incident_id": df.incident_id.values, "narrative": text.loc[df.incident_id].values,
                          "relevance": "", "judge_notes": ""})
    blind.to_csv(jd / "to_judge.csv", index=False)

    dev_ids = set(pd.read_csv(GOLD / "splits" / "dev_queries.csv").query_id)
    R = {"task": "retrieval", "status": "STOPPED: unjudged pairs exported, waiting for judgements",
         "seed": SEED, "k": K, "depth": DEPTH, "systems": SYSTEMS,
         "cross_encoder": {"model": CE_MODEL, "revision": CE_REVISION},
         "recency_formula": f"{rag.SIM_WEIGHT}*sigmoid(ce_logit) + {rag.RECENCY_WEIGHT}*recency({rag.RECENCY_MIN_YEAR}-{rag.RECENCY_MAX_YEAR})",
         "systems_1_3_identical_to_gold_eval_pool_queries": same_as_pool,
         "judged_at_10": cov,
         "unjudged": {"pairs": int(len(prov)), "queries": int(prov.query_id.nunique()),
                      "dev_pairs": int(prov.query_id.isin(dev_ids).sum()),
                      "test_pairs": int((~prov.query_id.isin(dev_ids)).sum()),
                      "by_system_rows": un.system.value_counts().reindex(SYSTEMS, fill_value=0).astype(int).to_dict()},
         "latency_ms_this_vm": {s: {"p50": round(float(lat[s].quantile(.5)), 1), "p95": round(float(lat[s].quantile(.95)), 1)}
                                for s in SYSTEMS},
         "env": {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                 "cpu_count": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__},
         "phase1": phase1_info(REPO),
         "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                           [GOLD / "judgements" / "test_judged.csv", GOLD / "splits" / "qrels_dev_phase1_manual.csv",
                            GOLD / "pool" / "system_runs_top10.csv"]},
         "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                           [out / "runs_top10.csv", jd / "unjudged_provenance.csv", jd / "to_judge.csv"]}}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({k: R[k] for k in ["systems_1_3_identical_to_gold_eval_pool_queries", "judged_at_10", "unjudged",
                                         "latency_ms_this_vm"]}, indent=2))


def write_report(R):
    u = R["unjudged"]
    L = ["# Task 3: Retrieval (STOPPED, waiting for judgements)", "",
         "All numbers come from `results.json`, written by `run.py` in the same run. "
         "No retrieval metrics are reported yet: the systems retrieve pairs that were never judged, "
         "and the plan forbids treating them as non-relevant.", "",
         "## Step 1: baseline reproduction (done)",
         "See `baseline/REPORT.md`. The paper's HySOR numbers reproduce within ±0.01 under the published protocol.", "",
         "## Step 2: systems",
         "| # | System | Definition |", "|---|---|---|",
         "| 1 | bm25 | Phase 1 `bm25_baseline.bm25_retrieve` over `asrs_faiss_meta.summary` |",
         "| 2 | faiss | raw FAISS inner product, Phase 1 index, all-MiniLM-L6-v2 |",
         f"| 3 | hybrid | Phase 1 `rag_query_engine.hybrid_search(k=20)`, top-{R['k']} |",
         f"| 4 | rrf | `hysor_rrf.rrf_fuse` of systems 1 and 2 at depth {R['depth']}, k = 60 |",
         f"| 5 | ce | union of bm25 top-{R['depth']} and faiss top-{R['depth']} re-ranked by `{R['cross_encoder']['model']}` |",
         f"| 6 | ce_recency | system 5 scored as `{R['recency_formula']}`, Phase 1 weights, not tuned |", "",
         "Systems 1–3 return the same top-10 lists as the `gold_eval_rebuild` pool on "
         + ", ".join(f"{s} {n}/100" for s, n in R["systems_1_3_identical_to_gold_eval_pool_queries"].items()) + " queries.", "",
         "## Judgement coverage of the top-10 (test qrels + Phase 1 manual dev labels)",
         "| System | judged@10 | unjudged rows |", "|---|---|---|"]
    L += [f"| {s} | {R['judged_at_10'][s]:.4f} | {u['by_system_rows'][s]} |" for s in R["systems"]]
    L += ["", f"Unique unjudged (query, incident) pairs: **{u['pairs']}** over {u['queries']} queries "
          f"({u['test_pairs']} test, {u['dev_pairs']} dev). They include the 95 dev-gap BM25 pairs from `gold_eval_rebuild` "
          "that were never returned.", "",
          "## Latency on this VM (ms per query, CPU)",
          f"`{R['env']['cpu']}`, {R['env']['cpu_count']} cores. For rrf/ce/ce_recency the time includes the bm25 and faiss retrieval they need.", "",
          "| System | p50 | p95 |", "|---|---|---|"]
    L += [f"| {s} | {v['p50']} | {v['p95']} |" for s, v in R["latency_ms_this_vm"].items()]
    L += ["", "## What is needed to continue",
          "Fill `relevance` (0/1/2, same guideline as `phase2/gold_eval_rebuild/judgements/`) in `judgements/to_judge.csv` "
          "and return it as `judgements/judged.csv`. It is blind: no system names, ranks or scores.",
          "", "## Failures / caveats",
          "- Latency is measured on this VM; it will differ on other hardware.",
          "- The test qrels these systems are scored against are LLM-judged (see `phase2/gold_eval_rebuild/final/REPORT.md`)."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
