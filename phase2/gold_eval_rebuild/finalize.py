#!/usr/bin/env python3
"""Phase 2 / gold_eval_rebuild, step 2: final metrics from the returned judgements.

Inputs (kept on disk):
  judgements/test_to_judge.csv      blind export written by run.py (identity reference)
  judgements/test_judged.csv        the same rows with `relevance` filled in, as returned by the user
  judgements/spotcheck_100_human.csv  user's own blind judgements of a 100-pair test sample
  splits/qrels_dev_phase1_manual.csv  Phase 1 manual dev labels
  pool/system_runs_top10.csv        bm25 / faiss / hybrid top-10 written by run.py

Also computes the HySOR RRF run (src/hysor_rrf.py, imported read-only).
Writes qrels/qrels_test.csv, pool/rrf_runs_top10.csv, final/results.json, final/REPORT.md.
Run from anywhere:  python phase2/gold_eval_rebuild/finalize.py --phase1 <path to uatie-edm-phase1>
"""

import hashlib
import importlib.util
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SEED = 42
K = 10
RRF_DEPTH = 50          # hysor_rrf.py default list depth fused per system
N_PERM = 10000          # paired randomisation test permutations
SYSTEMS = ["bm25", "faiss", "hybrid", "rrf"]
ID_COLS = ["pair_id", "query_id", "query_text", "information_need", "incident_id", "narrative"]

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root  # noqa: E402

REPO = phase1_root()  # Phase 1 checkout (uatie-edm-phase1), read-only
J = TASK / "judgements"
OUT = TASK / "final"
QRELS = TASK / "qrels"


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_judged():
    blind, judged = read(J / "test_to_judge.csv"), read(J / "test_judged.csv")
    errs = []
    if list(judged.columns) != list(blind.columns):
        errs.append(f"columns differ: {list(judged.columns)}")
    if len(judged) != len(blind):
        errs.append(f"row count {len(judged)} != {len(blind)}")
    else:
        for c in ID_COLS:
            n = int((judged[c] != blind[c]).sum())
            if n:
                errs.append(f"{n} rows changed in column {c}")
    bad = judged[~judged.relevance.isin(["0", "1", "2"])]
    if len(bad):
        errs.append(f"{len(bad)} rows with relevance not in {{0,1,2}}: {bad.pair_id.head().tolist()}")
    if judged.pair_id.duplicated().any():
        errs.append("duplicate pair_id")
    if errs:
        sys.exit("test_judged.csv failed validation:\n  " + "\n  ".join(errs))
    judged["relevance"] = judged.relevance.astype(int)
    return judged


def rrf_runs(queries):
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("hysor_rrf", REPO / "src" / "hysor_rrf.py")
    rrf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rrf)
    bm25, ids = rrf.build_bm25_index()
    rows = []
    for _, q in queries.iterrows():
        b = rrf.retrieve_bm25_ranked(bm25, ids, q.query_text, k=RRF_DEPTH)
        d = rrf.retrieve_dense_ranked(q.query_text, k=RRF_DEPTH)
        for r, inc in enumerate(rrf.rrf_fuse(b, d, k=K, rrf_k=rrf.RRF_K), 1):
            rows.append((q.query_id, "rrf", r, str(inc)))
    return pd.DataFrame(rows, columns=["query_id", "system", "rank", "incident_id"]), rrf.RRF_K


def per_query(runs, qrels, qids, thr):
    """Per-query P@5, P@10, MRR@10, AP@10, nDCG@10, judged@10. Unjudged = non-relevant."""
    lab = {(a, b): c for a, b, c in qrels[["query_id", "incident_id", "relevance"]].itertuples(index=False)}
    rows = []
    for sysname in SYSTEMS:
        sr = runs[runs.system == sysname]
        for qid in qids:
            ranked = sr[sr.query_id == qid].sort_values("rank").incident_id.tolist()[:K]
            g = [lab.get((qid, i)) for i in ranked]
            judged = sum(x is not None for x in g)
            g = [x or 0 for x in g]
            b = [x >= thr for x in g]
            allq = sorted((v for (qq, _), v in lab.items() if qq == qid), reverse=True)
            R = sum(v >= thr for v in allq)
            hits, ap = 0, 0.0
            for k, x in enumerate(b, 1):
                if x:
                    hits += 1
                    ap += hits / k
            ap = ap / min(R, K) if R else 0.0
            gain = lambda v: (2 ** v - 1) if v >= thr else 0
            dcg = sum(gain(v) / np.log2(k + 2) for k, v in enumerate(g))
            idcg = sum(gain(v) / np.log2(k + 2) for k, v in enumerate(allq[:K]))
            rows.append(dict(system=sysname, query_id=qid, R=R, judged_at_10=judged / K,
                             p5=np.mean(b[:5]), p10=np.mean(b[:K]),
                             mrr=next((1 / k for k, x in enumerate(b, 1) if x), 0.0),
                             map10=ap, ndcg10=dcg / idcg if idcg else 0.0))
    return pd.DataFrame(rows)


def summarise(pq):
    m = ["p5", "p10", "mrr", "map10", "ndcg10", "judged_at_10"]
    s = pq.groupby("system")[m].mean().reindex(SYSTEMS).round(4)
    return {k: {c: float(v[c]) for c in m} for k, v in s.iterrows()}


def randomisation(pq, metric, rng):
    out = {}
    for a, b in itertools.combinations(SYSTEMS, 2):
        x = pq[pq.system == a].set_index("query_id")[metric]
        y = pq[pq.system == b].set_index("query_id")[metric].reindex(x.index)
        d = (x - y).to_numpy()
        obs = abs(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
        p = (np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12) + 1) / (N_PERM + 1)
        out[f"{a}_vs_{b}"] = {"mean_diff": round(float(d.mean()), 4), "p_value": round(float(p), 4)}
    return out


def agreement(judged):
    h = read(J / "spotcheck_100_human.csv")
    m = h.merge(judged, on="pair_id", suffixes=("", "_j"))
    assert len(m) == len(h), "spot-check pairs missing from the test pool"
    for c in ["query_id", "incident_id", "narrative"]:
        assert (m[c] == m[c + "_j"]).all(), f"spot-check {c} differs from export"
    u, l = m.your_relevance.astype(int), m.relevance.astype(int)
    ub, lb = (u > 0).astype(int), (l > 0).astype(int)
    return {
        "n_pairs": len(m), "n_queries": int(m.query_id.nunique()),
        "exact_agreement": round(float((u == l).mean()), 4),
        "cohen_kappa": round(float(cohen_kappa_score(u, l)), 4),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(u, l, weights="quadratic")), 4),
        "binary_ge1_agreement": round(float((ub == lb).mean()), 4),
        "binary_ge1_kappa": round(float(cohen_kappa_score(ub, lb)), 4),
        "confusion_rows_user_cols_test_labels_0_1_2": confusion_matrix(u, l, labels=[0, 1, 2]).tolist(),
        "user_relevant_rate": round(float(ub.mean()), 4),
        "test_label_relevant_rate": round(float(lb.mean()), 4),
        "disagreements": int((u != l).sum()),
    }


def main():
    set_seeds()
    judged = validate_judged()
    notes = judged.judge_notes.str.split(",").str[0].value_counts().to_dict()
    QRELS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    qrels_test = judged[["query_id", "incident_id", "relevance"]].sort_values(["query_id", "incident_id"])
    qrels_test.to_csv(QRELS / "qrels_test.csv", index=False)
    qrels_dev = pd.read_csv(TASK / "splits" / "qrels_dev_phase1_manual.csv", dtype={"relevance": int})

    test_q = pd.read_csv(TASK / "splits" / "test_queries.csv")
    dev_q = pd.read_csv(TASK / "splits" / "dev_queries.csv")
    rrf, rrf_k = rrf_runs(pd.concat([dev_q, test_q]))
    rrf.to_csv(TASK / "pool" / "rrf_runs_top10.csv", index=False)
    runs = pd.read_csv(TASK / "pool" / "system_runs_top10.csv")[["query_id", "system", "rank", "incident_id"]]
    runs = pd.concat([runs, rrf], ignore_index=True)

    rng = np.random.default_rng(SEED)
    R = {"task": "gold_eval_rebuild/finalize", "seed": SEED, "k": K, "rrf_depth": RRF_DEPTH, "rrf_k": rrf_k,
         "n_permutations": N_PERM, "systems": SYSTEMS,
         "test_label_source": {"judge_notes_prefix_counts": notes,
                               "rows": len(judged),
                               "relevance_counts": {str(k): int(v) for k, v in judged.relevance.value_counts().sort_index().items()}},
         "spotcheck_agreement": agreement(judged)}
    cal = TASK / "llm_judge" / "calibration_results.json"
    if cal.exists():
        R["dev_calibration_reference"] = json.loads(cal.read_text(encoding="utf-8"))

    bucket = test_q.set_index("query_id").scenario_bucket
    for split, qids, qr in [("test", test_q.query_id.tolist(), qrels_test), ("dev", dev_q.query_id.tolist(), qrels_dev)]:
        for name, thr in [("relevant_ge1", 1), ("relevant_ge2", 2)]:
            pq = per_query(runs, qr, qids, thr)
            R.setdefault(split, {})[name] = {
                "metrics": summarise(pq),
                "queries_with_relevant": int((pq[pq.system == "bm25"].R > 0).sum()),
                "randomisation_ndcg10": randomisation(pq, "ndcg10", rng),
            }
            if split == "test" and thr == 1:
                pq.round(4).to_csv(OUT / "per_query_test_ge1.csv", index=False)
                pb = pq.assign(bucket=pq.query_id.map(bucket)).groupby(["bucket", "system"]).ndcg10.mean().round(4)
                R["test"]["ndcg10_by_bucket_ge1"] = {b: pb[b].reindex(SYSTEMS).to_dict() for b in pb.index.levels[0]}

    ins = [J / "test_to_judge.csv", J / "test_judged.csv", J / "spotcheck_100_human.csv",
           TASK / "splits" / "qrels_dev_phase1_manual.csv", TASK / "pool" / "system_runs_top10.csv"]
    R["phase1"] = phase1_info(REPO)
    R["input_sha256"] = {rel(p): sha256(p) for p in ins}
    R["output_sha256"] = {rel(p): sha256(p) for p in [QRELS / "qrels_test.csv", TASK / "pool" / "rrf_runs_top10.csv",
                                                       OUT / "per_query_test_ge1.csv"]}
    (OUT / "results.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    write_report(R)
    print(json.dumps({s: R["test"]["relevant_ge1"]["metrics"][s] for s in SYSTEMS}, indent=2))


def table(m):
    L = ["| System | P@5 | P@10 | MRR@10 | MAP@10 | nDCG@10 | judged@10 |", "|---|---|---|---|---|---|---|"]
    for s, v in m.items():
        L.append(f"| {s} | {v['p5']:.4f} | {v['p10']:.4f} | {v['mrr']:.4f} | {v['map10']:.4f} | {v['ndcg10']:.4f} | {v['judged_at_10']:.4f} |")
    return L


def sig(d):
    return [f"- {k.replace('_vs_', ' vs ')}: mean diff {v['mean_diff']:+.4f}, p = {v['p_value']}" for k, v in d.items()]


def write_report(R):
    a, t, d = R["spotcheck_agreement"], R["test"], R["dev"]
    src = R["test_label_source"]
    L = ["# gold_eval_rebuild: FINAL REPORT", "",
         "All numbers come from `final/results.json`, which `finalize.py` writes in the same run.", "",
         "## Label provenance (read first)",
         f"- Test qrels = `judgements/test_judged.csv` ({src['rows']} rows, relevance counts {src['relevance_counts']}), as returned by the user.",
         f"- Every row's `judge_notes` begins with: {src['judge_notes_prefix_counts']}. These are **LLM-generated labels**, not manual judgements.",
         f"- User's own blind spot-check: {a['n_pairs']} test pairs over {a['n_queries']} queries (`judgements/spotcheck_100_human.csv`). "
         f"Agreement with the test labels: exact {a['exact_agreement']}, Cohen's κ {a['cohen_kappa']}, quadratic-weighted κ {a['quadratic_weighted_kappa']}, "
         f"binary (≥1) agreement {a['binary_ge1_agreement']}, binary κ {a['binary_ge1_kappa']}.",
         f"- Confusion (rows = user, cols = test labels, order 0/1/2): {a['confusion_rows_user_cols_test_labels_0_1_2']}. "
         f"Relevant rate: user {a['user_relevant_rate']}, test labels {a['test_label_relevant_rate']}.",
         f"- On the {a['disagreements']} disagreeing pairs the user chose to keep the LLM label. The test qrels are therefore entirely LLM-judged.",
         ]
    if "dev_calibration_reference" in R:
        c = R["dev_calibration_reference"]
        L.append(f"- Separate dev calibration of an LLM judge (Devin, `llm_judge/`) against Phase 1 manual labels: κ {c['cohen_kappa']}, binary κ {c['binary_relevant_ge1_kappa']}.")
    L += ["- These labels are independent of MiniLM (the circularity this task removes), but they are not human gold. "
          "Report them as LLM-judged, with the agreement numbers above.", "",
          "## Method",
          f"- Systems: `bm25`, `faiss`, `hybrid` (top-10 from `run.py`, the same lists that built the pool) and `rrf` "
          f"(`src/hysor_rrf.py`, BM25 + dense top-{R['rrf_depth']} each, RRF k = {R['rrf_k']}, top-{R['k']}).",
          "- `rrf` was not in the pool, so some of its results are unjudged; `judged@10` gives coverage. Unjudged = non-relevant for every system.",
          "- P@k, MRR@10 and MAP@10 are binary. nDCG@10 uses gain 2^rel − 1, with the ideal ranking taken from all judged pairs for the query. "
          "MAP@10 divides by min(R, 10), where R = judged relevant pairs for the query.",
          f"- Main threshold relevance ≥ 1; strict sensitivity check relevance ≥ 2. Significance: two-sided paired randomisation test on per-query nDCG@10, "
          f"{R['n_permutations']} sign flips, seed {R['seed']}, no multiple-comparison correction.", "",
          f"## Test, Q021–Q100, relevance ≥ 1 ({t['relevant_ge1']['queries_with_relevant']}/80 queries have a relevant pair)"]
    L += table(t["relevant_ge1"]["metrics"]) + [""] + sig(t["relevant_ge1"]["randomisation_ndcg10"])
    L += ["", f"## Test, relevance ≥ 2 (strict; {t['relevant_ge2']['queries_with_relevant']}/80 queries have a relevant pair)"]
    L += table(t["relevant_ge2"]["metrics"]) + [""] + sig(t["relevant_ge2"]["randomisation_ndcg10"])
    L += ["", "## Test nDCG@10 by bucket (relevance ≥ 1)", "| Bucket | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    for b, v in t["ndcg10_by_bucket_ge1"].items():
        L.append(f"| {b} | " + " | ".join(f"{v[s]:.4f}" for s in SYSTEMS) + " |")
    L += ["", f"## Dev, Q001–Q020, Phase 1 manual labels, relevance ≥ 1 ({d['relevant_ge1']['queries_with_relevant']}/20 queries have a relevant pair)",
          "Manual labels cover 362 of the 457 dev pool pairs; the 95 in `judgements/dev_gap_to_judge.csv` are unjudged and count as non-relevant."]
    L += table(d["relevant_ge1"]["metrics"]) + [""] + sig(d["relevant_ge1"]["randomisation_ndcg10"])
    L += ["", "## Failures / caveats",
          "- Test labels are LLM-judged and agree only moderately with the user's blind spot-check (see provenance). "
          "Absolute values are inflated relative to the user's stricter judging (relevant rate on the spot-check sample: test labels "
          f"{a['test_label_relevant_rate']} vs user {a['user_relevant_rate']}).",
          "- Dev is a single bucket (`runway_change`) and its labels are human; dev and test numbers are not directly comparable.",
          "- `rrf` judged@10 < 1 biases it downward relative to the pooled systems.",
          "- Nothing was tuned on test."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
