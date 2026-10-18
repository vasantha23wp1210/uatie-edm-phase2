#!/usr/bin/env python3
"""Task 3, step 3: score the six systems once every top-10 pair is judged.

Inputs:  runs/runs_top10.csv, runs/latency_ms.csv            (run.py)
         judgements/to_judge.csv  (blind export, identity reference) and judgements/judged.csv (same rows, filled)
         ../gold_eval_rebuild/qrels/qrels_test.csv, ../gold_eval_rebuild/splits/qrels_dev_phase1_manual.csv
         baseline/results.json                                (paper protocol rerun)
Writes:  eval/qrels_dev.csv, eval/qrels_test.csv, eval/per_query.csv, eval/error_analysis.csv,
         eval/results.json, eval/REPORT.md
The best system is chosen on dev (nDCG@10, rel >= 1); test is only scored, never used for any choice.
Run:  python phase2/retrieval/evaluate.py --phase1 <path to uatie-edm-phase1>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root, sha256  # noqa: E402

REPO = phase1_root()
GOLD = ROOT / "phase2" / "gold_eval_rebuild"
OUT = TASK / "eval"
SEED, K, N_PERM = 42, 10, 10000
SYSTEMS = ["bm25", "faiss", "hybrid", "rrf", "ce", "ce_recency"]
PRIMARY = "ce"  # fixed in advance by the Phase 2 plan ("BM25 + cross-encoder re-ranking"), not chosen on dev or test
PLAN_TARGETS = {"p5": 0.80, "map10": 0.80}
BASELINE = "hybrid"
METRICS = ["p1", "p3", "p5", "p10", "mrr", "map10", "recall10", "ndcg5", "ndcg10"]
ID_COLS = ["pair_id", "query_id", "query_text", "information_need", "incident_id", "narrative"]
ERROR_QUERIES = ["Q025", "Q026", "Q030", "Q036", "Q055", "Q075", "Q092"]


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_judged():
    ref, got = read(TASK / "judgements" / "to_judge.csv"), read(TASK / "judgements" / "judged.csv")
    assert len(got) == len(ref), f"judged.csv has {len(got)} rows, expected {len(ref)}"
    assert got.pair_id.is_unique, "duplicate pair_id in judged.csv"
    m = ref[ID_COLS].merge(got, on="pair_id", how="left", suffixes=("", "_j"))
    for c in ID_COLS[1:]:
        bad = m[m[c] != m[c + "_j"]].pair_id.tolist()
        assert not bad, f"{c} changed for {bad[:5]}"
    bad = m[~m.relevance.isin(["0", "1", "2"])].pair_id.tolist()
    assert not bad, f"relevance must be 0/1/2: {bad[:5]}"
    m["relevance"] = m.relevance.astype(int)
    return m


def gains(v, thr):
    return (2 ** v - 1) if v >= thr else 0


def per_query(runs, qrels, qids, thr):
    lab = {(a, b): int(c) for a, b, c in qrels[["query_id", "incident_id", "relevance"]].itertuples(index=False)}
    byq = {}
    for (q, _), v in lab.items():
        byq.setdefault(q, []).append(v)
    rows = []
    for s in SYSTEMS:
        sr = runs[runs.system == s]
        for q in qids:
            allq = sorted(byq.get(q, []), reverse=True)
            R = sum(v >= thr for v in allq)
            if R == 0:
                continue
            ranked = sr[sr.query_id == q].sort_values("rank").incident_id.tolist()[:K]
            g = [lab.get((q, i)) for i in ranked]
            unj = sum(x is None for x in g)
            g = [x or 0 for x in g]
            b = [x >= thr for x in g]
            hits, ap = 0, 0.0
            for k, x in enumerate(b, 1):
                if x:
                    hits += 1
                    ap += hits / k
            dcg = lambda xs, n: sum(gains(v, thr) / np.log2(k + 2) for k, v in enumerate(xs[:n]))
            rows.append(dict(system=s, query_id=q, R=R, unjudged_at_10=unj,
                             p1=float(b[0]), p3=np.mean(b[:3]), p5=np.mean(b[:5]), p10=np.mean(b[:K]),
                             mrr=next((1 / k for k, x in enumerate(b, 1) if x), 0.0),
                             map10=ap / min(R, K), recall10=sum(b) / R,
                             ndcg5=dcg(g, 5) / dcg(allq, 5), ndcg10=dcg(g, K) / dcg(allq, K)))
    return pd.DataFrame(rows)


def summarise(pq):
    s = pq.groupby("system")[METRICS].mean().reindex(SYSTEMS).round(4)
    out = {k: {c: float(v[c]) for c in METRICS} for k, v in s.iterrows()}
    for k in out:
        out[k]["n_queries"] = int((pq.system == k).sum())
    return out


def randomisation(pq, a, b, metric, rng):
    x = pq[pq.system == a].set_index("query_id")[metric]
    y = pq[pq.system == b].set_index("query_id")[metric].reindex(x.index)
    d = (x - y).to_numpy()
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    p = (np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12) + 1) / (N_PERM + 1)
    return {"mean_diff": round(float(d.mean()), 4), "p_value": round(float(p), 4)}


def main():
    judged = validate_judged()
    runs = pd.read_csv(TASK / "runs" / "runs_top10.csv")
    lat = pd.read_csv(TASK / "runs" / "latency_ms.csv")
    dev_q = pd.read_csv(GOLD / "splits" / "dev_queries.csv")
    test_q = pd.read_csv(GOLD / "splits" / "test_queries.csv")
    bucket = pd.concat([dev_q, test_q]).set_index("query_id").scenario_bucket
    new = judged[["query_id", "incident_id", "relevance"]]
    qd = pd.concat([pd.read_csv(GOLD / "splits" / "qrels_dev_phase1_manual.csv").assign(source="phase1_manual"),
                    new[new.query_id.isin(dev_q.query_id)].assign(source="task3_judged")])
    qt = pd.concat([pd.read_csv(GOLD / "qrels" / "qrels_test.csv").assign(source="gold_eval_rebuild"),
                    new[new.query_id.isin(test_q.query_id)].assign(source="task3_judged")])
    for name, q in [("dev", qd), ("test", qt)]:
        dup = q.duplicated(["query_id", "incident_id"]).sum()
        assert dup == 0, f"{name}: {dup} pairs judged twice"
    OUT.mkdir(exist_ok=True)
    qd.to_csv(OUT / "qrels_dev.csv", index=False)
    qt.to_csv(OUT / "qrels_test.csv", index=False)

    rng = np.random.default_rng(SEED)
    R = {"task": "retrieval/eval", "seed": SEED, "k": K, "n_perm": N_PERM, "baseline_system": BASELINE,
         "label_provenance": {
             "task3_judged_notes": judged.judge_notes.value_counts().to_dict(),
             "dev_qrels_rows": qd.source.value_counts().to_dict(),
             "test_qrels_rows": qt.source.value_counts().to_dict()},
         "metrics": {}, "per_bucket_test_ge1": {}, "significance_test_vs_baseline": {}}
    pqs = []
    for split, qids, qr in [("dev", dev_q.query_id, qd), ("test", test_q.query_id, qt)]:
        for thr in (1, 2):
            pq = per_query(runs, qr, qids, thr).assign(split=split, thr=thr)
            assert pq.unjudged_at_10.sum() == 0, f"{split} thr{thr}: unjudged pairs remain"
            pqs.append(pq)
            R["metrics"][f"{split}_rel_ge{thr}"] = summarise(pq)
    pq_all = pd.concat(pqs)
    pq_all.round(4).to_csv(OUT / "per_query.csv", index=False)
    best = max(SYSTEMS, key=lambda s: R["metrics"]["dev_rel_ge1"][s]["ndcg10"])
    R["best_system_chosen_on_dev_ndcg10"] = best
    R["primary_system"] = PRIMARY
    R["primary_system_rule"] = ("fixed in advance by the Phase 2 plan (BM25 + cross-encoder re-ranking); "
                                "the dev-selected system is reported alongside")
    tp = R["metrics"]["test_rel_ge1"][PRIMARY]
    R["target_primary_plan"] = {m: {"value": tp[m], "target": t, "met": tp[m] >= t} for m, t in PLAN_TARGETS.items()}

    t1 = pq_all[(pq_all.split == "test") & (pq_all.thr == 1)]
    for thr in (1, 2):
        t = pq_all[(pq_all.split == "test") & (pq_all.thr == thr)]
        R["significance_test_vs_baseline"][f"rel_ge{thr}"] = {
            f"{s}_vs_{BASELINE}": {m: randomisation(t, s, BASELINE, m, rng) for m in ["p5", "map10", "ndcg10"]}
            for s in SYSTEMS if s != BASELINE}
    t1b = t1.assign(bucket=t1.query_id.map(bucket))
    R["per_bucket_test_ge1"] = {
        b: {s: round(float(g[g.system == s].ndcg10.mean()), 4) for s in SYSTEMS} | {"n": int(g.query_id.nunique())}
        for b, g in t1b.groupby("bucket")}
    tb, tbase = R["metrics"]["test_rel_ge1"][best], R["metrics"]["test_rel_ge1"][BASELINE]
    R["target_best_ge_baseline_p5_map_ndcg10"] = {
        m: {"best": tb[m], "baseline": tbase[m], "met": tb[m] >= tbase[m]} for m in ["p5", "map10", "ndcg10"]}
    R["latency_ms_this_vm"] = {s: {"p50": round(float(lat[s].quantile(.5)), 1),
                                   "p95": round(float(lat[s].quantile(.95)), 1)} for s in SYSTEMS}
    base = json.loads((TASK / "baseline" / "results.json").read_text())
    R["paper_protocol_side_by_side"] = {"paper": base["paper"], "published_protocol_rerun": base["fresh_rerun"]["metrics"],
                                        "published_protocol_queries": base["fresh_rerun"]["queries_evaluated"]}

    meta = pd.read_csv(REPO / "data" / "processed" / "asrs_faiss_meta.csv", dtype=str, keep_default_na=False)
    tcol = "summary" if "summary" in meta.columns else meta.columns[-1]
    text = meta.drop_duplicates("incident_id").set_index("incident_id")[tcol] if "incident_id" in meta.columns else None
    bq = t1[t1.system == best].set_index("query_id")
    extra = [q for q in bq.sort_values(["ndcg10", "query_id"]).index if q not in ERROR_QUERIES][:10 - len(ERROR_QUERIES)]
    lab = {(a, b): int(c) for a, b, c in qt[["query_id", "incident_id", "relevance"]].itertuples(index=False)}
    rows = []
    for q in ERROR_QUERIES + extra:
        ranked = runs[(runs.system == best) & (runs.query_id == q)].sort_values("rank").incident_id.tolist()[:5]
        for r, i in enumerate(ranked, 1):
            rows.append(dict(query_id=q, query_text=test_q.set_index("query_id").query_text[q],
                             bucket=bucket[q], R_ge1=int(bq.R[q]) if q in bq.index else 0,
                             ndcg10=round(float(bq.ndcg10[q]), 4) if q in bq.index else None,
                             rank=r, incident_id=i, relevance=lab.get((q, i)),
                             summary=str(text.get(i, ""))[:200] if text is not None else ""))
    pd.DataFrame(rows).to_csv(OUT / "error_analysis.csv", index=False)
    R["error_analysis_queries"] = ERROR_QUERIES + extra
    R["phase1"] = phase1_info(REPO)
    R["inputs_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [
        TASK / "judgements" / "judged.csv", TASK / "runs" / "runs_top10.csv",
        GOLD / "qrels" / "qrels_test.csv", GOLD / "splits" / "qrels_dev_phase1_manual.csv"]}
    R["output_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [
        OUT / "per_query.csv", OUT / "error_analysis.csv", OUT / "qrels_dev.csv", OUT / "qrels_test.csv"]}
    (OUT / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({k: R[k] for k in ["primary_system", "target_primary_plan", "best_system_chosen_on_dev_ndcg10", "target_best_ge_baseline_p5_map_ndcg10",
                                         "label_provenance"]}, indent=2))


NAMES = {"p1": "P@1", "p3": "P@3", "p5": "P@5", "p10": "P@10", "mrr": "MRR@10", "map10": "MAP@10",
         "recall10": "R@10", "ndcg5": "nDCG@5", "ndcg10": "nDCG@10"}


def table(m):
    L = ["| System | n | " + " | ".join(NAMES[c] for c in METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
    for s, v in m.items():
        L.append(f"| {s} | {v['n_queries']} | " + " | ".join(f"{v[c]:.4f}" for c in METRICS) + " |")
    return L


def write_report(R):
    prov = R["label_provenance"]
    L = ["# Task 3 retrieval: evaluation", "",
         "All numbers come from `eval/results.json`, written by `evaluate.py` in the same run.", "",
         "## Label provenance (read first)", "",
         f"- Task 3 pairs (`judgements/judged.csv`): {prov['task3_judged_notes']}.",
         f"- Test qrels rows: {prov['test_qrels_rows']}. The `gold_eval_rebuild` test labels are LLM-judged (Claude).",
         f"- Dev qrels rows: {prov['dev_qrels_rows']}.",
         "- Any label marked `pending user review` makes the scores below **provisional**. Replace "
         "`judgements/judged.csv` with reviewed labels and rerun this script; no code change is needed.", "",
         "Every top-10 result of every system is judged (no unjudged pair is counted as non-relevant). "
         "Queries with no relevant label at the given threshold are left out; `n` is the number scored. "
         "MAP@10 normalises by min(R, 10). nDCG gain is 2^rel - 1 for labels at or above the threshold.", ""]
    for key, title in [("test_rel_ge1", "Test (Q021-Q100), relevant = rel >= 1"),
                       ("test_rel_ge2", "Test, relevant = rel 2"),
                       ("dev_rel_ge1", "Dev (Q001-Q020), relevant = rel >= 1"), ("dev_rel_ge2", "Dev, relevant = rel 2")]:
        L += [f"## {title}", ""] + table(R["metrics"][key]) + [""]
    pr = R["primary_system"]
    L += ["## Primary Phase 2 system and plan targets (test, rel >= 1)", "",
          f"Primary system: `{pr}`, {R['primary_system_rule']}. `ce` re-ranks the union of BM25 top-{50} and FAISS top-{50}, "
          "so its candidate pool includes BM25. MAP is MAP@10.", "",
          "| Metric | value | target | met |", "|---|---|---|---|"]
    for m, v in R["target_primary_plan"].items():
        L.append(f"| {NAMES[m]} | {v['value']:.4f} | >= {v['target']:.2f} | {'yes' if v['met'] else 'NO'} |")
    dv = R["metrics"]["dev_rel_ge1"]
    L += ["", f"On dev, `{pr}` scores P@5 {dv[pr]['p5']:.4f} / MAP@10 {dv[pr]['map10']:.4f}, below `faiss` "
          f"({dv['faiss']['p5']:.4f} / {dv['faiss']['map10']:.4f}). Most dev labels are Phase 1 manual judgements of "
          "Phase 1 hybrid/FAISS output, which favours those systems; this is why the dev rule picks a different system.", ""]
    best = R["best_system_chosen_on_dev_ndcg10"]
    L += ["## Target: best Phase 2 system >= Phase 1 baseline (test, rel >= 1)", "",
          f"Best system chosen on **dev** nDCG@10: `{best}`. Baseline: `{R['baseline_system']}` (Phase 1 hybrid engine).", "",
          "| Metric | best | baseline | met |", "|---|---|---|---|"]
    for m, v in R["target_best_ge_baseline_p5_map_ndcg10"].items():
        L.append(f"| {NAMES[m]} | {v['best']:.4f} | {v['baseline']:.4f} | {'yes' if v['met'] else 'NO'} |")
    L += ["", f"## Paired randomisation test vs `{R['baseline_system']}` (test, two-sided, {R['n_perm']} permutations, "
          "no multiple-comparison correction)", "", "| System | thr | P@5 diff (p) | MAP@10 diff (p) | nDCG@10 diff (p) |",
          "|---|---|---|---|---|"]
    for thr, d in R["significance_test_vs_baseline"].items():
        for k, v in d.items():
            L.append(f"| {k.split('_vs_')[0]} | {thr} | " + " | ".join(
                f"{v[m]['mean_diff']:+.4f} ({v[m]['p_value']})" for m in ["p5", "map10", "ndcg10"]) + " |")
    L += ["", "## nDCG@10 per scenario bucket (test, rel >= 1)", "",
          "| Bucket | n | " + " | ".join(SYSTEMS) + " |", "|---" * (len(SYSTEMS) + 2) + "|"]
    for b, v in R["per_bucket_test_ge1"].items():
        L.append(f"| {b} | {v['n']} | " + " | ".join(f"{v[s]:.4f}" for s in SYSTEMS) + " |")
    L += ["", "## Latency per query on this VM (ms, retrieval only)", "", "| System | p50 | p95 |", "|---|---|---|"]
    for s, v in R["latency_ms_this_vm"].items():
        L.append(f"| {s} | {v['p50']} | {v['p95']} |")
    side = R["paper_protocol_side_by_side"]
    L += ["", "## Paper protocol vs this protocol (Phase 1 hybrid)", "",
          f"Published protocol: {side['published_protocol_queries']} queries, top-20 lists, MiniLM-generated labels for "
          "Q021-Q100 (circular). This protocol: unbiased pooled labels, top-10, test queries only.", "",
          "| Metric | Paper | Published-protocol rerun | This protocol (test, rel >= 1) |", "|---|---|---|---|"]
    h = R["metrics"]["test_rel_ge1"]["hybrid"]
    for pk, mk in [("P@1", "p1"), ("P@3", "p3"), ("P@5", "p5"), ("P@10", "p10"), ("nDCG@5", "ndcg5"),
                   ("nDCG@10", "ndcg10"), ("MRR", "mrr"), ("MAP", "map10")]:
        L.append(f"| {pk} | {side['paper'][pk]} | {side['published_protocol_rerun'][pk]} | {h[mk]:.4f} |")
    L += ["", "## Error analysis", "",
          f"`eval/error_analysis.csv`: top-5 of `{best}` with labels for {', '.join(R['error_analysis_queries'])} "
          "(the seven named in the plan plus the lowest-nDCG@10 other test queries).", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
