#!/usr/bin/env python3
"""Phase 2 / gold_eval_rebuild.

Rebuilds the gold evaluation pool without the Phase 1 biases:
  * Q021-Q100 auto labels (MiniLM cosine, auto_label_relavance.py) are discarded.
  * Every system contributes the same pool depth (POOL_DEPTH) per query.
  * Judgement CSVs are blind (no system / rank / score) and shuffled with a fixed seed.

Run from anywhere:  python phase2/gold_eval_rebuild/run.py --phase1 <path to uatie-edm-phase1>
Phase 1 code is imported read-only from that checkout; nothing in it is modified.
"""

import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
POOL_DEPTH = 10            # top-k taken from every system for the pool
HYBRID_FETCH_K = 20        # Phase 1 eval called hybrid_search(k=20) (build_retrieval_results_v2.TOP_N)
DEV_IDS = [f"Q{i:03d}" for i in range(1, 21)]
TEST_IDS = [f"Q{i:03d}" for i in range(21, 101)]
SYSTEMS = ["bm25", "faiss", "hybrid"]

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root  # noqa: E402

REPO = phase1_root()  # Phase 1 checkout (uatie-edm-phase1), read-only
SPLITS = TASK_DIR / "splits"
JUDGE = TASK_DIR / "judgements"
POOL = TASK_DIR / "pool"

GOLD_Q = REPO / "evaluation" / "gold_queries.csv"
GOLD_REL = REPO / "evaluation" / "gold_relevance.csv"
P1_HYBRID = REPO / "evaluation" / "retrieval_results.csv"
P1_BM25 = REPO / "evaluation" / "bm25_results.csv"
META = REPO / "data" / "processed" / "asrs_faiss_meta.csv"
INDEX = REPO / "data" / "processed" / "asrs_faiss.index"


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_splits(q):
    SPLITS.mkdir(parents=True, exist_ok=True)
    assert set(DEV_IDS) | set(TEST_IDS) == set(q.query_id), "query ids do not match Q001-Q100"
    dev = q[q.query_id.isin(DEV_IDS)]
    test = q[q.query_id.isin(TEST_IDS)]
    dev.to_csv(SPLITS / "dev_queries.csv", index=False)
    test.to_csv(SPLITS / "test_queries.csv", index=False)

    rel = pd.read_csv(GOLD_REL, usecols=["query_id", "incident_id", "relevance"])
    qrels_dev = (rel[rel.query_id.isin(DEV_IDS)]
                 .astype({"relevance": int})
                 .sort_values(["query_id", "incident_id"]))
    qrels_dev.to_csv(SPLITS / "qrels_dev_phase1_manual.csv", index=False)
    discarded = rel[rel.query_id.isin(TEST_IDS)]
    return dev, test, qrels_dev, discarded


def run_systems(q):
    """Return long DataFrame: query_id, system, rank, incident_id, score."""
    os.chdir(REPO)  # Phase 1 modules use repo-relative paths
    rag = load_module("rag_query_engine", REPO / "src" / "rag" / "rag_query_engine.py")
    bm25_mod = load_module("bm25_baseline", REPO / "src" / "rag" / "bm25_baseline.py")

    rag.initialize_resources(verbose=False)
    if not rag.SEMANTIC_OK:
        raise RuntimeError("Phase 1 engine failed to load FAISS/embedding model")
    meta = pd.read_csv(META, low_memory=False)
    ids = meta["incident_id"].astype(str).to_numpy()
    bm25, _ = bm25_mod.build_bm25(meta, "summary")

    rows = []
    for _, r in q.iterrows():
        qid, text = r.query_id, str(r.query_text)
        for res in bm25_mod.bm25_retrieve(bm25, text, meta, "incident_id", POOL_DEPTH):
            rows.append((qid, "bm25", res["rank"], res["incident_id"], res["score"]))
        sims, idx = rag.index.search(rag.embed_query(text), POOL_DEPTH)
        for rank, (i, s) in enumerate(zip(idx[0], sims[0]), 1):
            rows.append((qid, "faiss", rank, ids[i], float(s)))
        for res in rag.hybrid_search(text, k=HYBRID_FETCH_K)[:POOL_DEPTH]:
            rows.append((qid, "hybrid", res["rank"], str(res["incident_id"]), res["composite_score"]))
    runs = pd.DataFrame(rows, columns=["query_id", "system", "rank", "incident_id", "score"])
    return runs, meta, rag


def build_pool(runs):
    prov = (runs.assign(tag=runs.system + "@" + runs["rank"].astype(str))
            .groupby(["query_id", "incident_id"])
            .agg(systems=("system", lambda s: "|".join(sorted(set(s)))),
                 ranks=("tag", lambda s: "|".join(sorted(s))),
                 n_systems=("system", "nunique"))
            .reset_index())
    return prov


def blind_export(pairs, q, meta, out_path, rng):
    text = meta.set_index("incident_id")["summary"]
    qinfo = q.set_index("query_id")
    parts = []
    for qid in sorted(pairs.query_id.unique()):
        sub = pairs[pairs.query_id == qid].sort_values("incident_id")
        sub = sub.iloc[rng.permutation(len(sub))]
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True) if parts else pairs.iloc[0:0]
    out = pd.DataFrame({
        "pair_id": [f"P{i:05d}" for i in range(1, len(df) + 1)],
        "query_id": df.query_id.values,
        "query_text": qinfo.loc[df.query_id, "query_text"].values,
        "information_need": qinfo.loc[df.query_id, "relevance_notes"].values,
        "incident_id": df.incident_id.values,
        "narrative": text.loc[df.incident_id].values,
        "relevance": "",
        "judge_notes": "",
    })
    out.to_csv(out_path, index=False)
    return out


def overlap_stats(runs):
    out = {}
    tops = {s: runs[runs.system == s].groupby("query_id").incident_id.apply(set) for s in SYSTEMS}
    for a, b in [("bm25", "faiss"), ("bm25", "hybrid"), ("faiss", "hybrid")]:
        inter = [len(tops[a][k] & tops[b][k]) for k in tops[a].index]
        out[f"{a}_vs_{b}_mean_overlap_at_{POOL_DEPTH}"] = round(float(np.mean(inter)), 4)
    return out


def phase1_bias(rel, runs):
    """Recompute the Phase 1 pooling bias from the Phase 1 files."""
    key = lambda d: set(zip(d.query_id.astype(str), d.incident_id.astype(str)))
    labeled = key(rel)
    h, b = pd.read_csv(P1_HYBRID), pd.read_csv(P1_BM25)
    res = {
        "phase1_hybrid_top20_pairs": len(key(h)),
        "phase1_hybrid_top20_pairs_labeled": len(key(h) & labeled),
        "phase1_bm25_top20_pairs": len(key(b)),
        "phase1_bm25_top20_pairs_labeled": len(key(b) & labeled),
    }
    # reproducibility of Phase 1 runs at depth POOL_DEPTH
    for sys_name, p1 in [("hybrid", h), ("bm25", b)]:
        p1k = p1[p1["rank"] <= POOL_DEPTH].groupby("query_id").incident_id.apply(list)
        new = runs[runs.system == sys_name].sort_values("rank").groupby("query_id").incident_id.apply(list)
        common = [k for k in new.index if k in p1k.index]
        set_ov = [len(set(new[k]) & set(p1k[k])) / POOL_DEPTH for k in common]
        exact = [new[k] == p1k[k] for k in common]
        res[f"phase1_{sys_name}_reproduction_mean_set_overlap_at_{POOL_DEPTH}"] = round(float(np.mean(set_ov)), 4)
        res[f"phase1_{sys_name}_reproduction_exact_ranklist_queries"] = int(sum(exact))
        res[f"phase1_{sys_name}_reproduction_queries_compared"] = len(common)
    # BM25 mismatches: are they only reorderings among tied scores?
    sc = lambda d: d.sort_values("rank").groupby("query_id").score.apply(lambda x: [round(float(v), 6) for v in x])
    p1s = sc(b[b["rank"] <= POOL_DEPTH])
    news = sc(runs[runs.system == "bm25"])
    differ = [k for k in news.index if k in p1s.index and
              runs[(runs.system == "bm25") & (runs.query_id == k)].sort_values("rank").incident_id.tolist()
              != b[(b.query_id == k) & (b["rank"] <= POOL_DEPTH)].sort_values("rank").incident_id.tolist()]
    res["phase1_bm25_reproduction_mismatch_queries"] = len(differ)
    res["phase1_bm25_reproduction_mismatch_with_identical_score_sequence"] = int(sum(news[k] == p1s[k] for k in differ))
    return res


def per_query_stats(s):
    return {"total": int(s.sum()), "mean_per_query": round(float(s.mean()), 2),
            "min_per_query": int(s.min()), "max_per_query": int(s.max())}


def write_report(R):
    p, j = R["pool"], R["judgement_export"]
    b = R["phase1_bias"]
    lines = [
        "# gold_eval_rebuild: REPORT",
        "",
        "All numbers below come from `results.json`, which `run.py` writes in the same run.",
        "",
        "## Method",
        f"- Split (chosen by the user): dev = Q001–Q020 ({R['splits']['dev_queries']} queries, all `runway_change`), "
        f"test = Q021–Q100 ({R['splits']['test_queries']} queries).",
        f"- Dev qrels = the Phase 1 manual labels for Q001–Q020 ({R['splits']['dev_qrels_rows']} pairs), copied as-is.",
        f"- The Phase 1 labels for Q021–Q100 ({R['splits']['discarded_auto_label_rows']} pairs, from "
        "`auto_label_relavance.py` MiniLM cosine thresholds) are **discarded**. That is the same model as the FAISS index, "
        "so those labels are circular.",
        f"- Pool: the top-{POOL_DEPTH} from each of 3 systems for every query, merged and de-duplicated by incident_id.",
        "  - `bm25`: Phase 1 `src/rag/bm25_baseline.py` (`tokenize`, `BM25Okapi` over `asrs_faiss_meta.summary`), imported unchanged.",
        "  - `faiss`: Phase 1 `asrs_faiss.index` (IndexFlatIP, exact) plus `rag_query_engine.embed_query`, in raw cosine order.",
        f"  - `hybrid`: Phase 1 `rag_query_engine.hybrid_search(k={HYBRID_FETCH_K})`, top-{POOL_DEPTH}. This is the same call "
        "the Phase 1 eval made: FAISS top-20 re-sorted by 0.65·cos + 0.35·recency.",
        f"- Judgement CSVs are blind: no system, rank or score columns. Rows are grouped by query and shuffled within each query "
        f"(seed {SEED}). Pool provenance is kept separately in `pool/pool_provenance.csv`.",
        "",
        "## Numbers",
        f"- Pool pairs: {p['all']['total']} total (mean {p['all']['mean_per_query']} per query, "
        f"min {p['all']['min_per_query']}, max {p['all']['max_per_query']}).",
        f"- Test pairs to judge: **{j['test_to_judge']}** in `judgements/test_to_judge.csv`.",
        f"- Dev pool pairs already covered by the Phase 1 manual labels: {j['dev_pool_pairs_already_judged']} / "
        f"{j['dev_pool_pairs']}. Unjudged dev gap: **{j['dev_gap_to_judge']}** in `judgements/dev_gap_to_judge.csv`.",
        f"- Mean overlap at {POOL_DEPTH} per query: " + ", ".join(f"{k} = {v}" for k, v in R["overlap"].items()) + ".",
        f"- Unique pool contribution (pairs found by only one system): {R['unique_contribution']}.",
        f"- Pool pairs with a narrative identical to another pair for the same query: {j['duplicate_narrative_pairs']}.",
        "",
        "## Phase 1 bias (recomputed from Phase 1 files)",
        f"- Phase 1 hybrid top-20 pairs labelled: {b['phase1_hybrid_top20_pairs_labeled']} / {b['phase1_hybrid_top20_pairs']}.",
        f"- Phase 1 BM25 top-20 pairs labelled: {b['phase1_bm25_top20_pairs_labeled']} / {b['phase1_bm25_top20_pairs']}.",
        "- In Phase 1 every unlabelled pair counted as non-relevant, so BM25 was penalised for results nobody judged.",
        "",
        "## Reproduction of Phase 1 runs (top-%d)" % POOL_DEPTH,
        f"- BM25: mean set overlap with `evaluation/bm25_results.csv` = {b[f'phase1_bm25_reproduction_mean_set_overlap_at_{POOL_DEPTH}']}; "
        f"identical ranked lists on {b['phase1_bm25_reproduction_exact_ranklist_queries']} / {b['phase1_bm25_reproduction_queries_compared']} queries.",
        f"  Of the {b['phase1_bm25_reproduction_mismatch_queries']} mismatching queries, "
        f"{b['phase1_bm25_reproduction_mismatch_with_identical_score_sequence']} have exactly the same top-{POOL_DEPTH} BM25 score sequence "
        "(to 6 dp) as Phase 1. The only difference is the order of documents with tied scores, including ties at the rank-10 cut-off. "
        "rank_bm25 + pandas sorting does not fix the order of ties; within this environment the order is stable from run to run.",
        f"- Hybrid: mean set overlap with `evaluation/retrieval_results.csv` = {b[f'phase1_hybrid_reproduction_mean_set_overlap_at_{POOL_DEPTH}']}; "
        f"identical ranked lists on {b['phase1_hybrid_reproduction_exact_ranklist_queries']} / {b['phase1_hybrid_reproduction_queries_compared']} queries.",
        "",
        "## Failures / caveats",
        f"- `rag_query_engine.RECENCY_MAX_YEAR = datetime.now().year` (= {R['env']['recency_max_year']} in this run). "
        "The hybrid ranking therefore depends on the year the script runs in. Phase 1 code is not modified, so this is recorded rather than fixed.",
        "- The hybrid's candidates are FAISS's top-20, so `hybrid` adds to the pool only what the recency re-sort moves "
        "from FAISS ranks 11–20 into its top-10.",
        "- Dev covers a single bucket (`runway_change`). Any tuning on dev may not carry over to the other 6 buckets in test.",
        f"- The embedding model `{R['env']['embed_model']}` is loaded from the HF Hub without a pinned revision in Phase 1 code. "
        f"The revision resolved in this run is recorded in results.json (`{R['env']['embed_model_revision']}`).",
        "- Retrieval metrics are not computed by `run.py`; `finalize.py` computes them from the returned judgements (`final/REPORT.md`).",
    ]
    cal_path = TASK_DIR / "llm_judge" / "calibration_results.json"
    if cal_path.exists():
        c = json.loads(cal_path.read_text(encoding="utf-8"))
        lines += [
            f"- LLM-judge attempt (Devin, blind, see `llm_judge/`) on the {c['n_pairs']} dev pool pairs that have manual labels: "
            f"Cohen's kappa = {c['cohen_kappa']}, quadratic-weighted kappa = {c['quadratic_weighted_kappa']}, "
            f"binary (>=1) kappa = {c['binary_relevant_ge1_kappa']}, exact agreement = {c['exact_agreement']}. "
            "Agreement this low ruled out Devin as the test judge; see `final/REPORT.md` for the labels actually used.",
        ]
    lines += [
        "",
        "## Next step (user)",
        "Fill `relevance` (0 = not relevant, 1 = partially relevant, 2 = highly relevant) in `judgements/test_to_judge.csv` "
        "(and ideally `judgements/dev_gap_to_judge.csv`), then return the files. Do not open `pool/pool_provenance.csv` "
        "while judging.",
    ]
    (TASK_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_revision():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = sorted((hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").glob("*"))
    return snaps[-1].name if snaps else "unknown"


def main():
    set_seeds()
    rng = np.random.default_rng(SEED)
    q = pd.read_csv(GOLD_Q)
    dev, test, qrels_dev, discarded = write_splits(q)

    runs, meta, rag = run_systems(q)
    POOL.mkdir(parents=True, exist_ok=True)
    runs.to_csv(POOL / f"system_runs_top{POOL_DEPTH}.csv", index=False)
    prov = build_pool(runs)
    prov.to_csv(POOL / "pool_provenance.csv", index=False)

    JUDGE.mkdir(parents=True, exist_ok=True)
    judged = set(zip(qrels_dev.query_id, qrels_dev.incident_id))
    is_judged = prov.apply(lambda r: (r.query_id, r.incident_id) in judged, axis=1)
    test_pairs = prov[prov.query_id.isin(TEST_IDS)]
    dev_pairs = prov[prov.query_id.isin(DEV_IDS)]
    dev_gap = dev_pairs[~is_judged[dev_pairs.index]]
    test_out = blind_export(test_pairs, q, meta, JUDGE / "test_to_judge.csv", rng)
    dev_out = blind_export(dev_gap, q, meta, JUDGE / "dev_gap_to_judge.csv", rng)

    both = pd.concat([test_out, dev_out])
    dup_narr = int(both.duplicated(["query_id", "narrative"], keep=False).sum())
    sizes = prov.groupby("query_id").size()
    uniq = prov[prov.n_systems == 1].systems.value_counts().reindex(SYSTEMS, fill_value=0)

    rel_all = pd.read_csv(GOLD_REL)
    R = {
        "task": "gold_eval_rebuild",
        "seed": SEED,
        "pool_depth": POOL_DEPTH,
        "hybrid_fetch_k": HYBRID_FETCH_K,
        "systems": SYSTEMS,
        "splits": {
            "dev_query_ids": DEV_IDS, "test_query_ids": TEST_IDS,
            "dev_queries": len(dev), "test_queries": len(test),
            "dev_buckets": dev.scenario_bucket.value_counts().to_dict(),
            "test_buckets": test.scenario_bucket.value_counts().to_dict(),
            "dev_qrels_rows": len(qrels_dev),
            "dev_qrels_relevance_counts": {str(k): int(v) for k, v in qrels_dev.relevance.value_counts().sort_index().items()},
            "discarded_auto_label_rows": len(discarded),
        },
        "pool": {
            "all": per_query_stats(sizes),
            "dev": per_query_stats(sizes[sizes.index.isin(DEV_IDS)]),
            "test": per_query_stats(sizes[sizes.index.isin(TEST_IDS)]),
        },
        "overlap": overlap_stats(runs),
        "unique_contribution": {k: int(v) for k, v in uniq.items()},
        "judgement_export": {
            "test_to_judge": len(test_out),
            "dev_pool_pairs": len(dev_pairs),
            "dev_pool_pairs_already_judged": int(is_judged[dev_pairs.index].sum()),
            "dev_gap_to_judge": len(dev_out),
            "duplicate_narrative_pairs": dup_narr,
        },
        "phase1_bias": phase1_bias(rel_all, runs),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "embed_model": rag.EMBED_MODEL, "embed_model_revision": model_revision(),
            "recency_max_year": rag.RECENCY_MAX_YEAR,
        },
        "phase1": phase1_info(REPO),
        "input_sha256": {p.relative_to(REPO).as_posix(): sha256(p) for p in [GOLD_Q, GOLD_REL, META, INDEX, P1_HYBRID, P1_BM25]},
        "output_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(
            list(SPLITS.glob("*.csv")) + list(JUDGE.glob("*.csv")) + list(POOL.glob("*.csv")))},
    }
    (TASK_DIR / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({k: R[k] for k in ["pool", "judgement_export", "overlap", "unique_contribution"]}, indent=2))


if __name__ == "__main__":
    main()
