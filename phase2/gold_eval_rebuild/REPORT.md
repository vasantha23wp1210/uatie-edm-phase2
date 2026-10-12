# gold_eval_rebuild: REPORT

All numbers below come from `results.json`, which `run.py` writes in the same run.

## Method
- Split (chosen by the user): dev = Q001–Q020 (20 queries, all `runway_change`), test = Q021–Q100 (80 queries).
- Dev qrels = the Phase 1 manual labels for Q001–Q020 (483 pairs), copied as-is.
- The Phase 1 labels for Q021–Q100 (1887 pairs, from `auto_label_relavance.py` MiniLM cosine thresholds) are **discarded**. That is the same model as the FAISS index, so those labels are circular.
- Pool: the top-10 from each of 3 systems for every query, merged and de-duplicated by incident_id.
  - `bm25`: Phase 1 `src/rag/bm25_baseline.py` (`tokenize`, `BM25Okapi` over `asrs_faiss_meta.summary`), imported unchanged.
  - `faiss`: Phase 1 `asrs_faiss.index` (IndexFlatIP, exact) plus `rag_query_engine.embed_query`, in raw cosine order.
  - `hybrid`: Phase 1 `rag_query_engine.hybrid_search(k=20)`, top-10. This is the same call the Phase 1 eval made: FAISS top-20 re-sorted by 0.65·cos + 0.35·recency.
- Judgement CSVs are blind: no system, rank or score columns. Rows are grouped by query and shuffled within each query (seed 42). Pool provenance is kept separately in `pool/pool_provenance.csv`.

## Numbers
- Pool pairs: 2205 total (mean 22.05 per query, min 16, max 28).
- Test pairs to judge: **1748** in `judgements/test_to_judge.csv`.
- Dev pool pairs already covered by the Phase 1 manual labels: 362 / 457. Unjudged dev gap: **95** in `judgements/dev_gap_to_judge.csv`.
- Mean overlap at 10 per query: bm25_vs_faiss_mean_overlap_at_10 = 1.5, bm25_vs_hybrid_mean_overlap_at_10 = 1.29, faiss_vs_hybrid_mean_overlap_at_10 = 6.21.
- Unique pool contribution (pairs found by only one system): {'bm25': 826, 'faiss': 334, 'hybrid': 355}.
- Pool pairs with a narrative identical to another pair for the same query: 0.

## Phase 1 bias (recomputed from Phase 1 files)
- Phase 1 hybrid top-20 pairs labelled: 2000 / 2000.
- Phase 1 BM25 top-20 pairs labelled: 702 / 2000.
- In Phase 1 every unlabelled pair counted as non-relevant, so BM25 was penalised for results nobody judged.

## Reproduction of Phase 1 runs (top-10)
- BM25: mean set overlap with `evaluation/bm25_results.csv` = 0.994; identical ranked lists on 69 / 100 queries.
  Of the 31 mismatching queries, 31 have exactly the same top-10 BM25 score sequence (to 6 dp) as Phase 1. The only difference is the order of documents with tied scores, including ties at the rank-10 cut-off. rank_bm25 + pandas sorting does not fix the order of ties; within this environment the order is stable from run to run.
- Hybrid: mean set overlap with `evaluation/retrieval_results.csv` = 1.0; identical ranked lists on 100 / 100 queries.

## Failures / caveats
- `rag_query_engine.RECENCY_MAX_YEAR = datetime.now().year` (= 2026 in this run). The hybrid ranking therefore depends on the year the script runs in. Phase 1 code is not modified, so this is recorded rather than fixed.
- The hybrid's candidates are FAISS's top-20, so `hybrid` adds to the pool only what the recency re-sort moves from FAISS ranks 11–20 into its top-10.
- Dev covers a single bucket (`runway_change`). Any tuning on dev may not carry over to the other 6 buckets in test.
- The embedding model `all-MiniLM-L6-v2` is loaded from the HF Hub without a pinned revision in Phase 1 code. The revision resolved in this run is recorded in results.json (`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`).
- Retrieval metrics are not computed by `run.py`; `finalize.py` computes them from the returned judgements (`final/REPORT.md`).
- LLM-judge attempt (Devin, blind, see `llm_judge/`) on the 362 dev pool pairs that have manual labels: Cohen's kappa = 0.0619, quadratic-weighted kappa = 0.4311, binary (>=1) kappa = 0.1151, exact agreement = 0.3591. Agreement this low ruled out Devin as the test judge; see `final/REPORT.md` for the labels actually used.

## Next step (user)
Fill `relevance` (0 = not relevant, 1 = partially relevant, 2 = highly relevant) in `judgements/test_to_judge.csv` (and ideally `judgements/dev_gap_to_judge.csv`), then return the files. Do not open `pool/pool_provenance.csv` while judging.
