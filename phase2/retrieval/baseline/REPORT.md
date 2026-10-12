# Task 3 step 1: HySOR baseline reproduction

All numbers come from `baseline/results.json`, written by `baseline.py` in the same run.

## Method
- Fresh rerun of Phase 1 `rag_query_engine.hybrid_search(k=20)` for all 100 gold queries: score = 0.65·cosine + 0.35·recency, recency range 1990–2026, model `all-MiniLM-L6-v2`.
- Scored with the published protocol, Phase 1 `bm25_baseline.compute_metrics` unchanged: relevance ≥ 1, AP over the top-20 list divided by all relevant labels of the query, binary nDCG, labels = `evaluation/gold_relevance.csv` as committed (Q001–Q020 manual, Q021–Q100 MiniLM cosine ≥ 0.52 auto-labels).
- The stored Phase 1 ranking (`evaluation/retrieval_results_enriched.csv`) is scored the same way. Fresh and stored top-20 lists are identical on 100 / 100 queries.
- Target: P@5, MRR, MAP, nDCG@10 within ±0.01 of the paper.

## Result
| Metric | Paper | Fresh rerun | Δ | Stored ranking | Δ |
|---|---|---|---|---|---|
| P@1 | 0.896 | 0.8961 | +0.0001 | 0.8961 | +0.0001 |
| P@3 | 0.909 | 0.9091 | +0.0001 | 0.9091 | +0.0001 |
| P@5 | 0.891 | 0.8909 | -0.0001 | 0.8909 | -0.0001 |
| P@10 | 0.888 | 0.8883 | +0.0003 | 0.8883 | +0.0003 |
| nDCG@5 | 0.899 | 0.8992 | +0.0002 | 0.8992 | +0.0002 |
| nDCG@10 | 0.901 | 0.9013 | +0.0003 | 0.9013 | +0.0003 |
| MRR | 0.934 | 0.9340 | +0.0000 | 0.9340 | +0.0000 |
| MAP | 0.896 | 0.8957 | -0.0003 | 0.8957 | -0.0003 |

**Reproduced within ±0.01: yes** (P@5 True, MRR True, MAP True, nDCG@10 True).

## Differences from the paper's description (not hidden by the match)
- Queries evaluated: 77 of 100, not 100. `compute_metrics` returns NaN for queries with no relevance ≥ 1 label and the mean skips them: Q014, Q069, Q071, Q072, Q073, Q074, Q075, Q076, Q078, Q079, Q080, Q081, Q082, Q083, Q084, Q085, Q086, Q087, Q088, Q089, Q090, Q099, Q100.
- The paper says all systems return top-5; the published numbers come from top-20 lists (P@10 and MAP need more than 5).
- Q021–Q100 labels are MiniLM cosine auto-labels from the same model as the FAISS index, so this baseline is circular by construction. Task 3 evaluates on the graded judgements from `phase2/gold_eval_rebuild` instead.
- `RECENCY_MAX_YEAR = datetime.now().year` (2026 in this run): the ranking depends on the run year.
