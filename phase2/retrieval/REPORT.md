# Task 3: Retrieval (STOPPED, waiting for judgements)

All numbers come from `results.json`, written by `run.py` in the same run. No retrieval metrics are reported yet: the systems retrieve pairs that were never judged, and the plan forbids treating them as non-relevant.

## Step 1: baseline reproduction (done)
See `baseline/REPORT.md`. The paper's HySOR numbers reproduce within ±0.01 under the published protocol.

## Step 2: systems
| # | System | Definition |
|---|---|---|
| 1 | bm25 | Phase 1 `bm25_baseline.bm25_retrieve` over `asrs_faiss_meta.summary` |
| 2 | faiss | raw FAISS inner product, Phase 1 index, all-MiniLM-L6-v2 |
| 3 | hybrid | Phase 1 `rag_query_engine.hybrid_search(k=20)`, top-10 |
| 4 | rrf | `hysor_rrf.rrf_fuse` of systems 1 and 2 at depth 50, k = 60 |
| 5 | ce | union of bm25 top-50 and faiss top-50 re-ranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 6 | ce_recency | system 5 scored as `0.65*sigmoid(ce_logit) + 0.35*recency(1990-2026)`, Phase 1 weights, not tuned |

Systems 1–3 return the same top-10 lists as the `gold_eval_rebuild` pool on bm25 100/100, faiss 100/100, hybrid 100/100 queries.

## Judgement coverage of the top-10 (test qrels + Phase 1 manual dev labels)
| System | judged@10 | unjudged rows |
|---|---|---|
| bm25 | 0.9050 | 95 |
| faiss | 1.0000 | 0 |
| hybrid | 1.0000 | 0 |
| rrf | 0.7870 | 213 |
| ce | 0.5260 | 474 |
| ce_recency | 0.4680 | 532 |

Unique unjudged (query, incident) pairs: **975** over 100 queries (719 test, 256 dev). They include the 95 dev-gap BM25 pairs from `gold_eval_rebuild` that were never returned.

## Latency on this VM (ms per query, CPU)
`x86_64`, 8 cores. For rrf/ce/ce_recency the time includes the bm25 and faiss retrieval they need.

| System | p50 | p95 |
|---|---|---|
| bm25 | 153.0 | 231.8 |
| faiss | 18.6 | 33.1 |
| hybrid | 19.0 | 31.4 |
| rrf | 175.6 | 254.5 |
| ce | 500.7 | 638.7 |
| ce_recency | 501.1 | 639.1 |

## What is needed to continue
Fill `relevance` (0/1/2, same guideline as `phase2/gold_eval_rebuild/judgements/`) in `judgements/to_judge.csv` and return it as `judgements/judged.csv`. It is blind: no system names, ranks or scores.

## Failures / caveats
- Latency is measured on this VM; it will differ on other hardware.
- The test qrels these systems are scored against are LLM-judged (see `phase2/gold_eval_rebuild/final/REPORT.md`).
