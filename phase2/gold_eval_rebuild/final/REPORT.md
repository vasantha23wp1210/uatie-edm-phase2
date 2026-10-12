# gold_eval_rebuild: FINAL REPORT

All numbers come from `final/results.json`, which `finalize.py` writes in the same run.

## Label provenance (read first)
- Test qrels = `judgements/test_judged.csv` (1748 rows, relevance counts {'0': 374, '1': 736, '2': 638}), as returned by the user.
- Every row's `judge_notes` begins with: {'LLM-judged (Claude)': 1748}. These are **LLM-generated labels**, not manual judgements.
- User's own blind spot-check: 100 test pairs over 63 queries (`judgements/spotcheck_100_human.csv`). Agreement with the test labels: exact 0.55, Cohen's κ 0.3394, quadratic-weighted κ 0.5758, binary (≥1) agreement 0.75, binary κ 0.4292.
- Confusion (rows = user, cols = test labels, order 0/1/2): [[17, 20, 3], [1, 14, 13], [1, 7, 24]]. Relevant rate: user 0.6, test labels 0.81.
- On the 45 disagreeing pairs the user chose to keep the LLM label. The test qrels are therefore entirely LLM-judged.
- Separate dev calibration of an LLM judge (Devin, `llm_judge/`) against Phase 1 manual labels: κ 0.0619, binary κ 0.1151.
- These labels are independent of MiniLM (the circularity this task removes), but they are not human gold. Report them as LLM-judged, with the agreement numbers above.

## Method
- Systems: `bm25`, `faiss`, `hybrid` (top-10 from `run.py`, the same lists that built the pool) and `rrf` (`src/hysor_rrf.py`, BM25 + dense top-50 each, RRF k = 60, top-10).
- `rrf` was not in the pool, so some of its results are unjudged; `judged@10` gives coverage. Unjudged = non-relevant for every system.
- P@k, MRR@10 and MAP@10 are binary. nDCG@10 uses gain 2^rel − 1, with the ideal ranking taken from all judged pairs for the query. MAP@10 divides by min(R, 10), where R = judged relevant pairs for the query.
- Main threshold relevance ≥ 1; strict sensitivity check relevance ≥ 2. Significance: two-sided paired randomisation test on per-query nDCG@10, 10000 sign flips, seed 42, no multiple-comparison correction.

## Test, Q021–Q100, relevance ≥ 1 (80/80 queries have a relevant pair)
| System | P@5 | P@10 | MRR@10 | MAP@10 | nDCG@10 | judged@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.8450 | 0.8238 | 0.9356 | 0.7858 | 0.7081 | 1.0000 |
| faiss | 0.8075 | 0.8000 | 0.9128 | 0.7372 | 0.6593 | 1.0000 |
| hybrid | 0.7900 | 0.7950 | 0.8754 | 0.7272 | 0.6299 | 1.0000 |
| rrf | 0.7750 | 0.6775 | 0.9542 | 0.6114 | 0.6354 | 0.7962 |

- bm25 vs faiss: mean diff +0.0488, p = 0.0936
- bm25 vs hybrid: mean diff +0.0781, p = 0.0148
- bm25 vs rrf: mean diff +0.0727, p = 0.0004
- faiss vs hybrid: mean diff +0.0293, p = 0.0564
- faiss vs rrf: mean diff +0.0238, p = 0.1769
- hybrid vs rrf: mean diff -0.0055, p = 0.8029

## Test, relevance ≥ 2 (strict; 73/80 queries have a relevant pair)
| System | P@5 | P@10 | MRR@10 | MAP@10 | nDCG@10 | judged@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.4850 | 0.4250 | 0.6579 | 0.4275 | 0.5344 | 1.0000 |
| faiss | 0.4175 | 0.3938 | 0.6210 | 0.3737 | 0.4975 | 1.0000 |
| hybrid | 0.3825 | 0.3712 | 0.5688 | 0.3266 | 0.4539 | 1.0000 |
| rrf | 0.4650 | 0.3763 | 0.6738 | 0.3950 | 0.5093 | 0.7962 |

- bm25 vs faiss: mean diff +0.0369, p = 0.3322
- bm25 vs hybrid: mean diff +0.0805, p = 0.0365
- bm25 vs rrf: mean diff +0.0251, p = 0.3575
- faiss vs hybrid: mean diff +0.0437, p = 0.0155
- faiss vs rrf: mean diff -0.0118, p = 0.6214
- hybrid vs rrf: mean diff -0.0554, p = 0.0731

## Test nDCG@10 by bucket (relevance ≥ 1)
| Bucket | bm25 | faiss | hybrid | rrf |
|---|---|---|---|---|
| crm_workload | 0.6430 | 0.6072 | 0.5504 | 0.5420 |
| maintenance | 0.7686 | 0.8034 | 0.7854 | 0.7392 |
| phraseology | 0.6907 | 0.6518 | 0.5923 | 0.6364 |
| runway_incursion | 0.6518 | 0.6934 | 0.6394 | 0.6073 |
| unstable_approach | 0.7630 | 0.5611 | 0.6030 | 0.6361 |
| weather | 0.6939 | 0.7442 | 0.6738 | 0.6500 |

## Dev, Q001–Q020, Phase 1 manual labels, relevance ≥ 1 (19/20 queries have a relevant pair)
Manual labels cover 362 of the 457 dev pool pairs; the 95 in `judgements/dev_gap_to_judge.csv` are unjudged and count as non-relevant.
| System | P@5 | P@10 | MRR@10 | MAP@10 | nDCG@10 | judged@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.2200 | 0.1350 | 0.4117 | 0.0878 | 0.1313 | 0.5250 |
| faiss | 0.9100 | 0.9250 | 0.9500 | 0.9072 | 0.7606 | 1.0000 |
| hybrid | 0.9100 | 0.9200 | 0.9500 | 0.9058 | 0.7296 | 1.0000 |
| rrf | 0.5500 | 0.5100 | 0.8750 | 0.3973 | 0.4778 | 0.7650 |

- bm25 vs faiss: mean diff -0.6293, p = 0.0001
- bm25 vs hybrid: mean diff -0.5983, p = 0.0001
- bm25 vs rrf: mean diff -0.3465, p = 0.0001
- faiss vs hybrid: mean diff +0.0309, p = 0.154
- faiss vs rrf: mean diff +0.2827, p = 0.0001
- hybrid vs rrf: mean diff +0.2518, p = 0.0001

## Failures / caveats
- Test labels are LLM-judged and agree only moderately with the user's blind spot-check (see provenance). Absolute values are inflated relative to the user's stricter judging (relevant rate on the spot-check sample: test labels 0.81 vs user 0.6).
- Dev is a single bucket (`runway_change`) and its labels are human; dev and test numbers are not directly comparable.
- `rrf` judged@10 < 1 biases it downward relative to the pooled systems.
- Nothing was tuned on test.
