# Task 3 retrieval: evaluation

All numbers come from `eval/results.json`, written by `evaluate.py` in the same run.

## Label provenance (read first)

- Task 3 pairs (`judgements/judged.csv`): {'LLM-judged (Devin), pending user review': 975}.
- Test qrels rows: {'gold_eval_rebuild': 1748, 'task3_judged': 719}. The `gold_eval_rebuild` test labels are LLM-judged (Claude).
- Dev qrels rows: {'phase1_manual': 483, 'task3_judged': 256}.
- Any label marked `pending user review` makes the scores below **provisional**. Replace `judgements/judged.csv` with reviewed labels and rerun this script; no code change is needed.

Every top-10 result of every system is judged (no unjudged pair is counted as non-relevant). Queries with no relevant label at the given threshold are left out; `n` is the number scored. MAP@10 normalises by min(R, 10). nDCG gain is 2^rel - 1 for labels at or above the threshold.

## Test (Q021-Q100), relevant = rel >= 1

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 80 | 0.9125 | 0.8708 | 0.8450 | 0.8238 | 0.9356 | 0.7797 | 0.3530 | 0.6612 | 0.6567 |
| faiss | 80 | 0.8625 | 0.8042 | 0.8075 | 0.8000 | 0.9128 | 0.7318 | 0.3389 | 0.5987 | 0.6109 |
| hybrid | 80 | 0.8000 | 0.7917 | 0.7900 | 0.7950 | 0.8754 | 0.7218 | 0.3327 | 0.5641 | 0.5864 |
| rrf | 80 | 0.9500 | 0.8917 | 0.8575 | 0.8288 | 0.9625 | 0.7842 | 0.3502 | 0.6892 | 0.6824 |
| ce | 80 | 0.9625 | 0.9292 | 0.8925 | 0.8525 | 0.9812 | 0.8232 | 0.3779 | 0.7977 | 0.7798 |
| ce_recency | 80 | 0.8250 | 0.8417 | 0.8200 | 0.7962 | 0.8912 | 0.7443 | 0.3452 | 0.6439 | 0.6600 |

## Test, relevant = rel 2

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 76 | 0.6184 | 0.5351 | 0.5105 | 0.4474 | 0.6926 | 0.4038 | 0.3477 | 0.5417 | 0.5172 |
| faiss | 76 | 0.5395 | 0.4649 | 0.4395 | 0.4145 | 0.6537 | 0.3508 | 0.3374 | 0.4721 | 0.4781 |
| hybrid | 76 | 0.4737 | 0.4342 | 0.4026 | 0.3908 | 0.5987 | 0.3136 | 0.3117 | 0.4289 | 0.4397 |
| rrf | 76 | 0.6184 | 0.5746 | 0.5421 | 0.4803 | 0.7136 | 0.4453 | 0.3847 | 0.5791 | 0.5629 |
| ce | 76 | 0.8026 | 0.7105 | 0.6605 | 0.5671 | 0.8772 | 0.5917 | 0.5069 | 0.7304 | 0.7069 |
| ce_recency | 76 | 0.5526 | 0.5132 | 0.4974 | 0.4684 | 0.6763 | 0.4174 | 0.4182 | 0.5235 | 0.5436 |

## Dev (Q001-Q020), relevant = rel >= 1

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 19 | 0.2632 | 0.2807 | 0.2316 | 0.3158 | 0.4474 | 0.1707 | 0.1170 | 0.1607 | 0.2096 |
| faiss | 19 | 1.0000 | 0.9825 | 0.9579 | 0.9737 | 1.0000 | 0.9550 | 0.3990 | 0.7210 | 0.7579 |
| hybrid | 19 | 1.0000 | 1.0000 | 0.9579 | 0.9684 | 1.0000 | 0.9535 | 0.3952 | 0.6983 | 0.7286 |
| rrf | 19 | 0.8421 | 0.7895 | 0.7158 | 0.6789 | 0.9211 | 0.5622 | 0.2705 | 0.5627 | 0.5629 |
| ce | 19 | 0.9474 | 0.7193 | 0.6737 | 0.6526 | 0.9737 | 0.5376 | 0.2587 | 0.5082 | 0.5314 |
| ce_recency | 19 | 0.6316 | 0.4912 | 0.5158 | 0.5158 | 0.7294 | 0.3813 | 0.2019 | 0.4119 | 0.4131 |

## Dev, relevant = rel 2

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 19 | 0.0526 | 0.0877 | 0.0947 | 0.1316 | 0.1737 | 0.0847 | 0.1516 | 0.1067 | 0.1468 |
| faiss | 19 | 0.5789 | 0.5439 | 0.4842 | 0.4263 | 0.7180 | 0.4157 | 0.5466 | 0.5510 | 0.5692 |
| hybrid | 19 | 0.6316 | 0.4035 | 0.4632 | 0.3947 | 0.7107 | 0.3729 | 0.4765 | 0.5073 | 0.5173 |
| rrf | 19 | 0.4737 | 0.3860 | 0.3684 | 0.3053 | 0.6296 | 0.3061 | 0.4248 | 0.4454 | 0.4497 |
| ce | 19 | 0.4211 | 0.3509 | 0.3263 | 0.3211 | 0.5458 | 0.2517 | 0.3664 | 0.3627 | 0.3918 |
| ce_recency | 19 | 0.5263 | 0.2982 | 0.2842 | 0.2474 | 0.5684 | 0.2067 | 0.2477 | 0.3279 | 0.3152 |

## Target: best Phase 2 system >= Phase 1 baseline (test, rel >= 1)

Best system chosen on **dev** nDCG@10: `faiss`. Baseline: `hybrid` (Phase 1 hybrid engine).

| Metric | best | baseline | met |
|---|---|---|---|
| P@5 | 0.8075 | 0.7900 | yes |
| MAP@10 | 0.7318 | 0.7218 | yes |
| nDCG@10 | 0.6109 | 0.5864 | yes |

## Paired randomisation test vs `hybrid` (test, two-sided, 10000 permutations, no multiple-comparison correction)

| System | thr | P@5 diff (p) | MAP@10 diff (p) | nDCG@10 diff (p) |
|---|---|---|---|---|
| bm25 | rel_ge1 | +0.0550 (0.1317) | +0.0580 (0.0951) | +0.0703 (0.0136) |
| faiss | rel_ge1 | +0.0175 (0.5042) | +0.0100 (0.6286) | +0.0246 (0.0719) |
| rrf | rel_ge1 | +0.0675 (0.0201) | +0.0624 (0.0085) | +0.0960 (0.0001) |
| ce | rel_ge1 | +0.1025 (0.0009) | +0.1015 (0.0017) | +0.1934 (0.0001) |
| ce_recency | rel_ge1 | +0.0300 (0.3275) | +0.0225 (0.5074) | +0.0737 (0.0104) |
| bm25 | rel_ge2 | +0.1079 (0.0072) | +0.0902 (0.0053) | +0.0775 (0.0299) |
| faiss | rel_ge2 | +0.0368 (0.163) | +0.0373 (0.0202) | +0.0384 (0.0178) |
| rrf | rel_ge2 | +0.1395 (0.0002) | +0.1317 (0.0002) | +0.1233 (0.0002) |
| ce | rel_ge2 | +0.2579 (0.0001) | +0.2781 (0.0001) | +0.2672 (0.0001) |
| ce_recency | rel_ge2 | +0.0947 (0.0059) | +0.1038 (0.0017) | +0.1039 (0.004) |

## nDCG@10 per scenario bucket (test, rel >= 1)

| Bucket | n | bm25 | faiss | hybrid | rrf | ce | ce_recency |
|---|---|---|---|---|---|---|---|
| crm_workload | 10 | 0.5172 | 0.4899 | 0.4406 | 0.5640 | 0.7425 | 0.6455 |
| maintenance | 10 | 0.7614 | 0.7952 | 0.7789 | 0.8724 | 0.9328 | 0.8507 |
| phraseology | 20 | 0.6396 | 0.6021 | 0.5494 | 0.6635 | 0.7349 | 0.6592 |
| runway_incursion | 10 | 0.6162 | 0.6488 | 0.5942 | 0.6677 | 0.7520 | 0.5534 |
| unstable_approach | 20 | 0.7055 | 0.5246 | 0.5624 | 0.6598 | 0.7620 | 0.6157 |
| weather | 10 | 0.6687 | 0.7000 | 0.6536 | 0.7084 | 0.8171 | 0.6811 |

## Latency per query on this VM (ms, retrieval only)

| System | p50 | p95 |
|---|---|---|
| bm25 | 153.0 | 231.8 |
| faiss | 18.6 | 33.1 |
| hybrid | 19.0 | 31.4 |
| rrf | 175.6 | 254.5 |
| ce | 500.7 | 638.7 |
| ce_recency | 501.1 | 639.1 |

## Paper protocol vs this protocol (Phase 1 hybrid)

Published protocol: 77 queries, top-20 lists, MiniLM-generated labels for Q021-Q100 (circular). This protocol: unbiased pooled labels, top-10, test queries only.

| Metric | Paper | Published-protocol rerun | This protocol (test, rel >= 1) |
|---|---|---|---|
| P@1 | 0.896 | 0.8961 | 0.8000 |
| P@3 | 0.909 | 0.9091 | 0.7917 |
| P@5 | 0.891 | 0.8909 | 0.7900 |
| P@10 | 0.888 | 0.8883 | 0.7950 |
| nDCG@5 | 0.899 | 0.8992 | 0.5641 |
| nDCG@10 | 0.901 | 0.9013 | 0.5864 |
| MRR | 0.934 | 0.934 | 0.8754 |
| MAP | 0.896 | 0.8957 | 0.7218 |

## Error analysis

`eval/error_analysis.csv`: top-5 of `faiss` with labels for Q025, Q026, Q030, Q036, Q055, Q075, Q092, Q098, Q058, Q049 (the seven named in the plan plus the lowest-nDCG@10 other test queries).

