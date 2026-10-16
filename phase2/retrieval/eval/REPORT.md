# Task 3 retrieval: evaluation

All numbers come from `eval/results.json`, written by `evaluate.py` in the same run.

## Label provenance (read first)

- Task 3 pairs (`judgements/judged.csv`): {'LLM-judged (Claude), pending user review': 975}.
- Test qrels rows: {'gold_eval_rebuild': 1748, 'task3_judged': 719}. The `gold_eval_rebuild` test labels are LLM-judged (Claude).
- Dev qrels rows: {'phase1_manual': 483, 'task3_judged': 256}.
- Any label marked `pending user review` makes the scores below **provisional**. Replace `judgements/judged.csv` with reviewed labels and rerun this script; no code change is needed.

Every top-10 result of every system is judged (no unjudged pair is counted as non-relevant). Queries with no relevant label at the given threshold are left out; `n` is the number scored. MAP@10 normalises by min(R, 10). nDCG gain is 2^rel - 1 for labels at or above the threshold.

## Test (Q021-Q100), relevant = rel >= 1

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 80 | 0.9125 | 0.8708 | 0.8450 | 0.8238 | 0.9356 | 0.7798 | 0.3515 | 0.6618 | 0.6605 |
| faiss | 80 | 0.8625 | 0.8042 | 0.8075 | 0.8000 | 0.9128 | 0.7326 | 0.3416 | 0.6002 | 0.6160 |
| hybrid | 80 | 0.8000 | 0.7917 | 0.7900 | 0.7950 | 0.8754 | 0.7241 | 0.3359 | 0.5652 | 0.5914 |
| rrf | 80 | 0.9500 | 0.8917 | 0.8625 | 0.8325 | 0.9604 | 0.7925 | 0.3514 | 0.6969 | 0.6910 |
| ce | 80 | 0.9625 | 0.9333 | 0.8975 | 0.8525 | 0.9812 | 0.8297 | 0.3755 | 0.7901 | 0.7727 |
| ce_recency | 80 | 0.8500 | 0.8417 | 0.8100 | 0.7900 | 0.8989 | 0.7434 | 0.3402 | 0.6433 | 0.6529 |

## Test, relevant = rel 2

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 76 | 0.6184 | 0.5351 | 0.5105 | 0.4474 | 0.6926 | 0.4076 | 0.3545 | 0.5417 | 0.5209 |
| faiss | 76 | 0.5395 | 0.4649 | 0.4395 | 0.4145 | 0.6537 | 0.3566 | 0.3479 | 0.4736 | 0.4838 |
| hybrid | 76 | 0.4737 | 0.4342 | 0.4026 | 0.3908 | 0.5987 | 0.3180 | 0.3221 | 0.4293 | 0.4445 |
| rrf | 76 | 0.6184 | 0.5746 | 0.5474 | 0.4829 | 0.7140 | 0.4537 | 0.3979 | 0.5834 | 0.5705 |
| ce | 76 | 0.7895 | 0.6886 | 0.6395 | 0.5500 | 0.8640 | 0.5756 | 0.4940 | 0.7106 | 0.6905 |
| ce_recency | 76 | 0.5395 | 0.5044 | 0.4947 | 0.4500 | 0.6646 | 0.4039 | 0.4041 | 0.5179 | 0.5293 |

## Dev (Q001-Q020), relevant = rel >= 1

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 19 | 0.2632 | 0.2807 | 0.2316 | 0.3316 | 0.4561 | 0.1821 | 0.1199 | 0.1574 | 0.2201 |
| faiss | 19 | 1.0000 | 0.9825 | 0.9579 | 0.9737 | 1.0000 | 0.9550 | 0.3932 | 0.7146 | 0.7517 |
| hybrid | 19 | 1.0000 | 1.0000 | 0.9579 | 0.9684 | 1.0000 | 0.9535 | 0.3886 | 0.6919 | 0.7230 |
| rrf | 19 | 0.8421 | 0.7719 | 0.7263 | 0.6947 | 0.9211 | 0.5777 | 0.2721 | 0.5544 | 0.5684 |
| ce | 19 | 0.9474 | 0.7368 | 0.6842 | 0.6632 | 0.9737 | 0.5499 | 0.2578 | 0.4985 | 0.5417 |
| ce_recency | 19 | 0.6316 | 0.5088 | 0.5263 | 0.5368 | 0.7294 | 0.4036 | 0.2044 | 0.4116 | 0.4250 |

## Dev, relevant = rel 2

| System | n | P@1 | P@3 | P@5 | P@10 | MRR@10 | MAP@10 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 19 | 0.0526 | 0.0877 | 0.0947 | 0.1474 | 0.1750 | 0.0831 | 0.1328 | 0.0966 | 0.1481 |
| faiss | 19 | 0.5789 | 0.5439 | 0.4842 | 0.4263 | 0.7180 | 0.4153 | 0.5380 | 0.5510 | 0.5689 |
| hybrid | 19 | 0.6316 | 0.4035 | 0.4632 | 0.3947 | 0.7107 | 0.3743 | 0.4702 | 0.5073 | 0.5179 |
| rrf | 19 | 0.4737 | 0.3860 | 0.3684 | 0.3211 | 0.6296 | 0.2945 | 0.3986 | 0.4250 | 0.4410 |
| ce | 19 | 0.4211 | 0.3509 | 0.3158 | 0.3421 | 0.5458 | 0.2660 | 0.3684 | 0.3471 | 0.4064 |
| ce_recency | 19 | 0.5263 | 0.2982 | 0.2842 | 0.2632 | 0.5697 | 0.2187 | 0.2483 | 0.3279 | 0.3273 |

## Primary Phase 2 system and plan targets (test, rel >= 1)

Primary system: `ce`, fixed in advance by the Phase 2 plan (BM25 + cross-encoder re-ranking); the dev-selected system is reported alongside. `ce` re-ranks the union of BM25 top-50 and FAISS top-50, so its candidate pool includes BM25. MAP is MAP@10.

| Metric | value | target | met |
|---|---|---|---|
| P@5 | 0.8975 | >= 0.80 | yes |
| MAP@10 | 0.8297 | >= 0.80 | yes |

On dev, `ce` scores P@5 0.6842 / MAP@10 0.5499, below `faiss` (0.9579 / 0.9550). Most dev labels are Phase 1 manual judgements of Phase 1 hybrid/FAISS output, which favours those systems; this is why the dev rule picks a different system.

## Target: best Phase 2 system >= Phase 1 baseline (test, rel >= 1)

Best system chosen on **dev** nDCG@10: `faiss`. Baseline: `hybrid` (Phase 1 hybrid engine).

| Metric | best | baseline | met |
|---|---|---|---|
| P@5 | 0.8075 | 0.7900 | yes |
| MAP@10 | 0.7326 | 0.7241 | yes |
| nDCG@10 | 0.6160 | 0.5914 | yes |

## Paired randomisation test vs `hybrid` (test, two-sided, 10000 permutations, no multiple-comparison correction)

| System | thr | P@5 diff (p) | MAP@10 diff (p) | nDCG@10 diff (p) |
|---|---|---|---|---|
| bm25 | rel_ge1 | +0.0550 (0.1317) | +0.0557 (0.1156) | +0.0690 (0.0169) |
| faiss | rel_ge1 | +0.0175 (0.5042) | +0.0085 (0.6849) | +0.0245 (0.0742) |
| rrf | rel_ge1 | +0.0725 (0.0128) | +0.0684 (0.0092) | +0.0996 (0.0001) |
| ce | rel_ge1 | +0.1075 (0.0006) | +0.1056 (0.0016) | +0.1813 (0.0001) |
| ce_recency | rel_ge1 | +0.0200 (0.5679) | +0.0193 (0.5935) | +0.0614 (0.0368) |
| bm25 | rel_ge2 | +0.1079 (0.0072) | +0.0896 (0.0064) | +0.0763 (0.0341) |
| faiss | rel_ge2 | +0.0368 (0.163) | +0.0386 (0.0168) | +0.0393 (0.0165) |
| rrf | rel_ge2 | +0.1447 (0.0001) | +0.1357 (0.0002) | +0.1260 (0.0002) |
| ce | rel_ge2 | +0.2368 (0.0001) | +0.2576 (0.0001) | +0.2459 (0.0001) |
| ce_recency | rel_ge2 | +0.0921 (0.0086) | +0.0859 (0.0077) | +0.0848 (0.0187) |

## nDCG@10 per scenario bucket (test, rel >= 1)

| Bucket | n | bm25 | faiss | hybrid | rrf | ce | ce_recency |
|---|---|---|---|---|---|---|---|
| crm_workload | 10 | 0.5334 | 0.5062 | 0.4553 | 0.5748 | 0.7350 | 0.6228 |
| maintenance | 10 | 0.7583 | 0.7908 | 0.7763 | 0.8761 | 0.9377 | 0.8457 |
| phraseology | 20 | 0.6428 | 0.6078 | 0.5546 | 0.6855 | 0.7445 | 0.6757 |
| runway_incursion | 10 | 0.6202 | 0.6596 | 0.6075 | 0.6546 | 0.7117 | 0.5121 |
| unstable_approach | 20 | 0.7072 | 0.5266 | 0.5643 | 0.6694 | 0.7565 | 0.6148 |
| weather | 10 | 0.6719 | 0.7024 | 0.6547 | 0.7129 | 0.7952 | 0.6614 |

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
| nDCG@5 | 0.899 | 0.8992 | 0.5652 |
| nDCG@10 | 0.901 | 0.9013 | 0.5914 |
| MRR | 0.934 | 0.934 | 0.8754 |
| MAP | 0.896 | 0.8957 | 0.7241 |

## Error analysis

`eval/error_analysis.csv`: top-5 of `faiss` with labels for Q025, Q026, Q030, Q036, Q055, Q075, Q092, Q098, Q058, Q049 (the seven named in the plan plus the lowest-nDCG@10 other test queries).

