# 95% intervals for headline numbers

Wilson for proportions; bootstrap over queries for retrieval (10,000, seed 42). From `ci.py`; no new runs.

| Quantity | value | 95% CI | n |
|---|---|---|---|
| Intent test accuracy, `distilbert_scaled` | 0.6816 | [0.6143, 0.7421] | 201 |
| Intent test accuracy, `logreg_minilm_scaled` | 0.5871 | [0.518, 0.6529] | 201 |
| Intent test accuracy, `logreg_minilm` | 0.403 | [0.3376, 0.472] | 201 |
| Unsupported claims (3B judge), ground llama3.2:3b test | 0.1094 | [0.0819, 0.1445] | 384 claims |
| Unsupported claims (3B judge), ground llama3.1:8b test | 0.1124 | [0.0833, 0.15] | 347 claims |
| Unsupported claims (3B judge), cockpit (all 100 queries) | 0.1228 | [0.097, 0.1543] | 505 claims |
| Unsupported claims (3B judge), dev free llama3.2:1b | 0.1364 | [0.0844, 0.2129] | 110 claims |
| Unsupported claims (3B judge), dev free llama3.2:3b | 0.0617 | [0.0267, 0.1365] | 81 claims |
| Unsupported claims (3B judge), dev free llama3.1:8b | 0.0385 | [0.0151, 0.0947] | 104 claims |
| Unsupported claims (3B judge), dev anchored llama3.2:3b | 0.24 | [0.1575, 0.3478] | 75 claims |
| Unsupported claims (3B judge), dev anchored llama3.1:8b | 0.1845 | [0.1214, 0.2702] | 103 claims |
| Slot test set (test), precision | 1.0 | [0.8754, 1.0] | 27 |
| Slot test set (test), recall | 0.871 | [0.7115, 0.9487] | 31 |
| Slot test set (test), items_fully_correct | 0.9 | [0.7695, 0.9604] | 40 |
| Retrieval test p5, `bm25` | 0.845 | [0.7875, 0.8975] | 80 queries |
| Retrieval test map10, `bm25` | 0.7798 | [0.7167, 0.838] | 80 queries |
| Retrieval test p5, `ce` | 0.8975 | [0.855, 0.935] | 80 queries |
| Retrieval test map10, `ce` | 0.8297 | [0.7738, 0.8813] | 80 queries |
| Retrieval test p5, `ce_recency` | 0.81 | [0.7475, 0.8675] | 80 queries |
| Retrieval test map10, `ce_recency` | 0.7434 | [0.6735, 0.8102] | 80 queries |
| Retrieval test p5, `faiss` | 0.8075 | [0.7475, 0.8625] | 80 queries |
| Retrieval test map10, `faiss` | 0.7326 | [0.6645, 0.7967] | 80 queries |
| Retrieval test p5, `hybrid` | 0.79 | [0.7225, 0.8525] | 80 queries |
| Retrieval test map10, `hybrid` | 0.7241 | [0.6532, 0.7894] | 80 queries |
| Retrieval test p5, `rrf` | 0.8625 | [0.81, 0.91] | 80 queries |
| Retrieval test map10, `rrf` | 0.7925 | [0.7336, 0.8493] | 80 queries |

## Reading

- Overlapping intervals mean the data do not separate the two numbers; e.g. the 3B and 8B test hallucination rates overlap almost completely.
- Dev has 20 queries: a one-claim or one-query difference on dev is within noise, which is why dev choices (retrieval `faiss`, generator `llama3.1:8b`) did not carry over to test.
- Claim intervals treat claims as independent; claims from one answer are correlated, so the true intervals are wider.
- Intervals do not include label error (provisional LLM labels) or judge error.
