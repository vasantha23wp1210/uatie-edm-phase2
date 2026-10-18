# Phase 1 vs Phase 2

Every number below is read from a task's `results.json`. Labels marked provisional are LLM-judged (Claude), pending user review.

## Retrieval (test, rel >= 1, provisional)

| System     |    p1 |    p5 |   p10 |   mrr |   map10 |   recall10 |   ndcg10 |
|:-----------|------:|------:|------:|------:|--------:|-----------:|---------:|
| hybrid     | 0.8   | 0.79  | 0.795 | 0.875 |   0.724 |      0.336 |    0.591 |
| bm25       | 0.912 | 0.845 | 0.824 | 0.936 |   0.78  |      0.351 |    0.66  |
| faiss      | 0.863 | 0.807 | 0.8   | 0.913 |   0.733 |      0.342 |    0.616 |
| rrf        | 0.95  | 0.863 | 0.833 | 0.96  |   0.792 |      0.351 |    0.691 |
| ce         | 0.963 | 0.897 | 0.853 | 0.981 |   0.83  |      0.376 |    0.773 |
| ce_recency | 0.85  | 0.81  | 0.79  | 0.899 |   0.743 |      0.34  |    0.653 |

Primary system (fixed in advance by the Phase 2 plan): `ce`; dev-selected system: `faiss`. Phase 1 = `hybrid`.

## NER (Phase 1 = rules, Phase 2 = spaCy; provisional)

| Entity        |   Support |   Rules F1 |   spaCy F1 |
|:--------------|----------:|-----------:|-----------:|
| RUNWAY        |        14 |      0.963 |      1     |
| FLIGHT_LEVEL  |         5 |      1     |      0.909 |
| ALTITUDE      |        10 |      0.778 |      0.667 |
| HEADING       |         6 |      1     |      0.923 |
| FREQUENCY     |         6 |      0.857 |      1     |
| CALLSIGN      |         0 |      0     |      0     |
| AIRCRAFT_TYPE |        30 |      0.717 |      0.881 |
| AIRPORT       |        20 |      0.385 |      0.743 |
| ALL           |        91 |      0.762 |      0.864 |

## Phraseology crossmap

| Tier             |   Phase 1 |   Phase 2 TF-IDF |   Phase 2 embedding |
|:-----------------|----------:|-----------------:|--------------------:|
| No Match         |       260 |               25 |                   0 |
| Partial Match    |       124 |               41 |                  25 |
| Close Match      |        65 |               47 |                  63 |
| Exact/Near-Exact |        65 |              401 |                 426 |

CAA-unique %: Phase 1 50.58, Phase 2 TF-IDF 4.86, embedding 0.0. Ontology RDF triples: Phase 1 7883, Phase 2 8579.

Verification of 20 sampled matches (human (user); notes refined with Claude): assigned tier correct for 16 of 20 (No Match 4/5, Partial Match 2/5, Close Match 5/5, Exact/Near-Exact 5/5); verdicts changed during the Claude review: M18, M20.

## Intent (agreement with provisional labels)

| Model                       |   Test accuracy |   Test macro-F1 |
|:----------------------------|----------------:|----------------:|
| keyword_rules_phase1        |           0.393 |           0.462 |
| logreg_minilm               |           0.403 |           0.445 |
| logreg_minilm_scaled        |           0.587 |           0.587 |
| logreg_minilm_silver_scaled |           0.483 |           0.496 |
| distilbert                  |           0.393 |           0.45  |
| distilbert_scaled           |           0.682 |           0.601 |
| distilbert_silver_scaled    |           0.637 |           0.607 |

## Slot filling

Status over 100 queries: {'incomplete': 63, 'no_slots': 37}; slots filled 0 of 91.

## Generation

Model `llama3.2:3b`; status {'ok': 38, 'ok_after_filtering': 35, 'insufficient_evidence': 16, 'no_grounded_sentence': 11}; citation validity 1.0; hallucination rate (LLM judge) 0.0977 after the grounding filter, 0.101 before it (same judge; not independent evidence).

User ratings (human (user); notes refined with Claude) of 30 answers from the earlier run (`generation/out/run_prev/`): mean 2.7, median 3.0, distribution {'1': 1, '2': 9, '3': 18, '4': 2, '5': 0}; 0 of 30 rated answers are identical in the current CE-context run, so the current answers have no user ratings.

## Latency (this VM, CPU)

```
{
 "pipeline_ms": {
  "p50": 541.5,
  "p95": 739.0,
  "n": 100
 },
 "llm_ms": {
  "p50": 11242.0,
  "p95": 21172.8,
  "n": 100
 },
 "end_to_end_ms": {
  "p50": 11780.1,
  "p95": 21836.7,
  "n": 100
 }
}
```

- `pipeline_ms`: BM25 top-50 + FAISS top-50, cross-encoder re-ranking of the union, spaCy NER on the query text, TF-IDF phrase match and slot filling, measured here (intent lookup not included).
- `llm_ms`: per-query LLM time recorded by `generation/generate.py`; it is 0 for 16 insufficient-evidence queries (no LLM call); run conditions and run-to-run variation are in `generation/REPORT.md`.
