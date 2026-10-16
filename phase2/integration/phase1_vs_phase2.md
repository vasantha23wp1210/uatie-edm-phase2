# Phase 1 vs Phase 2

Every number below is read from a task's `results.json`. Labels marked provisional are LLM-judged (Devin), pending user review.

## Retrieval (test, rel >= 1, provisional)

| System     |    p1 |    p5 |   p10 |   mrr |   map10 |   recall10 |   ndcg10 |
|:-----------|------:|------:|------:|------:|--------:|-----------:|---------:|
| hybrid     | 0.8   | 0.79  | 0.795 | 0.875 |   0.722 |      0.333 |    0.586 |
| bm25       | 0.912 | 0.845 | 0.824 | 0.936 |   0.78  |      0.353 |    0.657 |
| faiss      | 0.863 | 0.807 | 0.8   | 0.913 |   0.732 |      0.339 |    0.611 |
| rrf        | 0.95  | 0.858 | 0.829 | 0.963 |   0.784 |      0.35  |    0.682 |
| ce         | 0.963 | 0.892 | 0.853 | 0.981 |   0.823 |      0.378 |    0.78  |
| ce_recency | 0.825 | 0.82  | 0.796 | 0.891 |   0.744 |      0.345 |    0.66  |

Dev-selected system: `faiss`. Phase 1 = `hybrid`.

## NER (Phase 1 = rules, Phase 2 = spaCy; provisional)

| Entity        |   Support |   Rules F1 |   spaCy F1 |
|:--------------|----------:|-----------:|-----------:|
| RUNWAY        |        13 |      1     |      0.96  |
| FLIGHT_LEVEL  |         5 |      1     |      1     |
| ALTITUDE      |        10 |      0.667 |      0.375 |
| HEADING       |         6 |      1     |      0.923 |
| FREQUENCY     |         6 |      0.857 |      0.923 |
| CALLSIGN      |         0 |      0     |      0     |
| AIRCRAFT_TYPE |        30 |      0.717 |      0.9   |
| AIRPORT       |        20 |      0.385 |      0.706 |
| ALL           |        90 |      0.755 |      0.83  |

## Phraseology crossmap

| Tier             |   Phase 1 |   Phase 2 TF-IDF |   Phase 2 embedding |
|:-----------------|----------:|-----------------:|--------------------:|
| No Match         |       260 |               25 |                   0 |
| Partial Match    |       124 |               41 |                  25 |
| Close Match      |        65 |               47 |                  63 |
| Exact/Near-Exact |        65 |              401 |                 426 |

CAA-unique %: Phase 1 50.58, Phase 2 TF-IDF 4.86, embedding 0.0. Ontology RDF triples: Phase 1 7883, Phase 2 8579.

## Intent (agreement with provisional labels)

| Model                |   Test accuracy |   Test macro-F1 |
|:---------------------|----------------:|----------------:|
| keyword_rules_phase1 |           0.333 |           0.377 |
| logreg_minilm        |           0.433 |           0.398 |
| distilbert           |           0.403 |           0.41  |

## Slot filling

Status over 100 queries: {'incomplete': 62, 'no_slots': 37, 'complete': 1}; slots filled 3 of 91.

## Generation

Model `llama3.2:3b`; status {'insufficient_evidence': 41, 'ok': 36, 'ok_after_filtering': 23}; citation validity 1.0; hallucination rate (LLM judge) 0.0709.

## Latency (this VM, CPU)

```
{
 "pipeline_ms": {
  "p50": 8.8,
  "p95": 18.4,
  "n": 100
 },
 "llm_ms": {
  "p50": 10540.4,
  "p95": 255333.7,
  "n": 100
 },
 "end_to_end_ms": {
  "p50": 10550.7,
  "p95": 255342.5,
  "n": 100
 }
}
```

- `pipeline_ms`: query embedding + FAISS top-5, spaCy NER on query + top-1 summary, TF-IDF phrase match and slot filling, measured here (cross-encoder and intent lookup not included).
- `llm_ms`: per-query LLM time recorded by `generation/generate.py`; it is 0 for 41 insufficient-evidence queries (no LLM call), and its tail includes queries run while other CPU jobs shared this VM (see `generation/REPORT.md`).
