# Task 7: grounded generation

## Method

- LangChain `ChatOllama`, model `llama3.2:3b`, options {'temperature': 0, 'seed': 42, 'num_ctx': 4096, 'num_predict': 300}, JSON output.
- Context: ce top-5 (the primary Task 3 system: BM25 + FAISS candidates re-ranked by the cross-encoder), Phase 1 risk level of the top-1 incident, candidate phraseology (Task 6 filled phrase + top-3 crossmap phrases).
- Grounding filter: keep a summary sentence only if >= 50% of its content words occur in the validly cited reports (all five if none cited); keep a hazard if >= 50% occur in the retrieved reports. Dropped 16 summary sentences and 7 hazards. The unfiltered answer is judged too (below).
- Insufficient evidence when the top-1 sigmoid(cross-encoder logit) < 0.9837 (max balanced accuracy on dev; dev queries only).
- Validation: citations outside the retrieved IDs and phrases not verbatim in the crossmap (or its Task 6 filled form) are removed and counted; an answer left with no valid citation is `no_valid_citation`.
- Judge: `llama3.2:3b` splits each answer into claims and marks each supported / unsupported by the retrieved summaries.

## Numbers

| Status | Answers |
|---|---:|
| ok | 38 |
| ok_after_filtering | 35 |
| insufficient_evidence | 16 |
| no_grounded_sentence | 11 |

- Citation validity: 1.0 (0 of 392 citations rejected).
- Phraseology rejected as not verbatim: 56 of 210.
- Before the grounding filter (raw answers, same judge): 0.101 (53 of 525 claims). The filter is lexical and fixed in advance; the judge is the same 3B model, so the before/after drop is not independent evidence.
- Hallucination rate: 0.0977 of claims (46 of 471); mean per answer 0.1005. Judge invalid JSON: 1.
- LLM latency on this VM (CPU): p50 11750.5 ms, p95 22712.0 ms (num_predict 300). Streaming: time to first token p50 2672.9 ms; the JSON answer is only usable once complete, so streaming does not shorten time to a validated answer. Retrieval incl. cross-encoder: p50 518.4 ms, p95 678.6 ms. Target <= 5 s end to end: MISSED on CPU; meeting it needs a GPU (or a smaller model, untested here).
- 0 of 84 answered queries took > 60 s.

## Run-to-run variation

`compare_runs.py` compares the two archived FAISS-context runs, which had identical inputs (`out/run_prev/` vs `out/run_faiss_claude/`; this CE run has different inputs and is not part of it). Same retrieval lists 100/100, same candidate phraseology 100/100, threshold [0.6164] vs [0.6164], same insufficient-evidence set: True.

- Identical raw LLM output: 13 of 59 generated answers, despite temperature 0 and a fixed seed; Ollama on CPU is not bit-reproducible across runs here.
- Answer status changed for 3 queries.
- Hallucination rate: 0.0709 (29/409) in the first run vs 0.1026 (43/419) in the repeat, so a single run's rate is not stable to two decimals.

## User ratings (earlier run)

Ratings: human (user); notes refined with Claude (`rate/answers_rated_run_prev.csv`, scored by `score_ratings.py`). They rate the 30 answers exported from the earlier run (`out/run_prev/`); every row's text was checked against that run.

- Distribution (rating: answers): 1: 1, 2: 9, 3: 18, 4: 2, 5: 0. Mean 2.7 (SD 0.65), median 3.0; 2 of 30 rated 4 or 5, 10 rated 1 or 2.
- Mean by split: dev 2.67 (n=9), test 2.71 (n=21); by status: ok 2.67 (n=15), ok_after_filtering 2.73 (n=15); by risk: CRITICAL 2.82 (n=17), HIGH 2.62 (n=8), MEDIUM 2.4 (n=5).
- Spearman between rating and the LLM judge's unsupported-claim fraction for the same answers: rho -0.192 (p 0.31, n 30); no significant association at n = 30, so the LLM judge's rate cannot stand in for user ratings.
- 0 of the 30 rated answers are identical in the current (CE-context) run, so these ratings describe the earlier FAISS-context run, not `out/answers.jsonl`; the current answers have no user ratings.

## Failures and limits

- The judge is an LLM, and by default the same 3B model as the generator; the hallucination rate is an automatic estimate, not a human judgement.
- Claims are judged against ASRS summaries only (no full narratives in the Phase 1 data).
- Retrieval relevance labels behind the dev threshold include provisional LLM labels (LLM-judged (Claude), pending user review).
- `rate/answers_to_rate.csv`: 30 answers (seed 42) from the current run; not rated (the user ratings above are of the earlier run).

## Reproduce

```
ollama pull llama3.2:3b
python phase2/generation/generate.py --phase1 ../uatie-edm-phase1
python phase2/generation/judge.py --phase1 ../uatie-edm-phase1
```
