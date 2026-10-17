# Task 7, step 3: generator comparison on dev only

All numbers come from `results.json`, written by `compare_models.py`. Test queries were not generated here.

## Method

- Same context (cross-encoder top-5), prompt, Ollama options, validation and grounding filter as `generate.py`; only the generator model changes.
- Judge fixed at `llama3.2:3b` for every model. It may favour answers from its own model (`llama3.2:3b`).
- Rule fixed before any run: lowest filtered hallucination on dev; models within 1 percentage point(s) of the lowest tie, and the lowest LLM p50 latency among them wins.
- CPU only; run with no other job on the machine. Ollama on CPU is not bit-reproducible across runs.

## Dev results

| Model | answers judged | claims | unsupported | hallucination (filtered) | hallucination (raw) | LLM p50 ms | LLM p95 ms | TTFT p50 ms | statuses |
|---|---|---|---|---|---|---|---|---|---|
| llama3.2:1b | 19 | 110 | 15 | 0.1364 | 0.2043 | 8287.7 | 10807.5 | 1960.0 | {'ok_after_filtering': 12, 'ok': 6, 'no_valid_citation': 1, 'insufficient_evidence': 1} |
| llama3.2:3b | 15 | 81 | 5 | 0.0617 | 0.0485 | 16217.1 | 21206.2 | 3373.1 | {'ok_after_filtering': 8, 'ok': 7, 'no_grounded_sentence': 4, 'insufficient_evidence': 1} |
| llama3.1:8b | 18 | 104 | 4 | 0.0385 | 0.0521 | 43441.2 | 62963.3 | 8957.4 | {'ok': 10, 'ok_after_filtering': 8, 'no_grounded_sentence': 1, 'insufficient_evidence': 1} |

Chosen on dev: **`llama3.1:8b`** (within margin: llama3.1:8b).

## Limitations

- 20 dev queries only; the rates rest on the claim counts above and are noisy.
- The judge is an LLM, not a human, and is one of the compared models.
