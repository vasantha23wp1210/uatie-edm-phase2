# Deployment profiles: ground (LLM) and cockpit (CPU only)

All numbers come from `results.json`, written by `report.py` from the measured outputs listed in its docstring.

## Profiles

- **Ground** (ATC, safety analysts): full Task 7 pipeline with an LLM-written answer (`phase2/generation`). GPU recommended; only CPU latency is measured in this repository.
- **Cockpit** (CPU only, no LLM): `cockpit.py`. Same retrieval (BM25 + FAISS, cross-encoder top-5) and dev-chosen insufficient-evidence threshold; the summary is the best TF-IDF sentence of each top-3 report copied verbatim and cited; the phrase is the first verified crossmap candidate; risk from the Phase 1 matrix. Design fixed before measuring.
- **NPU / DPU**: the encoder and cross-encoder are exported to ONNX int8 (`onnx_export.py`). ONNX Runtime NPU execution providers can load the same files; that path is not measured (no NPU or DPU on this machine).

## Cockpit profile (100 gold queries, dev + test)

- Statuses: {'ok': 84, 'insufficient_evidence': 16}.
- End-to-end latency (retrieval + NER + phrase + extractive): p50 616.8 ms, p95 956.9 ms, max 1334.6 ms. Target <= 5 s: met (p95).
- Retrieval p50 604.9 ms, p95 944.4 ms.
- Hallucination (same claim judge `llama3.2:3b` as Task 7, summary only): 0.1228 (62 of 505 claims, 84 answers; judge invalid JSON 0). Target <= 5%: missed.
- Fabricated summary text: 0 by construction. The cockpit copies report sentences and does not generate text; a deterministic exact-match check confirms 252 of 252 sentences. This is a stronger statement than any LLM-judged rate below.
- String check (deterministic): 252 of 252 summary sentences are, in order, a verbatim sentence of the report they cite. Each claim the judge marks unsupported therefore comes from text copied from the retrieved reports: these are judge errors (or claims the judge split or reworded), so this rate is a lower bound on the judge's false-unsupported rate on fully grounded text. The same judge scores the ground profile, so its hallucination rates carry the same judge error.
- Hazard tags are Phase 1 database fields shown as-is and are not judged.
- Second judge `llama3.1:8b` on the same verbatim answers (control): 0.1259 (91 of 723 claims marked unsupported; invalid JSON 2). A lower rate here means fewer false-unsupported verdicts on known-grounded text. Neither judge is independent proof; a human check of a sample of claims is needed to fix the true rate.

## ONNX int8 (dev only)

- Size MB: {'encoder': {'fp32': 86.2, 'int8': 21.8}, 'cross_encoder': {'fp32': 86.8, 'int8': 22.0}}.
- Query encoder p50: torch fp32 14.8 ms, ONNX int8 2.7 ms; FAISS top-50 overlap 0.708.
- Cross-encoder p50 per query: torch fp32 1671.6 ms, ONNX int8 519.8 ms; top-5 overlap 0.53; dev P@5 0.66 (fp32) vs 0.65 (int8).
- not measured: no NPU/DPU on this machine; ONNX Runtime QNN / OpenVINO execution providers load the same files.

## Ground profile: generator comparison (dev only)

| Model | hallucination (filtered) | claims | LLM p50 ms | LLM p95 ms |
|---|---|---|---|---|
| llama3.2:1b | 0.1364 | 110 | 8287.7 | 10807.5 |
| llama3.2:3b | 0.0617 | 81 | 16217.1 | 21206.2 |
| llama3.1:8b | 0.0385 | 104 | 43441.2 | 62963.3 |

Chosen on dev: `llama3.1:8b` (lowest filtered hallucination on dev; models within 1 percentage point(s) of the lowest tie, and the lowest LLM p50 latency among them wins). Dev is 20 queries, so one claim moves a rate by about 1 point; the judge is the 3B model, which may favour its own answers.

## Ground profile: sentence-anchored prompt (dev only)

Each claim must cite a report sentence ID (`R2.S3`); claims with an unknown ID or too little word overlap with their sentence are dropped (`phase2/generation/compare_anchored/REPORT.md`).

| Candidate | hallucination (filtered) | claims | LLM p50 ms |
|---|---|---|---|
| free llama3.2:3b | 0.0617 | 81 | 16217.1 |
| anchored llama3.2:3b | 0.24 | 75 | 11604.9 |
| free llama3.1:8b | 0.0385 | 104 | 43441.2 |
| anchored llama3.1:8b | 0.1845 | 103 | 25113.6 |

Chosen on dev: free llama3.1:8b (lowest filtered hallucination on dev; within 1 point(s) of the lowest tie, lowest LLM p50 wins).

## Ground profile: chosen `llama3.1:8b` on test (run once)

- Statuses: {'ok': 36, 'ok_after_filtering': 22, 'insufficient_evidence': 15, 'no_grounded_sentence': 7}.
- Hallucination (judge `llama3.2:3b`): 0.1124 (39 of 347 claims, 58 answers; invalid JSON 0). Subject to the judge error shown above.
- LLM latency p50 41283.3 ms, p95 64152.0 ms; first token p50 7544.7 ms (CPU). Target <= 5 s: missed on CPU.
- Baseline `llama3.2:3b` (in `phase2/generation/out`) on the same test queries, same judge: 0.1094 (42 of 384 claims) after the filter, 0.1132 before it; LLM p50 11613.9 ms, p95 24377.5 ms. The chosen model was kept; it was not switched after seeing test.
- Primary run in `phase2/generation/out` (dev + test, 100 queries): model `llama3.2:3b`, hallucination 0.0977 after the filter.

## Limitations

- The cockpit answer is less fluent than an LLM answer and has no key-hazards list beyond the Phase 1 tags.
- Answer relevance needs new user ratings for these answers; none exist yet.
- GPU and NPU latencies are not measured here.
