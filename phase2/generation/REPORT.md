# Task 7: grounded generation

## Method

- LangChain `ChatOllama`, model `llama3.2:3b`, options {'temperature': 0, 'seed': 42, 'num_ctx': 4096, 'num_predict': 350}, JSON output.
- Context: faiss top-5 (the Task 3 system chosen on dev), Phase 1 risk level of the top-1 incident, candidate phraseology (Task 6 filled phrase + top-3 crossmap phrases).
- Insufficient evidence when the top-1 FAISS score < 0.6164 (max balanced accuracy on dev; dev queries only).
- Validation: citations outside the retrieved IDs and phrases not verbatim in the crossmap (or its Task 6 filled form) are removed and counted; an answer left with no valid citation is `no_valid_citation`.
- Judge: `llama3.2:3b` splits each answer into claims and marks each supported / unsupported by the retrieved summaries.

## Numbers

| Status | Answers |
|---|---:|
| insufficient_evidence | 41 |
| ok | 36 |
| ok_after_filtering | 23 |

- Citation validity: 1.0 (0 of 285 citations rejected).
- Phraseology rejected as not verbatim: 33 of 129.
- Hallucination rate: 0.0709 of claims (29 of 409); mean per answer 0.0782. Judge invalid JSON: 0.
- LLM latency on this VM (CPU): p50 12146.2 ms, p95 268515.1 ms.
- 17 of 59 answered queries took > 60 s (Q001, Q002, Q003, Q005, Q007, Q008, Q009, Q010, Q011, Q012, Q015, Q016, Q017, Q018, Q061, Q066, Q067); on this VM these coincided with other CPU jobs (Task 1 DistilBERT training, Task 1 ontology population), so p95 reflects contention.

## Failures and limits

- The judge is an LLM, and by default the same 3B model as the generator; the hallucination rate is an automatic estimate, not a human judgement.
- Claims are judged against ASRS summaries only (no full narratives in the Phase 1 data).
- Retrieval relevance labels behind the dev threshold include provisional LLM labels (LLM-judged (Devin), pending user review).
- `rate/answers_to_rate.csv`: 30 answers (seed 42) for your 1-5 rating; pending user rating.

## Reproduce

```
ollama pull llama3.2:3b
python phase2/generation/generate.py --phase1 ../uatie-edm-phase1
python phase2/generation/judge.py --phase1 ../uatie-edm-phase1
```
