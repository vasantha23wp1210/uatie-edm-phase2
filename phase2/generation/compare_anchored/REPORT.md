# Task 7, step 4: sentence-anchored answers vs free answers, dev only

All numbers come from `results.json`, written by `select_anchored.py`. Test queries were not generated here.

## Method

- `generate.py --prompt anchored`: every retrieved report is split into sentences with IDs (`R2.S3`). The LLM returns each claim and hazard with the ID of the sentence it restates. A claim is kept only if its ID exists and at least half of its content words occur in that one sentence; then the same grounding filter and phraseology check as the free prompt apply.
- If the model writes a sentence ID as the claim text, the cited sentence itself is used (counted as `text_was_id`). This fallback and one line of format example were added after the first 4 dev answers of the first anchored run showed this format error; that run was discarded and nothing else was changed.
- Same retrieval context, Ollama options, judge (`llama3.2:3b`) and dev queries as `compare/`.
- Rule fixed before the anchored runs: lowest filtered hallucination on dev; within 1 point(s) of the lowest tie, lowest LLM p50 wins.

## Dev results

| Candidate | claims | unsupported | hallucination (filtered) | hallucination (raw) | LLM p50 ms | LLM p95 ms | anchors | statuses |
|---|---|---|---|---|---|---|---|---|
| free llama3.2:3b | 81 | 5 | 0.0617 | 0.0485 | 16217.1 | 21206.2 | - | {'ok_after_filtering': 8, 'ok': 7, 'no_grounded_sentence': 4, 'insufficient_evidence': 1} |
| anchored llama3.2:3b | 75 | 18 | 0.24 | 0.3119 | 11604.9 | 13858.1 | {'claims': 41, 'hazards': 56, 'bad_anchor': 0, 'text_was_id': 0, 'unsupported_by_anchor': 23} | {'ok': 9, 'ok_after_filtering': 8, 'no_grounded_sentence': 2, 'insufficient_evidence': 1} |
| free llama3.1:8b | 104 | 4 | 0.0385 | 0.0521 | 43441.2 | 62963.3 | - | {'ok': 10, 'ok_after_filtering': 8, 'no_grounded_sentence': 1, 'insufficient_evidence': 1} |
| anchored llama3.1:8b | 103 | 19 | 0.1845 | 0.1887 | 25113.6 | 39352.3 | {'claims': 38, 'hazards': 57, 'bad_anchor': 0, 'text_was_id': 0, 'unsupported_by_anchor': 6} | {'ok': 13, 'ok_after_filtering': 6, 'insufficient_evidence': 1} |

Chosen on dev: **free llama3.1:8b** (within margin: free llama3.1:8b).

## Limitations

- 20 dev queries; differences of a few claims are within noise.
- The judge is `llama3.2:3b`; it also marks about 12% of claims unsupported in the cockpit answers, which are copied word for word from the cited reports (`phase2/edge/REPORT.md`), so these rates carry that error.
- The anchor check is lexical (word overlap with one sentence), not semantic entailment.
