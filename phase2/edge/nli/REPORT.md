# Judge calibration with controls, and an NLI judge

All numbers come from `results.json`, written by `nli_judge.py`.

## Method

- Controls (no gold labels): 252 positives, i.e. cockpit summary sentences copied verbatim from the cited report, and 252 negatives, one altered copy of each ({'outcome': 129, 'unrelated': 63, 'negation': 38, 'number': 22}). Altered copies are easier to catch than real hallucinations, so sensitivity here is an upper bound.
- NLI judge: `cross-encoder/nli-deberta-v3-base` (revision `6c749ce342`), CPU, deterministic. A summary sentence is supported if P(entailment) reaches the threshold for its best premise (a cited report, or one sentence of it). A first run with whole reports as premises only rejected 17 of 252 verbatim positives, all from multi-sentence reports, so sentence premises were added before the results below; that check looked at controls from dev and test queries (controls only, no task labels). Threshold chosen on the controls from dev queries (max balanced accuracy): 0.05. Key-hazard strings are not judged by NLI; the LLM judge splits summary + hazards into claims, so units differ.
- Rogan-Gladen correction: true rate = (observed + specificity - 1) / (sensitivity + specificity - 1), clamped to [0, 1], using held-out control sensitivity and specificity.

## Judge accuracy on held-out controls (test queries)

| Judge | sensitivity (negatives caught) | 95% CI | specificity (positives accepted) | 95% CI |
|---|---|---|---|---|
| NLI (195 neg / 195 pos) | 0.9692 | [0.9345, 0.9858] | 1.0 | [0.9807, 1.0] |
| LLM `llama3.2:3b` (96 judged, 4 invalid) | 0.8913 | [0.7696, 0.9527] | 0.54 | [0.404, 0.6703] |
| LLM `llama3.1:8b` (99 judged, 1 invalid) | 0.898 | [0.7824, 0.9556] | 0.84 | [0.7149, 0.9166] |

NLI negatives caught by kind (held out): {'negation': '29/33', 'number': '18/18', 'outcome': '95/95', 'unrelated': '47/49'}.
LLM `llama3.2:3b` negatives caught by kind: {'negation': '8/8', 'number': '2/4', 'outcome': '21/24', 'unrelated': '10/10'}.
LLM `llama3.1:8b` negatives caught by kind: {'negation': '7/9', 'number': '3/4', 'outcome': '26/27', 'unrelated': '8/9'}.

## Unsupported rate by answer set

| Answer set | sentences | NLI unsupported | NLI rate [95% CI] | NLI corrected | LLM 3B judge rate [95% CI] (claims) | LLM 3B corrected |
|---|---|---|---|---|---|---|
| cockpit (extractive) | 251 | 0 | 0.0 [0.0, 0.0151] | 0.0 | 0.1228 [0.097, 0.1543] (505) | 0.0 |
| ground llama3.2:3b (dev+test) | 81 | 36 | 0.4444 [0.3412, 0.5527] | 0.4585 | 0.0977 [0.074, 0.1278] (471) | 0.0 |
| ground llama3.1:8b (test) | 83 | 43 | 0.5181 [0.4122, 0.6224] | 0.5346 | 0.1124 [0.0833, 0.15] (347) | 0.0 |
| dev free llama3.2:1b | 19 | 5 | 0.2632 [0.1181, 0.4879] | 0.2716 | 0.1364 [0.0844, 0.2129] (110) | 0.0 |
| dev free llama3.2:3b | 16 | 5 | 0.3125 [0.1416, 0.556] | 0.3224 | 0.0617 [0.0267, 0.1365] (81) | 0.0 |
| dev free llama3.1:8b | 22 | 8 | 0.3636 [0.1973, 0.5705] | 0.3752 | 0.0385 [0.0151, 0.0947] (104) | 0.0 |
| dev anchored llama3.2:3b | 27 | 11 | 0.4074 [0.2451, 0.5927] | 0.4203 | 0.24 [0.1575, 0.3478] (75) | 0.0 |
| dev anchored llama3.1:8b | 37 | 0 | 0.0 [0.0, 0.0941] | 0.0 | 0.1845 [0.1214, 0.2702] (103) | 0.0 |

## Reading these numbers

- On controls the NLI judge is near-perfect, but it marks 44-52% of ground LLM sentences unsupported. Ground answers paraphrase and combine several reports (e.g. "reported in multiple incidents"), which no single premise entails. That says NLI is strict on abstractive text, not that half the answers are invented; its specificity on verbatim positives does not carry over to paraphrased sentences.
- The 3B LLM judge accepts only 0.54 of verbatim positives, so it flags about 0.46 of fully grounded sentences. Every observed 3B rate is below that false-flag rate, so the Rogan-Gladen estimate clamps to 0 for every answer set. The correction is therefore uninformative here, not evidence that the ground answers are fully grounded.
- LLM-judge rates are per claim (summary and hazards split into claims), NLI rates per summary sentence. The cockpit row counts 251 sentences with the sentence splitter used for answer sets; the controls use the 252 sentences of the exact-match check.
- Hallucination <= 5% for the ground profile therefore stays not verifiable with these judges.

## Limitations

- Positives are verbatim, so they are the easiest possible supported claims; real paraphrased claims are harder, and specificity on them may be lower.
- Negatives are rule-made alterations, not real model errors.
- The cockpit row is a sanity check: its sentences are verbatim by construction (the controls' positives).
- The human claim check (`phase2/edge/rate/claims_to_check.csv`) is still the reference; none of these judges replaces it.
