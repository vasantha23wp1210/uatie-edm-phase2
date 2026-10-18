# Human claim check (50 ground-answer claims, 3B ground run, test)

All numbers come from `results_claims.json`, written by `score_claims.py`.

- Provenance: human (user), blind to the judge key. Levels (user's definitions): yes = fully supported, partial = 50-75% correct, no / almost no = unsupported. Rows marked only `dupe` (repeated query) carry no verdict and are excluded: 11 of 50; 39 rated.
- Rated levels: {'supported': 19, 'partial': 18, 'unsupported': 2}.
- Sampling: 25 judge-supported + 25 judge-unsupported claims from 384 test claims, of which the 3B judge called 42 unsupported; pool estimates re-weight the two strata (10,000 parametric bootstrap draws per stratum, seed 42).

| Rate | rated sample | judge-supported stratum | judge-unsupported stratum | pool estimate [95% CI] |
|---|---|---|---|---|
| Unsupported | 2/39 | 0/19 | 2/20 | 0.0109 [0.0, 0.0273] |
| Not fully supported (unsupported + partial) | 20/39 | 8/19 | 12/20 | 0.4406 [0.2477, 0.6336] |

## Judges against the human labels (rated claims)

| Judge | reference | flagged and human-bad | flagged, human-ok | missed | sensitivity | specificity | agreement |
|---|---|---|---|---|---|---|---|
| llm_3b | unsupported | 2 | 18 | 0 | 1.0 | 0.5135 | 0.5385 |
| llm_3b | not_fully_supported | 12 | 8 | 8 | 0.6 | 0.5789 | 0.5897 |
| nli | unsupported | 0 | 10 | 2 | 0.0 | 0.7297 | 0.6923 |
| nli | not_fully_supported | 3 | 7 | 17 | 0.15 | 0.6316 | 0.3846 |

## Limits

- One rater, 39 rated claims; intervals are wide.
- `partial` mixes claim support with off-topic cited reports in the notes; both readings are reported.
- LLM-judge sensitivity/specificity here are on a sample balanced by the judge's own verdict; they describe this sample.
- The claims come from the 3B ground run (generation/out), not the 8B run.
