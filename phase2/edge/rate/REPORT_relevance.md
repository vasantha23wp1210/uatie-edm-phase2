# Answer relevance: user ratings of the current answers

All numbers come from `results_relevance.json`, written by `score_relevance.py`.

- Provenance: human (user) ratings; to confirm whether the notes were refined with an LLM.
- Cockpit: 20 extractive answers (`cockpit_to_rate.csv`); ground: 30 answers of the dev-chosen `llama3.1:8b` on test (`ground_to_rate.csv`, seed 42). Both files match their exports row for row.
- Scale 1-5 (5 = directly answers the query). Target: mean >= 4.0. Mean intervals: bootstrap over answers (10,000 resamples, seed 42).

| Profile | n | mean [95% CI] | SD | median | share >= 4 | distribution 1/2/3/4/5 | target met |
|---|---|---|---|---|---|---|---|
| cockpit | 20 | 3.4 [2.95, 3.85] | 1.0463 | 4.0 | 0.55 | 1/3/5/9/2 | False |
| ground | 30 | 3.6667 [3.4, 3.9] | 0.7112 | 4.0 | 0.6667 | 0/2/8/18/2 | False |

## Themes in the notes (keyword counts, fixed patterns in the script)

- cockpit: phraseology irrelevant or off topic: 7 of 20; generic restatement of the query: 0 of 20; retrieved reports do not match the query: 5 of 20
- ground: phraseology irrelevant or off topic: 7 of 30; generic restatement of the query: 7 of 30; retrieved reports do not match the query: 1 of 30

## Lowest-rated answers

- cockpit Q004 (1): new runway assigned close to touchdown: No report involves a runway reassignment near touchdown; keyword matches only (touchdown, new runway construction, wake)
- cockpit Q019 (2): runway change to shorter runway on approach: Only the LAS late runway change fits, and shorter runway isn't stated; others are traffic conflicts; frequency phraseology irrelevant
- cockpit Q015 (2): runway change issued while vectoring to final: Only APA (runway change on short final) is close; other two are vectoring errors without a runway change
- ground Q023 (2): ATC phraseology ambiguity during approach clearance: Generic restatement only; TCAS RA / ROGER phraseology has nothing to do with approach clearance ambiguity
- ground Q046 (2): tailwind landing causing unstable approach: Answer is a dangling fragment that never explains tailwind causing an unstable approach; phraseology off
- ground Q074 (3): icing encounter on approach requiring go around: First sentence restates the query generically; high winds / crew errors and flap drive failure are off topic

## Limits

- One rater; no second rater, so no agreement figure.
- Cockpit and ground were rated on different query samples (20 vs 30), so the two means are not a paired comparison.
- The earlier 2.70 (30 answers of the FAISS-context run, `generation/results_ratings.json`) describes a different system and is kept for reference only.
- Theme counts are keyword matches on free-text notes, not a coded analysis.
