# Task 2: Named entity recognition (STOPPED, waiting for corrections)

Numbers come from `export_results.json`, written by `export.py`. No model has been trained yet.

## Export
- 20,000 ASRS summaries sampled (seed 42), split into 28,978 candidate sentences (30–300 chars, deduplicated).
- Exported 260: 200 with at least one rule entity (round-robin over types), 60 with none. Split: 200 train / 60 test.
- Rules: spaCy `EntityRuler` patterns in `rules.py`.

## Rule-entity support (before your corrections)
| Entity | Candidate sentences | Export |
|---|---|---|
| RUNWAY | 1132 | 54 |
| FLIGHT_LEVEL | 673 | 44 |
| ALTITUDE | 1325 | 54 |
| HEADING | 16 | 16 |
| FREQUENCY | 33 | 33 |
| CALLSIGN | 0 | 0 |
| AIRCRAFT_TYPE | 9729 | 95 |
| AIRPORT | 398 | 34 |

ASRS narratives are de-identified (airports often `ZZZ`, callsigns `ACR X`), so CALLSIGN and AIRPORT are expected to be rare or templated. Across all 111,484 ASRS summaries, 369 mention the word callsign/call sign but only 4 contain a templated callsign (`acr x`-style); real callsigns are removed by de-identification. Support per entity will be reported after correction; any entity with fewer than 10 test mentions will be reported as too rare to evaluate.

## What you need to do
Edit the `corrected` column of `judgements/ner_to_correct.csv`: fix, add or remove `[span](LABEL)` markup using only RUNWAY, FLIGHT_LEVEL, ALTITUDE, HEADING, FREQUENCY, CALLSIGN, AIRCRAFT_TYPE, AIRPORT. Do not change any other text. Return it as `judgements/ner_corrected.csv`.
