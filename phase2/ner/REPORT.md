# Task 2: Named entity recognition

All numbers come from `results.json`, written by `train.py`.

## Label provenance (read first)

- Corrected markup: {'LLM-judged (Claude), pending user review': 260}. While it is `pending user review`, the scores below measure agreement with provisional LLM corrections, not with human gold.

## Test (n = 60 sentences, exact span + label)

| Entity | test support | rules P | rules R | rules F1 | spaCy P | spaCy R | spaCy F1 |
|---|---|---|---|---|---|---|---|
| RUNWAY | 14 | 1.0000 | 0.9286 | 0.9630 | 1.0000 | 1.0000 | 1.0000 |
| FLIGHT_LEVEL | 5 | 1.0000 | 1.0000 | 1.0000 | 0.8333 | 1.0000 | 0.9091 |
| ALTITUDE | 10 | 0.8750 | 0.7000 | 0.7778 | 0.7500 | 0.6000 | 0.6667 |
| HEADING | 6 | 1.0000 | 1.0000 | 1.0000 | 0.8571 | 1.0000 | 0.9231 |
| FREQUENCY | 6 | 0.7500 | 1.0000 | 0.8571 | 1.0000 | 1.0000 | 1.0000 |
| CALLSIGN | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| AIRCRAFT_TYPE | 30 | 0.8261 | 0.6333 | 0.7170 | 0.8966 | 0.8667 | 0.8814 |
| AIRPORT | 20 | 0.8333 | 0.2500 | 0.3846 | 0.8667 | 0.6500 | 0.7429 |
| **micro (all)** | 91 | 0.8841 | 0.6703 | 0.7625 | 0.8941 | 0.8352 | 0.8636 |

Train entity counts: {'RUNWAY': 44, 'FLIGHT_LEVEL': 39, 'ALTITUDE': 44, 'HEADING': 14, 'FREQUENCY': 22, 'CALLSIGN': 0, 'AIRCRAFT_TYPE': 95, 'AIRPORT': 70}.

Too rare to evaluate (test support < 5): CALLSIGN.
CALLSIGN in particular cannot be evaluated: ASRS de-identifies callsigns, and none of the 260 sentences contains one.

- spaCy blank `en` + `ner`, 30 epochs, dropout 0.2, seed 42, CPU. 0 training spans were not token-aligned and were expanded to token boundaries.
- Per-sentence gold/rules/model entities: `out/test_predictions.csv`.
