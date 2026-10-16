# Task 2: Named entity recognition

All numbers come from `results.json`, written by `train.py`.

## Label provenance (read first)

- Corrected markup: {'LLM-judged (Devin), pending user review': 260}. While it is `pending user review`, the scores below measure agreement with provisional LLM corrections, not with human gold.

## Test (n = 60 sentences, exact span + label)

| Entity | test support | rules P | rules R | rules F1 | spaCy P | spaCy R | spaCy F1 |
|---|---|---|---|---|---|---|---|
| RUNWAY | 13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9231 | 0.9600 |
| FLIGHT_LEVEL | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ALTITUDE | 10 | 0.7500 | 0.6000 | 0.6667 | 0.5000 | 0.3000 | 0.3750 |
| HEADING | 6 | 1.0000 | 1.0000 | 1.0000 | 0.8571 | 1.0000 | 0.9231 |
| FREQUENCY | 6 | 0.7500 | 1.0000 | 0.8571 | 0.8571 | 1.0000 | 0.9231 |
| CALLSIGN | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| AIRCRAFT_TYPE | 30 | 0.8261 | 0.6333 | 0.7170 | 0.9000 | 0.9000 | 0.9000 |
| AIRPORT | 20 | 0.8333 | 0.2500 | 0.3846 | 0.8571 | 0.6000 | 0.7059 |
| **micro (all)** | 90 | 0.8696 | 0.6667 | 0.7547 | 0.8765 | 0.7889 | 0.8304 |

Train entity counts: {'RUNWAY': 41, 'FLIGHT_LEVEL': 40, 'ALTITUDE': 44, 'HEADING': 14, 'FREQUENCY': 22, 'CALLSIGN': 0, 'AIRCRAFT_TYPE': 95, 'AIRPORT': 72}.

Too rare to evaluate (test support < 5): CALLSIGN.
CALLSIGN in particular cannot be evaluated: ASRS de-identifies callsigns, and none of the 260 sentences contains one.

- spaCy blank `en` + `ner`, 30 epochs, dropout 0.2, seed 42, CPU. 0 training spans were not token-aligned and were expanded to token boundaries.
- Per-sentence gold/rules/model entities: `out/test_predictions.csv`.
