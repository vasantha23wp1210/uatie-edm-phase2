# Task 6: slot filling

Fills bracketed slots in the matched CAA phrase from Task 2 NER entities; unknown, unmapped or ambiguous slots stay bracketed (never guessed).
Needs: `phase2/retrieval/eval/results.json`, `phase2/retrieval/runs/runs_top10.csv`, `phase2/ner/out/model`, `phase2/phraseology/out/CAA_ICAO_crossmap_v2.csv`.

Reproduce: `python phase2/slots/fill.py --phase1 ../uatie-edm-phase1`

Outputs: `out/slot_fills.csv`, `results.json`, `REPORT.md`.
