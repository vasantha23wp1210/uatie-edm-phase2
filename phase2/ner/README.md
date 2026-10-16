# phase2/ner (Task 2)

Status: done on provisional corrections. All 260 sentences are `LLM-judged (Devin), pending user review`.

```powershell
python phase2\ner\export.py --phase1 ..\uatie-edm-phase1     # judgements\ner_to_correct.csv, splits\ner_split.csv
python phase2\ner\llm_judge\apply_edits.py   # judgements\ner_corrected.csv from llm_judge\edits.csv (text checked unchanged)
python phase2\ner\train.py --phase1 ..\uatie-edm-phase1      # results.json, REPORT.md, out\model, out\test_predictions.csv
```

Entities are written `[span](LABEL)`: RUNWAY, FLIGHT_LEVEL, ALTITUDE, HEADING, FREQUENCY, CALLSIGN, AIRCRAFT_TYPE,
AIRPORT. 200 train / 60 test. Rules (`rules.py`) are the baseline.

After your review, overwrite `judgements/ner_corrected.csv` with your reviewed copy and rerun only `train.py`; do
**not** rerun `apply_edits.py`, which rebuilds the provisional file.
