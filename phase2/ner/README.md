# phase2/ner (Task 2)

Status: done on provisional corrections. All 260 sentences are `LLM-judged (Claude), pending user review` (method: `phase2/labels/LABELLING_PROMPT.md`). Devin's earlier provisional set is kept in `judgements/ner_corrected_devin.csv`.

```powershell
python phase2\ner\export.py --phase1 ..\uatie-edm-phase1     # judgements\ner_to_correct.csv, splits\ner_split.csv
python phase2\ner\llm_judge\apply_edits.py   # optional: judgements\ner_corrected_devin.csv (Devin's superseded set; text checked unchanged)
python phase2\ner\train.py --phase1 ..\uatie-edm-phase1      # results.json, REPORT.md, out\model, out\test_predictions.csv
```

Entities are written `[span](LABEL)`: RUNWAY, FLIGHT_LEVEL, ALTITUDE, HEADING, FREQUENCY, CALLSIGN, AIRCRAFT_TYPE,
AIRPORT. 200 train / 60 test. Rules (`rules.py`) are the baseline.

After your review, overwrite `judgements/ner_corrected.csv` with your reviewed copy and rerun only `train.py`.
`apply_edits.py` only rebuilds Devin's superseded `ner_corrected_devin.csv`.
