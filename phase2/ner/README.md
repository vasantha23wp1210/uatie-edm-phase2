# phase2/ner (Task 2)

Status: **stopped**, waiting for your corrections of 260 sentences.

```powershell
python phase2\ner\export.py --phase1 ..\uatie-edm-phase1   # judgements\ner_to_correct.csv, splits\ner_split.csv, REPORT.md
```

Edit the `corrected` column: entities are written `[span](LABEL)`. Labels: RUNWAY, FLIGHT_LEVEL, ALTITUDE, HEADING,
FREQUENCY, CALLSIGN, AIRCRAFT_TYPE, AIRPORT. Only add, remove or move markup; don't change the text. Return it as
`judgements/ner_corrected.csv`. Rules used for pre-annotation (and the rule-only baseline) are in `rules.py`.
