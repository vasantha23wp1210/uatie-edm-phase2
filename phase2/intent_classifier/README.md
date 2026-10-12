# phase2/intent_classifier (Task 1)

Status: **stopped**, waiting for your 400 labels.

```powershell
python phase2\intent_classifier\export.py --phase1 ..\uatie-edm-phase1   # judgements\intent_to_label.csv, splits\gold_key.csv, REPORT.md
```

Fill `your_intent` in `judgements/intent_to_label.csv` with a value from `judgements/allowed_labels.txt` and return it as
`judgements/intent_labelled.csv`. The rule label is hidden in `splits/gold_key.csv`; don't open it while labelling.
