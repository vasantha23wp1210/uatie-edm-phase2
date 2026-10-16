# phase2/intent_classifier (Task 1)

Status: done on provisional labels. All 400 gold labels are `LLM-judged (Claude), pending user review` (method: `phase2/labels/LABELLING_PROMPT.md`; Claude's notes on 33 ambiguous rows: `judgements/intent_ambiguity_notes.csv`). Devin's earlier provisional set is kept in `judgements/intent_labelled_devin.csv`.

```powershell
python phase2\intent_classifier\export.py --phase1 ..\uatie-edm-phase1               # judgements\intent_to_label.csv, splits\gold_key.csv
python phase2\intent_classifier\llm_judge\combine.py                   # optional: judgements\intent_labelled_devin.csv (Devin's superseded set)
python phase2\intent_classifier\train.py --phase1 ..\uatie-edm-phase1                # results.json, REPORT.md, out\intent_v2.csv (~70 min CPU)
python phase2\intent_classifier\ontology_population.py --phase1 ..\uatie-edm-phase1  # results_ontology.json, REPORT.md section
```

- `train.py`: keyword-rule baseline, logistic regression on MiniLM embeddings, DistilBERT (3 epochs); threshold and
  model chosen on dev only. `--skip-distilbert` for a fast run.
- `ontology_population.py`: reruns Phase 1 `src/rag/ontology_population.py` unmodified with `intent_v2`.

After your review, overwrite the judged file with your reviewed copy (same columns; set the notes column to `human` on rows you checked) and rerun only the last step(s); `combine.py` only rebuilds Devin's superseded `*_devin.csv` file, so it never overwrites the file you review.
