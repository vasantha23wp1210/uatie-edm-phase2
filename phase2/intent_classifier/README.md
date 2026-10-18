# phase2/intent_classifier (Task 1)

Status: done on provisional labels. The 400 gold labels are `LLM-judged (Claude), section 5 relabel` (method: `phase2/labels/LABELLING_PROMPT.md` sections 2 and 5; borderline rows: `judgements/intent_ambiguity_notes.csv`). The first Claude pass is kept in `judgements/intent_labelled_claude_v1.csv` (with `intent_ambiguity_notes_claude_v1.csv`) and Devin's earlier set in `judgements/intent_labelled_devin.csv`. Training also uses 2,000 Claude labels in `scaled/labelled_batch{1..4}.csv` (exported by `export_train.py`; no gold record included).

```powershell
python phase2\intent_classifier\export.py --phase1 ..\uatie-edm-phase1               # judgements\intent_to_label.csv, splits\gold_key.csv
python phase2\intent_classifier\export_train.py --phase1 ..\uatie-edm-phase1         # scaled\to_label_batch{1..4}.csv, scaled\train_key.csv
python phase2\intent_classifier\llm_judge\combine.py                   # optional: judgements\intent_labelled_devin.csv (Devin's superseded set)
python phase2\intent_classifier\train.py --phase1 ..\uatie-edm-phase1                # results.json, REPORT.md, out\intent_v2.csv (CPU; time in results.json)
python phase2\intent_classifier\ontology_population.py --phase1 ..\uatie-edm-phase1  # results_ontology.json, REPORT.md section
```

- `train.py`: keyword-rule baseline; logistic regression on MiniLM embeddings and DistilBERT (3 epochs), each trained on
  silver (rules), scaled (2,000 Claude labels) and silver + scaled; threshold and model chosen on dev only. `--skip-distilbert` for a fast run.
- `ontology_population.py`: reruns Phase 1 `src/rag/ontology_population.py` unmodified with `intent_v2`.

After your review, overwrite the judged file with your reviewed copy (same columns; set the notes column to `human` on rows you checked) and rerun only the last step(s); `combine.py` only rebuilds Devin's superseded `*_devin.csv` file, so it never overwrites the file you review.
