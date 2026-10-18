# Task 1: Intent classifier

All numbers come from `results.json`, written by `train.py`.

## Label provenance (read first)

- Gold dev/test labels: {'LLM-judged (Claude), section 5 relabel': 400}. No label is human-reviewed, so every gold score below is **agreement with provisional LLM labels**, not accuracy against human ground truth.
- The gold labels were revised under `phase2/labels/LABELLING_PROMPT.md` section 5 before any scaled-label model was trained: 29 of 400 rows changed from `judgements/intent_labelled_claude_v1.csv` (dev 17, test 12); 91 rows are marked borderline (`judgements/intent_ambiguity_notes.csv`). Claude's own summary describes the changes relative to its first pass, so the relabel was not blind to the earlier labels.
- Scaled training labels: 2,000 Claude labels under the same section 5 rules, not human-reviewed. An earlier attempt drifted 13 rows out of alignment; the user rebuilt the labels keyed to `label_id`. `train.py` checks that IDs, source, summaries and row order equal the export. Agreement with the hidden Phase 1 rule label on rule-labelled rows: 0.443-0.540 per batch at the correct row, 0.046-0.119 when shifted by 1-3 rows; lowest 50-row block 0.191.
- The scaled set was sampled 1,000 rule-labelled records (90-91 per rule class) + 1,000 rule-Unclassified records, so its class mix is not the corpus distribution.
- Silver training labels are the Phase 1 keyword rules. No training set contains a gold record. Threshold and model are chosen on gold dev only; gold test is scored once.

## Scores

| Model | train set | train records | threshold (dev) | dev macro-F1 | test accuracy | test macro-F1 (12 labels) | test Unclassified % |
|---|---|---|---|---|---|---|---|
| keyword_rules_phase1 | rules | - | - | 0.4151 | 0.3930 | 0.4622 | 49.75 |
| logreg_minilm | silver | 68928 | 0.6 | 0.4117 | 0.4030 | 0.4449 | 48.76 |
| logreg_minilm_scaled | scaled | 2000 | 0.15 | 0.5609 | 0.5871 | 0.5867 | 4.48 |
| logreg_minilm_silver_scaled | silver_scaled | 70928 | 0.4 | 0.4308 | 0.4826 | 0.4958 | 28.36 |
| distilbert | silver_capped | 3300 | 0.9 | 0.4431 | 0.3930 | 0.4496 | 42.29 |
| distilbert_scaled | scaled | 2000 | 0.4 | 0.5938 | 0.6816 | 0.6005 | 19.9 |
| distilbert_silver_scaled | silver_capped+scaled | 5300 | 0.45 | 0.5639 | 0.6368 | 0.6070 | 13.43 |

Model chosen on dev macro-F1: `distilbert_scaled`. Target test accuracy >= 0.85 met: **False** (0.6816).

## intent_v2 over all records

- 113,468 records; Unclassified 20.54% (Phase 1 rules: 39.08%). Target < 15% met: **False**.
- Agreement with the rule label where the rules classified: 51.05%.
- Rule-Unclassified records now classified: 33,530.
- Gold rows whose full-corpus label differs from the scored prediction (batching): 0.
- Output: `out/intent_v2.csv` (`intent_v2`, `intent_conf`, `intent_source`).

## Per-class test scores (`distilbert_scaled`)

| Class | P | R | F1 | support |
|---|---|---|---|---|
| ATC_Operational_Error | 0.7143 | 0.3846 | 0.5000 | 13 |
| Emergency_Event | 1.0000 | 0.7353 | 0.8475 | 34 |
| Runway_Incursion | 0.5000 | 0.2857 | 0.3636 | 7 |
| Ground_Incident | 0.5909 | 0.8667 | 0.7027 | 15 |
| Communication_Failure | 0.6250 | 0.6250 | 0.6250 | 8 |
| Airspace_Violation | 0.7857 | 0.7333 | 0.7586 | 15 |
| IMC_Spatial_Disorientation | 0.0000 | 0.0000 | 0.0000 | 2 |
| Weather_Deviation | 0.6923 | 0.6923 | 0.6923 | 13 |
| Crew_Resource_Management | 1.0000 | 0.7222 | 0.8387 | 18 |
| Maintenance_Error | 0.6944 | 0.8621 | 0.7692 | 29 |
| Procedural_Non_Compliance | 0.7895 | 0.5172 | 0.6250 | 29 |
| Unclassified | 0.3500 | 0.7778 | 0.4828 | 18 |

On test records the rules left Unclassified (n = 100): accuracy 0.7400, macro-F1 0.5225.

Most frequent test errors (`distilbert_scaled`, gold -> predicted): Procedural_Non_Compliance -> Unclassified 10; Emergency_Event -> Maintenance_Error 6; Weather_Deviation -> Unclassified 4; ATC_Operational_Error -> Procedural_Non_Compliance 4; Runway_Incursion -> Ground_Incident 3; Unclassified -> Maintenance_Error 2.

Full confusion matrices (rows gold, columns predicted, label order as in `LABELS`) are in `results.json`.

## Known limitations

- Silver labels come from the keyword rules, which agree poorly with the gold labels (rules test accuracy 0.3930).
- The confidence threshold chosen on dev maps low-confidence predictions to Unclassified (`distilbert_scaled` threshold 0.4; 19.9% of test predictions are Unclassified).
- Gold and scaled labels come from the same LLM under the same rules, so a high score partly measures consistency with Claude, not correctness.
- Gold classes are imbalanced; classes with support < 10 on test give unstable per-class scores.
- DistilBERT runs on CPU with 3 epochs and no hyper-parameter search (see `results.json` for sizes and time).

## Ontology population (Phase 1 `src/rag/ontology_population.py`, run unmodified)

| Run | intent | phraseology_ttl | incident_individuals | unclassified_records | intent_triples | strong_phraseme_triples | weak_phraseme_triples | incident_graph_triples | merged_triples |
|---|---|---|---|---|---|---|---|---|---|
| phase1_repro | phase1 rules | phase1 | 113,468 | 44,340 | 113,468 | 655 | 9,640 | 983,098 | 989,421 |
| intent_v2_phase1_ttl | intent_v2 | phase1 | 113,468 | 23,301 | 113,468 | 387 | 6,715 | 979,905 | 986,237 |
| phase2 | intent_v2 | phase2 | 113,468 | 23,301 | 113,468 | 387 | 6,715 | 979,905 | 986,933 |

- `intent_v2_phase1_ttl` changes only the intent column, `phase2` also swaps in the Phase 2 phraseology TTL, so the difference between rows isolates each change.
- Intent change alone (phase1_repro -> intent_v2_phase1_ttl): strong phraseme links 655 -> 387, weak 9,640 -> 6,715. The Phase 1 linker attaches phrasemes by intent, so fewer records carrying the linked intents gives fewer links.
- Phraseology TTL change alone (intent_v2_phase1_ttl -> phase2): merged triples 986,237 -> 986,933 (+696); incident-graph triples unchanged.
- Source: `results_ontology.json`; large TTL outputs are regenerated, not committed.
