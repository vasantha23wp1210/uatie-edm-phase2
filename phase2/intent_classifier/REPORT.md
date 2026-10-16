# Task 1: Intent classifier

All numbers come from `results.json`, written by `train.py`.

## Label provenance (read first)

- Gold labels: {'LLM-judged (Claude), pending user review': 400}. While they are `pending user review`, every gold score below is **agreement with provisional LLM labels**, not accuracy against human ground truth.
- Training data is the Phase 1 keyword-rule labels (silver), excluding the 400 gold records. Threshold and model are chosen on gold dev only; gold test is scored once.

## Test scores (gold test)

| Model | threshold | accuracy | macro-F1 (12 labels) | Unclassified predicted % |
|---|---|---|---|---|
| keyword_rules_phase1 | None | 0.3831 | 0.4516 | 49.75 |
| logreg_minilm | 0.3 | 0.4776 | 0.4618 | 7.96 |
| distilbert | 0.9 | 0.3930 | 0.4446 | 42.29 |

Model chosen on dev macro-F1: `logreg_minilm`. Target test accuracy >= 0.85 met: **False**.

## intent_v2 over all records

- 113,468 records; Unclassified 9.41% (Phase 1 rules: 39.08%). Target < 15% met: **True**.
- Agreement with the rule label where the rules classified: 76.95%.
- Rule-Unclassified records now classified: 36,066.
- Output: `out/intent_v2.csv` (`intent_v2`, `intent_conf`, `intent_source`).

## Per-class test scores (`logreg_minilm`)

| Class | P | R | F1 | support |
|---|---|---|---|---|
| ATC_Operational_Error | 0.5833 | 0.5000 | 0.5385 | 14 |
| Emergency_Event | 0.7727 | 0.5312 | 0.6296 | 32 |
| Runway_Incursion | 0.4286 | 0.8571 | 0.5714 | 7 |
| Ground_Incident | 0.6667 | 0.7500 | 0.7059 | 16 |
| Communication_Failure | 0.3750 | 0.6667 | 0.4800 | 9 |
| Airspace_Violation | 0.6667 | 0.4000 | 0.5000 | 15 |
| IMC_Spatial_Disorientation | 0.0714 | 0.5000 | 0.1250 | 2 |
| Weather_Deviation | 0.4444 | 0.5714 | 0.5000 | 14 |
| Crew_Resource_Management | 0.4211 | 0.6667 | 0.5161 | 12 |
| Maintenance_Error | 0.5000 | 0.4667 | 0.4828 | 30 |
| Procedural_Non_Compliance | 0.6000 | 0.2647 | 0.3673 | 34 |
| Unclassified | 0.1250 | 0.1250 | 0.1250 | 16 |

On test records the rules left Unclassified (n = 100): accuracy 0.3600, macro-F1 0.3165.

Confusion matrices (rows gold, columns predicted, label order as in `LABELS`) are in `results.json`.

## Known limitations

- Silver training labels come from the keyword rules, so the models partly learn the rules' errors. The provisional gold labels differ from the rules a lot (rules test accuracy 0.3831), and gold-frequent classes are rare in silver (Procedural_Non_Compliance: 513 of 68,928 silver records vs 37 of 199 dev labels).
- The confidence threshold chosen on dev maps low-confidence predictions to Unclassified (`logreg_minilm` threshold 0.3; 7.96% of test predictions are Unclassified).
- Gold classes are imbalanced; classes with support < 10 on test give unstable per-class scores.
- DistilBERT is trained on a class-capped subsample on CPU (see `results.json` for size and time).

## Ontology population (Phase 1 `src/rag/ontology_population.py`, run unmodified)

| Run | intent | phraseology_ttl | incident_individuals | unclassified_records | intent_triples | strong_phraseme_triples | weak_phraseme_triples | incident_graph_triples | merged_triples |
|---|---|---|---|---|---|---|---|---|---|
| phase1_repro | phase1 rules | phase1 | 113,468 | 44,340 | 113,468 | 655 | 9,640 | 983,098 | 989,421 |
| intent_v2_phase1_ttl | intent_v2 | phase1 | 113,468 | 10,676 | 113,468 | 600 | 10,646 | 984,049 | 990,370 |
| phase2 | intent_v2 | phase2 | 113,468 | 10,676 | 113,468 | 600 | 10,646 | 984,049 | 991,066 |

- `intent_v2_phase1_ttl` changes only the intent column, `phase2` also swaps in the Phase 2 phraseology TTL, so the difference between rows isolates each change.
- Intent change alone (phase1_repro -> intent_v2_phase1_ttl): strong phraseme links 655 -> 600, weak 9,640 -> 10,646. The Phase 1 linker attaches phrasemes by intent, so fewer records carrying the linked intents gives fewer links.
- Phraseology TTL change alone (intent_v2_phase1_ttl -> phase2): merged triples 990,370 -> 991,066 (+696); incident-graph triples unchanged.
- Source: `results_ontology.json`; large TTL outputs are regenerated, not committed.
