# Task 1: Intent classifier

All numbers come from `results.json`, written by `train.py`.

## Label provenance (read first)

- Gold labels: {'LLM-judged (Devin), pending user review': 400}. While they are `pending user review`, every gold score below is **agreement with provisional LLM labels**, not accuracy against human ground truth.
- Training data is the Phase 1 keyword-rule labels (silver), excluding the 400 gold records. Threshold and model are chosen on gold dev only; gold test is scored once.

## Test scores (gold test)

| Model | threshold | accuracy | macro-F1 (12 labels) | Unclassified predicted % |
|---|---|---|---|---|
| keyword_rules_phase1 | None | 0.3333 | 0.3765 | 49.75 |
| logreg_minilm | 0.3 | 0.4328 | 0.3983 | 7.96 |
| distilbert | 0.85 | 0.4030 | 0.4097 | 34.33 |

Model chosen on dev macro-F1: `distilbert`. Target test accuracy >= 0.85 met: **False**.

## intent_v2 over all records

- 113,468 records; Unclassified 38.14% (Phase 1 rules: 39.08%). Target < 15% met: **False**.
- Agreement with the rule label where the rules classified: 77.17%.
- Rule-Unclassified records now classified: 13,154.
- Output: `out/intent_v2.csv` (`intent_v2`, `intent_conf`, `intent_source`).

## Per-class test scores (`distilbert`)

| Class | P | R | F1 | support |
|---|---|---|---|---|
| ATC_Operational_Error | 0.7143 | 0.2941 | 0.4167 | 17 |
| Emergency_Event | 0.8000 | 0.4706 | 0.5926 | 17 |
| Runway_Incursion | 0.4167 | 0.8333 | 0.5556 | 6 |
| Ground_Incident | 0.5200 | 0.7222 | 0.6047 | 18 |
| Communication_Failure | 0.6667 | 0.5000 | 0.5714 | 12 |
| Airspace_Violation | 0.1000 | 0.2000 | 0.1333 | 5 |
| IMC_Spatial_Disorientation | 0.1000 | 1.0000 | 0.1818 | 1 |
| Weather_Deviation | 0.2500 | 0.3750 | 0.3000 | 8 |
| Crew_Resource_Management | 0.4000 | 0.6667 | 0.5000 | 6 |
| Maintenance_Error | 0.9375 | 0.3125 | 0.4688 | 48 |
| Procedural_Non_Compliance | 0.8182 | 0.2250 | 0.3529 | 40 |
| Unclassified | 0.1594 | 0.4783 | 0.2391 | 23 |

On test records the rules left Unclassified (n = 100): accuracy 0.2600, macro-F1 0.1460.

Confusion matrices (rows gold, columns predicted, label order as in `LABELS`) are in `results.json`.

## Known limitations

- Silver training labels come from the keyword rules, so the models partly learn the rules' errors. The provisional gold labels differ from the rules a lot (rules test accuracy 0.3333), and gold-frequent classes are rare in silver (Procedural_Non_Compliance: 513 of 68,928 silver records vs 43 of 199 dev labels).
- The confidence threshold chosen on dev maps low-confidence predictions to Unclassified, which keeps the Unclassified rate high (DistilBERT threshold 0.85).
- Gold classes are imbalanced; classes with support < 10 on test give unstable per-class scores.
- DistilBERT is trained on a class-capped subsample on CPU (see `results.json` for size and time).

## Ontology population (Phase 1 `src/rag/ontology_population.py`, run unmodified)

| Run | intent | phraseology_ttl | incident_individuals | unclassified_records | intent_triples | strong_phraseme_triples | weak_phraseme_triples | incident_graph_triples | merged_triples |
|---|---|---|---|---|---|---|---|---|---|
| phase1_repro | phase1 rules | phase1 | 113,468 | 44,340 | 113,468 | 655 | 9,640 | 983,098 | 989,421 |
| intent_v2_phase1_ttl | intent_v2 | phase1 | 113,468 | 43,275 | 113,468 | 359 | 6,560 | 979,722 | 986,046 |
| phase2 | intent_v2 | phase2 | 113,468 | 43,275 | 113,468 | 359 | 6,560 | 979,722 | 986,742 |

- `intent_v2_phase1_ttl` changes only the intent column, `phase2` also swaps in the Phase 2 phraseology TTL, so the difference between rows isolates each change.
- Intent change alone (phase1_repro -> intent_v2_phase1_ttl): strong phraseme links 655 -> 359, weak 9,640 -> 6,560. The Phase 1 linker attaches phrasemes by intent, so fewer records carrying the linked intents gives fewer links.
- Phraseology TTL change alone (intent_v2_phase1_ttl -> phase2): merged triples 986,046 -> 986,742 (+696); incident-graph triples unchanged.
- Source: `results_ontology.json`; large TTL outputs are regenerated, not committed.
