# Provisional label sets: Claude vs Devin

Every Phase 2 task now uses the Claude label sets (`judgements/*.csv`, method in `LABELLING_PROMPT.md`).
Devin's earlier provisional sets are kept as `judgements/*_devin.csv`. Both are LLM-generated and **pending user
review**, so the agreement below says how consistent two LLM judges are, not how accurate either is.

| File | Rows | Agreement | Cohen's κ |
|---|---:|---:|---:|
| Retrieval relevance (0/1/2) | 975 | 89.1% | 0.8359 |
| Intent (all) | 400 | 75.5% | 0.7222 |
| Intent (dev) | 199 | 74.4% | 0.7081 |
| Intent (test) | 201 | 76.6% | 0.7358 |

- Retrieval: linear-weighted κ 0.8818, binary rel ≥ 1 κ 0.8846, binary rel 2 κ 0.879; 0 pairs differ by 2 points. Confusion (rows Devin, columns Claude, 0/1/2): [[313, 30, 0], [21, 223, 26], [0, 29, 333]].
- Intent, largest disagreements (Devin → Claude): Maintenance_Error → Emergency_Event (27); Unclassified → Airspace_Violation (20); Procedural_Non_Compliance → Crew_Resource_Management (10); Unclassified → Weather_Deviation (6); ATC_Operational_Error → Airspace_Violation (5).
- Intent: Claude left notes on 33 ambiguous rows (`phase2/intent_classifier/judgements/intent_ambiguity_notes.csv`).
- NER, entity-level agreement F1 (exact span and label): all 260 sentences 0.9821 (419 Claude vs 418 Devin entities, 11 sentences differ); test 60 sentences 0.9834.
- Earlier calibration against the user's own labels (see `phase2/gold_eval_rebuild/`): Devin κ 0.0619 on 362 dev pairs; Claude κ 0.34 on the user's 100-pair test spot-check. This is why the Claude sets were adopted.
