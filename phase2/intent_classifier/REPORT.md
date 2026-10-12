# Task 1: Intent classifier (STOPPED, waiting for labels)

Numbers come from `export_results.json`, written by `export.py`. No classifier has been trained yet.

## Phase 1 starting point
- Keyword rules (`asrs_nlp_pipeline.classify_intent`, unchanged) leave 39.08% of 113,468 records Unclassified (target: < 15%).

## Gold sample
- 200 rule-labelled records stratified over the 11 classes and 200 rule-Unclassified records, seed 42, from 113,235 records with a summary of at least 40 characters.
- Gold dev/test split within each stratum: {'test': 201, 'dev': 199}. The confidence threshold is picked on gold dev only; accuracy is reported on gold test, and on all 400 where the plan asks for it.
- These 400 records are excluded from training.

| Stratum (rule label, hidden from you) | Records |
|---|---|
| ATC_Operational_Error | 19 |
| Emergency_Event | 19 |
| Runway_Incursion | 18 |
| Ground_Incident | 18 |
| Communication_Failure | 18 |
| Airspace_Violation | 18 |
| IMC_Spatial_Disorientation | 18 |
| Weather_Deviation | 18 |
| Crew_Resource_Management | 18 |
| Maintenance_Error | 18 |
| Procedural_Non_Compliance | 18 |
| Unclassified | 200 |

## What you need to do
Fill `your_intent` in `judgements/intent_to_label.csv` with one value from `judgements/allowed_labels.txt` (use `Unclassified` when none of the 11 fits). Return it as `judgements/intent_labelled.csv`. Do not change other columns. The file does not show the rule label.
