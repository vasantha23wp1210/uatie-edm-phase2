# Intent: breakdown by borderline status, with 95% intervals

From `out/gold_predictions.csv` (already scored; nothing retrained, no threshold changed). Borderline rows are those Claude marked borderline in the section 5 relabel. Intervals: Wilson 95%.

| Model | split | all | non-borderline | borderline |
|---|---|---|---|---|
| `logreg_minilm` | dev | 0.3518 [0.2888, 0.4203] (n=199) | 0.3846 [0.3119, 0.4629] (n=156) | 0.2326 [0.1315, 0.3774] (n=43) |
| `logreg_minilm` | test | 0.403 [0.3376, 0.472] (n=201) | 0.4118 [0.3369, 0.491] (n=153) | 0.375 [0.2522, 0.5164] (n=48) |
| `logreg_minilm_scaled` | dev | 0.603 [0.5337, 0.6684] (n=199) | 0.6795 [0.6027, 0.7476] (n=156) | 0.3256 [0.2049, 0.4748] (n=43) |
| `logreg_minilm_scaled` | test | 0.5871 [0.518, 0.6529] (n=201) | 0.6667 [0.5887, 0.7365] (n=153) | 0.3333 [0.2168, 0.4746] (n=48) |
| `logreg_minilm_silver_scaled` | dev | 0.402 [0.3364, 0.4714] (n=199) | 0.4359 [0.3605, 0.5143] (n=156) | 0.2791 [0.1675, 0.4269] (n=43) |
| `logreg_minilm_silver_scaled` | test | 0.4826 [0.4145, 0.5513] (n=201) | 0.5163 [0.4377, 0.5942] (n=153) | 0.375 [0.2522, 0.5164] (n=48) |
| `distilbert` | dev | 0.3869 [0.322, 0.4561] (n=199) | 0.3846 [0.3119, 0.4629] (n=156) | 0.3953 [0.2637, 0.5442] (n=43) |
| `distilbert` | test | 0.393 [0.3281, 0.462] (n=201) | 0.4183 [0.3431, 0.4975] (n=153) | 0.3125 [0.1995, 0.4533] (n=48) |
| `distilbert_scaled` (chosen) | dev | 0.6583 [0.59, 0.7206] (n=199) | 0.6859 [0.6093, 0.7535] (n=156) | 0.5581 [0.4111, 0.6957] (n=43) |
| `distilbert_scaled` (chosen) | test | 0.6816 [0.6143, 0.7421] (n=201) | 0.719 [0.643, 0.7841] (n=153) | 0.5625 [0.4227, 0.693] (n=48) |
| `distilbert_silver_scaled` | dev | 0.598 [0.5286, 0.6636] (n=199) | 0.6346 [0.5567, 0.7061] (n=156) | 0.4651 [0.3251, 0.6108] (n=43) |
| `distilbert_silver_scaled` | test | 0.6368 [0.5683, 0.7002] (n=201) | 0.6732 [0.5954, 0.7425] (n=153) | 0.5208 [0.3833, 0.6553] (n=48) |

Most frequent test confusions of `distilbert_scaled` (gold -> predicted):

- Procedural_Non_Compliance -> Unclassified: 10
- Emergency_Event -> Maintenance_Error: 6
- ATC_Operational_Error -> Procedural_Non_Compliance: 4
- Weather_Deviation -> Unclassified: 4
- Runway_Incursion -> Ground_Incident: 3
- Emergency_Event -> Unclassified: 2
- Airspace_Violation -> Communication_Failure: 2
- Unclassified -> Maintenance_Error: 2
- Unclassified -> Ground_Incident: 2
- Procedural_Non_Compliance -> Maintenance_Error: 2

26 of 64 test errors are predictions of Unclassified (below the dev threshold or the class itself).

## Limits

- Borderline flags come from the same LLM that wrote the gold labels; there is no human agreement figure, so no measured ceiling for accuracy.
- Splitting by borderline status was not used to choose anything.
