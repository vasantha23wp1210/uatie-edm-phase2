"""Agreement between Devin's blind labels and the Phase 1 manual dev labels (calibration)."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
TASK = HERE.parent

blind = pd.read_csv(HERE / "calibration_to_judge.csv", usecols=["pair_id", "query_id", "incident_id"])
llm = pd.concat([pd.read_csv(f) for f in sorted((HERE / "labels").glob("calibration_b*.csv"))])
assert llm.pair_id.is_unique and set(llm.pair_id) == set(blind.pair_id), "calibration labels incomplete"
llm = blind.merge(llm, on="pair_id")
llm.sort_values("pair_id").to_csv(HERE / "calibration_llm_labels.csv", index=False)
human = pd.read_csv(TASK / "splits" / "qrels_dev_phase1_manual.csv")
m = llm.merge(human, on=["query_id", "incident_id"], suffixes=("_llm", "_human"))
assert len(m) == len(blind)
a, h = m.relevance_llm.astype(int), m.relevance_human.astype(int)
per_q = {q: round(float((g.relevance_llm == g.relevance_human).mean()), 4) for q, g in m.groupby("query_id")}
res = {
    "n_pairs": len(m),
    "exact_agreement": round(float((a == h).mean()), 4),
    "cohen_kappa": round(float(cohen_kappa_score(h, a)), 4),
    "quadratic_weighted_kappa": round(float(cohen_kappa_score(h, a, weights="quadratic")), 4),
    "binary_relevant_ge1_kappa": round(float(cohen_kappa_score(h >= 1, a >= 1)), 4),
    "binary_relevant_ge1_agreement": round(float(((h >= 1) == (a >= 1)).mean()), 4),
    "label_counts_human": {int(k): int(v) for k, v in h.value_counts().sort_index().items()},
    "label_counts_llm": {int(k): int(v) for k, v in a.value_counts().sort_index().items()},
    "confusion_rows_human_cols_llm_0_1_2": confusion_matrix(h, a, labels=[0, 1, 2]).tolist(),
    "per_query_exact_agreement": per_q,
}
(HERE / "calibration_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
