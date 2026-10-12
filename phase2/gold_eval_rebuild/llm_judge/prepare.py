"""Blind CSVs for LLM-judge calibration (dev) and the unused human spot-check (test). Prints counts only."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SPOT_N = 150
HERE = Path(__file__).resolve().parent
TASK = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(TASK.parents[1]))
from phase1 import phase1_root  # noqa: E402
REPO = phase1_root()
COLS = ["pair_id", "query_id", "query_text", "information_need", "incident_id", "narrative", "relevance", "judge_notes"]


def spot_check(rng):
    test = pd.read_csv(TASK / "judgements" / "test_to_judge.csv")
    bucket = pd.read_csv(TASK / "splits" / "test_queries.csv").set_index("query_id")["scenario_bucket"]
    test["bucket"] = test.query_id.map(bucket)
    sizes = test.bucket.value_counts().sort_index()
    quota = sizes / sizes.sum() * SPOT_N
    alloc = np.floor(quota).astype(int)
    rem = (quota - alloc).sort_values(ascending=False)
    alloc[rem.index[: SPOT_N - alloc.sum()]] += 1
    picks = [test[test.bucket == b].iloc[rng.choice(n, alloc[b], replace=False)] for b, n in sizes.items()]
    out = pd.concat(picks).iloc[rng.permutation(SPOT_N)]
    out[COLS].to_csv(HERE / "spot_check_to_judge.csv", index=False)
    return alloc.to_dict()


def calibration(rng):
    prov = pd.read_csv(TASK / "pool" / "pool_provenance.csv")
    qrels = pd.read_csv(TASK / "splits" / "qrels_dev_phase1_manual.csv", usecols=["query_id", "incident_id"])
    pairs = prov[["query_id", "incident_id"]].merge(qrels, on=["query_id", "incident_id"])
    pairs = pairs.sort_values(["query_id", "incident_id"]).iloc[rng.permutation(len(pairs))]
    q = pd.read_csv(TASK / "splits" / "dev_queries.csv").set_index("query_id")
    text = pd.read_csv(REPO / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["incident_id", "summary"],
                       low_memory=False).set_index("incident_id")["summary"]
    out = pd.DataFrame({
        "pair_id": [f"C{i:05d}" for i in range(1, len(pairs) + 1)],
        "query_id": pairs.query_id.values,
        "query_text": q.loc[pairs.query_id, "query_text"].values,
        "information_need": q.loc[pairs.query_id, "relevance_notes"].values,
        "incident_id": pairs.incident_id.values,
        "narrative": text.loc[pairs.incident_id].values,
        "relevance": "",
        "judge_notes": "",
    })
    out.to_csv(HERE / "calibration_to_judge.csv", index=False)
    return len(out)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print("spot_check allocation:", spot_check(rng))  # drawn first: fixes the calibration order
    print("calibration pairs:", calibration(rng))
