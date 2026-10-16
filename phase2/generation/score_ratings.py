#!/usr/bin/env python3
"""Task 7, step 3: score the user's 1-5 answer ratings.

Reads rate/answers_rated_run_prev.csv (the user's ratings of the 30 answers exported from the earlier run in
out/run_prev/), checks every row against out/run_prev/answers.jsonl, and writes results_ratings.json. Rerun
judge.py afterwards to add the numbers to REPORT.md and results.json.
Run:  python phase2/generation/score_ratings.py [--rated PATH] [--run out/run_prev]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TASK = Path(__file__).resolve().parent
PROVENANCE = "human (user); notes refined with Claude"


def jl(p):
    return {r["query_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def unsupported(j):
    c = [x["supported"] is True or str(x["supported"]).lower() == "true" for x in j.get("claims", [])]
    return round(1 - float(np.mean(c)), 4) if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rated", default=str(TASK / "rate" / "answers_rated_run_prev.csv"))
    ap.add_argument("--run", default=str(TASK / "out" / "run_prev"))
    a = ap.parse_args()
    run = Path(a.run)
    d = pd.read_csv(a.rated, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    A, Jd, cur = jl(run / "answers.jsonl"), jl(run / "judgements.jsonl"), jl(TASK / "out" / "answers.jsonl")
    assert len(d) == len(set(d.rate_id)), "duplicate rate_id"
    bad = [r.rate_id for r in d.itertuples() if r.your_rating_1_to_5.strip() not in {"1", "2", "3", "4", "5"}]
    assert not bad, f"ratings not an integer 1-5: {bad}"
    for r in d.itertuples():
        ans = A[r.query_id]["answer"]
        assert ans["summary"] == r.summary and "; ".join(ans["key_hazards"]) == r.key_hazards, \
            f"{r.rate_id}: answer text differs from {run}"
    d["rating"] = d.your_rating_1_to_5.astype(int)
    d["split"] = [A[q]["split"] for q in d.query_id]
    d["status"] = [A[q]["status"] for q in d.query_id]
    d["unsupported_frac"] = [unsupported(Jd.get(q, {})) for q in d.query_id]
    d["same_in_current_run"] = [bool(cur.get(q, {}).get("answer")) and cur[q]["answer"] == A[q]["answer"]
                                for q in d.query_id]
    r = d.rating
    s = d.dropna(subset=["unsupported_frac"])
    rho, p = spearmanr(s.rating, s.unsupported_frac)
    R = {"provenance": PROVENANCE, "rated_file": Path(a.rated).name, "rated_run": run.name,
         "rows": len(d), "distribution": {str(k): int((r == k).sum()) for k in range(1, 6)},
         "mean": round(float(r.mean()), 2), "median": float(r.median()), "sd": round(float(r.std(ddof=1)), 2),
         "rated_ge4": int((r >= 4).sum()), "rated_le2": int((r <= 2).sum()),
         "by_split": {k: {"n": int(len(g)), "mean": round(float(g.rating.mean()), 2)} for k, g in d.groupby("split")},
         "by_status": {k: {"n": int(len(g)), "mean": round(float(g.rating.mean()), 2)} for k, g in d.groupby("status")},
         "by_risk": {k: {"n": int(len(g)), "mean": round(float(g.rating.mean()), 2)} for k, g in d.groupby("risk")},
         "spearman_rating_vs_llm_judge_unsupported": {"rho": round(float(rho), 3), "p": round(float(p), 3),
                                                      "n": int(len(s))},
         "same_answer_in_current_run": sorted(d.rate_id[d.same_in_current_run]),
         "ratings": {x.rate_id: int(x.rating) for x in d.itertuples()}}
    (TASK / "results_ratings.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["rows", "distribution", "mean", "median", "rated_ge4",
                                        "spearman_rating_vs_llm_judge_unsupported", "same_answer_in_current_run"]}))


if __name__ == "__main__":
    main()
