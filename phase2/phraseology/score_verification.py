#!/usr/bin/env python3
"""Tasks 4-5: score the user's verdicts on the 20 sampled crossmap matches.

Reads verify/matches_verified.csv. A verdict says whether the TF-IDF tier assigned to the pair is right
(so `correct` on a No Match row means the phrases are not equivalent). Checks the rows against
verify/matches_to_verify.csv and writes results_verification.json; rerun report.py afterwards.
Run:  python phase2/phraseology/score_verification.py [--verified PATH]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parent
TIERS = ["No Match", "Partial Match", "Close Match", "Exact/Near-Exact"]
KEY = ["sample_id", "Section_Number", "CAA_Phraseology", "Best_ICAO_Match", "Best_ICAO_Source", "Match_Type"]
PROVENANCE = "human (user); notes refined with Claude"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=str(TASK / "verify" / "matches_verified.csv"))
    a = ap.parse_args()
    v = pd.read_csv(a.verified, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    e = pd.read_csv(TASK / "verify" / "matches_to_verify.csv", dtype=str, keep_default_na=False)
    assert len(v) == len(e) and (v[KEY].values == e[KEY].values).all(), "rows differ from matches_to_verify.csv"
    assert (v.Match_Score.astype(float) == e.Match_Score.astype(float)).all(), "Match_Score differs"
    v["verdict"] = v.your_verdict.str.strip().str.lower()
    bad = list(v.sample_id[~v.verdict.isin(["correct", "incorrect"])])
    assert not bad, f"verdict not correct/incorrect: {bad}"
    changed = list(v.sample_id[v.notes.str.contains("changed from", case=False)])
    per = {t: {"rows": int((v.Match_Type == t).sum()),
               "tier_correct": int(((v.Match_Type == t) & (v.verdict == "correct")).sum())} for t in TIERS}
    keep = v[~v.sample_id.isin(changed)]
    R = {"provenance": PROVENANCE, "verdict_meaning": "assigned TF-IDF tier is right for the pair",
         "rows": len(v), "tier_correct": int((v.verdict == "correct").sum()), "per_tier": per,
         "incorrect": list(v.sample_id[v.verdict == "incorrect"]),
         "verdict_changed_during_claude_review": changed,
         "tier_correct_excluding_changed": {"rows": len(keep), "tier_correct": int((keep.verdict == "correct").sum())}}
    (TASK / "results_verification.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(R))


if __name__ == "__main__":
    main()
