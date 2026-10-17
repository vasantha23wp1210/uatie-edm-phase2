#!/usr/bin/env python3
"""Exports two blind sheets for human review (seed 42), so the LLM judge can be checked against a person:
- claims_to_check.csv: 50 claims from the ground baseline test answers (25 the 3B judge called supported,
  25 it called unsupported), with the cited report text. Fill your_supported with yes / no.
  The judge verdicts are in claims_key.csv; do not open it while rating.
- cockpit_to_rate.csv: 20 cockpit answers. Fill your_relevance with 1-5.
Run:  python phase2/edge/rate/export.py --phase1 ../uatie-edm-phase1
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parents[1]
sys.path.insert(0, str(P2.parent))
from phase1 import phase1_root  # noqa: E402

SEED, N_CLAIMS, N_ANSWERS = 42, 25, 20


def jl(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def main():
    text_of = pd.read_csv(phase1_root() / "data" / "processed" / "asrs_faiss_meta.csv",
                          usecols=["incident_id", "summary"]).astype(str).set_index("incident_id").summary
    G = P2 / "generation" / "out"
    A = {a["query_id"]: a for a in jl(G / "answers.jsonl")}
    rows = []
    for j in jl(G / "judgements.jsonl"):
        a = A[j["query_id"]]
        if a["split"] != "test":
            continue
        for k, c in enumerate(j["claims"]):
            rows.append({"query_id": j["query_id"], "query_text": a["query_text"], "claim": c["claim"],
                         "cited_reports": " || ".join(f"{i}: {text_of[i]}" for i in a["answer"]["citations"]),
                         "judge_supported": str(c["supported"]).lower() == "true", "k": k})
    rng = random.Random(SEED)
    pick = []
    for v in (True, False):
        pool = [r for r in rows if r["judge_supported"] is v]
        pick += rng.sample(pool, min(N_CLAIMS, len(pool)))
    rng.shuffle(pick)
    sheet = pd.DataFrame(pick)
    sheet.insert(0, "claim_id", [f"C{i + 1:03d}" for i in range(len(sheet))])
    sheet[["claim_id", "judge_supported"]].to_csv(HERE / "claims_key.csv", index=False)
    sheet.assign(your_supported="", notes="")[["claim_id", "query_id", "query_text", "claim", "cited_reports",
                                               "your_supported", "notes"]].to_csv(HERE / "claims_to_check.csv",
                                                                                  index=False, encoding="utf-8")
    C = [a for a in jl(P2 / "edge" / "out" / "answers.jsonl") if a["status"] == "ok"]
    C = random.Random(SEED).sample(C, min(N_ANSWERS, len(C)))
    pd.DataFrame([{"query_id": a["query_id"], "query_text": a["query_text"], "answer": a["answer"]["summary"],
                   "phraseology": " | ".join(a["answer"].get("recommended_phraseology", [])), "citations": " ".join(a["answer"]["citations"]),
                   "your_relevance": "", "notes": ""} for a in C]).to_csv(HERE / "cockpit_to_rate.csv", index=False,
                                                                          encoding="utf-8")
    print(len(sheet), "claims;", len(C), "cockpit answers")


if __name__ == "__main__":
    main()
