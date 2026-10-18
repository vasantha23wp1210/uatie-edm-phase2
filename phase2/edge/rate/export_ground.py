#!/usr/bin/env python3
"""Exports a blind relevance sheet for the current ground profile (llama3.1:8b, free prompt, test answers),
replacing the stale 2.70 rating of an earlier system. Seed 42, 30 answers with status ok / ok_after_filtering.
Fill your_relevance with 1-5 (5 = answers the query with points the cited reports support).
Writes ground_to_rate.csv. Run:  python phase2/edge/rate/export_ground.py
"""
import json
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parents[1]
SEED, N = 42, 30
SRC = P2 / "generation" / "out_llama3.1_8b_test" / "answers.jsonl"


def main():
    A = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines()]
    A = [a for a in A if a["status"] in ("ok", "ok_after_filtering")]
    A = random.Random(SEED).sample(A, min(N, len(A)))
    pd.DataFrame([{"query_id": a["query_id"], "query_text": a["query_text"], "answer": a["answer"]["summary"],
                   "key_hazards": " | ".join(a["answer"].get("key_hazards", [])),
                   "phraseology": " | ".join(a["answer"].get("recommended_phraseology", [])),
                   "citations": " ".join(a["answer"]["citations"]), "your_relevance": "", "notes": ""}
                  for a in A]).to_csv(HERE / "ground_to_rate.csv", index=False, encoding="utf-8")
    print(len(A), "ground answers from", SRC.relative_to(P2.parent))


if __name__ == "__main__":
    main()
