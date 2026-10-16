"""Rebuild judgements/judged.csv from judgements/to_judge.csv + llm_judge/labels/b*.csv (pair_id, relevance).
Every row is marked as a provisional LLM judgement. Run: python phase2/retrieval/llm_judge/combine.py [--out PATH]"""
import sys
from pathlib import Path

import pandas as pd

T = Path(__file__).resolve().parents[1]
NOTE = "LLM-judged (Devin), pending user review"
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else T / "judgements" / "judged.csv"
df = pd.read_csv(T / "judgements" / "to_judge.csv", dtype=str, keep_default_na=False)
lab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((T / "llm_judge" / "labels").glob("b*.csv"))])
assert lab.pair_id.is_unique and set(lab.pair_id) == set(df.pair_id), "labels must cover every pair exactly once"
assert lab.relevance.isin(["0", "1", "2"]).all()
df["relevance"] = df.pair_id.map(dict(zip(lab.pair_id, lab.relevance)))
df["judge_notes"] = NOTE
df.to_csv(out, index=False)
print(f"{out}: {len(df)} rows", df.relevance.value_counts().sort_index().to_dict())
