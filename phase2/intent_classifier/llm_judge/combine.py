"""Rebuild judgements/intent_labelled.csv from judgements/intent_to_label.csv + llm_judge/labels/i*.csv
(label_id, your_intent). Every row is marked as a provisional LLM label.
Run: python phase2/intent_classifier/llm_judge/combine.py [--out PATH]"""
import sys
from pathlib import Path

import pandas as pd

T = Path(__file__).resolve().parents[1]
NOTE = "LLM-judged (Devin), pending user review"
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else T / "judgements" / "intent_labelled.csv"
df = pd.read_csv(T / "judgements" / "intent_to_label.csv", dtype=str, keep_default_na=False)
lab = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((T / "llm_judge" / "labels").glob("i*.csv"))])
allowed = set((T / "judgements" / "allowed_labels.txt").read_text(encoding="utf-8").split())
assert lab.label_id.is_unique and set(lab.label_id) == set(df.label_id), "labels must cover every row exactly once"
assert lab.your_intent.isin(allowed).all(), "label outside allowed_labels.txt"
df["your_intent"] = df.label_id.map(dict(zip(lab.label_id, lab.your_intent)))
df["notes"] = NOTE
df.to_csv(out, index=False)
print(f"{out}: {len(df)} rows")
