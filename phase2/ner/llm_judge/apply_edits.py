"""Build judgements/ner_corrected.csv from llm_judge/edits.csv (sent_id, old, new substring replacements on
pre_annotated). Rows without edits keep the pre-annotation. Fails if any edit does not match exactly once or the
markup-stripped text differs from the original sentence."""
import re
from pathlib import Path

import pandas as pd

T = Path(__file__).resolve().parents[1]
d = pd.read_csv(T / "judgements" / "ner_to_correct.csv", dtype=str, keep_default_na=False)
e = pd.read_csv(T / "llm_judge" / "edits.csv", dtype=str, keep_default_na=False)
assert set(e.sent_id) <= set(d.sent_id)
out = []
for r in d.itertuples():
    s = r.pre_annotated
    for x in e[e.sent_id == r.sent_id].itertuples():
        assert s.count(x.old) == 1, (r.sent_id, x.old)
        s = s.replace(x.old, x.new)
    assert re.sub(r"\[([^\]]+)\]\([A-Z_]+\)", r"\1", s) == r.text, r.sent_id
    out.append(s)
d["corrected"] = out
d["notes"] = "LLM-judged (Devin), pending user review"
d.to_csv(T / "judgements" / "ner_corrected.csv", index=False)
print(len(d), "rows;", e.sent_id.nunique(), "sentences edited;", len(e), "edits")
