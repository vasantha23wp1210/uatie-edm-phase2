#!/usr/bin/env python3
"""Run the NLI judge (edge/nli, same model, revision and dev-control threshold) on the 50 human-checked claims.

Premises: each cited report and each of its sentences; a claim's score is the max P(entailment) over them.
Writes rate/claims_nli.csv (claim_id, p_entail, nli_supported). Run:  python phase2/edge/rate/nli_claims.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nli"))
import nli_judge as N  # noqa: E402


def main():
    thr = json.loads((HERE.parent / "nli" / "results.json").read_text(encoding="utf-8"))["nli"]["threshold_chosen_on_dev_controls"]
    C = pd.read_csv(HERE / "claims_to_check.csv", dtype=str).fillna("")
    items, owner = [], []
    for r in C.itertuples():
        for part in r.cited_reports.split(" || "):
            items.append((part.split(": ", 1)[1] if ": " in part else part, r.claim))
            owner.append(r.claim_id)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(N.NLI, revision=N.NLI_REVISION, device="cpu")
    p = pd.Series(N.entail(model, items)).groupby(pd.Series(owner)).max()
    out = pd.DataFrame({"claim_id": C.claim_id, "p_entail": [round(float(p[i]), 4) for i in C.claim_id]})
    out["nli_supported"] = out.p_entail >= thr
    out.to_csv(HERE / "claims_nli.csv", index=False)
    print(f"threshold {thr}; NLI supported {int(out.nli_supported.sum())} of {len(out)}")


if __name__ == "__main__":
    main()
