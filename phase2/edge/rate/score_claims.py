#!/usr/bin/env python3
"""Score the user's blind check of 50 ground-answer claims against the 3B LLM judge and the NLI judge.

Input: rate/claims_checked.csv (user labels, matched to claims_to_check.csv by claim_id), rate/claims_key.csv
(3B judge verdicts, hidden while rating), rate/claims_nli.csv (nli_claims.py).
Label scheme (user's definitions): yes = fully supported; partial = 50-75% correct; no / almost no = unsupported.
`dupe` marks a repeated query; rows marked only `dupe` / `dupe*N` carry no verdict and are excluded.
The 50 claims were sampled 25 judge-supported + 25 judge-unsupported from the 384 test claims of the 3B ground
run (generation/out), so population rates are re-weighted by the judge's split in that pool.
Writes rate/results_claims.json and rate/REPORT_claims.md. Run:  python phase2/edge/rate/score_claims.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parents[1]

SEED, BOOT = 42, 10000


def level(s):
    s = re.sub(r"\s*-?\s*dupe(\*\d+)?\s*$", "", s.strip().lower()).strip()
    return {"yes": "supported", "partial": "partial", "no": "unsupported", "almost no": "unsupported", "": None}[s]


def pool():
    sp = {a["query_id"]: a["split"] for a in map(json.loads, open(P2 / "generation" / "out" / "answers.jsonl", encoding="utf-8"))}
    cl = [c for j in map(json.loads, open(P2 / "generation" / "out" / "judgements.jsonl", encoding="utf-8"))
          if sp[j["query_id"]] == "test" for c in j["claims"]]
    return len(cl), sum(str(c["supported"]).lower() != "true" for c in cl)


def main():
    exp = pd.read_csv(HERE / "claims_to_check.csv", dtype=str).fillna("").set_index("claim_id")
    got = pd.read_csv(HERE / "claims_checked.csv", dtype=str).fillna("").set_index("claim_id").loc[exp.index]
    assert (got.claim.str.strip() == exp.claim.str.strip()).all(), "claim text differs from the export"
    key = pd.read_csv(HERE / "claims_key.csv", dtype=str).set_index("claim_id").loc[exp.index]
    nli = pd.read_csv(HERE / "claims_nli.csv").set_index("claim_id").loc[exp.index]
    D = pd.DataFrame({"human": got.your_supported.map(level), "raw": got.your_supported,
                      "judge_supported": key.judge_supported.str.lower() == "true",
                      "nli_supported": nli.nli_supported.astype(bool), "p_entail": nli.p_entail})
    rated = D[D.human.notna()]
    n_pool, n_pool_uns = pool()
    w_uns = n_pool_uns / n_pool
    R = {"provenance": "human (user), blind to the judge key", "claims_exported": len(D), "excluded_dupe_only": int(D.human.isna().sum()),
         "rated": len(rated), "human_levels": rated.human.value_counts().to_dict(),
         "pool": {"test_claims": n_pool, "judge_unsupported": n_pool_uns}, "rates": {}, "judges": {}}
    strata = {v: rated[rated.judge_supported == v] for v in (True, False)}
    rng = np.random.default_rng(SEED)
    for name, bad in (("unsupported", {"unsupported"}), ("not_fully_supported", {"unsupported", "partial"})):
        k = {v: int(s.human.isin(bad).sum()) for v, s in strata.items()}
        n = {v: len(s) for v, s in strata.items()}
        est = (k[True] / n[True]) * (1 - w_uns) + (k[False] / n[False]) * w_uns
        bt = [(rng.binomial(n[True], k[True] / n[True]) / n[True]) * (1 - w_uns) +
              (rng.binomial(n[False], k[False] / n[False]) / n[False]) * w_uns for _ in range(BOOT)]
        R["rates"][name] = {"in_rated_sample": [sum(k.values()), len(rated)],
                            "among_judge_supported": [k[True], n[True]], "among_judge_unsupported": [k[False], n[False]],
                            "pool_estimate": round(est, 4), "pool_estimate_ci95_bootstrap": [round(float(x), 4) for x in np.percentile(bt, [2.5, 97.5])]}
    for jn, col in (("llm_3b", "judge_supported"), ("nli", "nli_supported")):
        out = {}
        for name, bad in (("unsupported", {"unsupported"}), ("not_fully_supported", {"unsupported", "partial"})):
            h = rated.human.isin(bad); j = ~rated[col]
            tp, fp, fn, tn = int((h & j).sum()), int((~h & j).sum()), int((h & ~j).sum()), int((~h & ~j).sum())
            out[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
                         "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
                         "agreement": round((tp + tn) / len(rated), 4)}
        R["judges"][jn] = out
    R["note"] = "LLM-judge sensitivity/specificity here are on a sample balanced by the judge's own verdict; they describe this sample."
    (HERE / "results_claims.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    ru, rn = R["rates"]["unsupported"], R["rates"]["not_fully_supported"]
    L = ["# Human claim check (50 ground-answer claims, 3B ground run, test)", "",
         "All numbers come from `results_claims.json`, written by `score_claims.py`.", "",
         f"- Provenance: {R['provenance']}. Levels (user's definitions): yes = fully supported, partial = 50-75% correct, "
         "no / almost no = unsupported. Rows marked only `dupe` (repeated query) carry no verdict and are excluded: "
         f"{R['excluded_dupe_only']} of {R['claims_exported']}; {R['rated']} rated.",
         f"- Rated levels: {R['human_levels']}.",
         f"- Sampling: 25 judge-supported + 25 judge-unsupported claims from {n_pool} test claims, of which the 3B judge "
         f"called {n_pool_uns} unsupported; pool estimates re-weight the two strata ({BOOT:,} parametric bootstrap draws per stratum, seed {SEED}).", "",
         "| Rate | rated sample | judge-supported stratum | judge-unsupported stratum | pool estimate [95% CI] |", "|---|---|---|---|---|"]
    for nm, v in (("Unsupported", ru), ("Not fully supported (unsupported + partial)", rn)):
        L.append(f"| {nm} | {v['in_rated_sample'][0]}/{v['in_rated_sample'][1]} | {v['among_judge_supported'][0]}/{v['among_judge_supported'][1]} | "
                 f"{v['among_judge_unsupported'][0]}/{v['among_judge_unsupported'][1]} | {v['pool_estimate']} {v['pool_estimate_ci95_bootstrap']} |")
    L += ["", "## Judges against the human labels (rated claims)", "",
          "| Judge | reference | flagged and human-bad | flagged, human-ok | missed | sensitivity | specificity | agreement |",
          "|---|---|---|---|---|---|---|---|"]
    for jn, out in R["judges"].items():
        for nm, v in out.items():
            L.append(f"| {jn} | {nm} | {v['tp']} | {v['fp']} | {v['fn']} | {v['sensitivity']} | {v['specificity']} | {v['agreement']} |")
    L += ["", "## Limits", "", "- One rater, 39 rated claims; intervals are wide.",
          "- `partial` mixes claim support with off-topic cited reports in the notes; both readings are reported.",
          f"- {R['note']}", "- The claims come from the 3B ground run (generation/out), not the 8B run."]
    (HERE / "REPORT_claims.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("rated", "human_levels", "rates", "judges")}, indent=1))


if __name__ == "__main__":
    main()
