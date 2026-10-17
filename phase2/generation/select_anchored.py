#!/usr/bin/env python3
"""Task 7, step 4: sentence-anchored prompt vs the free prompt, dev only (no test query is generated here).

Inputs (already run, dev split, same judge llama3.2:3b):
  compare/<model>/results.json            free prompt   (compare_models.py)
  compare_anchored/<model>/results.json   anchored prompt (generate.py --prompt anchored; judge.py)
Rule, fixed before the anchored runs: the same rule as compare_models.py over all four candidates - lowest filtered
hallucination on dev; candidates within 1 percentage point tie, and the lowest LLM p50 latency among them wins.
Only the chosen candidate is then run on test (outside this script).
Writes compare_anchored/results.json and compare_anchored/REPORT.md.
"""
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent
MARGIN, MODELS = 0.01, ["llama3.2:3b", "llama3.1:8b"]


def row(d):
    r = json.loads((d / "results.json").read_text())
    g = json.loads((d / "results_generation.json").read_text())
    h, hr = r["hallucination_filtered"], r["hallucination_raw_before_grounding_filter"]
    return {"status_counts": g["status_counts"], "answers_judged": h["answers"], "claims": h["claims"],
            "unsupported": h["unsupported"], "hallucination_filtered": h["rate"], "claims_raw": hr["claims"],
            "unsupported_raw": hr["unsupported"], "hallucination_raw": hr["rate"],
            "judge_invalid_json": r["judge_invalid_json"], "anchors": g.get("anchors"),
            "llm_latency_ms_p50": g["llm_latency_ms_p50"], "llm_latency_ms_p95": g["llm_latency_ms_p95"]}


def main():
    cands = {}
    for m in MODELS:
        cands[f"free {m}"] = row(TASK / "compare" / m.replace(":", "_"))
        cands[f"anchored {m}"] = row(TASK / "compare_anchored" / m.replace(":", "_"))
    names = list(cands)
    ok = [k for k in names if cands[k]["hallucination_filtered"] is not None]
    low = min(cands[k]["hallucination_filtered"] for k in ok)
    near = [k for k in ok if cands[k]["hallucination_filtered"] <= low + MARGIN]
    chosen = min(near, key=lambda k: (cands[k]["llm_latency_ms_p50"], names.index(k)))
    R = {"split": "dev", "judge_model": "llama3.2:3b", "candidates": cands, "within_margin": near, "chosen": chosen,
         "rule": f"lowest filtered hallucination on dev; within {MARGIN * 100:.0f} point(s) of the lowest tie, lowest "
                 "LLM p50 wins"}
    out = TASK / "compare_anchored"
    (out / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Task 7, step 4: sentence-anchored answers vs free answers, dev only", "",
         "All numbers come from `results.json`, written by `select_anchored.py`. Test queries were not generated here.",
         "", "## Method", "",
         "- `generate.py --prompt anchored`: every retrieved report is split into sentences with IDs (`R2.S3`). The LLM "
         "returns each claim and hazard with the ID of the sentence it restates. A claim is kept only if its ID exists "
         "and at least half of its content words occur in that one sentence; then the same grounding filter and "
         "phraseology check as the free prompt apply.",
         "- If the model writes a sentence ID as the claim text, the cited sentence itself is used (counted as "
         "`text_was_id`). This fallback and one line of format example were added after the first 4 dev answers of "
         "the first anchored run showed this format error; that run was discarded and nothing else was changed.",
         "- Same retrieval context, Ollama options, judge (`llama3.2:3b`) and dev queries as `compare/`.",
         f"- Rule fixed before the anchored runs: {R['rule']}.", "",
         "## Dev results", "",
         "| Candidate | claims | unsupported | hallucination (filtered) | hallucination (raw) | LLM p50 ms | "
         "LLM p95 ms | anchors | statuses |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in cands.items():
        L.append(f"| {k} | {v['claims']} | {v['unsupported']} | {v['hallucination_filtered']} | "
                 f"{v['hallucination_raw']} | {v['llm_latency_ms_p50']} | {v['llm_latency_ms_p95']} | "
                 f"{v['anchors'] or '-'} | {v['status_counts']} |")
    L += ["", f"Chosen on dev: **{chosen}** (within margin: {', '.join(near)}).", "",
          "## Limitations", "",
          "- 20 dev queries; differences of a few claims are within noise.",
          "- The judge is `llama3.2:3b`; it also marks about 12% of claims unsupported in the cockpit answers, which "
          "are copied word for word from the cited reports (`phase2/edge/REPORT.md`), so these rates carry that error.",
          "- The anchor check is lexical (word overlap with one sentence), not semantic entailment."]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"chosen": chosen, "within_margin": near,
                      "rates": {k: cands[k]["hallucination_filtered"] for k in names}}, indent=1))


if __name__ == "__main__":
    main()
