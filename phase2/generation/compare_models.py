#!/usr/bin/env python3
"""Task 7, step 3: dev-only generator comparison (no test query is generated here).

Each model generates answers for the 20 dev queries with the same context, prompt, options and grounding filter as
generate.py; judge.py then scores them with a fixed judge (llama3.2:3b) for every model.
Choice rule, fixed before any run: lowest filtered hallucination rate (unsupported / all claims) on dev; if other
models are within 1 percentage point of the lowest, the one with the lowest LLM p50 latency among them wins.
Only the chosen model is then run on test (generate.py --model <chosen>), outside this script.
Writes compare/<model>/{answers.jsonl, judgements*.jsonl, results_generation.json, results.json},
compare/results.json, compare/REPORT.md.
Run:  python phase2/generation/compare_models.py --phase1 PATH [--models llama3.2:1b,llama3.2:3b,llama3.1:8b]
"""
import argparse
import json
import platform
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
OUT = TASK / "compare"
JUDGE = "llama3.2:3b"
MARGIN = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--models", default="llama3.2:1b,llama3.2:3b,llama3.1:8b")
    a = ap.parse_args()
    models = a.models.split(",")
    extra = ["--phase1", a.phase1] if a.phase1 else []
    rows = {}
    for m in models:
        d = OUT / m.replace(":", "_")
        for script, args in [("generate.py", ["--model", m, "--splits", "dev", "--out-dir", str(d)]),
                             ("judge.py", ["--judge-model", JUDGE, "--answers-dir", str(d)])]:
            print("running", script, m, flush=True)
            subprocess.run([sys.executable, str(TASK / script)] + extra + args, check=True)
        r = json.loads((d / "results.json").read_text())
        g, h, hr = r["generation"], r["hallucination_filtered"], r["hallucination_raw_before_grounding_filter"]
        rows[m] = {"dev_queries": g["queries"], "status_counts": g["status_counts"],
                   "insufficient_evidence_threshold": g["insufficient_evidence_threshold"],
                   "answers_judged": h["answers"], "claims": h["claims"], "unsupported": h["unsupported"],
                   "hallucination_filtered": h["rate"], "hallucination_raw": hr["rate"],
                   "claims_raw": hr["claims"], "unsupported_raw": hr["unsupported"],
                   "judge_invalid_json": r["judge_invalid_json"], "citation_validity": g["citation_validity"],
                   "phraseology_rejected": g["phraseology_rejected"], "phraseology_raw": g["phraseology_raw"],
                   "llm_latency_ms_p50": g["llm_latency_ms_p50"], "llm_latency_ms_p95": g["llm_latency_ms_p95"],
                   "ttft_ms_p50": g["ttft_ms_p50"], "retrieval_ms_p50": g["retrieval_ms_p50"]}
    ok = [m for m in models if rows[m]["hallucination_filtered"] is not None]
    low = min(rows[m]["hallucination_filtered"] for m in ok)
    near = [m for m in ok if rows[m]["hallucination_filtered"] <= low + MARGIN]
    chosen = min(near, key=lambda m: (rows[m]["llm_latency_ms_p50"], models.index(m)))
    R = {"split": "dev", "judge_model": JUDGE, "models": rows,
         "rule": f"lowest filtered hallucination on dev; models within {MARGIN * 100:.0f} percentage point(s) of the "
                 "lowest tie, and the lowest LLM p50 latency among them wins",
         "within_margin": near, "chosen": chosen,
         "env": {"python": platform.python_version(), "cpu_count": os.cpu_count()}}
    (OUT / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Task 7, step 3: generator comparison on dev only", "",
         "All numbers come from `results.json`, written by `compare_models.py`. Test queries were not generated here.",
         "", "## Method", "",
         "- Same context (cross-encoder top-5), prompt, Ollama options, validation and grounding filter as `generate.py`; "
         "only the generator model changes.",
         f"- Judge fixed at `{JUDGE}` for every model. It may favour answers from its own model (`llama3.2:3b`).",
         f"- Rule fixed before any run: {R['rule']}.",
         "- CPU only; run with no other job on the machine. Ollama on CPU is not bit-reproducible across runs.", "",
         "## Dev results", "",
         "| Model | answers judged | claims | unsupported | hallucination (filtered) | hallucination (raw) | "
         "LLM p50 ms | LLM p95 ms | TTFT p50 ms | statuses |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in rows.items():
        L.append(f"| {m} | {v['answers_judged']} | {v['claims']} | {v['unsupported']} | {v['hallucination_filtered']} | "
                 f"{v['hallucination_raw']} | {v['llm_latency_ms_p50']} | {v['llm_latency_ms_p95']} | {v['ttft_ms_p50']} | "
                 f"{v['status_counts']} |")
    L += ["", f"Chosen on dev: **`{chosen}`** (within margin: {', '.join(near)}).", "",
          "## Limitations", "",
          "- 20 dev queries only; the rates rest on the claim counts above and are noisy.",
          "- The judge is an LLM, not a human, and is one of the compared models."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"chosen": chosen, "within_margin": near}, indent=1))


if __name__ == "__main__":
    main()
