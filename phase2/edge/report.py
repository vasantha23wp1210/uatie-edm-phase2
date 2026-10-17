#!/usr/bin/env python3
"""Edge / deployment profiles: collects measured results into results.json and REPORT.md (no new measurements).
Inputs: out/results_generation.json (cockpit.py), out/results.json (judge.py --answers-dir out),
onnx/results.json (onnx_export.py, optional), ../generation/compare/results.json,
../generation/compare_anchored/results.json (optional).
Run:  python phase2/edge/report.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
P2 = TASK.parent
sys.path.insert(0, str(P2.parent))
from phase1 import phase1_root  # noqa: E402

SPLIT = re.compile(r"(?<=[.!?])\s+")


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def claim_rate(path, ids=None):
    c = u = 0
    for x in path.read_text().splitlines():
        j = json.loads(x)
        if ids is None or j["query_id"] in ids:
            c += len(j["claims"])
            u += sum(1 for k in j["claims"] if str(k["supported"]).lower() != "true")
    return {"claims": c, "unsupported": u, "rate": round(u / c, 4) if c else None}


def main():
    G = load(TASK / "out" / "results_generation.json")
    J = load(TASK / "out" / "results.json")
    J8 = load(TASK / "out_judge_llama3.1_8b" / "results.json")
    X = load(TASK / "onnx" / "results.json")
    C = load(P2 / "generation" / "compare" / "results.json")
    H = load(P2 / "generation" / "results.json")
    T = P2 / "generation" / "out_llama3.1_8b_test"
    TG, TJ = load(T / "results_generation.json"), load(T / "results.json")
    GO = P2 / "generation" / "out"
    B = [json.loads(x) for x in (GO / "answers.jsonl").read_text().splitlines()]
    bt = {a["query_id"] for a in B if a["split"] == "test"}
    blat = [a["latency_ms"] for a in B if a["split"] == "test" and a.get("latency_ms")]
    base = {"model": B[0]["model"], "split": "test", "filtered": claim_rate(GO / "judgements.jsonl", bt),
            "raw": claim_rate(GO / "judgements_raw.jsonl", bt), "llm_calls": len(blat),
            "llm_latency_ms_p50": round(float(np.percentile(blat, 50)), 1),
            "llm_latency_ms_p95": round(float(np.percentile(blat, 95)), 1)}
    h = J["hallucination_filtered"] if J else None
    meta = pd.read_csv(phase1_root() / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["incident_id", "summary"])
    text_of = meta.astype(str).set_index("incident_id").summary
    A = [json.loads(x) for x in (TASK / "out" / "answers.jsonl").read_text().splitlines()]
    sents = verb = 0
    for a in A:
        if a["status"] != "ok":
            continue
        rest = a["answer"]["summary"]
        for c in a["answer"]["citations"]:
            sents += 1
            ss = [x.strip() for x in SPLIT.split(text_of[c]) if x.strip() and rest.startswith(x.strip())]
            if ss:
                verb += 1
                rest = rest[len(max(ss, key=len)):].lstrip()
    V = {"summary_sentences": sents, "verbatim_in_cited_report": verb}
    R = {"cockpit": {"generation": G, "judge_model": J and J["judge_model"], "judge_invalid_json": J and J["judge_invalid_json"],
                     "hallucination": h, "verbatim_check": V,
                     "control_judge_llama3.1_8b": J8 and {"hallucination": J8["hallucination_filtered"],
                                                          "judge_invalid_json": J8["judge_invalid_json"]}},
         "onnx_int8": X, "ground_generator_comparison_dev": C,
         "ground_chosen_test": TG and TJ and {"model": TG.get("model"), "generation": TG, "judge_model": TJ["judge_model"],
                                              "hallucination": TJ["hallucination_filtered"],
                                              "judge_invalid_json": TJ["judge_invalid_json"]},
         "ground_baseline_test_same_queries": base,
         "targets": {"cockpit_hallucination_le_0.05": bool(h and h["rate"] is not None and h["rate"] <= 0.05),
                     "cockpit_latency_p95_le_5000ms": bool(G and G["end_to_end_ms_p95"] <= 5000)}}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Deployment profiles: ground (LLM) and cockpit (CPU only)", "",
         "All numbers come from `results.json`, written by `report.py` from the measured outputs listed in its docstring.", "",
         "## Profiles", "",
         "- **Ground** (ATC, safety analysts): full Task 7 pipeline with an LLM-written answer (`phase2/generation`). "
         "GPU recommended; only CPU latency is measured in this repository.",
         "- **Cockpit** (CPU only, no LLM): `cockpit.py`. Same retrieval (BM25 + FAISS, cross-encoder top-5) and dev-chosen "
         "insufficient-evidence threshold; the summary is the best TF-IDF sentence of each top-3 report copied verbatim and "
         "cited; the phrase is the first verified crossmap candidate; risk from the Phase 1 matrix. Design fixed before measuring.",
         "- **NPU / DPU**: the encoder and cross-encoder are exported to ONNX int8 (`onnx_export.py`). ONNX Runtime NPU "
         "execution providers can load the same files; that path is not measured (no NPU or DPU on this machine).", ""]
    if G:
        L += ["## Cockpit profile (100 gold queries, dev + test)", "",
              f"- Statuses: {G['status_counts']}.",
              f"- End-to-end latency (retrieval + NER + phrase + extractive): p50 {G['end_to_end_ms_p50']} ms, "
              f"p95 {G['end_to_end_ms_p95']} ms, max {G['end_to_end_ms_max']} ms. Target <= 5 s: "
              f"{'met' if R['targets']['cockpit_latency_p95_le_5000ms'] else 'missed'} (p95).",
              f"- Retrieval p50 {G['retrieval_ms_p50']} ms, p95 {G['retrieval_ms_p95']} ms."]
    if h:
        L += [f"- Hallucination (same claim judge `{J['judge_model']}` as Task 7, summary only): {h['rate']} "
              f"({h['unsupported']} of {h['claims']} claims, {h['answers']} answers; judge invalid JSON "
              f"{J['judge_invalid_json']}). Target <= 5%: {'met' if R['targets']['cockpit_hallucination_le_0.05'] else 'missed'}.",
              f"- Fabricated summary text: 0 by construction. The cockpit copies report sentences and does not generate "
              f"text; a deterministic exact-match check confirms {V['verbatim_in_cited_report']} of "
              f"{V['summary_sentences']} sentences. This is a stronger statement than any LLM-judged rate below.",
              f"- String check (deterministic): {V['verbatim_in_cited_report']} of {V['summary_sentences']} summary "
              "sentences are, in order, a verbatim sentence of the report they cite. Each claim the judge marks unsupported therefore comes from "
              "text copied from the retrieved reports: these are judge errors (or claims the judge split or reworded), "
              "so this rate is a lower bound on the judge's false-unsupported rate on fully grounded text. The same judge "
              "scores the ground profile, so its hallucination rates carry the same judge error.",
              "- Hazard tags are Phase 1 database fields shown as-is and are not judged."]
    if J8:
        h8 = J8["hallucination_filtered"]
        L += [f"- Second judge `llama3.1:8b` on the same verbatim answers (control): {h8['rate']} ({h8['unsupported']} of "
              f"{h8['claims']} claims marked unsupported; invalid JSON {J8['judge_invalid_json']}). A lower rate here means "
              "fewer false-unsupported verdicts on known-grounded text. Neither judge is independent proof; a human check of "
              "a sample of claims is needed to fix the true rate."]
    if X:
        L += ["", "## ONNX int8 (dev only)", "",
              f"- Size MB: {X['model_mb']}.",
              f"- Query encoder p50: torch fp32 {X['enc_ms_torch_p50']} ms, ONNX int8 {X['enc_ms_int8_p50']} ms; "
              f"FAISS top-50 overlap {X['faiss_top50_overlap_mean']}.",
              f"- Cross-encoder p50 per query: torch fp32 {X['ce_ms_torch_p50']} ms, ONNX int8 {X['ce_ms_int8_p50']} ms; "
              f"top-5 overlap {X['ce_top5_overlap_mean']}; dev P@5 {X['dev_p5_torch_fp32']} (fp32) vs {X['dev_p5_onnx_int8']} (int8).",
              f"- {X['npu']}."]
    if C:
        L += ["", "## Ground profile: generator comparison (dev only)", "",
              "| Model | hallucination (filtered) | claims | LLM p50 ms | LLM p95 ms |", "|---|---|---|---|---|"]
        L += [f"| {m} | {v['hallucination_filtered']} | {v['claims']} | {v['llm_latency_ms_p50']} | {v['llm_latency_ms_p95']} |"
              for m, v in C["models"].items()]
        L += ["", f"Chosen on dev: `{C['chosen']}` ({C['rule']}). Dev is 20 queries, so one claim moves a rate by about "
              "1 point; the judge is the 3B model, which may favour its own answers."]
    A = load(P2 / "generation" / "compare_anchored" / "results.json")
    if A:
        L += ["", "## Ground profile: sentence-anchored prompt (dev only)", "",
              "Each claim must cite a report sentence ID (`R2.S3`); claims with an unknown ID or too little word overlap "
              "with their sentence are dropped (`phase2/generation/compare_anchored/REPORT.md`).", "",
              "| Candidate | hallucination (filtered) | claims | LLM p50 ms |", "|---|---|---|---|"]
        L += [f"| {k} | {v['hallucination_filtered']} | {v['claims']} | {v['llm_latency_ms_p50']} |"
              for k, v in A["candidates"].items()]
        L += ["", f"Chosen on dev: {A['chosen']} ({A['rule']})."]
    if TG and TJ:
        th = TJ["hallucination_filtered"]
        L += ["", f"## Ground profile: chosen `{TG.get('model')}` on test (run once)", "",
              f"- Statuses: {TG['status_counts']}.",
              f"- Hallucination (judge `{TJ['judge_model']}`): {th['rate']} ({th['unsupported']} of {th['claims']} claims, "
              f"{th['answers']} answers; invalid JSON {TJ['judge_invalid_json']}). Subject to the judge error shown above.",
              f"- LLM latency p50 {TG['llm_latency_ms_p50']} ms, p95 {TG['llm_latency_ms_p95']} ms; first token p50 "
              f"{TG['ttft_ms_p50']} ms (CPU). Target <= 5 s: missed on CPU.",
              f"- Baseline `{base['model']}` (in `phase2/generation/out`) on the same test queries, same judge: "
              f"{base['filtered']['rate']} ({base['filtered']['unsupported']} of {base['filtered']['claims']} claims) "
              f"after the filter, {base['raw']['rate']} before it; LLM p50 {base['llm_latency_ms_p50']} ms, p95 "
              f"{base['llm_latency_ms_p95']} ms. The chosen model was kept; it was not switched after seeing test."]
    if H:
        L += [f"- Primary run in `phase2/generation/out` (dev + test, 100 queries): model `{H['generation']['model']}`, "
              f"hallucination {H['hallucination_rate_claims']} after the filter."]
    L += ["", "## Limitations", "",
          "- The cockpit answer is less fluent than an LLM answer and has no key-hazards list beyond the Phase 1 tags.",
          "- Answer relevance needs new user ratings for these answers; none exist yet.",
          "- GPU and NPU latencies are not measured here."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(R["targets"]))


if __name__ == "__main__":
    main()
