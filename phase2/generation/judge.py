#!/usr/bin/env python3
"""Task 7, step 2: claim-level LLM judge, rating export and REPORT.md.

For each answer that called the LLM, the judge model splits summary + key_hazards into atomic claims and marks
each as supported or not by the retrieved incident summaries. Hallucination rate = unsupported / all claims.
The default judge is the generator model (llama3.2:3b), which can favour its own output; pass --judge-model to
use another. Exports 30 answers (seed 42) to rate/answers_to_rate.csv for the user; ratings stay pending.
Writes out/judgements.jsonl (resumable), results.json, REPORT.md.
Run:  python phase2/generation/judge.py [--phase1 PATH] [--judge-model llama3.2:3b] [--base-url URL]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_root, sha256  # noqa: E402

SEED, N_RATE = 42, 30
OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": 500}
PROMPT = """You check an answer against incident reports.
Incident reports:
{docs}

Answer:
{answer}

Split the answer into short atomic factual claims. For each claim decide if the incident reports support it.
Return JSON: {{"claims": [{{"claim": "...", "supported": true or false}}]}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--judge-model", default=os.environ.get("UATIE_JUDGE_LLM", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()
    repo = phase1_root()
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["incident_id", "summary"])
    summ = meta.astype(str).set_index("incident_id").summary
    A = [json.loads(x) for x in (TASK / "out" / "answers.jsonl").read_text().splitlines()]
    G = json.loads((TASK / "results_generation.json").read_text())
    assert len(A) == G["queries"], "answers.jsonl incomplete: rerun generate.py"

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=args.judge_model, base_url=args.base_url, format="json", keep_alive="2h", **OPTIONS)
    f = TASK / "out" / "judgements.jsonl"
    done = {json.loads(x)["query_id"] for x in f.read_text().splitlines()} if f.exists() else set()
    for a in A:
        if a["query_id"] in done or not a.get("answer") or a["status"] == "insufficient_evidence":
            continue
        ans = a["answer"]
        text = ans["summary"] + "\nKey hazards: " + "; ".join(ans["key_hazards"])
        docs = "\n".join(f"[{i}] {summ.get(i, '')}" for i in a["retrieved"])
        raw = llm.invoke(PROMPT.format(docs=docs, answer=text)).content
        try:
            claims = [c for c in json.loads(raw).get("claims", []) if isinstance(c, dict) and "supported" in c]
            rec = {"query_id": a["query_id"], "claims": claims, "status": "ok"}
        except (json.JSONDecodeError, AttributeError):
            rec = {"query_id": a["query_id"], "claims": [], "status": "invalid_json", "raw": raw}
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(a["query_id"], rec["status"], len(rec["claims"]), flush=True)

    J = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    claims = [c for j in J for c in j["claims"]]
    sup = [c["supported"] is True or str(c["supported"]).lower() == "true" for c in claims]
    per_q = [1 - np.mean([c["supported"] is True or str(c["supported"]).lower() == "true" for c in j["claims"]])
             for j in J if j["claims"]]

    ok = [a for a in A if a.get("answer") and a["status"] != "insufficient_evidence"]
    rng = np.random.RandomState(SEED)
    pick = sorted(rng.choice(len(ok), min(N_RATE, len(ok)), replace=False))
    (TASK / "rate").mkdir(exist_ok=True)
    pd.DataFrame([{"rate_id": f"R{k + 1:02d}", "query_id": ok[i]["query_id"], "query_text": ok[i]["query_text"],
                   "risk": ok[i]["risk"], "summary": ok[i]["answer"]["summary"],
                   "key_hazards": "; ".join(ok[i]["answer"]["key_hazards"]),
                   "recommended_phraseology": " | ".join(ok[i]["answer"]["recommended_phraseology"]),
                   "citations": " ".join(ok[i]["answer"]["citations"]), "your_rating_1_to_5": "", "notes": ""}
                  for k, i in enumerate(pick)]).to_csv(TASK / "rate" / "answers_to_rate.csv", index=False)

    R = {"generation": G, "judge_model": args.judge_model, "judge_options": OPTIONS,
         "judge_is_generator": args.judge_model == G["model"], "answers_judged": len(J),
         "judge_invalid_json": sum(j["status"] != "ok" for j in J), "claims": len(claims),
         "claims_unsupported": int(len(sup) - sum(sup)),
         "hallucination_rate_claims": round(1 - sum(sup) / len(sup), 4) if sup else None,
         "hallucination_rate_mean_per_answer": round(float(np.mean(per_q)), 4) if per_q else None,
         "rating_export": {"rows": len(pick), "status": "pending user rating"},
         "judgements_sha256": sha256(f) if f.exists() else None}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    sc = G["status_counts"]
    L = ["# Task 7: grounded generation", "", "## Method", "",
         f"- LangChain `ChatOllama`, model `{G['model']}`, options {G['ollama_options']}, JSON output.",
         f"- Context: {G['retrieval_system']} top-{G['top_k']} (the Task 3 system chosen on dev), Phase 1 risk level "
         "of the top-1 incident, candidate phraseology (Task 6 filled phrase + top-3 crossmap phrases).",
         f"- Insufficient evidence when the top-1 FAISS score < {G['insufficient_evidence_threshold']} "
         f"({G['threshold_rule']}; dev queries only).",
         "- Validation: citations outside the retrieved IDs and phrases not verbatim in the crossmap (or its Task 6 "
         "filled form) are removed and counted; an answer left with no valid citation is `no_valid_citation`.",
         f"- Judge: `{args.judge_model}` splits each answer into claims and marks each supported / unsupported "
         "by the retrieved summaries.", "", "## Numbers", "", "| Status | Answers |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in sc.items()]
    L += ["", f"- Citation validity: {G['citation_validity']} ({G['citations_rejected']} of {G['citations_raw']} "
          "citations rejected).",
          f"- Phraseology rejected as not verbatim: {G['phraseology_rejected']} of {G['phraseology_raw']}.",
          f"- Hallucination rate: {R['hallucination_rate_claims']} of claims ({R['claims_unsupported']} of "
          f"{R['claims']}); mean per answer {R['hallucination_rate_mean_per_answer']}. Judge invalid JSON: "
          f"{R['judge_invalid_json']}.",
          f"- LLM latency on this VM (CPU): p50 {G['llm_latency_ms_p50']} ms, p95 {G['llm_latency_ms_p95']} ms.",
          f"- {sum(a['latency_ms'] > 60000 for a in ok)} of {len(ok)} answered queries took > 60 s "
          f"({', '.join(a['query_id'] for a in ok if a['latency_ms'] > 60000)}); on this VM these coincided with "
          "other CPU jobs (Task 1 DistilBERT training, Task 1 ontology population), so p95 reflects contention.", "",
          "## Failures and limits", "",
          "- The judge is an LLM, and by default the same 3B model as the generator; the hallucination rate is an "
          "automatic estimate, not a human judgement." if R["judge_is_generator"] else
          "- The judge is an LLM; the hallucination rate is an automatic estimate, not a human judgement.",
          "- Claims are judged against ASRS summaries only (no full narratives in the Phase 1 data).",
          "- Retrieval relevance labels behind the dev threshold include provisional LLM labels "
          "(LLM-judged (Devin), pending user review).",
          f"- `rate/answers_to_rate.csv`: {len(pick)} answers (seed {SEED}) for your 1-5 rating; pending user rating.",
          "", "## Reproduce", "", "```", "ollama pull llama3.2:3b",
          "python phase2/generation/generate.py --phase1 ../uatie-edm-phase1",
          "python phase2/generation/judge.py --phase1 ../uatie-edm-phase1", "```"]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["answers_judged", "claims", "hallucination_rate_claims",
                                        "hallucination_rate_mean_per_answer", "judge_invalid_json"]}))


if __name__ == "__main__":
    main()
