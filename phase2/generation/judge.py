#!/usr/bin/env python3
"""Task 7, step 2: claim-level LLM judge, rating export and REPORT.md.

For each answer that called the LLM, the judge model splits summary + key_hazards into atomic claims and marks
each as supported or not by the retrieved incident summaries. Hallucination rate = unsupported / all claims.
The default judge is the generator model (llama3.2:3b), which can favour its own output; pass --judge-model to
use another. Exports 30 answers (seed 42) to rate/answers_to_rate.csv for the user, and adds the user's ratings
from results_ratings.json (score_ratings.py) when present. Writes out/judgements.jsonl (resumable), results.json,
REPORT.md.
Run:  python phase2/generation/judge.py [--phase1 PATH] [--judge-model llama3.2:3b] [--base-url URL] [--answers-dir DIR]
      (--answers-dir: judge DIR/answers.jsonl, write DIR/judgements*.jsonl and DIR/results.json only; no export/report)
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


def judge_all(llm, A, summ, field, path):
    """Judge A[i][field] for every answer that called the LLM; resumable, one JSON line per answer."""
    done = {json.loads(x)["query_id"] for x in path.read_text().splitlines()} if path.exists() else set()
    for a in A:
        ans = a.get(field)
        if a["query_id"] in done or not ans or not ans.get("summary") or a["status"] == "insufficient_evidence":
            continue
        text = ans["summary"] + "\nKey hazards: " + "; ".join(ans["key_hazards"])
        docs = "\n".join(f"[{i}] {summ.get(i, '')}" for i in a["retrieved"])
        raw = llm.invoke(PROMPT.format(docs=docs, answer=text)).content
        try:
            claims = [c for c in json.loads(raw).get("claims", []) if isinstance(c, dict) and "supported" in c]
            rec = {"query_id": a["query_id"], "claims": claims, "status": "ok"}
        except (json.JSONDecodeError, AttributeError):
            rec = {"query_id": a["query_id"], "claims": [], "status": "invalid_json", "raw": raw}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(field, a["query_id"], rec["status"], len(rec["claims"]), flush=True)


def raw_rate(path):
    J = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    cl = [c for j in J for c in j["claims"]]
    u = sum(not (c["supported"] is True or str(c["supported"]).lower() == "true") for c in cl)
    return {"answers": len(J), "claims": len(cl), "unsupported": u, "rate": round(u / len(cl), 4) if cl else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--judge-model", default=os.environ.get("UATIE_JUDGE_LLM", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--answers-dir")
    args = ap.parse_args()
    d = Path(args.answers_dir).resolve() if args.answers_dir else TASK / "out"
    repo = phase1_root()
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["incident_id", "summary"])
    summ = meta.astype(str).set_index("incident_id").summary
    A = [json.loads(x) for x in (d / "answers.jsonl").read_text().splitlines()]
    G = json.loads((d / "results_generation.json" if args.answers_dir else TASK / "results_generation.json").read_text())
    assert len(A) == G["queries"], "answers.jsonl incomplete: rerun generate.py"

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=args.judge_model, base_url=args.base_url, format="json", keep_alive="2h", **OPTIONS)
    f = d / "judgements.jsonl"
    fr = d / "judgements_raw.jsonl"
    for field, path in [("answer", f), ("raw_answer", fr)]:
        judge_all(llm, A, summ, field, path)
    if args.answers_dir:
        inv = sum(json.loads(x)["status"] != "ok" for x in f.read_text().splitlines()) if f.exists() else 0
        R = {"generation": G, "judge_model": args.judge_model, "judge_options": OPTIONS, "judge_invalid_json": inv,
             "hallucination_filtered": raw_rate(f), "hallucination_raw_before_grounding_filter": raw_rate(fr),
             "judgements_sha256": sha256(f) if f.exists() else None}
        (d / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
        return

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

    rf = TASK / "results_ratings.json"
    RT = json.loads(rf.read_text()) if rf.exists() else None
    R = {"generation": G, "judge_model": args.judge_model, "judge_options": OPTIONS,
         "judge_is_generator": args.judge_model == G["model"], "answers_judged": len(J),
         "judge_invalid_json": sum(j["status"] != "ok" for j in J), "claims": len(claims),
         "claims_unsupported": int(len(sup) - sum(sup)),
         "hallucination_rate_claims": round(1 - sum(sup) / len(sup), 4) if sup else None,
         "hallucination_rate_mean_per_answer": round(float(np.mean(per_q)), 4) if per_q else None,
         "rating_export": {"rows": len(pick), "status": "pending user rating"},
         "user_ratings_run_prev": RT,
         "judgements_sha256": sha256(f) if f.exists() else None,
         "hallucination_raw_before_grounding_filter": raw_rate(fr),
         "judgements_raw_sha256": sha256(fr) if fr.exists() else None}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    sc = G["status_counts"]
    L = ["# Task 7: grounded generation", "", "## Method", "",
         f"- LangChain `ChatOllama`, model `{G['model']}`, options {G['ollama_options']}, JSON output.",
         f"- Context: {G['retrieval_system']} top-{G['top_k']} (the primary Task 3 system: BM25 + FAISS candidates re-ranked by the cross-encoder), Phase 1 risk level "
         "of the top-1 incident, candidate phraseology (Task 6 filled phrase + top-3 crossmap phrases).",
         f"- Grounding filter: {G['grounding_rule']}. Dropped {G['grounding_dropped_sentences']} summary sentences "
         f"and {G['grounding_dropped_hazards']} hazards. The unfiltered answer is judged too (below).",
         f"- Insufficient evidence when the top-1 sigmoid(cross-encoder logit) < {G['insufficient_evidence_threshold']} "
         f"({G['threshold_rule']}; dev queries only).",
         "- Validation: citations outside the retrieved IDs and phrases not verbatim in the crossmap (or its Task 6 "
         "filled form) are removed and counted; an answer left with no valid citation is `no_valid_citation`.",
         f"- Judge: `{args.judge_model}` splits each answer into claims and marks each supported / unsupported "
         "by the retrieved summaries.", "", "## Numbers", "", "| Status | Answers |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in sc.items()]
    slow = [a['query_id'] for a in ok if a['latency_ms'] > 60000]
    L += ["", f"- Citation validity: {G['citation_validity']} ({G['citations_rejected']} of {G['citations_raw']} "
          "citations rejected).",
          f"- Phraseology rejected as not verbatim: {G['phraseology_rejected']} of {G['phraseology_raw']}.",
          f"- Before the grounding filter (raw answers, same judge): {R['hallucination_raw_before_grounding_filter']['rate']} "
          f"({R['hallucination_raw_before_grounding_filter']['unsupported']} of "
          f"{R['hallucination_raw_before_grounding_filter']['claims']} claims). The filter is lexical and fixed in "
          "advance; the judge is the same 3B model, so the before/after drop is not independent evidence.",
          f"- Hallucination rate: {R['hallucination_rate_claims']} of claims ({R['claims_unsupported']} of "
          f"{R['claims']}); mean per answer {R['hallucination_rate_mean_per_answer']}. Judge invalid JSON: "
          f"{R['judge_invalid_json']}.",
          f"- LLM latency on this VM (CPU): p50 {G['llm_latency_ms_p50']} ms, p95 {G['llm_latency_ms_p95']} ms "
          f"(num_predict {G['ollama_options']['num_predict']}). Streaming: time to first token p50 "
          f"{G.get('ttft_ms_p50')} ms; the JSON answer is only usable once complete, so streaming does not shorten "
          f"time to a validated answer. Retrieval incl. cross-encoder: p50 {G.get('retrieval_ms_p50')} ms, p95 "
          f"{G.get('retrieval_ms_p95')} ms. "
          + ("Target <= 5 s end to end: met." if G['llm_latency_ms_p50'] + G.get('retrieval_ms_p50', 0) <= 5000 else
             "Target <= 5 s end to end: MISSED on CPU; meeting it needs a GPU (or a smaller model, untested here)."),
          f"- {len(slow)} of {len(ok)} answered queries took > 60 s" + (f" ({', '.join(slow)})." if slow else "."),
          ""]
    V = TASK / "run_variation.json"
    if V.exists():
        V = json.loads(V.read_text())
        L += ["## Run-to-run variation", "",
              "`compare_runs.py` compares the two archived FAISS-context runs, which had identical inputs "
              "(`out/run_prev/` vs `out/run_faiss_claude/`; this CE run has different inputs and is not part of it). Same "
              f"retrieval lists {V['same_retrieved_lists']}/{V['queries']}, same candidate phraseology "
              f"{V['same_candidate_phraseology']}/{V['queries']}, threshold {V['threshold_prev']} vs "
              f"{V['threshold_cur']}, same insufficient-evidence set: {V['insufficient_sets_equal']}.", "",
              f"- Identical raw LLM output: {V['identical_raw_output']} of {V['generated']} generated answers, "
              "despite temperature 0 and a fixed seed; Ollama on CPU is not bit-reproducible across runs here.",
              f"- Answer status changed for {V['status_changed']} queries.",
              f"- Hallucination rate: {V['hallucination_prev']['rate']} ({V['hallucination_prev']['unsupported']}/"
              f"{V['hallucination_prev']['claims']}) in the first run vs {V['hallucination_cur']['rate']} "
              f"({V['hallucination_cur']['unsupported']}/{V['hallucination_cur']['claims']}) in the repeat, so a single run's "
              "rate is not stable to two decimals.", ""]
    if RT:
        dist = ", ".join(f"{k}: {v}" for k, v in RT["distribution"].items())
        sp = RT["spearman_rating_vs_llm_judge_unsupported"]
        L += ["## User ratings (earlier run)", "",
              f"Ratings: {RT['provenance']} (`rate/{RT['rated_file']}`, scored by `score_ratings.py`). They rate the "
              f"{RT['rows']} answers exported from the earlier run (`out/{RT['rated_run']}/`); every row's text was "
              "checked against that run.", "",
              f"- Distribution (rating: answers): {dist}. Mean {RT['mean']} (SD {RT['sd']}), median {RT['median']}; "
              f"{RT['rated_ge4']} of {RT['rows']} rated 4 or 5, {RT['rated_le2']} rated 1 or 2.",
              "- Mean by split: " + ", ".join(f"{k} {v['mean']} (n={v['n']})" for k, v in RT["by_split"].items())
              + "; by status: " + ", ".join(f"{k} {v['mean']} (n={v['n']})" for k, v in RT["by_status"].items())
              + "; by risk: " + ", ".join(f"{k} {v['mean']} (n={v['n']})" for k, v in RT["by_risk"].items()) + ".",
              f"- Spearman between rating and the LLM judge's unsupported-claim fraction for the same answers: "
              f"rho {sp['rho']} (p {sp['p']}, n {sp['n']})" + (
                  "; no significant association at n = 30, so the LLM judge's rate cannot stand in for user ratings."
                  if sp["p"] >= 0.05 else "."),
              f"- {len(RT['same_answer_in_current_run'])} of the {RT['rows']} rated answers are identical in the current "
              "(CE-context) run" + (f" ({', '.join(RT['same_answer_in_current_run'])})" if RT['same_answer_in_current_run']
                                    else "") + ", so these ratings describe the earlier FAISS-context run, not "
              "`out/answers.jsonl`; the current answers have no user ratings.", ""]
    L += [
          "## Failures and limits", "",
          "- The judge is an LLM, and by default the same 3B model as the generator; the hallucination rate is an "
          "automatic estimate, not a human judgement." if R["judge_is_generator"] else
          "- The judge is an LLM; the hallucination rate is an automatic estimate, not a human judgement.",
          "- Claims are judged against ASRS summaries only (no full narratives in the Phase 1 data).",
          "- Retrieval relevance labels behind the dev threshold include provisional LLM labels "
          "(LLM-judged (Claude), pending user review).",
          f"- `rate/answers_to_rate.csv`: {len(pick)} answers (seed {SEED}) from the current run; "
          + ("not rated (the user ratings above are of the earlier run)." if RT else "pending user rating."),
          "", "## Reproduce", "", "```", "ollama pull llama3.2:3b",
          "python phase2/generation/generate.py --phase1 ../uatie-edm-phase1",
          "python phase2/generation/judge.py --phase1 ../uatie-edm-phase1", "```"]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["answers_judged", "claims", "hallucination_rate_claims",
                                        "hallucination_rate_mean_per_answer", "judge_invalid_json"]}))


if __name__ == "__main__":
    main()
