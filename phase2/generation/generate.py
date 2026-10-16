#!/usr/bin/env python3
"""Task 7, step 1: grounded answer generation with LangChain + Ollama (default llama3.2:3b), with validation.

Per query: FAISS top-5 (the Task 3 system chosen on dev), Phase 1 risk level of the top-1 incident
(rag_query_engine.compute_risk_score), candidate phraseology (Task 6 filled phrase + top-3 crossmap phrases).
If the top-1 FAISS score is below a threshold chosen on DEV queries only, the answer is "insufficient evidence"
and the LLM is not called. Citations outside the retrieved IDs and phrases not verbatim in the crossmap (or its
Task 6 slot-filled form) are removed and counted.
Writes out/answers.jsonl (resumable), results_generation.json.
Run:  python phase2/generation/generate.py [--phase1 PATH] [--model llama3.2:3b] [--base-url http://localhost:11434]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
P2 = ROOT / "phase2"
sys.path.insert(0, str(ROOT))
from phase1 import phase1_root, sha256  # noqa: E402

SEED, TOPK, N_PHRASES = 42, 5, 3
OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": 350}
INSUFFICIENT = "Insufficient evidence in the retrieved incidents to answer this query."
PROMPT = """You are an aviation safety analyst. Answer ONLY from the incident reports below.
Query: {query}
Risk level (from the top incident): {risk}

Incident reports:
{docs}

Allowed ATC phraseology (copy exactly, or use none):
{phrases}

Return JSON with keys:
"summary": 2-3 sentences answering the query using only the reports,
"key_hazards": list of short hazard strings found in the reports,
"recommended_phraseology": list of phrases copied exactly from the allowed list,
"citations": list of incident IDs (from the reports above) that support the summary."""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pick_threshold(scores, positive):
    """Threshold on the top-1 score maximising balanced accuracy of 'top-5 has a relevant result' (dev only)."""
    s, y = np.asarray(scores), np.asarray(positive, bool)
    if y.all() or not y.any():
        return float(s[y].min() if y.any() else s.max()), "dev has one class only: min top-1 score of positive dev queries"
    best = max(sorted(set(s)), key=lambda t: (((s >= t) & y).sum() / y.sum() + ((s < t) & ~y).sum() / (~y).sum(), -t))
    return float(best), "max balanced accuracy on dev"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--model", default=os.environ.get("UATIE_LLM", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()
    repo = phase1_root()
    cwd = Path.cwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    rag = load_module("rag_query_engine", repo / "src" / "rag" / "rag_query_engine.py")
    rag.initialize_resources(verbose=False)
    os.chdir(cwd)

    system = json.loads((P2 / "retrieval" / "eval" / "results.json").read_text())["best_system_chosen_on_dev_ndcg10"]
    assert system == "faiss", f"generation implements FAISS retrieval only; dev chose {system}"
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["incident_id"] = meta.incident_id.astype(str)
    q = pd.concat([pd.read_csv(P2 / "gold_eval_rebuild" / "splits" / f"{s}_queries.csv").assign(split=s)
                   for s in ["dev", "test"]], ignore_index=True)
    runs = pd.read_csv(P2 / "retrieval" / "runs" / "runs_top10.csv")
    runs = runs[runs.system == system]
    qrels = pd.read_csv(P2 / "retrieval" / "eval" / "qrels_dev.csv")
    rel = set(zip(qrels[qrels.relevance >= 1].query_id, qrels[qrels.relevance >= 1].incident_id.astype(str)))
    cm = pd.read_csv(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv")
    slots = pd.read_csv(P2 / "slots" / "out" / "slot_fills.csv").set_index("query_id")
    allowed = set(cm.Phraseology.astype(str)) | set(slots.filled.astype(str))
    doc = (cm.Phraseology.astype(str) + " " + cm.Circumstances.fillna("").astype(str) + " " +
           cm.Section_Title.fillna("").astype(str))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(list(doc) + list(q.query_text))
    psim = cosine_similarity(vec.transform(q.query_text), vec.transform(doc))

    ret = {}
    for r in q.itertuples():
        sims, idx = rag.index.search(rag.embed_query(str(r.query_text)), TOPK)
        ids = [meta.incident_id.iat[i] for i in idx[0]]
        exp = runs[runs.query_id == r.query_id].sort_values("rank").incident_id.astype(str).tolist()[:TOPK]
        assert ids == exp, f"{r.query_id}: FAISS top-5 differs from Task 3 run"
        ret[r.query_id] = (ids, [float(x) for x in sims[0]], idx[0].tolist())
    dev = q[q.split == "dev"].query_id
    thr, how = pick_threshold([ret[x][1][0] for x in dev], [any((x, i) in rel for i in ret[x][0]) for x in dev])

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=args.model, base_url=args.base_url, format="json", keep_alive="2h", **OPTIONS)
    out = TASK / "out"
    out.mkdir(exist_ok=True)
    f = out / "answers.jsonl"
    done = {json.loads(line)["query_id"] for line in f.read_text().splitlines()} if f.exists() else set()
    for n, r in enumerate(q.itertuples(), 1):
        if r.query_id in done:
            continue
        ids, scores, pos = ret[r.query_id]
        top = meta.iloc[pos[0]]
        risk = rag.compute_risk_score(top.intent, top.phase_final, top.hazard_tags)
        cand = list(dict.fromkeys([str(slots.filled.get(r.query_id, ""))] +
                                  [str(cm.Phraseology.iat[j]) for j in np.argsort(-psim[n - 1])[:N_PHRASES]]))
        cand = [c for c in cand if c and c != "nan"]
        rec = {"query_id": r.query_id, "split": r.split, "query_text": r.query_text, "retrieved": ids,
               "scores": [round(s, 4) for s in scores], "risk": risk, "candidate_phraseology": cand,
               "model": args.model, "threshold": round(thr, 4)}
        t0 = time.perf_counter()
        if scores[0] < thr:
            rec.update(status="insufficient_evidence", answer={"summary": INSUFFICIENT, "key_hazards": [],
                                                               "recommended_phraseology": [], "citations": []})
        else:
            docs = "\n".join(f"[{i}] {meta.summary.iat[p]}" for i, p in zip(ids, pos))
            raw = llm.invoke(PROMPT.format(query=r.query_text, risk=risk, docs=docs,
                                           phrases="\n".join(f"- {c}" for c in cand))).content
            rec["raw"] = raw
            try:
                a = json.loads(raw)
                cites = [str(c) for c in a.get("citations", []) or []]
                phr = [str(p) for p in a.get("recommended_phraseology", []) or []]
                hz = a.get("key_hazards", []) or []
                a = {"summary": str(a.get("summary", "")), "key_hazards": [str(h) for h in hz] if isinstance(hz, list) else [str(hz)],
                     "recommended_phraseology": [p for p in phr if p in allowed],
                     "citations": [c for c in cites if c in ids]}
                rec.update(answer=a, citations_raw=cites, citations_rejected=[c for c in cites if c not in ids],
                           phraseology_raw=phr, phraseology_rejected=[p for p in phr if p not in allowed])
                rejected = rec["citations_rejected"] or rec["phraseology_rejected"]
                rec["status"] = "ok_after_filtering" if rejected else "ok"
                if not a["citations"]:
                    rec["status"] = "no_valid_citation"
            except (json.JSONDecodeError, AttributeError, TypeError):
                rec.update(status="invalid_json", answer=None)
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{n}/{len(q)}] {r.query_id} {rec['status']} {rec['latency_ms']:.0f} ms", flush=True)

    A = [json.loads(line) for line in f.read_text().splitlines()]
    called = [a for a in A if a["status"] != "insufficient_evidence"]
    craw = sum(len(a.get("citations_raw", [])) for a in called)
    praw = sum(len(a.get("phraseology_raw", [])) for a in called)
    lat = [a["latency_ms"] for a in called]
    R = {"model": args.model, "ollama_options": OPTIONS, "retrieval_system": system, "top_k": TOPK,
         "insufficient_evidence_threshold": round(thr, 4), "threshold_rule": how, "queries": len(A),
         "status_counts": pd.Series([a["status"] for a in A]).value_counts().to_dict(),
         "citations_raw": craw, "citations_rejected": sum(len(a.get("citations_rejected", [])) for a in called),
         "citation_validity": round(1 - sum(len(a.get("citations_rejected", [])) for a in called) / craw, 4) if craw else None,
         "phraseology_raw": praw, "phraseology_rejected": sum(len(a.get("phraseology_rejected", [])) for a in called),
         "llm_latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
         "llm_latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
         "answers_sha256": sha256(f)}
    (TASK / "results_generation.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
