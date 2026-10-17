#!/usr/bin/env python3
"""Cockpit profile: CPU-only answers without an LLM (design fixed before measuring; nothing tuned on test).

Per query: the primary Task 3 retrieval (BM25 top-50 + FAISS top-50, cross-encoder re-rank, top-5); the
insufficient-evidence threshold chosen on dev with generate.py's rule; extractive summary = for each of the top-3 reports,
the sentence with the highest TF-IDF cosine to the query, copied verbatim and cited; recommended phrase = the first
generate.py candidate (Task 6 query-only filled phrase, else the top crossmap phrase), verbatim from the crossmap;
risk from Phase 1 compute_risk_score; Phase 1 hazard tags of the top-1 record shown as database fields (not judged).
Latency per query covers retrieval, spaCy NER on the query, phrase match and the extractive step.
Writes out/answers.jsonl and out/results_generation.json in the generate.py format so judge.py --answers-dir can
score them with the same claim judge; then results.json and REPORT.md (run report.py after judging).
Run:  python phase2/edge/cockpit.py --phase1 PATH [--limit N]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASK = Path(__file__).resolve().parent
P2 = TASK.parent
sys.path.insert(0, str(P2.parent))
sys.path.insert(0, str(P2 / "generation"))
from phase1 import phase1_root, sha256  # noqa: E402
from generate import INSUFFICIENT, N_PHRASES, TOPK, load_module, pick_threshold  # noqa: E402

N_SENT = 3
SPLIT = re.compile(r"(?<=[.!?])\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    repo = phase1_root()
    import os
    cwd = Path.cwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    rag = load_module("rag_query_engine", repo / "src" / "rag" / "rag_query_engine.py")
    rag.initialize_resources(verbose=False)
    bm25_mod = load_module("bm25_baseline", repo / "src" / "rag" / "bm25_baseline.py")
    os.chdir(cwd)
    t3 = load_module("p2_retrieval_run", P2 / "retrieval" / "run.py")
    t3.set_seeds()
    import spacy
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(t3.CE_MODEL, device="cpu", revision=t3.CE_REVISION)
    nlp = spacy.load(P2 / "ner" / "out" / "model")
    G = json.loads((P2 / "generation" / "results_generation.json").read_text())
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["incident_id"] = meta.incident_id.astype(str)
    text_of = meta.set_index("incident_id")["summary"].astype(str)
    posof = {i: p for p, i in enumerate(meta.incident_id)}
    bm25, _ = bm25_mod.build_bm25(meta, "summary")
    q = pd.concat([pd.read_csv(P2 / "gold_eval_rebuild" / "splits" / f"{s}_queries.csv").assign(split=s)
                   for s in ["dev", "test"]], ignore_index=True)
    runs = pd.read_csv(P2 / "retrieval" / "runs" / "runs_top10.csv")
    runs = runs[runs.system == "ce"]
    cm = pd.read_csv(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv")
    slots = pd.read_csv(P2 / "slots" / "out" / "slot_fills.csv").set_index("query_id")
    doc = (cm.Phraseology.astype(str) + " " + cm.Circumstances.fillna("").astype(str) + " " +
           cm.Section_Title.fillna("").astype(str))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(list(doc) + list(q.query_text))
    Xdoc = vec.transform(doc)
    ce.predict([("warm up", "warm up")])
    qr = pd.read_csv(P2 / "retrieval" / "eval" / "qrels_dev.csv")
    rel = set(zip(qr[qr.relevance >= 1].query_id, qr[qr.relevance >= 1].incident_id.astype(str)))

    def retrieve(qt):
        b = [str(x["incident_id"]) for x in bm25_mod.bm25_retrieve(bm25, qt, meta, "incident_id", t3.DEPTH)]
        _, idx = rag.index.search(rag.embed_query(qt), t3.DEPTH)
        union = list(dict.fromkeys(b + [str(meta.incident_id.iat[i]) for i in idx[0] if i >= 0]))
        logits = ce.predict([(qt, text_of[i]) for i in union], batch_size=32, show_progress_bar=False)
        order = sorted(range(len(union)), key=lambda j: (-float(logits[j]), j))[:TOPK]
        return [union[j] for j in order], float(1 / (1 + np.exp(-float(logits[order[0]]))))

    dv = [(r.query_id, *retrieve(str(r.query_text))) for r in q[q.split == "dev"].itertuples()]
    thr, _ = pick_threshold([s for _, _, s in dv], [any((x, i) in rel for i in ids) for x, ids, _ in dv])
    assert round(thr, 4) == G["insufficient_evidence_threshold"], "dev threshold differs from generate.py"
    nlp("warm up")
    out = TASK / "out"
    out.mkdir(exist_ok=True)
    f = out / "answers.jsonl"
    f.write_text("")
    for n, r in enumerate(q.itertuples(), 1):
        if a.limit and n > a.limit:
            break
        qt = str(r.query_text)
        t0 = time.perf_counter()
        ids, top1 = retrieve(qt)
        t1 = time.perf_counter()
        nlp(qt)
        sim = cosine_similarity(vec.transform([qt]), Xdoc)[0]
        cand = list(dict.fromkeys([str(slots.filled.get(r.query_id, ""))] +
                                  [str(cm.Phraseology.iat[j]) for j in np.argsort(-sim)[:N_PHRASES]]))
        cand = [c for c in cand if c and c != "nan"]
        t2 = time.perf_counter()
        exp = runs[runs.query_id == r.query_id].sort_values("rank").incident_id.astype(str).tolist()[:TOPK]
        assert ids == exp, f"{r.query_id}: ce top-5 differs from Task 3 run"
        top = meta.iloc[posof[ids[0]]]
        risk = rag.compute_risk_score(top.intent, top.phase_final, top.hazard_tags)
        rec = {"query_id": r.query_id, "split": r.split, "query_text": qt, "retrieved": ids,
               "top1_score": round(top1, 4), "threshold": round(thr, 4), "risk": risk, "model": "none (extractive)",
               "hazard_tags_top1": str(top.hazard_tags)}
        if top1 < thr:
            rec.update(status="insufficient_evidence", answer={"summary": INSUFFICIENT, "key_hazards": [],
                                                               "recommended_phraseology": [], "citations": []})
        else:
            sents, cites = [], []
            for i in ids[:N_SENT]:
                ss = [s.strip() for s in SPLIT.split(text_of[i]) if s.strip()]
                if not ss:
                    continue
                k = int(np.argmax(cosine_similarity(vec.transform([qt]), vec.transform(ss))[0]))
                sents.append(ss[k])
                cites.append(i)
            rec.update(status="ok", answer={"summary": " ".join(sents), "key_hazards": [],
                                            "recommended_phraseology": cand[:1], "citations": cites})
        t3_ = time.perf_counter()
        ms = lambda x, y: round((y - x) * 1000, 1)
        rec.update(retrieval_ms=ms(t0, t1), ner_phrase_ms=ms(t1, t2), extract_ms=ms(t2, t3_), latency_ms=ms(t0, t3_))
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{n}/{len(q)}] {r.query_id} {rec['status']} {rec['latency_ms']:.0f} ms", flush=True)
    A = [json.loads(x) for x in f.read_text().splitlines()]
    lat = [x["latency_ms"] for x in A]
    R = {"model": "none (extractive)", "queries": len(A), "splits": ["dev", "test"],
         "insufficient_evidence_threshold": round(thr, 4),
         "threshold_rule": "generate.py pick_threshold on dev (exact value, not the rounded one)",
         "status_counts": pd.Series([x["status"] for x in A]).value_counts().to_dict(),
         "sentences_per_answer": N_SENT, "citation_validity": 1.0, "citations_rejected": 0,
         "citations_raw": sum(len(x["answer"]["citations"]) for x in A), "phraseology_raw": 0, "phraseology_rejected": 0,
         "llm_latency_ms_p50": None, "llm_latency_ms_p95": None, "ttft_ms_p50": None,
         "end_to_end_ms_p50": round(float(np.percentile(lat, 50)), 1),
         "end_to_end_ms_p95": round(float(np.percentile(lat, 95)), 1),
         "end_to_end_ms_max": round(float(max(lat)), 1),
         "retrieval_ms_p50": round(float(np.percentile([x["retrieval_ms"] for x in A], 50)), 1),
         "retrieval_ms_p95": round(float(np.percentile([x["retrieval_ms"] for x in A], 95)), 1),
         "answers_sha256": sha256(f)}
    (out / "results_generation.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["status_counts", "end_to_end_ms_p50", "end_to_end_ms_p95"]}))


if __name__ == "__main__":
    main()
