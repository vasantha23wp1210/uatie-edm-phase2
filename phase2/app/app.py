"""Task 8: Streamlit app with a Phase 1 / Phase 2 toggle. Phase 1 code is imported read-only.

Run:  streamlit run phase2/app/app.py -- --phase1 ../uatie-edm-phase1
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
P2 = ROOT / "phase2"
for p in [ROOT, P2 / "generation", P2 / "slots"]:
    sys.path.insert(0, str(p))
from phase1 import phase1_root  # noqa: E402
from generate import INSUFFICIENT, OPTIONS, PROMPT, ground, load_module  # noqa: E402
from fill import fill  # noqa: E402

TOPK = 5
CE_DEPTH = 50
N_SENT = 3
SPLIT = re.compile(r"(?<=[.!?])\s+")
st.set_page_config(page_title="UATIE-EDM Phase 1 / Phase 2", layout="wide")


@st.cache_resource
def backend():
    repo = phase1_root()
    cwd = Path.cwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    rag = load_module("rag_query_engine", repo / "src" / "rag" / "rag_query_engine.py")
    rag.initialize_resources(verbose=False)
    bm25_mod = load_module("bm25_baseline", repo / "src" / "rag" / "bm25_baseline.py")
    os.chdir(cwd)
    import spacy
    from sentence_transformers import CrossEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["incident_id"] = meta.incident_id.astype(str)
    bm25, _ = bm25_mod.build_bm25(meta, "summary")
    iv2 = pd.read_csv(P2 / "intent_classifier" / "out" / "intent_v2.csv").set_index("faiss_idx")
    cm = pd.read_csv(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv")
    doc = (cm.Phraseology.astype(str) + " " + cm.Circumstances.fillna("").astype(str) + " " +
           cm.Section_Title.fillna("").astype(str))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(doc)
    gq = pd.concat([pd.read_csv(P2 / "gold_eval_rebuild" / "splits" / f"{s}_queries.csv") for s in ["dev", "test"]])
    vec_x = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(list(doc) + list(gq.query_text))
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu",
                      revision="233902d25c440f23af6f7d6e94d2946bac0bee0a")
    G = json.loads((P2 / "generation" / "results_generation.json").read_text())
    G_faiss = json.loads((P2 / "generation" / "out" / "run_faiss_claude" / "results_generation.json").read_text())
    G["threshold_by_system"] = {"ce": G["insufficient_evidence_threshold"],
                                "faiss": G_faiss["insufficient_evidence_threshold"]}
    return {"rag": rag, "bm25_mod": bm25_mod, "bm25": bm25, "meta": meta, "iv2": iv2, "cm": cm, "vec": vec, "X": vec.transform(doc), "vec_x": vec_x, "ce": ce,
            "nlp": spacy.load(P2 / "ner" / "out" / "model"), "G": G}


def phase2_retrieve(B, q, system):
    rag, meta = B["rag"], B["meta"]
    sims, idx = rag.index.search(rag.embed_query(q), CE_DEPTH if system == "ce" else TOPK)
    pos, sc = [int(i) for i in idx[0] if i >= 0], [float(s) for s in sims[0]]
    if system != "ce":
        return pos[:TOPK], sc[:TOPK], sims[0][0]
    row = {i: n for n, i in enumerate(meta.incident_id)}
    b = [row[str(x["incident_id"])] for x in B["bm25_mod"].bm25_retrieve(B["bm25"], q, meta, "incident_id", CE_DEPTH)]
    pos = list(dict.fromkeys(b + pos))
    logits = B["ce"].predict([(q, str(meta.summary.iat[p])) for p in pos], batch_size=32, show_progress_bar=False)
    order = np.argsort(-logits, kind="stable")[:TOPK]
    pos, sc = [pos[j] for j in order], [float(logits[j]) for j in order]
    return pos, sc, 1 / (1 + np.exp(-sc[0]))


def answer(B, q, pos, top1, risk, phrases, model):
    from langchain_ollama import ChatOllama
    meta, G = B["meta"], B["G"]
    ids = [meta.incident_id.iat[p] for p in pos]
    if top1 < G["threshold_by_system"][system]:
        return {"summary": INSUFFICIENT, "key_hazards": [], "recommended_phraseology": [], "citations": []}, []
    llm = ChatOllama(model=model, base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                     format="json", keep_alive="2h", **OPTIONS)
    docs = "\n".join(f"[{i}] {meta.summary.iat[p]}" for i, p in zip(ids, pos))
    raw = llm.invoke(PROMPT.format(query=q, risk=risk, docs=docs, phrases="\n".join(f"- {c}" for c in phrases)))
    a = json.loads(raw.content)
    cites = [str(c) for c in a.get("citations", []) or []]
    allowed = set(B["cm"].Phraseology.astype(str)) | set(phrases)
    out = {"summary": str(a.get("summary", "")), "key_hazards": [str(h) for h in a.get("key_hazards", []) or []],
           "recommended_phraseology": [p for p in a.get("recommended_phraseology", []) or [] if p in allowed],
           "citations": [c for c in cites if c in ids]}
    out, _, _ = ground(out, ids, {i: str(meta.summary.iat[p]) for i, p in zip(ids, pos)})
    return out, [c for c in cites if c not in ids]


def extractive(B, q, pos):
    """Cockpit profile (phase2/edge/cockpit.py rule): best TF-IDF sentence of each top-3 report, verbatim."""
    from sklearn.metrics.pairwise import cosine_similarity
    meta, sents, cites = B["meta"], [], []
    for p in pos[:N_SENT]:
        ss = [x.strip() for x in SPLIT.split(str(meta.summary.iat[p])) if x.strip()]
        if ss:
            k = int(np.argmax(cosine_similarity(B["vec_x"].transform([q]), B["vec_x"].transform(ss))[0]))
            sents.append(ss[k])
            cites.append(meta.incident_id.iat[p])
    return {"summary": " ".join(sents), "key_hazards": [], "recommended_phraseology": [], "citations": cites}


B = backend()
st.sidebar.header("Mode")
phase = st.sidebar.radio("Pipeline", ["Phase 1", "Phase 2"], index=1)
system = st.sidebar.selectbox("Phase 2 retrieval", ["ce", "faiss"], index=0,
                              help="ce = primary system fixed by the Phase 2 plan: BM25 top-50 + FAISS top-50 re-ranked "
                                   "by the cross-encoder; faiss = system chosen on dev in Task 3 (comparison)") if phase == "Phase 2" else None
profile = st.sidebar.radio("Deployment profile", ["Ground (LLM, GPU recommended)", "Cockpit (CPU only, no LLM)"],
                           help="Ground: ATC / safety analysts, LLM-written answer. Cockpit: extractive answer copied "
                                "verbatim from the cited reports, verified crossmap phrase, CPU only.") if phase == "Phase 2" else None
cockpit = bool(profile and profile.startswith("Cockpit"))
if cockpit:
    system = "ce"
model = st.sidebar.text_input("Ollama model", os.environ.get("UATIE_LLM", "llama3.2:3b"))
gen = st.sidebar.checkbox("Generate answer (Ollama)", value=True) if phase == "Phase 2" and not cockpit else False
st.sidebar.caption("Phase 2 labels and metrics are provisional: LLM-judged (Claude), pending user review.")

st.title(f"UATIE-EDM: {phase}")
q = st.text_input("Query", "pilot did not read back runway assignment correctly")
if st.button("Search") and q.strip():
    t0 = time.perf_counter()
    rag, meta = B["rag"], B["meta"]
    if phase == "Phase 1":
        res = rag.hybrid_search(q, k=TOPK)
        st.caption(f"{(time.perf_counter() - t0) * 1000:.0f} ms")
        st.dataframe(pd.DataFrame(res), use_container_width=True)
    else:
        pos, sc, top1 = phase2_retrieve(B, q, system)
        t_ret = time.perf_counter()
        ids = [meta.incident_id.iat[p] for p in pos]
        top = meta.iloc[pos[0]]
        risk = rag.compute_risk_score(top.intent, top.phase_final, top.hazard_tags)
        text = q
        ents = [(e.text, e.label_) for e in B["nlp"](text).ents]
        j = int((B["X"] @ B["vec"].transform([q]).T).toarray().ravel().argmax())
        phrase = str(B["cm"].Phraseology.iat[j])
        filled, slots = fill(phrase, ents)
        t_nlp = time.perf_counter()
        c1, c2 = st.columns(2)
        c1.metric("Risk (top-1, Phase 1 matrix)", risk)
        c1.write("**Entities (NER)**: " + (", ".join(f"`{t}` {l}" for t, l in ents) or "none"))
        c2.write(f"**Matched phraseology** (CAA {B['cm'].Section_Number.iat[j]}): `{phrase}`")
        c2.write(f"**Filled**: `{filled}`")
        c2.caption("; ".join(f"({s['slot']}): {s['status']}" for s in slots) or "no slots")
        if cockpit:
            ok = float(top1) >= B["G"]["threshold_by_system"]["ce"]
            a = extractive(B, q, pos) if ok else {"summary": INSUFFICIENT, "citations": []}
            st.subheader("Answer (cockpit: extractive, verbatim from cited reports)")
            st.write(a["summary"])
            if ok:
                st.code(filled)
                st.write(f"**Phase 1 hazard tags (top-1):** `{top.hazard_tags}`")
            st.markdown("**Citations:** " + " ".join(f"[{c}](#{c.lower()})" for c in a["citations"]))
            t_ans = time.perf_counter()
            st.caption(f"retrieval {(t_ret - t0) * 1000:.0f} ms | NER + phrase {(t_nlp - t_ret) * 1000:.0f} ms | "
                       f"extractive {(t_ans - t_nlp) * 1000:.0f} ms")
        if gen:
            with st.spinner("Generating with Ollama..."):
                a, rejected = answer(B, q, pos, float(top1), risk, [filled, phrase], model)
            st.subheader("Answer")
            st.write(a["summary"])
            if a["key_hazards"]:
                st.write("**Key hazards:** " + "; ".join(a["key_hazards"]))
            for p in a["recommended_phraseology"]:
                st.code(p)
            st.markdown("**Citations:** " + " ".join(f"[{c}](#{c.lower()})" for c in a["citations"]))
            if rejected:
                st.warning(f"Rejected citations outside the retrieved set: {rejected}")
        st.caption(f"{(time.perf_counter() - t0) * 1000:.0f} ms end to end")
        st.subheader(f"Top-{TOPK} ({system})")
        for i, p, s in zip(ids, pos, sc):
            v2 = B["iv2"].loc[p] if p in B["iv2"].index else None
            st.markdown(f"<a id='{i.lower()}'></a>", unsafe_allow_html=True)
            st.markdown(f"**{i}** score {s:.3f} | intent_v2 `{v2.intent_v2 if v2 is not None else '-'}` "
                        f"({v2.intent_conf if v2 is not None else '-'}) | phase {meta.phase_final.iat[p]}")
            st.write(str(meta.summary.iat[p]))
