#!/usr/bin/env python3
"""Task 8: full evaluation. Optionally reruns every task, measures end-to-end latency, then builds
phase1_vs_phase2.md, LaTeX booktabs tables and 300-DPI figures from the tasks' results.json files only.

Run:  python phase2/integration/run_full_eval.py --phase1 ../uatie-edm-phase1 [--rerun] [--latency]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
P2 = ROOT / "phase2"
sys.path.insert(0, str(ROOT))
OUT = TASK / "out"
STEPS = ["retrieval/evaluate.py", "intent_classifier/train.py", "ner/train.py", "phraseology/extract.py",
         "phraseology/crossmap.py", "phraseology/ontology.py", "phraseology/report.py", "slots/fill.py",
         "generation/generate.py", "generation/judge.py"]
SYS = ["hybrid", "bm25", "faiss", "rrf", "ce", "ce_recency"]
MET = ["p1", "p5", "p10", "mrr", "map10", "recall10", "ndcg10"]


def J(rel):
    p = P2 / rel
    return json.loads(p.read_text()) if p.exists() else {}


def tex(df, caption, label):
    cols = "l" + "r" * (df.shape[1] - 1)
    rows = [" & ".join(str(v) for v in r) + r" \\" for r in df.astype(str).values]
    return "\n".join([r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(df.columns) + r" \\", r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]).replace("_", r"\_")


def latency(phase1):
    """Non-LLM stages measured here, per query; LLM time per query from generation/out/answers.jsonl."""
    sys.path.insert(0, str(P2 / "generation"))
    sys.path.insert(0, str(P2 / "slots"))
    from generate import load_module
    from fill import fill
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from phase1 import phase1_root
    repo, cwd = phase1_root(), Path.cwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    rag = load_module("rag_query_engine", repo / "src" / "rag" / "rag_query_engine.py")
    rag.initialize_resources(verbose=False)
    os.chdir(cwd)
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["summary"])
    cm = pd.read_csv(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv")
    doc = cm.Phraseology.astype(str) + " " + cm.Circumstances.fillna("").astype(str)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(doc)
    X = vec.transform(doc)
    nlp = spacy.load(P2 / "ner" / "out" / "model")
    q = pd.concat([pd.read_csv(P2 / "gold_eval_rebuild" / "splits" / f"{s}_queries.csv") for s in ["dev", "test"]])
    llm = {}
    f = P2 / "generation" / "out" / "answers.jsonl"
    if f.exists():
        llm = {a["query_id"]: a["latency_ms"] for a in map(json.loads, f.read_text().splitlines())}
    rag.index.search(rag.embed_query("warm up"), 5)
    rows = []
    for r in q.itertuples():
        t0 = time.perf_counter()
        _, idx = rag.index.search(rag.embed_query(r.query_text), 5)
        ents = [(e.text, e.label_) for e in nlp(r.query_text + "\n" + str(meta.summary.iat[idx[0][0]])).ents]
        j = int((X @ vec.transform([r.query_text]).T).toarray().ravel().argmax())
        fill(str(cm.Phraseology.iat[j]), ents)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"query_id": r.query_id, "pipeline_ms": round(ms, 1), "llm_ms": llm.get(r.query_id),
                     "end_to_end_ms": round(ms + llm[r.query_id], 1) if r.query_id in llm else None})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "latency_per_query.csv", index=False)
    pct = lambda s: {"p50": round(float(np.percentile(s, 50)), 1), "p95": round(float(np.percentile(s, 95)), 1),
                     "n": int(len(s))}
    return {k: pct(d[k].dropna()) for k in ["pipeline_ms", "llm_ms", "end_to_end_ms"] if d[k].notna().any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--rerun", action="store_true", help="rerun every task script first (hours on CPU)")
    ap.add_argument("--latency", action="store_true", help="measure end-to-end latency over the 100 queries")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    extra = ["--phase1", a.phase1] if a.phase1 else []
    if a.rerun:
        for s in STEPS:
            print("running", s, flush=True)
            subprocess.run([sys.executable, str(P2 / s)] + ([] if "extract" in s or "report" in s else extra),
                           check=True)
    ret, ner, ph = J("retrieval/eval/results.json"), J("ner/results.json"), J("phraseology/results.json")
    it, sl, gen = J("intent_classifier/results.json"), J("slots/results.json"), J("generation/results.json")
    R = {"latency": latency(a.phase1) if a.latency else J("integration/results.json").get("latency")}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    md = ["# Phase 1 vs Phase 2", "", "Every number below is read from a task's `results.json`. Labels marked "
          "provisional are LLM-judged (Devin), pending user review.", ""]
    if ret:
        m = ret["metrics"]["test_rel_ge1"]
        t = pd.DataFrame([[s] + [f"{m[s][k]:.3f}" for k in MET] for s in SYS], columns=["System"] + MET)
        (OUT / "tables" / "retrieval_test.tex").write_text(tex(t, "Retrieval, test Q021--Q100, rel $\\geq$ 1 "
                                                             "(provisional labels)", "tab:retrieval"))
        md += ["## Retrieval (test, rel >= 1, provisional)", "", t.to_markdown(index=False), "",
               f"Dev-selected system: `{ret['best_system_chosen_on_dev_ndcg10']}`. Phase 1 = `hybrid`.", ""]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        w = 0.12
        for i, s in enumerate(SYS):
            ax.bar(np.arange(3) + i * w, [m[s][k] for k in ["p5", "map10", "ndcg10"]], w, label=s)
        ax.set_xticks(np.arange(3) + 2.5 * w, ["P@5", "MAP@10", "nDCG@10"])
        ax.set_ylim(0, 1)
        ax.legend(ncol=3, fontsize=7)
        ax.set_title("Retrieval, test (provisional labels)")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "retrieval_test.png", dpi=300)
        plt.close(fig)
    if ner:
        labs = [k for k in ner["spacy_ner"] if k != "ALL"] + ["ALL"]
        t = pd.DataFrame([[k, ner["spacy_ner"][k]["support"], f"{ner['rules_baseline'][k]['f1']:.3f}",
                           f"{ner['spacy_ner'][k]['f1']:.3f}"] for k in labs],
                         columns=["Entity", "Support", "Rules F1", "spaCy F1"])
        (OUT / "tables" / "ner.tex").write_text(tex(t, "NER, 60 held-out sentences (provisional labels)", "tab:ner"))
        md += ["## NER (Phase 1 = rules, Phase 2 = spaCy; provisional)", "", t.to_markdown(index=False), ""]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(labs))
        ax.bar(x - 0.2, [ner["rules_baseline"][k]["f1"] for k in labs], 0.4, label="rules")
        ax.bar(x + 0.2, [ner["spacy_ner"][k]["f1"] for k in labs], 0.4, label="spaCy")
        ax.set_xticks(x, labs, rotation=30, fontsize=7)
        ax.legend()
        ax.set_title("NER F1 (provisional labels)")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "ner_f1.png", dpi=300)
        plt.close(fig)
    if ph:
        c = ph["crossmap"]
        tiers = ["No Match", "Partial Match", "Close Match", "Exact/Near-Exact"]
        t = pd.DataFrame([[k, c["before_phase1_tfidf"][k], c["after_tfidf"][k], c["after_embedding"][k]]
                          for k in tiers], columns=["Tier", "Phase 1", "Phase 2 TF-IDF", "Phase 2 embedding"])
        (OUT / "tables" / "crossmap.tex").write_text(tex(t, "CAA--ICAO crossmap tiers (514 CAA phrases)",
                                                          "tab:crossmap"))
        o = ph["ontology"]
        md += ["## Phraseology crossmap", "", t.to_markdown(index=False), "",
               f"CAA-unique %: Phase 1 {c['caa_unique_pct']['before_phase1']}, Phase 2 TF-IDF {c['caa_unique_pct']['after_tfidf']}, embedding {c['caa_unique_pct']['after_embedding']}. Ontology RDF triples: Phase 1 {o['phase1_repro']['triples']}, "
               f"Phase 2 {o['phase2']['triples']}.", ""]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(4)
        for i, (k, lab) in enumerate([("before_phase1_tfidf", "Phase 1"), ("after_tfidf", "Phase 2 TF-IDF"),
                                      ("after_embedding", "Phase 2 embedding")]):
            ax.bar(x + (i - 1) * 0.27, [c[k][t_] for t_ in tiers], 0.27, label=lab)
        ax.set_xticks(x, tiers, fontsize=8)
        ax.legend(fontsize=7)
        ax.set_title("CAA phrases per match tier")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "crossmap_tiers.png", dpi=300)
        plt.close(fig)
    if it:
        rows = []
        for name, v in it.get("models", {}).items():
            if isinstance(v, dict) and "test" in v:
                rows.append([name, f"{v['test'].get('accuracy', float('nan')):.3f}",
                             f"{v['test'].get('macro_f1', float('nan')):.3f}"])
        kb = it.get("keyword_rules_baseline", {}).get("test") or it.get("rules_baseline", {}).get("test")
        if kb:
            rows.insert(0, ["Phase 1 keyword rules", f"{kb.get('accuracy', float('nan')):.3f}",
                            f"{kb.get('macro_f1', float('nan')):.3f}"])
        t = pd.DataFrame(rows, columns=["Model", "Test accuracy", "Test macro-F1"])
        (OUT / "tables" / "intent.tex").write_text(tex(t, "Intent, 201 test records (provisional labels)",
                                                        "tab:intent"))
        md += ["## Intent (agreement with provisional labels)", "", t.to_markdown(index=False), ""]
    if sl:
        md += ["## Slot filling", "", f"Status over {sl['queries']} queries: {sl['status_counts']}; "
               f"slots filled {sl['slots_filled']} of {sl['slot_occurrences_in_selected_phrases']}.", ""]
    if gen:
        g = gen["generation"]
        md += ["## Generation", "", f"Model `{g['model']}`; status {g['status_counts']}; citation validity "
               f"{g['citation_validity']}; hallucination rate (LLM judge) {gen['hallucination_rate_claims']}.", ""]
    if R["latency"]:
        md += ["## Latency (this VM, CPU)", "", "```", json.dumps(R["latency"], indent=1), "```", "",
               "- `pipeline_ms`: query embedding + FAISS top-5, spaCy NER on query + top-1 summary, TF-IDF phrase match and "
               "slot filling, measured here (cross-encoder and intent lookup not included).",
               "- `llm_ms`: per-query LLM time recorded by `generation/generate.py`; it is 0 for "
               f"{gen['generation']['status_counts'].get('insufficient_evidence', 0)} insufficient-evidence queries "
               "(no LLM call), and its tail includes queries run while other CPU jobs shared this VM "
               "(see `generation/REPORT.md`).", ""]
    (TASK / "phase1_vs_phase2.md").write_text("\n".join(md), encoding="utf-8")
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print("written:", sorted(p.name for p in (OUT / "tables").iterdir()), sorted(p.name for p in (OUT / "figures").iterdir()))


if __name__ == "__main__":
    main()
