#!/usr/bin/env python3
"""Task 4, step 2: rebuild the CAA -> ICAO crossmap against the expanded ICAO corpus.

TF-IDF matching repeats Phase 1 src/phraseology/caa_icao_crossmap.py exactly (clean_phrase, TfidfVectorizer
ngram (1, 2), min_df 1, cosine, bins -0.01/0.30/0.60/0.85/1.01). Embedding matching (all-MiniLM-L6-v2) is
reported separately; it reuses the TF-IDF bins, which were not calibrated for embeddings.
Inputs:  Phase 1 data/processed/CAA_ATC_section8_enriched.csv and CAA_ICAO_crossmap.csv (the "before" table);
         out/sources/doc4444_ch12.csv and doc9432_dialogues.csv from extract.py
Writes:  out/CAA_ICAO_crossmap_v2.csv, out/phase1_compat/CAA_ICAO_crossmap.csv, verify/matches_to_verify.csv,
         results_crossmap.json
Run:  python phase2/phraseology/crossmap.py [--phase1 PATH]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK))
from phase1 import phase1_root, sha256  # noqa: E402
from extract import EMERGENCY, GNSS  # noqa: E402

SEED = 42
BINS = [-0.01, 0.30, 0.60, 0.85, 1.01]
TIERS = ["No Match", "Partial Match", "Close Match", "Exact/Near-Exact"]
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
P1_COLS = ["Section_Number", "Section_Title", "ATC_Service", "Circumstances", "Phraseology", "Speaker",
           "Variable_Count", "Best_ICAO_Match", "Match_Score", "Match_Type"]


def clean_phrase(text):
    text = re.sub(r"\(.*?\)", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.upper().strip()


def icao_corpus():
    a = pd.read_csv(TASK / "out" / "sources" / "doc4444_ch12.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(TASK / "out" / "sources" / "doc9432_dialogues.csv", dtype=str, keep_default_na=False)
    a = a.assign(category=a.section_title)
    b = b.assign(category=b.chapter)
    cols = ["source", "section", "category", "speaker", "phrase"]
    c = pd.concat([a[cols], b[cols]], ignore_index=True).rename(columns={"phrase": "phraseology"})
    c["phraseology_clean"] = c.phraseology.map(clean_phrase)
    c = c[c.phraseology_clean.str.len() > 0].drop_duplicates(["source", "phraseology_clean"]).reset_index(drop=True)
    return c


def tier(s):
    return pd.cut(s, bins=BINS, labels=TIERS).astype(str)


def counts(s):
    v = s.value_counts()
    return {t: int(v.get(t, 0)) for t in TIERS}


def main():
    p1 = phase1_root()
    proc = p1 / "data" / "processed"
    caa = pd.read_csv(proc / "CAA_ATC_section8_enriched.csv")
    before = pd.read_csv(proc / "CAA_ICAO_crossmap.csv")
    icao = icao_corpus()
    caa["phraseology_clean"] = caa.Phraseology.map(clean_phrase)

    corpus = list(icao.phraseology_clean) + list(caa.phraseology_clean)
    m = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(corpus)
    sim = cosine_similarity(m[len(icao):], m[:len(icao)])
    best = sim.argmax(axis=1)
    caa["Best_ICAO_Match"] = icao.phraseology.iloc[best].values
    caa["Match_Score"] = sim.max(axis=1).round(4)
    caa["Match_Type"] = tier(caa.Match_Score)
    caa["Best_ICAO_Source"] = icao.source.iloc[best].values
    caa["Best_ICAO_Section"] = icao.section.iloc[best].values

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_MODEL, device="cpu")
    ei = st.encode(list(icao.phraseology_clean), normalize_embeddings=True, batch_size=64)
    ec = st.encode(list(caa.phraseology_clean), normalize_embeddings=True, batch_size=64)
    es = ec @ ei.T
    eb = es.argmax(axis=1)
    caa["Emb_Best_ICAO_Match"] = icao.phraseology.iloc[eb].values
    caa["Emb_Score"] = es.max(axis=1).round(4)
    caa["Emb_Match_Type"] = tier(caa.Emb_Score)
    caa["CAA_is_emergency"] = (caa.Circumstances.astype(str) + " " + caa.Phraseology.astype(str)).str.contains(EMERGENCY)
    caa["CAA_is_gnss"] = (caa.Circumstances.astype(str) + " " + caa.Phraseology.astype(str)).str.contains(GNSS)

    out = TASK / "out"
    (out / "phase1_compat").mkdir(parents=True, exist_ok=True)
    caa.drop(columns="phraseology_clean").to_csv(out / "CAA_ICAO_crossmap_v2.csv", index=False)
    caa[P1_COLS].to_csv(out / "phase1_compat" / "CAA_ICAO_crossmap.csv", index=False)

    rng = np.random.RandomState(SEED)
    samp = pd.concat([g.loc[rng.choice(g.index, min(5, len(g)), replace=False)]
                      for _, g in caa.groupby("Match_Type", sort=True)])
    (TASK / "verify").mkdir(exist_ok=True)
    pd.DataFrame({"sample_id": [f"M{i + 1:02d}" for i in range(len(samp))],
                  "Section_Number": samp.Section_Number.values, "CAA_Phraseology": samp.Phraseology.values,
                  "Best_ICAO_Match": samp.Best_ICAO_Match.values, "Best_ICAO_Source": samp.Best_ICAO_Source.values,
                  "Match_Score": samp.Match_Score.values, "Match_Type": samp.Match_Type.values,
                  "your_verdict": "", "notes": ""}).to_csv(TASK / "verify" / "matches_to_verify.csv", index=False)

    n = len(caa)
    icao_flags = pd.concat([pd.read_csv(TASK / "out" / "sources" / f, usecols=["source", "is_emergency", "is_gnss"])
                            for f in ["doc4444_ch12.csv", "doc9432_dialogues.csv"]])
    R = {"caa_rows": n, "icao_corpus_rows": len(icao),
         "icao_corpus_by_source": icao.source.value_counts().to_dict(),
         "before_phase1_tfidf": counts(before.Match_Type),
         "after_tfidf": counts(caa.Match_Type), "after_embedding": counts(caa.Emb_Match_Type),
         "caa_unique_pct": {"before_phase1": round(100 * (before.Match_Type == "No Match").mean(), 2),
                            "after_tfidf": round(100 * (caa.Match_Type == "No Match").mean(), 2),
                            "after_embedding": round(100 * (caa.Emb_Match_Type == "No Match").mean(), 2)},
         "emergency": {"caa_rows": int(caa.CAA_is_emergency.sum()),
                       "caa_rows_matched_tfidf_gt_0.30": int((caa.CAA_is_emergency & (caa.Match_Score > 0.30)).sum()),
                       "icao_rows_by_source": icao_flags[icao_flags.is_emergency].source.value_counts().to_dict()},
         "gnss": {"caa_rows": int(caa.CAA_is_gnss.sum()),
                  "caa_rows_matched_tfidf_gt_0.30": int((caa.CAA_is_gnss & (caa.Match_Score > 0.30)).sum()),
                  "icao_rows_by_source": icao_flags[icao_flags.is_gnss].source.value_counts().to_dict()},
         "tier_agreement_tfidf_vs_embedding": round(float((caa.Match_Type == caa.Emb_Match_Type).mean()), 4),
         "embedding_model": EMB_MODEL,
         "inputs_sha256": {**{f"phase1:data/processed/{f}": sha256(proc / f) for f in
                              ["CAA_ATC_section8_enriched.csv", "CAA_ICAO_crossmap.csv"]},
                           **{f"data/phraseology_sources/{f.name}": sha256(f) for f in
                              sorted((ROOT / "data" / "phraseology_sources").glob("*.pdf"))}},
         "output_sha256": {"phase2/phraseology/out/CAA_ICAO_crossmap_v2.csv": sha256(out / "CAA_ICAO_crossmap_v2.csv")}}
    (TASK / "results_crossmap.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["icao_corpus_by_source", "before_phase1_tfidf", "after_tfidf",
                                        "after_embedding", "caa_unique_pct", "emergency", "gnss"]}, indent=1))


if __name__ == "__main__":
    main()
