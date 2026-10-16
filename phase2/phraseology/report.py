#!/usr/bin/env python3
"""Tasks 4-5: merge results_crossmap.json and results_ontology.json into results.json and write REPORT.md."""
import json
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parent
TIERS = ["No Match", "Partial Match", "Close Match", "Exact/Near-Exact"]


def main():
    c = json.loads((TASK / "results_crossmap.json").read_text())
    o = json.loads((TASK / "results_ontology.json").read_text())
    src = TASK / "out" / "sources"
    a, b = pd.read_csv(src / "doc4444_ch12.csv"), pd.read_csv(src / "doc9432_dialogues.csv")
    ext = {"doc4444_ch12": {"rows": len(a), "speaker": a.speaker.value_counts().to_dict(),
                            "emergency": int(a.is_emergency.sum()), "gnss": int(a.is_gnss.sum()),
                            "sections": int(a.section.nunique())},
           "doc9432": {"rows": len(b), "speaker": b.speaker.value_counts().to_dict(),
                       "emergency": int(b.is_emergency.sum()), "gnss": int(b.is_gnss.sum())}}
    v = pd.read_csv(TASK / "verify" / "matches_to_verify.csv", dtype=str, keep_default_na=False)
    reviewed = int((v.your_verdict.str.strip() != "").sum())
    R = {"extraction": ext, "crossmap": c, "ontology": o,
         "verification_sample": {"rows": len(v), "reviewed": reviewed,
                                 "status": "pending user review" if reviewed < len(v) else "reviewed"}}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")

    bt, at, ae = c["before_phase1_tfidf"], c["after_tfidf"], c["after_embedding"]
    n = c["caa_rows"]
    p1, p2, rep = o["phase1_committed_ttl"], o["phase2"], o["phase1_repro"]
    flat = {k: int(open(TASK / "out" / "ontology" / k / "phraseology_triples.csv").read().count("\n")) - 1
            for k in ["phase1_repro", "phase2"]}
    L = ["# Tasks 4-5: phraseology crossmap and ontology", "",
         "Sources: only the three PDFs in `data/phraseology_sources/` (SHA-256 in results.json). No downloads.", "",
         "## Method", "",
         "1. `extract.py`: Doc 4444 Ch. 12 (12.3-12.7) with pdfplumber column positions (item letters at x~264, "
         "phrases at x~282); markers * = pilot, † = aircraft/tow vehicle, ‡ = needs pilot acknowledgement. "
         "Doc 9432 dialogues: uppercase lines grouped by speaker, ATC if x0 < 303 pt else pilot.",
         "2. `crossmap.py`: Phase 1 `caa_icao_crossmap.py` method unchanged (clean_phrase, TfidfVectorizer "
         "ngram (1,2), min_df 1, cosine, tiers <=0.30 / 0.30-0.60 / 0.60-0.85 / >0.85) against the new ICAO corpus. "
         f"Embedding matching ({c['embedding_model']}) reported separately with the same bins.",
         "3. `ontology.py`: runs Phase 1 `build_ontology_triples.py` unmodified except PROCESSED_DIR, redirected in "
         "memory to `out/ontology/<run>/`.", "",
         "## Extraction", "", "| Source | Rows | Speakers | Emergency | GNSS/RNAV |", "|---|---:|---|---:|---:|",
         f"| Doc 4444 Ch. 12 | {ext['doc4444_ch12']['rows']} | {ext['doc4444_ch12']['speaker']} | "
         f"{ext['doc4444_ch12']['emergency']} | {ext['doc4444_ch12']['gnss']} |",
         f"| Doc 9432 | {ext['doc9432']['rows']} | {ext['doc9432']['speaker']} | {ext['doc9432']['emergency']} | "
         f"{ext['doc9432']['gnss']} |", "",
         f"ICAO matching corpus after de-duplication: {c['icao_corpus_rows']} phrases {c['icao_corpus_by_source']}.", "",
         f"## CAA tiers ({n} CAA phrases)", "",
         "| Tier | Before (Phase 1 TF-IDF) | After (TF-IDF) | After (embedding) |", "|---|---:|---:|---:|"]
    L += [f"| {t} | {bt[t]} | {at[t]} | {ae[t]} |" for t in TIERS]
    u = c["caa_unique_pct"]
    L += ["", f"CAA-unique (No Match) %: before {u['before_phase1']}, after TF-IDF {u['after_tfidf']}, "
          f"after embedding {u['after_embedding']}.", "",
          f"Emergency: {c['emergency']['caa_rows']} CAA rows match the emergency regex, "
          f"{c['emergency']['caa_rows_matched_tfidf_gt_0.30']} of them now match ICAO above 0.30; ICAO emergency "
          f"rows {c['emergency']['icao_rows_by_source']}.",
          f"GNSS/RNAV: {c['gnss']['caa_rows']} CAA rows, {c['gnss']['caa_rows_matched_tfidf_gt_0.30']} matched above "
          f"0.30; ICAO GNSS rows {c['gnss']['icao_rows_by_source']}.", "",
          "## Ontology", "", "| | Phase 1 committed TTL | Phase 1 rerun | Phase 2 |", "|---|---:|---:|---:|",
          f"| RDF triples | {p1['triples']} | {rep['triples']} | {p2['triples']} |",
          f"| Flat triples (CSV rows) | | {flat['phase1_repro']} | {flat['phase2']} |",
          f"| CAA phrase individuals | | {rep['caa_phrase_individuals']} | {p2['caa_phrase_individuals']} |",
          f"| ICAO phrase individuals | | {rep['icao_phrase_individuals']} | {p2['icao_phrase_individuals']} |", "",
          f"The Phase 1 rerun is isomorphic to the committed TTL: {o['phase1_repro_isomorphic_to_committed']}.",
          "- The paper's 4,880 'triples' is the flat CSV row count; the RDF graph has "
          f"{rep['triples']} triples.",
          "- The paper's 703 'individuals' are not phraseology individuals. They are the 703 incident "
          "`owl:NamedIndividual`s in Phase 1's committed `src/ontology/uatie_fused.owl` (2,179 triples, counted with "
          "rdflib), built by `src/ontology/fuse_datasets.py` from <=2,000 rows per raw CSV with a keyword filter "
          "(Phase 1 README l. 120-133). `src/rag/ontology_population.py` instead creates 113,468 IncidentReport "
          "individuals (see `phase2/intent_classifier/REPORT.md`). Not rebuilt here.", "",
          "## Failures and limits", "",
          "- Phase 1's ICAO inputs (`icao_phraseology_auto.csv`, `icao_phraseology_master.csv`, "
          "`doc4444_12_3_ATC_phraseologies.csv`) are not in the Phase 1 checkout, so the 'before' corpus cannot be "
          "inspected; the before column is Phase 1's committed `CAA_ICAO_crossmap.csv`.",
          "- CAA Section 8 reproduces ICAO wording, so most CAA phrases now have a verbatim ICAO match; the drop in "
          "CAA-unique % mostly reflects the larger ICAO corpus, not new content.",
          "- Embedding scores use the TF-IDF bins, which were not calibrated for embeddings; every phrase scores "
          "above 0.30, so the embedding tiers are not comparable to TF-IDF.",
          "- Doc 9432 has no GNSS/RNAV dialogue (one RNAV mention, in the abbreviations list).",
          "- Extraction is heuristic: Doc 4444 sub-items can merge when the layout differs; Doc 9432 utterances split "
          "across page breaks may be cut. Not hand-checked line by line.",
          f"- Verification sample `verify/matches_to_verify.csv` ({len(v)} rows, 5 per tier, seed 42): "
          f"{reviewed} reviewed; {R['verification_sample']['status']}.", "",
          "## Reproduce", "", "```", "python phase2/phraseology/extract.py",
          "python phase2/phraseology/crossmap.py --phase1 ../uatie-edm-phase1",
          "python phase2/phraseology/ontology.py --phase1 ../uatie-edm-phase1",
          "python phase2/phraseology/report.py", "```"]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("REPORT.md written;", len(L), "lines")


if __name__ == "__main__":
    main()
