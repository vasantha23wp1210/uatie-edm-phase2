# Tasks 4-5: phraseology crossmap and ontology

Sources: only the three PDFs in `data/phraseology_sources/` (SHA-256 in results.json). No downloads.

## Method

1. `extract.py`: Doc 4444 Ch. 12 (12.3-12.7) with pdfplumber column positions (item letters at x~264, phrases at x~282); markers * = pilot, † = aircraft/tow vehicle, ‡ = needs pilot acknowledgement. Doc 9432 dialogues: uppercase lines grouped by speaker, ATC if x0 < 303 pt else pilot.
2. `crossmap.py`: Phase 1 `caa_icao_crossmap.py` method unchanged (clean_phrase, TfidfVectorizer ngram (1,2), min_df 1, cosine, tiers <=0.30 / 0.30-0.60 / 0.60-0.85 / >0.85) against the new ICAO corpus. Embedding matching (sentence-transformers/all-MiniLM-L6-v2) reported separately with the same bins.
3. `ontology.py`: runs Phase 1 `build_ontology_triples.py` unmodified except PROCESSED_DIR, redirected in memory to `out/ontology/<run>/`.

## Extraction

| Source | Rows | Speakers | Emergency | GNSS/RNAV |
|---|---:|---|---:|---:|
| Doc 4444 Ch. 12 | 580 | {'ATC': 487, 'Pilot': 91, 'Aircraft/tow vehicle': 2} | 4 | 17 |
| Doc 9432 | 677 | {'ATC': 345, 'Pilot': 332} | 14 | 0 |

ICAO matching corpus after de-duplication: 1180 phrases {'ICAO Doc 9432 (Manual of Radiotelephony)': 627, 'ICAO Doc 4444 (PANS-ATM) Ch. 12': 553}.

## CAA tiers (514 CAA phrases)

| Tier | Before (Phase 1 TF-IDF) | After (TF-IDF) | After (embedding) |
|---|---:|---:|---:|
| No Match | 260 | 25 | 0 |
| Partial Match | 124 | 41 | 25 |
| Close Match | 65 | 47 | 63 |
| Exact/Near-Exact | 65 | 401 | 426 |

CAA-unique (No Match) %: before 50.58, after TF-IDF 4.86, after embedding 0.0.

Emergency: 3 CAA rows match the emergency regex, 3 of them now match ICAO above 0.30; ICAO emergency rows {'ICAO Doc 9432 (Manual of Radiotelephony)': 14, 'ICAO Doc 4444 (PANS-ATM) Ch. 12': 4}.
GNSS/RNAV: 21 CAA rows, 16 matched above 0.30; ICAO GNSS rows {'ICAO Doc 4444 (PANS-ATM) Ch. 12': 17}.

## Ontology

| | Phase 1 committed TTL | Phase 1 rerun | Phase 2 |
|---|---:|---:|---:|
| RDF triples | 7883 | 7883 | 8579 |
| Flat triples (CSV rows) | | 4880 | 5115 |
| CAA phrase individuals | | 514 | 514 |
| ICAO phrase individuals | | 196 | 431 |

The Phase 1 rerun is isomorphic to the committed TTL: True.
- The paper's 4,880 'triples' is the flat CSV row count; the RDF graph has 7883 triples.
- The paper's 703 'individuals' are not phraseology individuals. They are the 703 incident `owl:NamedIndividual`s in Phase 1's committed `src/ontology/uatie_fused.owl` (2,179 triples, counted with rdflib), built by `src/ontology/fuse_datasets.py` from <=2,000 rows per raw CSV with a keyword filter (Phase 1 README l. 120-133). `src/rag/ontology_population.py` instead creates 113,468 IncidentReport individuals (see `phase2/intent_classifier/REPORT.md`). Not rebuilt here.

## Failures and limits

- Phase 1's ICAO inputs (`icao_phraseology_auto.csv`, `icao_phraseology_master.csv`, `doc4444_12_3_ATC_phraseologies.csv`) are not in the Phase 1 checkout, so the 'before' corpus cannot be inspected; the before column is Phase 1's committed `CAA_ICAO_crossmap.csv`.
- CAA Section 8 reproduces ICAO wording, so most CAA phrases now have a verbatim ICAO match; the drop in CAA-unique % mostly reflects the larger ICAO corpus, not new content.
- Embedding scores use the TF-IDF bins, which were not calibrated for embeddings; every phrase scores above 0.30, so the embedding tiers are not comparable to TF-IDF.
- Doc 9432 has no GNSS/RNAV dialogue (one RNAV mention, in the abbreviations list).
- Extraction is heuristic: Doc 4444 sub-items can merge when the layout differs; Doc 9432 utterances split across page breaks may be cut. Not hand-checked line by line.
- Verification sample `verify/matches_to_verify.csv` (20 rows, 5 per tier, seed 42): 0 reviewed; pending user review.

## Reproduce

```
python phase2/phraseology/extract.py
python phase2/phraseology/crossmap.py --phase1 ../uatie-edm-phase1
python phase2/phraseology/ontology.py --phase1 ../uatie-edm-phase1
python phase2/phraseology/report.py
```
