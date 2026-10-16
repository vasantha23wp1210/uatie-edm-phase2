# Claims that cannot be verified, or are only partly verifiable, from the repositories

Each item says what was checked and why it cannot be fully verified.

## Phase 1 / paper

1. **Composition of the Phase 1 ICAO phraseology corpus.** `src/phraseology/caa_icao_crossmap.py` reads three ICAO CSVs that are not in the Phase 1 checkout. The 196 ICAO phrase individuals in the committed TTL can be counted, but how they were extracted cannot be checked.
2. **"Top-5" evaluation.** The paper describes top-5 retrieval, but the published metrics are computed on top-20 lists (reproduced within ±0.01 in `phase2/gold_eval_rebuild`). The reported numbers can be reproduced; the stated protocol cannot.
3. **100-query evaluation.** Only 77 of the 100 queries have a relevance ≥ 1 label, so 23 do not contribute to the published means.
4. **Unbiased relevance.** The Q021–Q100 labels used in Phase 1 were generated with MiniLM, the same encoder as the retriever. They cannot serve as independent evidence of retrieval quality.
5. **"4,880 triples".** This is the row count of the flat triples CSV; the RDF graph has 7,883 triples (graph-isomorphic reproduction, `phase2/phraseology/results_ontology.json`).
6. **"703 individuals".** Verified: Phase 1's committed `src/ontology/uatie_fused.owl` has 703 incident `owl:NamedIndividual`s (2,179 triples), built by `src/ontology/fuse_datasets.py` from <=2,000 rows per raw CSV plus a keyword filter. They are not phraseology individuals, and not the output of `src/rag/ontology_population.py`, which creates 113,468 IncidentReport individuals. The raw CSVs needed to rebuild `uatie_fused.owl` were not rerun here.
7. **Provenance of the CAA ATC Section 8 PDF.** It is byte-identical to the Phase 1 copy (SHA-256 `a5ea96a6...a2f5`). Its official edition and date are not recorded in either repository.

## Phase 2 (provisional until user review)

8. **All Phase 2 metrics that depend on labels** (retrieval qrels for new pairs, the 400 intent labels, the 260 NER corrections) are `LLM-judged (Devin), pending user review`. They are not human gold.
9. **gold_eval_rebuild test labels** are LLM-judged (Claude). The spot-check against the user's manual labels gave κ 0.34.
10. **CALLSIGN NER performance.** It cannot be evaluated: 0 test entities, because ASRS de-identifies callsigns.
11. **Generation hallucination rate.** It is estimated by an LLM judge, by default the same 3B model as the generator. It is not a human judgement.
12. **Answer quality ratings.** `phase2/generation/rate/answers_to_rate.csv` is pending user rating; no quality score is reported until it is returned.
13. **Phraseology match correctness.** The 20 sampled matches in `phase2/phraseology/verify/matches_to_verify.csv` are pending user verification.
14. **Latency.** All timings are from this CPU-only VM, often with other jobs running. They do not transfer to other hardware.
