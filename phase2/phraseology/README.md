# Tasks 4-5: phraseology extraction, crossmap and ontology

Inputs: only the PDFs in `data/phraseology_sources/` (CAA ATC Section 8, ICAO Doc 4444, ICAO Doc 9432); Phase 1 checkout read-only.

The PDFs are not committed (ICAO documents are copyrighted; `.gitignore` excludes `data/phraseology_sources/*.pdf`). Place your copies there before running; their SHA-256 hashes are in `results_crossmap.json` under `inputs_sha256`, so a different edition shows up as a hash mismatch.

Reproduce (one command):

```
python phase2/phraseology/extract.py && python phase2/phraseology/crossmap.py --phase1 ../uatie-edm-phase1 && python phase2/phraseology/ontology.py --phase1 ../uatie-edm-phase1 && python phase2/phraseology/report.py
```

Outputs: `out/sources/*.csv`, `out/CAA_ICAO_crossmap_v2.csv`, `out/ontology/`, `verify/matches_to_verify.csv` (20 rows, pending user review), `results.json`, `REPORT.md`.
