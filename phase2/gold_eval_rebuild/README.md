# gold_eval_rebuild

Rebuilds the UATIE-EDM gold evaluation set so that it no longer has the Phase 1 problems:

* The Q021–Q100 labels in `evaluation/gold_relevance.csv` were produced by
  `src/rag/auto_label_relavance.py`, which thresholds cosine similarity from `all-MiniLM-L6-v2`. The FAISS
  index uses the same model, so those labels are circular. They are discarded here.
* In Phase 1, every hybrid result was labelled but most BM25 results were not, and unlabelled pairs counted as non-relevant.
  Here BM25, FAISS and the Phase 1 hybrid each contribute an equal top-10 to the pool.

Split (fixed and kept on disk in `splits/`): **dev = Q001–Q020** (manual Phase 1 labels),
**test = Q021–Q100** (judged in `judgements/test_judged.csv`; **LLM-judged**, see `final/REPORT.md` for provenance and agreement with the user's 100-pair blind spot-check).

Paths under `src/`, `evaluation/` and `data/` refer to the Phase 1 checkout (`uatie-edm-phase1`), which is read-only here.

## Reproduce (one command, run from the repo root)

```bash
python3.11 -m venv .venv && .venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r phase2/gold_eval_rebuild/requirements.txt && .venv/bin/python phase2/gold_eval_rebuild/run.py --phase1 ../uatie-edm-phase1
```

Inputs are the Phase 1 artefacts in that checkout (via Git LFS; run `git lfs pull` there if the files are pointers). Its commit and file hashes are recorded under `phase1` in `results.json`.
CPU only, seed 42, deterministic: two consecutive runs produce identical `output_sha256` values in `results.json`.

Step 2, after the judgements are back, writes the final metrics:

```bash
.venv/bin/python phase2/gold_eval_rebuild/finalize.py --phase1 ../uatie-edm-phase1
```

It checks `judgements/test_judged.csv` against the blind export (same rows and columns, relevance values only 0/1/2), writes `qrels/qrels_test.csv`, runs HySOR RRF (`src/hysor_rrf.py`, read-only), and scores bm25 / faiss / hybrid / rrf on test (Q021–Q100) and dev (Q001–Q020). It is deterministic: two runs give a byte-identical `final/results.json`.

## Outputs

| Path | What |
|---|---|
| `splits/dev_queries.csv`, `splits/test_queries.csv` | Query split |
| `splits/qrels_dev_phase1_manual.csv` | Phase 1 manual labels for Q001–Q020, unchanged |
| `judgements/test_to_judge.csv` | **Blind** test pool for you to label (`relevance` = 0/1/2) |
| `judgements/dev_gap_to_judge.csv` | Dev pool pairs that the manual Phase 1 labels don't cover |
| `pool/system_runs_top10.csv`, `pool/pool_provenance.csv` | Which system retrieved each pair. **Do not open while judging.** |
| `results.json`, `REPORT.md` | Numbers, method, caveats |
| `judgements/test_judged.csv` | Returned test judgements (LLM-judged, every row's `judge_notes` says so) |
| `judgements/spotcheck_100_human.csv` | User's own blind judgements of 100 test pairs (agreement check) |
| `qrels/qrels_test.csv` | Test qrels built from `test_judged.csv` |
| `pool/rrf_runs_top10.csv` | HySOR RRF top-10 per query |
| `final/results.json`, `final/REPORT.md`, `final/per_query_test_ge1.csv` | Final metrics, significance tests, per-bucket and per-query results |
| `llm_judge/` | Failed LLM-judge calibration (κ = 0.062 against manual dev labels); kept on record, not used |

## Judging guide

`relevance`: **2** = the narrative describes the event in the query or information need, **1** = partially or related
(right hazard but wrong phase, or similar), **0** = not relevant. Use `judge_notes` for anything unclear. Leave the other columns untouched.
