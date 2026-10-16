# Task 8: full evaluation

Reproduce (one command): `python phase2/integration/run_full_eval.py --phase1 ../uatie-edm-phase1 --latency`
Add `--rerun` to rerun every task first (several hours on CPU).

Outputs: `phase1_vs_phase2.md`, `results.json`, `out/tables/*.tex` (booktabs), `out/figures/*.png` (300 DPI), `out/latency_per_query.csv`.

Demo screenshots (`out/screenshots/`, Streamlit app on this VM, `llama3.2:3b`, captured with Playwright from the running app):

| File | Mode | Query |
|---|---|---|
| `01_phase2_sidebar.png` | Phase 2, `ce` (primary) | (startup) |
| `02_phase2_ce_answer.png` | Phase 2, `ce` (primary) | pilot did not read back runway assignment correctly |
| `03_phase2_ner_slots.png` | Phase 2, `ce` (primary) | B737 cleared to descend to FL240 then told to contact approach on 124.5 and expect runway 27L at KSFO |
| `04_phase2_faiss_comparison.png` | Phase 2, `faiss` (dev-selected comparison) | aircraft entered runway without clearance during taxi |
| `05_phase1_mode.png` | Phase 1 | aircraft entered runway without clearance during taxi |

In `03`, spaCy NER does not tag `FL240` in the query, so the `(level)` slot is filled with `fl050` from the top-1 incident summary. That value describes the retrieved incident, not the user's situation; see `slots/REPORT.md` (Failures and limits).
