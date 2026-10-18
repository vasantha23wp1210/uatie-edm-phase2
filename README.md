# uatie-edm-phase2

Phase 2 of the UATIE-EDM aviation safety retrieval project (M.Tech). Each task lives in its own folder and writes
`results.json` and `REPORT.md`. Python 3.11, CPU only, seed 42.

Phase 2 reads Phase 1 code and data from a separate, unmodified checkout of
[`uatie-edm-phase1`](https://github.com/vasantha23wp1210/uatie-edm-phase1). Point at it with `--phase1 <path>` or the
`UATIE_PHASE1` environment variable; by default a sibling folder `../uatie-edm-phase1` is used. The Phase 1 commit and
SHA-256 of every Phase 1 file used are recorded under `phase1` in each `results.json` (`phase1.py`).

| Task | What | Status |
|---|---|---|
| [`phase2/gold_eval_rebuild`](phase2/gold_eval_rebuild/README.md) | Unbiased pool, blind judgements, final bm25 / faiss / hybrid / RRF metrics | Done; test labels LLM-judged |
| [`phase2/retrieval`](phase2/retrieval/README.md) | Task 3: baseline reproduction + six retrieval systems | Done, provisional labels |
| [`phase2/intent_classifier`](phase2/intent_classifier/README.md) | Task 1: intent classifier, `intent_v2`, ontology population | Done, provisional labels; accuracy target missed, Unclassified < 15% met |
| [`phase2/ner`](phase2/ner/README.md) | Task 2: aviation NER | Done, provisional corrections |
| [`phase2/phraseology`](phase2/phraseology/README.md) | Tasks 4-5: ICAO extraction, crossmap, ontology | Done; 20 sampled matches verified by the user |
| [`phase2/slots`](phase2/slots/README.md) | Task 6: slot filling | Done |
| [`phase2/generation`](phase2/generation/README.md) | Task 7: grounded generation (LangChain + Ollama `llama3.2:3b`) | Done; 30 answers of the earlier run rated by the user |
| [`phase2/app`](phase2/app/README.md) | Task 8: Streamlit Phase 1 / Phase 2 app | Done |
| [`phase2/integration`](phase2/integration/README.md) | Task 8: full eval, Phase 1 vs Phase 2, tables, figures, screenshots | Done |

Every metric that depends on labels marked `LLM-judged (Claude), pending user review` is provisional. The labelling method and the agreement with Devin's earlier provisional labels are in [`phase2/labels/`](phase2/labels/REPORT.md). Files for your
review: `phase2/retrieval/judgements/judged.csv`, `phase2/intent_classifier/judgements/intent_labelled.csv`,
`phase2/ner/judgements/ner_corrected.csv`. User-reviewed (notes refined with Claude):
`phase2/phraseology/verify/matches_verified.csv` (20 match verdicts) and
`phase2/generation/rate/answers_rated_run_prev.csv` (30 answer ratings of the earlier generation run in
`phase2/generation/out/run_prev/`). Claims that could not be verified are listed in
[`phase2/integration/unverifiable_claims.md`](phase2/integration/unverifiable_claims.md).

## Setup (Windows PowerShell)

```powershell
git -C ..\uatie-edm-phase1 lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python phase2\gold_eval_rebuild\run.py --phase1 ..\uatie-edm-phase1
python phase2\gold_eval_rebuild\finalize.py --phase1 ..\uatie-edm-phase1
```

`.gitattributes` (`* -text`) keeps every file byte-exact on checkout, so recorded SHA-256 hashes match on Windows.
