# uatie-edm-phase2

Phase 2 of the UATIE-EDM aviation safety retrieval project (M.Tech). Each task lives in its own folder and writes
`results.json` and `REPORT.md`. Python 3.11, CPU only, seed 42.

Phase 2 reads Phase 1 code and data from a separate, unmodified checkout of
[`uatie-edm-phase1`](https://github.com/vasantha23wp1210/uatie-edm-phase1). Point at it with `--phase1 <path>` or the
`UATIE_PHASE1` environment variable; by default a sibling folder `../uatie-edm-phase1` is used. The Phase 1 commit and
SHA-256 of every Phase 1 file used are recorded under `phase1` in each `results.json` (`phase1.py`).

| Task | What | Status |
|---|---|---|
| [`phase2/retrieval`](phase2/retrieval/README.md) | Task 3: baseline reproduction + six retrieval systems | Baseline reproduced; waiting for judgements of 975 unjudged pairs |
| [`phase2/intent_classifier`](phase2/intent_classifier/README.md) | Task 1: intent classifier | Waiting for 400 labels |
| [`phase2/ner`](phase2/ner/README.md) | Task 2: aviation NER | Waiting for 260 corrected sentences |
| [`phase2/gold_eval_rebuild`](phase2/gold_eval_rebuild/README.md) | Unbiased 3-system pool, blind judgements, final metrics for bm25 / faiss / hybrid / HySOR RRF | Done; test labels are LLM-judged, see [`final/REPORT.md`](phase2/gold_eval_rebuild/final/REPORT.md) |

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
