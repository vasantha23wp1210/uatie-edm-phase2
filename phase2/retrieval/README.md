# phase2/retrieval (Task 3)

Status: step 1 (baseline reproduction) done; six systems run; **stopped** until `judgements/to_judge.csv` is judged.

```powershell
python phase2\retrieval\baseline.py --phase1 ..\uatie-edm-phase1   # baseline\REPORT.md, baseline\results.json
python phase2\retrieval\run.py --phase1 ..\uatie-edm-phase1        # runs\, judgements\to_judge.csv, REPORT.md, results.json
```

- `baseline.py`: fresh rerun of the Phase 1 HySOR engine scored with the paper's published protocol; checks P@5, MRR, MAP,
  nDCG@10 within ±0.01 of the paper.
- `run.py`: bm25, faiss, hybrid, rrf, cross-encoder re-rank, cross-encoder + recency on all 100 queries; top-10 runs,
  per-query latency, and a blind CSV of every retrieved pair without a judgement.
- Return the judged file as `judgements/judged.csv` (relevance 0/1/2). The metrics step reads it.
