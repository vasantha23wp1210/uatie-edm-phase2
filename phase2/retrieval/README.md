# phase2/retrieval (Task 3)

Status: done on provisional labels. All 975 previously unjudged pairs are `LLM-judged (Devin), pending user review`.

```powershell
python phase2\retrieval\baseline.py --phase1 ..\uatie-edm-phase1   # baseline\REPORT.md, baseline\results.json (paper protocol)
python phase2\retrieval\run.py --phase1 ..\uatie-edm-phase1        # runs\, judgements\to_judge.csv, REPORT.md, results.json
python phase2\retrieval\llm_judge\combine.py         # judgements\judged.csv from llm_judge\labels\b*.csv
python phase2\retrieval\evaluate.py --phase1 ..\uatie-edm-phase1   # eval\REPORT.md, eval\results.json, per_query.csv, error_analysis.csv
```

- `run.py`: bm25, faiss, hybrid, rrf, cross-encoder re-rank, cross-encoder + recency on all 100 queries; top-10 runs.
- `evaluate.py`: metrics on dev and test, rel 2 and rel >= 1, per bucket, paired significance, latency; best system
  chosen on dev nDCG@10 only.

After your review, overwrite the judged file with your reviewed copy (same columns; set the notes column to `human` on rows you checked) and rerun only the last step(s); do **not** rerun `combine.py`, which rebuilds the provisional LLM file.
