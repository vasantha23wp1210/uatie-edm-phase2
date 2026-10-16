# Task 7: grounded generation (LangChain + Ollama)

Default model `llama3.2:3b` (override with `--model` or `UATIE_LLM`; endpoint with `--base-url` or `OLLAMA_HOST`).

Reproduce (one command, after `ollama pull llama3.2:3b`):

```
python phase2/generation/generate.py --phase1 ../uatie-edm-phase1 && python phase2/generation/judge.py --phase1 ../uatie-edm-phase1
```

Both steps resume from `out/*.jsonl`; delete those files for a clean rerun.

Primary context: `ce` top-5 (BM25 top-50 + FAISS top-50 re-ranked by the cross-encoder), insufficient-evidence threshold chosen on dev only. A fixed lexical grounding filter drops unsupported summary sentences and hazards; the unfiltered `raw_answer` is kept and judged separately (`out/judgements_raw.jsonl`).

Archived runs: `out/run_prev/` and `out/run_faiss_claude/` are two earlier FAISS-context runs on identical inputs. Run-to-run check (optional): `python phase2/generation/compare_runs.py` compares those two and writes `run_variation.json`; rerun `judge.py` afterwards to add it to `REPORT.md`.
Outputs: `out/answers.jsonl`, `out/judgements.jsonl`, `out/judgements_raw.jsonl`, `results_generation.json`, `results.json`, `REPORT.md`, `rate/answers_to_rate.csv` (30 answers from the current run, not rated).

User ratings: `rate/answers_rated_run_prev.csv` rates the 30 answers exported from the earlier run (`out/run_prev/`). `python phase2/generation/score_ratings.py` checks each row against that run and writes `results_ratings.json`; rerun `judge.py` afterwards to add it to `REPORT.md`.
