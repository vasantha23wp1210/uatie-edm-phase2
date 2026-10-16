# Task 7: grounded generation (LangChain + Ollama)

Default model `llama3.2:3b` (override with `--model` or `UATIE_LLM`; endpoint with `--base-url` or `OLLAMA_HOST`).

Reproduce (one command, after `ollama pull llama3.2:3b`):

```
python phase2/generation/generate.py --phase1 ../uatie-edm-phase1 && python phase2/generation/judge.py --phase1 ../uatie-edm-phase1
```

Both steps resume from `out/*.jsonl`; delete those files for a clean rerun.
Outputs: `out/answers.jsonl`, `out/judgements.jsonl`, `results_generation.json`, `results.json`, `REPORT.md`, `rate/answers_to_rate.csv` (30 answers for your 1-5 rating; pending).
