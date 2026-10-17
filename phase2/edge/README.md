# Deployment profiles: ground and cockpit

| Profile | Users | Answer | Hardware |
|---|---|---|---|
| Ground | ATC, safety analysts | LLM-written, grounded and validated (`phase2/generation`) | GPU recommended; CPU latency measured |
| Cockpit | Flight crew | Extractive: verbatim sentences from the cited reports + verified crossmap phrase; no LLM | CPU only; ONNX int8 for NPU / DPU |

Scripts (Python 3.11, CPU only, seed 42):

1. `cockpit.py`: runs the cockpit profile on the 100 gold queries; writes `out/answers.jsonl`, `out/results_generation.json`.
2. `python phase2/generation/judge.py --answers-dir phase2/edge/out`: same claim judge as Task 7.
3. `onnx_export.py`: ONNX fp32 + int8 export of the query encoder and cross-encoder, dev-only parity and CPU latency (`onnx/results.json`). Models go to `onnx/models/` (not committed; rerun to rebuild).
4. `python phase2/generation/compare_models.py`: ground-profile generator comparison on dev only.
5. `report.py`: collects the measured results into `results.json` and `REPORT.md`.

The app (`phase2/app/app.py`) has a **Deployment profile** switch: Ground or Cockpit.

Human check (optional, fixes the judge problem): `python phase2/edge/rate/export.py` writes `rate/claims_to_check.csv` (50 ground-profile test claims, 25 the 3B judge called supported and 25 unsupported, shuffled; fill `your_supported` yes / no without opening `rate/claims_key.csv`) and `rate/cockpit_to_rate.csv` (20 cockpit answers; fill `your_relevance` 1-5).
