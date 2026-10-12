# llm_judge: failed calibration attempt (kept on record)

The question was whether an LLM judge (Devin) could label the test pool in place of manual judging.
Before any test pair was judged, I labelled the 362 dev pool pairs that already have manual Phase 1 labels,
blind and following `GUIDELINE.md`. Agreement with the human labels was then measured.

Reproduce the blind CSV with `python phase2/gold_eval_rebuild/llm_judge/prepare.py --phase1 <path>` and the agreement numbers with `python phase2/gold_eval_rebuild/llm_judge/agree.py`.
The labels in `labels/` were written by hand (by Devin) and cannot be regenerated with a seed.

Result (`calibration_results.json`): Cohen's kappa 0.0619, quadratic-weighted kappa 0.4311,
binary (>=1) kappa 0.1151, exact agreement 0.3591. Most disagreements are pairs the human labelled 1 and the LLM labelled 0.
Agreement was too low, so no LLM labels are used. The user is judging `judgements/test_to_judge.csv` by hand.

`spot_check_to_judge.csv` (150 seeded test pairs) was meant for a human check of LLM test labels. It is unused now that the test pool is judged by hand.
