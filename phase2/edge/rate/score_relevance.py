#!/usr/bin/env python3
"""Score the user's relevance ratings (1-5) of current cockpit and ground answers.

Inputs (returned by the user): rate/cockpit_rated.csv (20 cockpit answers), rate/ground_rated.csv (30 answers of
the chosen ground generator, llama3.1:8b on test). Each must match its export (cockpit_to_rate.csv,
ground_to_rate.csv) row for row by query_id and answer text.
Writes rate/results_relevance.json and rate/REPORT_relevance.md.
Run:  python phase2/edge/rate/score_relevance.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED, BOOT, TARGET = 42, 10000, 4.0
SETS = {"cockpit": ("cockpit_to_rate.csv", "cockpit_rated.csv"), "ground": ("ground_to_rate.csv", "ground_rated.csv")}
THEMES = {"phraseology irrelevant or off topic": r"phraseolog\w*[^.;]*(?:irrelevant|off topic|nothing to do|unrelated|wrong)"
          r"|(?:irrelevant|off topic|unrelated)[^.;]*phraseolog",
          "generic restatement of the query": r"generic|restat",
          "retrieved reports do not match the query": r"no report|keyword match|only .* fits|not .* match|"
          r"\d of 3 reports match"}


def load(name):
    return pd.read_csv(HERE / name, dtype=str, encoding="utf-8-sig").fillna("")


def score(kind):
    exp, got = (load(f) for f in SETS[kind])
    assert list(exp.query_id) == list(got.query_id), f"{kind}: query_id order differs from the export"
    assert (exp.answer.str.strip().values == got.answer.str.strip().values).all(), f"{kind}: answer text differs"
    r = pd.to_numeric(got.your_relevance, errors="coerce")
    assert r.notna().all() and r.between(1, 5).all(), f"{kind}: ratings must be 1-5"
    r = r.astype(float).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(r, size=(BOOT, len(r)), replace=True).mean(axis=1)
    notes = got.notes.str.lower()
    return {"n": len(r), "mean": round(float(r.mean()), 4), "sd": round(float(r.std(ddof=1)), 4),
            "median": float(np.median(r)), "mean_ci95_bootstrap": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "distribution": {str(k): int((r == k).sum()) for k in range(1, 6)},
            "share_ge_4": round(float((r >= 4).mean()), 4), "target_mean_ge_4_met": bool(r.mean() >= TARGET),
            "notes_theme_counts": {t: int(notes.str.contains(p, regex=True).sum()) for t, p in THEMES.items()},
            "lowest_rated": [{"query_id": q, "query_text": t, "rating": int(x), "note": n}
                             for q, t, x, n in sorted(zip(got.query_id, got.query_text, r, got.notes), key=lambda z: z[2])[:3]]}


def main():
    R = {"provenance": "human (user) ratings; to confirm whether the notes were refined with an LLM",
         "seed": SEED, "bootstrap": BOOT, "target_mean": TARGET, "sets": {k: score(k) for k in SETS}}
    (HERE / "results_relevance.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# Answer relevance: user ratings of the current answers", "",
         "All numbers come from `results_relevance.json`, written by `score_relevance.py`.", "",
         f"- Provenance: {R['provenance']}.",
         "- Cockpit: 20 extractive answers (`cockpit_to_rate.csv`); ground: 30 answers of the dev-chosen "
         "`llama3.1:8b` on test (`ground_to_rate.csv`, seed 42). Both files match their exports row for row.",
         f"- Scale 1-5 (5 = directly answers the query). Target: mean >= {TARGET}. Mean intervals: bootstrap over "
         f"answers ({BOOT:,} resamples, seed {SEED}).", "",
         "| Profile | n | mean [95% CI] | SD | median | share >= 4 | distribution 1/2/3/4/5 | target met |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in R["sets"].items():
        d = v["distribution"]
        L.append(f"| {k} | {v['n']} | {v['mean']} {v['mean_ci95_bootstrap']} | {v['sd']} | {v['median']} | "
                 f"{v['share_ge_4']} | {'/'.join(str(d[str(i)]) for i in range(1, 6))} | {v['target_mean_ge_4_met']} |")
    L += ["", "## Themes in the notes (keyword counts, fixed patterns in the script)", ""]
    for k, v in R["sets"].items():
        L.append(f"- {k}: " + "; ".join(f"{t}: {c} of {v['n']}" for t, c in v["notes_theme_counts"].items()))
    L += ["", "## Lowest-rated answers", ""]
    for k, v in R["sets"].items():
        for x in v["lowest_rated"]:
            L.append(f"- {k} {x['query_id']} ({x['rating']}): {x['query_text']}: {x['note']}")
    L += ["", "## Limits", "",
          "- One rater; no second rater, so no agreement figure.",
          "- Cockpit and ground were rated on different query samples (20 vs 30), so the two means are not a paired comparison.",
          "- The earlier 2.70 (30 answers of the FAISS-context run, `generation/results_ratings.json`) describes a different system "
          "and is kept for reference only.",
          "- Theme counts are keyword matches on free-text notes, not a coded analysis."]
    (HERE / "REPORT_relevance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "mean", "mean_ci95_bootstrap", "median", "share_ge_4", "notes_theme_counts")}
                      for k, v in R["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
