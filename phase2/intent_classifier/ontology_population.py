#!/usr/bin/env python3
"""Task 1, step 2: rerun Phase 1 src/rag/ontology_population.py with intent_v2, without modifying it.

The Phase 1 source is executed in memory with only PROCESSED_DIR redirected. Two runs:
  phase1_repro: Phase 1 annotated CSV (rule intents) + Phase 1 phraseology TTL
  phase2:       same CSV with `intent` replaced by intent_v2 (row-aligned, summaries checked) + Phase 2 phraseology TTL
Counts are parsed from the script's own printout. Large TTL outputs are not committed (see .gitignore).
Writes out/ontology/<run>/build.log, results_ontology.json.
Run:  python phase2/intent_classifier/ontology_population.py --phase1 ../uatie-edm-phase1
"""
import contextlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_root, sha256  # noqa: E402

ORIG = 'PROCESSED_DIR   = os.path.join("data", "processed")'
KEYS = {"incident_individuals": r"IncidentReport individuals\s*:\s*([\d,]+)",
        "intent_individuals": r"Intent individuals\s*:\s*([\d,]+)",
        "intent_triples": r"Intent triples\s*:\s*([\d,]+)",
        "strong_phraseme_triples": r"hasDirectPhrasemeError\s*:\s*([\d,]+)",
        "weak_phraseme_triples": r"hasCommunicationContext\s*:\s*([\d,]+)",
        "incident_graph_triples": r"Total triples \(incident g\)\s*:\s*([\d,]+)",
        "merged_triples": r"Total triples \(merged TTL\)\s*:\s*([\d,]+)",
        "unclassified_records": r"Unclassified\s*:\s*([\d,]+)"}


def run(src, outdir):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src.replace(ORIG, f"PROCESSED_DIR = {str(outdir)!r}"), "ontology_population.py", "exec"),
             {"__name__": "__main__"})
    log = buf.getvalue()
    (outdir / "build.log").write_text(log, encoding="utf-8")
    return {k: int(m.group(1).replace(",", "")) for k, p in KEYS.items() if (m := re.search(p, log))}


def main():
    repo = phase1_root()
    src = (repo / "src" / "rag" / "ontology_population.py").read_text(encoding="utf-8")
    assert src.count(ORIG) == 1, "Phase 1 ontology_population.py changed: PROCESSED_DIR line not found once"
    ann = pd.read_csv(repo / "data" / "processed" / "asrs_nlp_annotated.csv", low_memory=False)
    iv2 = pd.read_csv(TASK / "out" / "intent_v2.csv")
    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", usecols=["summary"], low_memory=False)
    assert len(ann) == len(iv2) == len(meta) and (iv2.faiss_idx.values == range(len(iv2))).all()
    assert (ann.summary.astype(str).values == meta.summary.astype(str).values).all(), "row alignment check failed"
    R = {"phase1_source_sha256": sha256(repo / "src" / "rag" / "ontology_population.py"), "runs": {}}
    base = TASK / "out" / "ontology"
    p1_ttl = repo / "data" / "processed" / "phraseology_ontology.ttl"
    p2_ttl = ROOT / "phase2" / "phraseology" / "out" / "ontology" / "phase2" / "phraseology_ontology.ttl"
    for name, phr, intent in [("phase1_repro", p1_ttl, None), ("intent_v2_phase1_ttl", p1_ttl, iv2.intent_v2.values),
                              ("phase2", p2_ttl, iv2.intent_v2.values)]:
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(phr, d / "phraseology_ontology.ttl")
        a = ann.copy()
        if intent is not None:
            a["intent"] = intent
        a.to_csv(d / "asrs_nlp_annotated.csv", index=False)
        cwd = Path.cwd()
        os.chdir(repo)
        try:
            R["runs"][name] = run(src, d)
        finally:
            os.chdir(cwd)
        R["runs"][name]["phraseology_ttl"] = "phase1" if phr == p1_ttl else "phase2"
        R["runs"][name]["intent"] = "phase1 rules" if intent is None else "intent_v2"
        for f in ["asrs_nlp_annotated.csv", "asrs_incidents.ttl", "uatie_edm_merged.ttl"]:
            R["runs"][name][f"{f}_sha256"] = sha256(d / f) if (d / f).exists() else None
        print(name, {k: v for k, v in R["runs"][name].items() if not k.endswith("sha256")}, flush=True)
    (TASK / "results_ontology.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)


def write_report(R):
    cols = ["intent", "phraseology_ttl", "incident_individuals", "unclassified_records", "intent_triples",
            "strong_phraseme_triples", "weak_phraseme_triples", "incident_graph_triples", "merged_triples"]
    p1, iv, p2 = (R["runs"][k] for k in ("phase1_repro", "intent_v2_phase1_ttl", "phase2"))
    L = ["## Ontology population (Phase 1 `src/rag/ontology_population.py`, run unmodified)", "",
         "| Run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {n} | " + " | ".join(f"{v.get(c):,}" if isinstance(v.get(c), int) else str(v.get(c)) for c in cols) + " |"
          for n, v in R["runs"].items()]
    L += ["", "- `intent_v2_phase1_ttl` changes only the intent column, `phase2` also swaps in the Phase 2 phraseology "
          "TTL, so the difference between rows isolates each change.",
          f"- Intent change alone (phase1_repro -> intent_v2_phase1_ttl): strong phraseme links "
          f"{p1['strong_phraseme_triples']:,} -> {iv['strong_phraseme_triples']:,}, weak "
          f"{p1['weak_phraseme_triples']:,} -> {iv['weak_phraseme_triples']:,}. The Phase 1 linker attaches "
          "phrasemes by intent, so fewer records carrying the linked intents gives fewer links.",
          f"- Phraseology TTL change alone (intent_v2_phase1_ttl -> phase2): merged triples "
          f"{iv['merged_triples']:,} -> {p2['merged_triples']:,} ({p2['merged_triples'] - iv['merged_triples']:+,}); "
          "incident-graph triples unchanged.", "- Source: `results_ontology.json`; large TTL "
          "outputs are regenerated, not committed.", ""]
    rp = TASK / "REPORT.md"
    r = rp.read_text(encoding="utf-8").split("## Ontology population")[0].rstrip() + "\n\n"
    rp.write_text(r + "\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report(json.loads((TASK / "results_ontology.json").read_text(encoding="utf-8")))
    else:
        main()
