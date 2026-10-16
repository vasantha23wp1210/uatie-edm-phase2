#!/usr/bin/env python3
"""Task 5: rebuild the phraseology TTL with Phase 1's src/ontology/build_ontology_triples.py, unmodified.

The Phase 1 script reads/writes a hard-coded data/processed/ directory. This runner reads its source, points
PROCESSED_DIR at a Phase 2 output folder (the only change, made in memory), and executes it twice:
  phase1_repro: on Phase 1's committed CAA_ICAO_crossmap.csv (reconciles against Phase 1's TTL)
  phase2:       on out/phase1_compat/CAA_ICAO_crossmap.csv from crossmap.py
Writes out/ontology/<run>/phraseology_ontology.ttl and phraseology_triples.csv, results_ontology.json.
Run:  python phase2/phraseology/ontology.py [--phase1 PATH]
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

from rdflib import OWL, RDF, Graph

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_root, sha256  # noqa: E402

ORIG = 'PROCESSED_DIR = os.path.join("data", "processed")'


def stats(g):
    subj_typed = set(g.subjects(RDF.type, None))
    return {"triples": len(g), "named_individuals": len(set(g.subjects(RDF.type, OWL.NamedIndividual))),
            "typed_subjects": len(subj_typed)}


def run(script, crossmap, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(crossmap, outdir / "CAA_ICAO_crossmap.csv")
    src = script.read_text(encoding="utf-8")
    assert src.count(ORIG) == 1, "Phase 1 script changed: PROCESSED_DIR line not found"
    ns = {"__name__": "__phase1_build_ontology__"}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        exec(compile(src.replace(ORIG, f"PROCESSED_DIR = {str(outdir)!r}"), str(script), "exec"), ns)
    (outdir / "build.log").write_text(log.getvalue(), encoding="utf-8")
    s = stats(ns["g"])
    s.update({"caa_phrase_individuals": len(ns["df"]), "icao_phrase_individuals": len(ns["icao_cache"]),
              "circumstance_nodes": len(ns["section_cache"]), "service_nodes": len(ns["service_cache"]),
              "speaker_nodes": len(ns["speaker_cache"]),
              "ttl_sha256": sha256(outdir / "phraseology_ontology.ttl")})
    return s


def main():
    p1 = phase1_root()
    script = p1 / "src" / "ontology" / "build_ontology_triples.py"
    proc = p1 / "data" / "processed"
    out = TASK / "out" / "ontology"
    committed = Graph().parse(proc / "phraseology_ontology.ttl")
    R = {"phase1_committed_ttl": stats(committed),
         "phase1_repro": run(script, proc / "CAA_ICAO_crossmap.csv", out / "phase1_repro"),
         "phase2": run(script, TASK / "out" / "phase1_compat" / "CAA_ICAO_crossmap.csv", out / "phase2"),
         "paper_claims": {"triples": 4880, "individuals": 703},
         "script_sha256": sha256(script)}
    repro = Graph().parse(out / "phase1_repro" / "phraseology_ontology.ttl")
    R["phase1_repro_isomorphic_to_committed"] = repro.isomorphic(committed)
    (TASK / "results_ontology.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
