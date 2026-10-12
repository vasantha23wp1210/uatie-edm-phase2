"""Locate and fingerprint the Phase 1 checkout (uatie-edm-phase1) that Phase 2 reads code and data from.

Order: --phase1 <path> on the command line, then the UATIE_PHASE1 environment variable,
then a sibling folder ../uatie-edm-phase1 next to this repo.
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Phase 1 files that Phase 2 tasks import or read; hashed into every results.json.
USED = [
    "src/rag/rag_query_engine.py", "src/rag/bm25_baseline.py", "src/hysor_rrf.py", "src/hysor_ablation.py",
    "evaluation/gold_queries.csv", "evaluation/gold_relevance.csv",
    "evaluation/retrieval_results.csv", "evaluation/bm25_results.csv",
    "data/processed/asrs_faiss_meta.csv", "data/processed/asrs_faiss.index",
]


def phase1_root():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--phase1")
    args, _ = ap.parse_known_args()
    raw = args.phase1 or os.environ.get("UATIE_PHASE1") or str(ROOT.parent / "uatie-edm-phase1")
    p = Path(raw).expanduser().resolve()
    bad = []
    for rel in USED:
        f = p / rel
        if not f.exists():
            bad.append(f"{rel}: missing")
        elif f.stat().st_size < 1024 and f.read_bytes().startswith(b"version https://git-lfs"):
            bad.append(f"{rel}: Git LFS pointer, not the real file")
    if bad:
        sys.exit(f"Phase 1 checkout not usable at {p}:\n  " + "\n  ".join(bad) +
                 "\nPass --phase1 <path to uatie-edm-phase1> (or set UATIE_PHASE1) and run `git lfs pull` there.")
    return p


def _git(p, *args):
    try:
        return subprocess.run(["git", "-C", str(p), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phase1_info(p):
    dirty = _git(p, "status", "--porcelain", "--", "src", "evaluation", "data")
    return {
        "commit": _git(p, "rev-parse", "HEAD") or "unknown (not a git checkout)",
        "src_evaluation_data_modified": None if dirty is None else bool(dirty),
        "sha256": {rel: sha256(p / rel) for rel in USED},
    }
