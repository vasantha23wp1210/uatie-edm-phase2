"""Task 7 run-to-run check: compare the two archived FAISS-context runs, out/run_prev/ and
out/run_faiss_claude/, which had identical inputs (same retrieval, dev threshold, prompt and Ollama
options: temperature 0, seed 42). The current CE-context run (out/) has different inputs and is not compared.
Writes run_variation.json, which judge.py adds to REPORT.md.
"""
import json
from pathlib import Path

TASK = Path(__file__).resolve().parent


def load(p):
    return {r["query_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def halluc(j):
    c = [x for r in j.values() for x in r["claims"]]
    sup = sum(x["supported"] is True or str(x["supported"]).lower() == "true" for x in c)
    return {"claims": len(c), "unsupported": len(c) - sup, "rate": round(1 - sup / len(c), 4) if c else None}


def main():
    prev, cur = TASK / "out" / "run_prev", TASK / "out" / "run_faiss_claude"
    a, b = load(prev / "answers.jsonl"), load(cur / "answers.jsonl")
    assert set(a) == set(b), "runs cover different queries"
    gen = sorted(q for q in b if b[q]["status"] != "insufficient_evidence")
    R = {"queries": len(b),
         "threshold_prev": sorted({r["threshold"] for r in a.values()}),
         "threshold_cur": sorted({r["threshold"] for r in b.values()}),
         "same_retrieved_lists": sum(a[q]["retrieved"] == b[q]["retrieved"] for q in b),
         "same_candidate_phraseology": sum(a[q]["candidate_phraseology"] == b[q]["candidate_phraseology"] for q in b),
         "insufficient_prev": sum(r["status"] == "insufficient_evidence" for r in a.values()),
         "insufficient_cur": sum(r["status"] == "insufficient_evidence" for r in b.values()),
         "insufficient_sets_equal": {q for q in a if a[q]["status"] == "insufficient_evidence"} ==
                                    {q for q in b if b[q]["status"] == "insufficient_evidence"},
         "generated": len(gen),
         "identical_raw_output": sum(a[q]["raw"] == b[q]["raw"] for q in gen),
         "status_changed": sum(a[q]["status"] != b[q]["status"] for q in b),
         "hallucination_prev": halluc(load(prev / "judgements.jsonl")),
         "hallucination_cur": halluc(load(cur / "judgements.jsonl"))}
    (TASK / "run_variation.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(R))


if __name__ == "__main__":
    main()
