#!/usr/bin/env python3
"""Judge calibration with controls, and a deterministic NLI judge (CPU).

Controls (built here, seed 42, no gold labels used):
- positive: every cockpit summary sentence with the report it was copied from (verbatim, so supported);
- negative: one altered copy of each positive, type by position: number changed, invented outcome appended,
  "reported" -> "reported no", or a sentence from an unrelated report (seed 42). Number/negation fall back to an
  invented outcome when the sentence has no digit / no " reported ".
Controls from dev queries calibrate the NLI threshold (max balanced accuracy); controls from test queries are
held out and give sensitivity (negatives caught) and specificity (positives accepted).
NLI judge: cross-encoder/nli-deberta-v3-base (fixed revision); a summary sentence is supported if P(entailment)
>= threshold for its best premise, where the premises are each cited report and each sentence of it. Sentence-level
premises were added after a first run with whole reports only rejected 17 of 252 verbatim positives, all from
multi-sentence reports (disclosed in REPORT.md). Key-hazard strings are not judged (fragments, not sentences).
--llm MODEL also runs the Task 7 LLM judge prompt on N_LLM held-out positives and N_LLM negatives (seed 42).
Writes nli/controls.csv, nli/answer_sentences.csv, nli/llm_controls_<model>.jsonl, nli/results.json, nli/REPORT.md.
Run:  python phase2/edge/nli/nli_judge.py --phase1 PATH [--llm llama3.2:3b --llm llama3.1:8b]
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parents[1]
sys.path.insert(0, str(P2.parent))
from phase1 import phase1_root, sha256  # noqa: E402

SEED, N_LLM = 42, 50
NLI, NLI_REVISION = "cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
SPLIT = re.compile(r"(?<=[.!?])\s+")
OUTCOMES = ["an engine fire and an emergency evacuation", "a runway excursion into the grass",
            "a collision with a ground vehicle", "a declared fuel emergency and diversion",
            "injuries to two flight attendants", "a rejected takeoff above v1"]
THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
SETS = {"cockpit (extractive)": P2 / "edge" / "out",
        "ground llama3.2:3b (dev+test)": P2 / "generation" / "out",
        "ground llama3.1:8b (test)": P2 / "generation" / "out_llama3.1_8b_test",
        "dev free llama3.2:1b": P2 / "generation" / "compare" / "llama3.2_1b",
        "dev free llama3.2:3b": P2 / "generation" / "compare" / "llama3.2_3b",
        "dev free llama3.1:8b": P2 / "generation" / "compare" / "llama3.1_8b",
        "dev anchored llama3.2:3b": P2 / "generation" / "compare_anchored" / "llama3.2_3b",
        "dev anchored llama3.1:8b": P2 / "generation" / "compare_anchored" / "llama3.1_8b"}


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def sents(t):
    return [s.strip() for s in SPLIT.split(str(t)) if s.strip()]


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def rogan_gladen(obs, se, sp):
    """True unsupported rate from an observed flagged rate, given sensitivity and specificity (clamped to [0, 1])."""
    if obs is None or se is None or sp is None or se + sp - 1 <= 0:
        return None
    return round(min(1.0, max(0.0, (obs + sp - 1) / (se + sp - 1))), 4)


def build_controls(text_of):
    rnd = random.Random(SEED)
    pool = list(text_of.index)
    rows = []
    for a in jl(P2 / "edge" / "out" / "answers.jsonl"):
        if a["status"] != "ok":
            continue
        summ = a["answer"]["summary"]
        for i in a["answer"]["citations"]:
            cand = [s for s in sents(text_of[i]) if s in summ]
            if cand:
                rows.append({"query_id": a["query_id"], "split": a["split"], "report": i,
                             "retrieved": "|".join(a["retrieved"]), "sentence": max(cand, key=len)})
    out = []
    for k, r in enumerate(rows):
        s, kind = r["sentence"], ["number", "outcome", "negation", "unrelated"][k % 4]
        if kind == "number" and not re.search(r"\d", s):
            kind = "outcome"
        if kind == "negation" and " reported " not in s:
            kind = "outcome"
        if kind == "number":
            neg = re.sub(r"\d+", lambda m: str(int(m.group(0)) + (1 if len(m.group(0)) < 3 else 10)), s)
        elif kind == "negation":
            neg = s.replace(" reported ", " reported no ", 1)
        elif kind == "unrelated":
            while True:
                j = rnd.choice(pool)
                if j not in r["retrieved"].split("|") and sents(text_of[j]):
                    neg = sents(text_of[j])[0]
                    break
        else:
            neg = s.rstrip(".") + ", resulting in " + OUTCOMES[k % len(OUTCOMES)] + "."
        out.append({**r, "control": "positive", "kind": "verbatim", "claim": s})
        out.append({**r, "control": "negative", "kind": kind, "claim": neg})
    return pd.DataFrame(out)


def entail(model, items):
    """items: (report text, claim); returns max P(entailment) over the report and each of its sentences."""
    pairs, owner = [], []
    for k, (t, c) in enumerate(items):
        for prem in [t] + sents(t):
            pairs.append((prem, c))
            owner.append(k)
    p = model.predict(pairs, batch_size=32, apply_softmax=True, show_progress_bar=False)[:, 1]
    best = np.zeros(len(items))
    np.maximum.at(best, np.asarray(owner), p)
    return best


def rates(d, thr):
    pos, neg = d[d.control == "positive"], d[d.control == "negative"]
    tp, tn = int((neg.p_entail < thr).sum()), int((pos.p_entail >= thr).sum())
    return {"sensitivity": round(tp / len(neg), 4), "sensitivity_ci95": wilson(tp, len(neg)), "negatives": len(neg),
            "specificity": round(tn / len(pos), 4), "specificity_ci95": wilson(tn, len(pos)), "positives": len(pos)}


def llm_controls(model_name, C, text_of, base_url):
    spec = importlib.util.spec_from_file_location("p2_judge", P2 / "generation" / "judge.py")
    jm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jm)
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model_name, base_url=base_url, format="json", keep_alive="2h", **jm.OPTIONS)
    held = C[C.split == "test"]
    pick = pd.concat([g.sample(min(N_LLM, len(g)), random_state=SEED) for _, g in held.groupby("control")])
    path = HERE / f"llm_controls_{model_name.replace(':', '_')}.jsonl"
    done = {(x["query_id"], x["report"], x["control"]) for x in jl(path)}
    for r in pick.itertuples():
        if (r.query_id, r.report, r.control) in done:
            continue
        docs = "\n".join(f"[{i}] {text_of[i]}" for i in r.retrieved.split("|"))
        raw = llm.invoke(jm.PROMPT.format(docs=docs, answer=r.claim)).content
        try:
            cl = [c for c in json.loads(raw).get("claims", []) if isinstance(c, dict) and "supported" in c]
            flag = None if not cl else any(not (c["supported"] is True or str(c["supported"]).lower() == "true")
                                           for c in cl)
        except (json.JSONDecodeError, AttributeError):
            flag = None
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"query_id": r.query_id, "report": r.report, "control": r.control, "kind": r.kind,
                                 "flagged_unsupported": flag}) + "\n")
        print(model_name, r.query_id, r.control, flag, flush=True)
    J = [x for x in jl(path) if x["flagged_unsupported"] is not None]
    pos, neg = [x for x in J if x["control"] == "positive"], [x for x in J if x["control"] == "negative"]
    tp, tn = sum(x["flagged_unsupported"] for x in neg), sum(not x["flagged_unsupported"] for x in pos)
    by = {k: f"{sum(x['flagged_unsupported'] for x in neg if x['kind'] == k)}/{sum(x['kind'] == k for x in neg)}"
          for k in sorted({x["kind"] for x in neg})}
    return {"controls_judged": len(J), "invalid_or_empty": len(jl(path)) - len(J),
            "sensitivity": round(tp / len(neg), 4) if neg else None, "sensitivity_ci95": wilson(tp, len(neg)),
            "specificity": round(tn / len(pos), 4) if pos else None, "specificity_ci95": wilson(tn, len(pos)),
            "negatives_caught_by_kind": by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--llm", action="append", default=[])
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--report-only", action="store_true", help="rewrite REPORT.md from results.json")
    a = ap.parse_args()
    if a.report_only:
        return write_report(json.loads((HERE / "results.json").read_text(encoding="utf-8")))
    text_of = pd.read_csv(phase1_root() / "data" / "processed" / "asrs_faiss_meta.csv",
                          usecols=["incident_id", "summary"]).astype(str).set_index("incident_id").summary
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(NLI, revision=NLI_REVISION, device="cpu")
    C = build_controls(text_of)
    C["p_entail"] = entail(model, [(text_of[r.report], r.claim) for r in C.itertuples()])
    C.to_csv(HERE / "controls.csv", index=False)
    cal = C[C.split == "dev"]
    bal = {t: (rates(cal, t)["sensitivity"] + rates(cal, t)["specificity"]) / 2 for t in THRESHOLDS}
    thr = max(THRESHOLDS, key=lambda t: (round(bal[t], 6), -t))
    held = C[C.split == "test"]
    nli = {"threshold_chosen_on_dev_controls": thr, "dev_controls": rates(cal, thr), "held_out_test_controls": rates(held, thr),
           "held_out_negatives_caught_by_kind": {k: f"{int((g.p_entail < thr).sum())}/{len(g)}" for k, g in
                                                 held[held.control == "negative"].groupby("kind")}}
    S, sets = [], {}
    for name, d in SETS.items():
        A = jl(d / "answers.jsonl")
        J = {x["query_id"]: x for x in jl(d / "judgements.jsonl")}
        llm = [c for x in J.values() for c in x["claims"]]
        llm_u = sum(not (c["supported"] is True or str(c["supported"]).lower() == "true") for c in llm)
        rows = []
        for x in A:
            ans = x.get("answer") or {}
            if x["status"] == "insufficient_evidence" or not ans.get("summary"):
                continue
            cites = ans.get("citations") or x["retrieved"]
            for s in sents(ans["summary"]):
                rows.append({"set": name, "query_id": x["query_id"], "split": x.get("split"), "sentence": s,
                             "cites": "|".join(cites)})
        if not rows:
            continue
        pairs = [(text_of[c], r["sentence"]) for r in rows for c in r["cites"].split("|")]
        p, k = entail(model, pairs), 0
        for r in rows:
            n = len(r["cites"].split("|"))
            r["p_entail_max"], k = float(p[k:k + n].max()), k + n
            r["nli_supported"] = r["p_entail_max"] >= thr
        u = sum(not r["nli_supported"] for r in rows)
        sets[name] = {"answers": len({r["query_id"] for r in rows}), "sentences": len(rows), "nli_unsupported": u,
                      "nli_rate": round(u / len(rows), 4), "nli_rate_ci95": wilson(u, len(rows)),
                      "llm_judge_3b_claims": len(llm), "llm_judge_3b_unsupported": llm_u,
                      "llm_judge_3b_rate": round(llm_u / len(llm), 4) if llm else None,
                      "llm_judge_3b_rate_ci95": wilson(llm_u, len(llm))}
        S += rows
    pd.DataFrame(S).to_csv(HERE / "answer_sentences.csv", index=False)
    held_nli = nli["held_out_test_controls"]
    for v in sets.values():
        v["nli_rate_corrected_rogan_gladen"] = rogan_gladen(v["nli_rate"], held_nli["sensitivity"], held_nli["specificity"])
    L = {m: llm_controls(m, C, text_of, a.base_url) for m in a.llm}
    if "llama3.2:3b" in L:
        for v in sets.values():
            v["llm_judge_3b_rate_corrected_rogan_gladen"] = rogan_gladen(
                v["llm_judge_3b_rate"], L["llama3.2:3b"]["sensitivity"], L["llama3.2:3b"]["specificity"])
    R = {"seed": SEED, "nli_model": NLI, "nli_revision": NLI_REVISION,
         "controls": {"positives": int((C.control == "positive").sum()), "negatives": int((C.control == "negative").sum()),
                      "negative_kinds": C[C.control == "negative"].kind.value_counts().to_dict(),
                      "controls_sha256": sha256(HERE / "controls.csv")},
         "nli": nli, "llm_judges_on_held_out_controls": L, "answer_sets": sets}
    (HERE / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({"nli": nli, "llm": L}, indent=1))


def ci(v):
    return [max(0.0, round(x, 4)) for x in v]


def write_report(R):
    n, c = R["nli"], R["controls"]
    L = ["# Judge calibration with controls, and an NLI judge", "",
         "All numbers come from `results.json`, written by `nli_judge.py`.", "", "## Method", "",
         f"- Controls (no gold labels): {c['positives']} positives, i.e. cockpit summary sentences copied verbatim "
         f"from the cited report, and {c['negatives']} negatives, one altered copy of each ({c['negative_kinds']}). "
         "Altered copies are easier to catch than real hallucinations, so sensitivity here is an upper bound.",
         f"- NLI judge: `{R['nli_model']}` (revision `{R['nli_revision'][:10]}`), CPU, deterministic. A summary sentence "
         "is supported if P(entailment) reaches the threshold for its best premise (a cited report, or one sentence of "
         "it). A first run with whole reports as premises only rejected 17 of 252 verbatim positives, all from "
         "multi-sentence reports, so sentence premises were added before the results below; that check looked at "
         "controls from dev and test queries (controls only, no task labels). Threshold chosen on the "
         f"controls from dev queries (max balanced accuracy): {n['threshold_chosen_on_dev_controls']}. Key-hazard "
         "strings are not judged by NLI; the LLM judge splits summary + hazards into claims, so units differ.",
         "- Rogan-Gladen correction: true rate = (observed + specificity - 1) / (sensitivity + specificity - 1), "
         "clamped to [0, 1], using held-out control sensitivity and specificity.", "",
         "## Judge accuracy on held-out controls (test queries)", "",
         "| Judge | sensitivity (negatives caught) | 95% CI | specificity (positives accepted) | 95% CI |",
         "|---|---|---|---|---|"]
    h = n["held_out_test_controls"]
    L.append(f"| NLI ({h['negatives']} neg / {h['positives']} pos) | {h['sensitivity']} | {h['sensitivity_ci95']} | "
             f"{h['specificity']} | {h['specificity_ci95']} |")
    for m, v in R["llm_judges_on_held_out_controls"].items():
        L.append(f"| LLM `{m}` ({v['controls_judged']} judged, {v['invalid_or_empty']} invalid) | {v['sensitivity']} | "
                 f"{v['sensitivity_ci95']} | {v['specificity']} | {v['specificity_ci95']} |")
    L += ["", f"NLI negatives caught by kind (held out): {n['held_out_negatives_caught_by_kind']}."]
    L += [f"LLM `{m}` negatives caught by kind: {v['negatives_caught_by_kind']}."
          for m, v in R["llm_judges_on_held_out_controls"].items()]
    L += ["", "## Unsupported rate by answer set", "",
          "| Answer set | sentences | NLI unsupported | NLI rate [95% CI] | NLI corrected | LLM 3B judge rate [95% CI] "
          "(claims) | LLM 3B corrected |", "|---|---|---|---|---|---|---|"]
    for k, v in R["answer_sets"].items():
        L.append(f"| {k} | {v['sentences']} | {v['nli_unsupported']} | {v['nli_rate']} {ci(v['nli_rate_ci95'])} | "
                 f"{v['nli_rate_corrected_rogan_gladen']} | {v['llm_judge_3b_rate']} {v['llm_judge_3b_rate_ci95']} "
                 f"({v['llm_judge_3b_claims']}) | {v.get('llm_judge_3b_rate_corrected_rogan_gladen', '-')} |")
    sp3 = R["llm_judges_on_held_out_controls"].get("llama3.2:3b", {}).get("specificity")
    L += ["", "## Reading these numbers", "",
          "- On controls the NLI judge is near-perfect, but it marks 44-52% of ground LLM sentences unsupported. Ground "
          "answers paraphrase and combine several reports (e.g. \"reported in multiple incidents\"), which no single "
          "premise entails. That says NLI is strict on abstractive text, not that half the answers are invented; its "
          "specificity on verbatim positives does not carry over to paraphrased sentences.",
          f"- The 3B LLM judge accepts only {sp3} of verbatim positives, so it flags about "
          f"{round(1 - sp3, 2) if sp3 is not None else '-'} of fully grounded sentences. Every observed 3B rate is below "
          "that false-flag rate, so the Rogan-Gladen estimate clamps to 0 for every answer set. The correction is "
          "therefore uninformative here, not evidence that the ground answers are fully grounded.",
          "- LLM-judge rates are per claim (summary and hazards split into claims), NLI rates per summary sentence. The "
          "cockpit row counts 251 sentences with the sentence splitter used for answer sets; the controls use the 252 "
          "sentences of the exact-match check.",
          "- Hallucination <= 5% for the ground profile therefore stays not verifiable with these judges."]
    L += ["", "## Limitations", "",
          "- Positives are verbatim, so they are the easiest possible supported claims; real paraphrased claims are "
          "harder, and specificity on them may be lower.",
          "- Negatives are rule-made alterations, not real model errors.",
          "- The cockpit row is a sanity check: its sentences are verbatim by construction (the controls' positives).",
          "- The human claim check (`phase2/edge/rate/claims_to_check.csv`) is still the reference; none of these judges "
          "replaces it."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
