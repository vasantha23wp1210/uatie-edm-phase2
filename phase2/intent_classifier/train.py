#!/usr/bin/env python3
"""Task 1, step 2: train intent classifiers and write intent_v2 for every record.

Training sets (none contains a gold dev/test record):
  silver        Phase 1 keyword-rule labels of all rule-classified records (11 classes)
  scaled        2,000 Claude labels, scaled/labelled_batch{1..4}.csv (12 labels incl. Unclassified, section 5 rules)
  silver_scaled silver + scaled
Gold labels (judgements/intent_labelled.csv, revised under section 5) are used only for threshold and model choice
(dev, 199 rows) and final scoring (test, 201 rows). Below the threshold a record becomes Unclassified.
Models: logistic regression on the Phase 1 384-d MiniLM vectors (read from asrs_faiss.index) and
distilbert-base-uncased fine-tuned for 3 epochs (CPU), on each training set (DistilBERT uses a class-capped silver
subsample). Only the model chosen on dev is run over all records.
Writes out/intent_v2.csv, out/gold_predictions.csv, results.json, REPORT.md.
Run:  python phase2/intent_classifier/train.py --phase1 <path to uatie-edm-phase1> [--skip-distilbert]
"""
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT))
from phase1 import phase1_info, phase1_root, sha256  # noqa: E402

REPO = phase1_root()
OUT = TASK / "out"
SEED = 42
CLASSES = ["ATC_Operational_Error", "Emergency_Event", "Runway_Incursion", "Ground_Incident", "Communication_Failure",
           "Airspace_Violation", "IMC_Spatial_Disorientation", "Weather_Deviation", "Crew_Resource_Management",
           "Maintenance_Error", "Procedural_Non_Compliance"]
UNK = "Unclassified"
LABELS = CLASSES + [UNK]
THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
BERT = "distilbert/distilbert-base-uncased"
BERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BERT_PER_CLASS, BERT_EPOCHS, BERT_LEN, BERT_BS, BERT_LR = 300, 3, 96, 32, 5e-5
SCALED = TASK / "scaled"
N_BATCHES = 4


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load_gold():
    key = pd.read_csv(TASK / "splits" / "gold_key.csv")
    lab = pd.read_csv(TASK / "judgements" / "intent_labelled.csv", dtype=str, keep_default_na=False)
    ref = pd.read_csv(TASK / "judgements" / "intent_to_label.csv", dtype=str, keep_default_na=False)
    assert len(lab) == len(ref) and lab.label_id.is_unique, "intent_labelled.csv rows differ from the export"
    m = ref[["label_id", "record_id", "summary"]].merge(lab, on="label_id", suffixes=("", "_l"))
    assert (m.record_id == m.record_id_l).all() and (m.summary == m.summary_l).all(), "id/summary changed"
    bad = m[~m.your_intent.isin(LABELS)].label_id.tolist()
    assert not bad, f"labels not in allowed_labels.txt: {bad[:5]}"
    return m.merge(key, on=["label_id", "record_id"])


def apply_thr(proba, classes, thr):
    i = proba.argmax(1)
    conf = proba[np.arange(len(i)), i]
    return np.where(conf >= thr, np.asarray(classes)[i], UNK), conf


def scores(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)), 4),
            "macro_f1_present_labels": round(float(f1_score(y, p, labels=sorted(set(y)), average="macro",
                                                            zero_division=0)), 4),
            "per_class": {c: {"precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(d), 4),
                              "support": int(s)} for c, a, b, d, s in zip(LABELS, pr, rc, f1, sup)},
            "confusion_rows_gold_cols_pred": confusion_matrix(y, p, labels=LABELS).tolist(),
            "unclassified_pred_pct": round(float(np.mean(np.asarray(p) == UNK) * 100), 2)}


def pick_threshold(proba, classes, y_dev):
    grid = {t: f1_score(y_dev, apply_thr(proba, classes, t)[0], labels=LABELS, average="macro", zero_division=0)
            for t in THRESHOLDS}
    best = max(grid, key=lambda t: (round(grid[t], 6), -t))
    return best, {str(t): round(float(v), 4) for t, v in grid.items()}


def load_scaled(meta, gold):
    key = pd.read_csv(SCALED / "train_key.csv")
    parts = []
    for b in range(1, N_BATCHES + 1):
        ref = pd.read_csv(SCALED / f"to_label_batch{b}.csv", dtype=str, keep_default_na=False)
        lab = pd.read_csv(SCALED / f"labelled_batch{b}.csv", dtype=str, keep_default_na=False)
        assert len(lab) == len(ref), f"labelled_batch{b}.csv: {len(lab)} rows, export has {len(ref)}"
        for c in ["label_id", "record_id", "source", "summary"]:
            assert (lab[c].values == ref[c].values).all(), f"labelled_batch{b}.csv: {c} or row order changed"
        bad = lab[~lab.your_intent.isin(LABELS)].label_id.tolist()
        assert not bad, f"labelled_batch{b}.csv: labels not in allowed_labels.txt: {bad[:5]}"
        parts.append(lab.assign(batch=b))
    s = pd.concat(parts, ignore_index=True).merge(key, on=["label_id", "record_id"], validate="one_to_one")
    assert len(s) == len(key) and s.label_id.is_unique
    assert (meta.record_id.to_numpy()[s.faiss_idx] == s.record_id.to_numpy()).all()
    assert (meta.summary.to_numpy()[s.faiss_idx] == s.summary.to_numpy()).all()
    assert not s.faiss_idx.isin(gold.faiss_idx).any(), "scaled training records overlap gold dev/test"
    val = {}
    for b, d in s.groupby("batch"):
        y, st = d.your_intent.to_numpy(), d.stratum.to_numpy()
        ok = st != UNK
        blocks = [float(np.mean(y[i:i + 50][ok[i:i + 50]] == st[i:i + 50][ok[i:i + 50]]))
                  for i in range(0, len(d), 50) if ok[i:i + 50].any()]
        val[f"batch{b}"] = {"rule_labelled_rows": int(ok.sum()),
                            "agreement_with_hidden_rule_label_by_row_offset": {
                                str(k): round(float(np.mean(np.roll(y, k)[ok] == st[ok])), 4) for k in range(-3, 4)},
                            "lowest_50_row_block_agreement": round(min(blocks), 4)}
    return s.assign(intent=s.your_intent), val


def bert_train(train_txt, train_y, classes):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(BERT, revision=BERT_REVISION, num_labels=len(classes))
    yi = torch.tensor([classes.index(c) for c in train_y])
    enc = tok(list(train_txt), truncation=True, max_length=BERT_LEN, padding="max_length", return_tensors="pt")
    opt = torch.optim.AdamW(model.parameters(), lr=BERT_LR)
    g = torch.Generator().manual_seed(SEED)
    losses = []
    model.train()
    for ep in range(BERT_EPOCHS):
        perm, tot = torch.randperm(len(yi), generator=g), 0.0
        for s in range(0, len(yi), BERT_BS):
            b = perm[s:s + BERT_BS]
            out = model(input_ids=enc.input_ids[b], attention_mask=enc.attention_mask[b], labels=yi[b])
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            tot += float(out.loss) * len(b)
        losses.append(round(tot / len(yi), 4))
        print(f"distilbert ({len(yi)} records) epoch {ep + 1}: loss {losses[-1]}", flush=True)
    model.eval()
    return model, tok, losses


def bert_predict(model, tok, texts):
    import torch
    ps = []
    with torch.no_grad():
        for s in range(0, len(texts), 256):
            e = tok(list(texts[s:s + 256]), truncation=True, max_length=BERT_LEN, padding=True, return_tensors="pt")
            ps.append(torch.softmax(model(**e).logits, -1).numpy())
    return np.vstack(ps)


def main():
    set_seeds()
    skip_bert = "--skip-distilbert" in sys.argv
    gold = load_gold()
    meta = pd.read_csv(REPO / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False,
                       usecols=["faiss_idx", "record_id", "summary", "intent"])
    meta["summary"] = meta.summary.fillna("").astype(str)
    import faiss
    ix = faiss.read_index(str(REPO / "data" / "processed" / "asrs_faiss.index"))
    X = ix.reconstruct_n(0, ix.ntotal)
    assert len(X) == len(meta) and (meta.faiss_idx.to_numpy() == np.arange(len(meta))).all()

    silver = meta[(meta.intent != UNK) & ~meta.faiss_idx.isin(gold.faiss_idx)]
    scaled, scaled_val = load_scaled(meta, gold)
    dev, test = gold[gold.gold_split == "dev"], gold[gold.gold_split == "test"]
    prev = pd.read_csv(TASK / "judgements" / "intent_labelled_claude_v1.csv", dtype=str,
                       keep_default_na=False).set_index("label_id").your_intent
    was = gold.label_id.map(prev)
    moved = Counter(zip(was[was != gold.your_intent], gold.your_intent[was != gold.your_intent]))
    R = {"task": "intent_classifier", "seed": SEED,
         "label_provenance": gold.notes.str.split(". Borderline").str[0].value_counts().to_dict(),
         "gold_relabel": {"previous_labels": "judgements/intent_labelled_claude_v1.csv",
                          "rows_changed": int((was != gold.your_intent).sum()),
                          "rows_changed_by_split": {s: int((was != gold.your_intent)[gold.gold_split == s].sum())
                                                    for s in ["dev", "test"]},
                          "borderline_rows": int(gold.notes.str.contains("Borderline").sum()),
                          "changes_previous_to_current": [{"from": a, "to": b, "rows": n}
                                                          for (a, b), n in moved.most_common()]},
         "silver_train_records": int(len(silver)), "silver_class_counts": silver.intent.value_counts().to_dict(),
         "scaled_train_records": int(len(scaled)), "scaled_class_counts": scaled.intent.value_counts().to_dict(),
         "scaled_label_provenance": scaled.notes.str.split(";").str[0].str.strip().replace("", "(blank)")
                                          .value_counts().head(5).to_dict(),
         "scaled_validation": scaled_val,
         "gold": {"dev": int(len(dev)), "test": int(len(test))},
         "gold_label_counts": {s: g.your_intent.value_counts().to_dict() for s, g in gold.groupby("gold_split")},
         "models": {}}

    rule = {s: scores(g.your_intent, g.stratum) for s, g in [("dev", dev), ("test", test)]}
    R["models"]["keyword_rules_phase1"] = {"threshold": None, "dev": rule["dev"], "test": rule["test"],
                                           "unclassified_pct_all_records": round(float((meta.intent == UNK).mean() * 100), 2)}

    cols = ["faiss_idx", "summary", "intent"]
    sets = {"silver": silver[cols], "scaled": scaled[cols], "silver_scaled": pd.concat([silver[cols], scaled[cols]])}
    gidx = gold.faiss_idx.to_numpy()
    cands = {}
    for name, key in [("logreg_minilm", "silver"), ("logreg_minilm_scaled", "scaled"),
                      ("logreg_minilm_silver_scaled", "silver_scaled")]:
        d, t0 = sets[key], time.time()
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(X[d.faiss_idx.to_numpy()], d.intent)
        cands[name] = {"P": lr.predict_proba(X[gidx]), "cls": list(lr.classes_),
                       "full": lambda lr=lr: lr.predict_proba(X),
                       "info": {"train_set": key, "train_records": int(len(d)), "train_seconds": round(time.time() - t0, 1)}}
        print(name, "trained", flush=True)

    if not skip_bert:
        sub = pd.concat([d.sample(min(len(d), BERT_PER_CLASS), random_state=SEED)
                         for _, d in silver[cols].groupby("intent", sort=True)])
        for name, key, d in [("distilbert", "silver_capped", sub), ("distilbert_scaled", "scaled", sets["scaled"]),
                             ("distilbert_silver_scaled", "silver_capped+scaled", pd.concat([sub, sets["scaled"]]))]:
            cls = [c for c in LABELS if c in set(d.intent)]
            t0 = time.time()
            model, tok, losses = bert_train(d.summary, d.intent, cls)
            secs = round(time.time() - t0, 1)
            cands[name] = {"P": bert_predict(model, tok, meta.summary.to_numpy()[gidx].tolist()), "cls": cls,
                           "full": lambda m=model, t=tok: bert_predict(m, t, meta.summary.tolist()),
                           "info": {"train_set": key, "train_records": int(len(d)), "model": BERT,
                                    "revision": BERT_REVISION, "epochs": BERT_EPOCHS, "per_class_cap_silver": BERT_PER_CLASS,
                                    "max_len": BERT_LEN, "lr": BERT_LR, "batch": BERT_BS, "classes": len(cls),
                                    "epoch_loss": losses, "train_seconds": secs}}
    else:
        for name in ["distilbert", "distilbert_scaled", "distilbert_silver_scaled"]:
            R["models"][name] = {"status": "skipped (--skip-distilbert)"}

    gi = {i: k for k, i in enumerate(gold.index)}
    orig_unk = (test.stratum == UNK).to_numpy()
    for name, c in cands.items():
        Pg, cls = c["P"], c["cls"]
        Pd, Pt = Pg[[gi[i] for i in dev.index]], Pg[[gi[i] for i in test.index]]
        thr, grid = pick_threshold(Pd, cls, dev.your_intent)
        pdv, _ = apply_thr(Pd, cls, thr)
        ptv, _ = apply_thr(Pt, cls, thr)
        R["models"][name] = {**c["info"], "threshold_chosen_on_dev": thr, "dev_macro_f1_by_threshold": grid,
                             "dev": scores(dev.your_intent, pdv), "test": scores(test.your_intent, ptv),
                             "test_on_rule_unclassified_records": scores(test.your_intent[orig_unk], ptv[orig_unk])}
        gold[f"pred_{name}"] = apply_thr(Pg, cls, thr)[0]

    best = max(cands, key=lambda m: R["models"][m]["dev"]["macro_f1"])
    R["model_chosen_on_dev_macro_f1"] = best
    thr = R["models"][best]["threshold_chosen_on_dev"]
    t0 = time.time()
    P = cands[best]["full"]()
    R["models"][best]["full_corpus_inference_seconds"] = round(time.time() - t0, 1)
    v2, conf = apply_thr(P, cands[best]["cls"], thr)
    R["gold_rows_where_full_corpus_label_differs_from_scored_prediction"] = int((v2[gidx] != gold[f"pred_{best}"]).sum())
    OUT.mkdir(exist_ok=True)
    out = pd.DataFrame({"faiss_idx": meta.faiss_idx, "record_id": meta.record_id, "intent_rule": meta.intent,
                        "intent_v2": v2, "intent_conf": np.round(conf, 4),
                        "intent_source": np.where(v2 == UNK, f"{best}:below_threshold", best)})
    out.to_csv(OUT / "intent_v2.csv", index=False)
    gold.drop(columns=["summary_l", "record_id_l"], errors="ignore").to_csv(OUT / "gold_predictions.csv", index=False)
    R["intent_v2_all_records"] = {
        "records": int(len(out)), "unclassified_pct": round(float((v2 == UNK).mean() * 100), 2),
        "target_unclassified_lt_15pct_met": bool((v2 == UNK).mean() * 100 < 15),
        "agreement_with_rules_on_rule_classified_pct": round(float(
            (out.intent_v2 == out.intent_rule)[out.intent_rule != UNK].mean() * 100), 2),
        "rule_unclassified_now_classified": int(((out.intent_rule == UNK) & (out.intent_v2 != UNK)).sum()),
        "distribution": out.intent_v2.value_counts().to_dict()}
    R["target_test_accuracy_ge_085_met"] = R["models"][best]["test"]["accuracy"] >= 0.85
    R["env"] = {"python": platform.python_version(), "cpu_count": os.cpu_count()}
    R["phase1"] = phase1_info(REPO)
    R["output_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [OUT / "intent_v2.csv",
                                                                             OUT / "gold_predictions.csv"]}
    (TASK / "results.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    write_report(R)
    print(json.dumps({k: R[k] for k in ["model_chosen_on_dev_macro_f1", "intent_v2_all_records",
                                         "target_test_accuracy_ge_085_met"]}, indent=2))
    for m, v in R["models"].items():
        if "test" in v:
            print(m, "dev macroF1", v["dev"]["macro_f1"], "test acc", v["test"]["accuracy"], "macroF1", v["test"]["macro_f1"])


def write_report(R):
    prov, gr = R["label_provenance"], R["gold_relabel"]
    sv = R["scaled_validation"]
    off0 = [v["agreement_with_hidden_rule_label_by_row_offset"]["0"] for v in sv.values()]
    offk = [x for v in sv.values() for k, x in v["agreement_with_hidden_rule_label_by_row_offset"].items() if k != "0"]
    L = ["# Task 1: Intent classifier", "", "All numbers come from `results.json`, written by `train.py`.", "",
         "## Label provenance (read first)", "",
         f"- Gold dev/test labels: {prov}. No label is human-reviewed, so every gold score below is **agreement with "
         "provisional LLM labels**, not accuracy against human ground truth.",
         f"- The gold labels were revised under `phase2/labels/LABELLING_PROMPT.md` section 5 before any scaled-label "
         f"model was trained: {gr['rows_changed']} of {R['gold']['dev'] + R['gold']['test']} rows changed from "
         f"`{gr['previous_labels']}` (dev {gr['rows_changed_by_split']['dev']}, test {gr['rows_changed_by_split']['test']}); "
         f"{gr['borderline_rows']} rows are marked borderline (`judgements/intent_ambiguity_notes.csv`). Claude's own "
         "summary describes the changes relative to its first pass, so the relabel was not blind to the earlier labels.",
         f"- Scaled training labels: {R['scaled_train_records']:,} Claude labels under the same section 5 rules, not "
         "human-reviewed. An earlier attempt drifted 13 rows out of alignment; the user rebuilt the labels keyed to "
         "`label_id`. `train.py` checks that IDs, source, summaries and row order equal the export. Agreement with the "
         f"hidden Phase 1 rule label on rule-labelled rows: {min(off0):.3f}-{max(off0):.3f} per batch at the correct row, "
         f"{min(offk):.3f}-{max(offk):.3f} when shifted by 1-3 rows; lowest agreement in any block of 50 consecutive "
         f"rows (about 25 rule-labelled rows each, so noisy): {min(v['lowest_50_row_block_agreement'] for v in sv.values()):.3f}.",
         "- The scaled set was sampled 1,000 rule-labelled records (90-91 per rule class) + 1,000 rule-Unclassified "
         "records, so its class mix is not the corpus distribution.",
         "- Silver training labels are the Phase 1 keyword rules. No training set contains a gold record. Threshold "
         "and model are chosen on gold dev only; gold test is scored once.", "",
         "## Scores", "",
         "| Model | train set | train records | threshold (dev) | dev macro-F1 | test accuracy | test macro-F1 (12 labels) "
         "| test Unclassified % |", "|---|---|---|---|---|---|---|---|"]
    for m, v in R["models"].items():
        if "test" in v:
            thr = v.get("threshold_chosen_on_dev", "-")
            L.append(f"| {m} | {v.get('train_set', 'rules')} | {v.get('train_records', '-')} | "
                     f"{'-' if thr is None else thr} | {v['dev']['macro_f1']:.4f} | {v['test']['accuracy']:.4f} | "
                     f"{v['test']['macro_f1']:.4f} | {v['test']['unclassified_pred_pct']} |")
        elif "status" in v:
            L.append(f"| {m} | {v['status']} | | | | | | |")
    best = R["model_chosen_on_dev_macro_f1"]
    a = R["intent_v2_all_records"]
    L += ["", f"Model chosen on dev macro-F1: `{best}`. Target test accuracy >= 0.85 met: "
          f"**{R['target_test_accuracy_ge_085_met']}** ({R['models'][best]['test']['accuracy']:.4f}).", "",
          "## intent_v2 over all records", "",
          f"- {a['records']:,} records; Unclassified {a['unclassified_pct']}% (Phase 1 rules: "
          f"{R['models']['keyword_rules_phase1']['unclassified_pct_all_records']}%). Target < 15% met: "
          f"**{a['target_unclassified_lt_15pct_met']}**.",
          f"- Agreement with the rule label where the rules classified: {a['agreement_with_rules_on_rule_classified_pct']}%.",
          f"- Rule-Unclassified records now classified: {a['rule_unclassified_now_classified']:,}.",
          f"- Gold rows whose full-corpus label differs from the scored prediction (batching): "
          f"{R['gold_rows_where_full_corpus_label_differs_from_scored_prediction']}.",
          "- Output: `out/intent_v2.csv` (`intent_v2`, `intent_conf`, `intent_source`).", "",
          f"## Per-class test scores (`{best}`)", "", "| Class | P | R | F1 | support |", "|---|---|---|---|---|"]
    for c, v in R["models"][best]["test"]["per_class"].items():
        L.append(f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    t = R["models"][best]["test_on_rule_unclassified_records"]
    rules_acc = R["models"]["keyword_rules_phase1"]["test"]["accuracy"]
    cm = np.array(R["models"][best]["test"]["confusion_rows_gold_cols_pred"])
    off = [(cm[i, j], LABELS[i], LABELS[j]) for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j and cm[i, j]]
    L += ["", f"On test records the rules left Unclassified (n = {t['n']}): accuracy {t['accuracy']:.4f}, "
          f"macro-F1 {t['macro_f1']:.4f}.", "",
          f"Most frequent test errors (`{best}`, gold -> predicted): "
          + "; ".join(f"{g} -> {p} {n}" for n, g, p in sorted(off, reverse=True)[:6]) + ".", "",
          "Full confusion matrices (rows gold, columns predicted, label order as in `LABELS`) are in `results.json`.", "",
          "## Known limitations", "",
          f"- Silver labels come from the keyword rules, which agree poorly with the gold labels (rules test accuracy "
          f"{rules_acc:.4f}).",
          f"- The confidence threshold chosen on dev maps low-confidence predictions to Unclassified (`{best}` threshold "
          f"{R['models'][best].get('threshold_chosen_on_dev')}; {R['models'][best]['test']['unclassified_pred_pct']}% of "
          "test predictions are Unclassified).",
          "- Gold and scaled labels come from the same LLM under the same rules, so a high score partly measures "
          "consistency with Claude, not correctness.",
          "- Gold classes are imbalanced; classes with support < 10 on test give unstable per-class scores.",
          "- DistilBERT runs on CPU with 3 epochs and no hyper-parameter search (see `results.json` for sizes and time)."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
