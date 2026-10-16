#!/usr/bin/env python3
"""Task 1, step 2: train intent classifiers and write intent_v2 for every record.

Training data: Phase 1 keyword-rule labels (silver) of all rule-classified records, minus the 400 gold records.
Gold labels (judgements/intent_labelled.csv) are used only for: threshold choice and model choice (dev, 199 rows)
and final scoring (test, 201 rows). Below the threshold a record becomes Unclassified.
Models: (a) logistic regression on the Phase 1 384-d MiniLM vectors (read from asrs_faiss.index);
        (b) distilbert-base-uncased fine-tuned for 3 epochs on a class-capped silver subsample (CPU).
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


def bert_train_predict(train_txt, train_y, texts_list):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(BERT, revision=BERT_REVISION, num_labels=len(CLASSES))
    yi = torch.tensor([CLASSES.index(c) for c in train_y])
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
        print(f"distilbert epoch {ep + 1}: loss {losses[-1]}", flush=True)
    model.eval()
    res = []
    with torch.no_grad():
        for texts in texts_list:
            ps = []
            for s in range(0, len(texts), 256):
                e = tok(list(texts[s:s + 256]), truncation=True, max_length=BERT_LEN, padding=True, return_tensors="pt")
                ps.append(torch.softmax(model(**e).logits, -1).numpy())
            res.append(np.vstack(ps))
    return res, losses


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
    dev, test = gold[gold.gold_split == "dev"], gold[gold.gold_split == "test"]
    R = {"task": "intent_classifier", "seed": SEED,
         "label_provenance": gold.notes.value_counts().to_dict(),
         "silver_train_records": int(len(silver)), "silver_class_counts": silver.intent.value_counts().to_dict(),
         "gold": {"dev": int(len(dev)), "test": int(len(test))},
         "gold_label_counts": {s: g.your_intent.value_counts().to_dict() for s, g in gold.groupby("gold_split")},
         "models": {}}

    rule = {s: scores(g.your_intent, g.stratum) for s, g in [("dev", dev), ("test", test)]}
    R["models"]["keyword_rules_phase1"] = {"threshold": None, "dev": rule["dev"], "test": rule["test"],
                                           "unclassified_pct_all_records": round(float((meta.intent == UNK).mean() * 100), 2)}

    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
    lr.fit(X[silver.faiss_idx], silver.intent)
    lr_train_s = round(time.time() - t0, 1)
    P_all = lr.predict_proba(X)
    probas = {"logreg_minilm": (P_all, list(lr.classes_))}

    if not skip_bert:
        sub = pd.concat([d.sample(min(len(d), BERT_PER_CLASS), random_state=SEED)
                         for _, d in silver.groupby("intent", sort=True)])
        t0 = time.time()
        (pg,), losses = bert_train_predict(sub.summary, sub.intent, [meta.summary.tolist()])
        bert_s = round(time.time() - t0, 1)
        probas["distilbert"] = (pg, CLASSES)

    gi = {i: k for k, i in enumerate(gold.index)}
    for name, (P, cls) in probas.items():
        Pg = P[gold.faiss_idx.to_numpy()] if len(P) == len(meta) else P
        Pd, Pt = Pg[[gi[i] for i in dev.index]], Pg[[gi[i] for i in test.index]]
        thr, grid = pick_threshold(Pd, cls, dev.your_intent)
        pdv, _ = apply_thr(Pd, cls, thr)
        ptv, _ = apply_thr(Pt, cls, thr)
        orig_unk = (test.stratum == UNK).to_numpy()
        R["models"][name] = {"threshold_chosen_on_dev": thr, "dev_macro_f1_by_threshold": grid,
                             "dev": scores(dev.your_intent, pdv), "test": scores(test.your_intent, ptv),
                             "test_on_rule_unclassified_records": scores(test.your_intent[orig_unk], ptv[orig_unk])}
        gold[f"pred_{name}"] = apply_thr(Pg, cls, thr)[0]
    R["models"]["logreg_minilm"]["train_seconds"] = lr_train_s
    if not skip_bert:
        R["models"]["distilbert"].update({"model": BERT, "revision": BERT_REVISION, "epochs": BERT_EPOCHS,
                                          "train_records": int(len(sub)), "per_class_cap": BERT_PER_CLASS,
                                          "max_len": BERT_LEN, "lr": BERT_LR, "batch": BERT_BS,
                                          "epoch_loss": losses, "train_plus_full_corpus_inference_seconds": bert_s})
    else:
        R["models"]["distilbert"] = {"status": "skipped (--skip-distilbert)"}

    cand = [m for m in probas]
    best = max(cand, key=lambda m: R["models"][m]["dev"]["macro_f1"])
    R["model_chosen_on_dev_macro_f1"] = best
    thr = R["models"][best]["threshold_chosen_on_dev"]
    P, cls = probas[best]
    v2, conf = apply_thr(P, cls, thr)
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
    for m in R["models"]:
        if "test" in R["models"][m]:
            print(m, "test acc", R["models"][m]["test"]["accuracy"], "macroF1", R["models"][m]["test"]["macro_f1"])


def write_report(R):
    prov = R["label_provenance"]
    L = ["# Task 1: Intent classifier", "", "All numbers come from `results.json`, written by `train.py`.", "",
         "## Label provenance (read first)", "",
         f"- Gold labels: {prov}. While they are `pending user review`, every gold score below is **agreement with "
         "provisional LLM labels**, not accuracy against human ground truth.",
         "- Training data is the Phase 1 keyword-rule labels (silver), excluding the 400 gold records. Threshold and "
         "model are chosen on gold dev only; gold test is scored once.", "",
         "## Test scores (gold test)", "", "| Model | threshold | accuracy | macro-F1 (12 labels) | Unclassified predicted % |",
         "|---|---|---|---|---|"]
    for m, v in R["models"].items():
        if "test" in v:
            L.append(f"| {m} | {v['threshold']if 'threshold' in v else v.get('threshold_chosen_on_dev')} | "
                     f"{v['test']['accuracy']:.4f} | {v['test']['macro_f1']:.4f} | {v['test']['unclassified_pred_pct']} |")
    best = R["model_chosen_on_dev_macro_f1"]
    a = R["intent_v2_all_records"]
    L += ["", f"Model chosen on dev macro-F1: `{best}`. Target test accuracy >= 0.85 met: "
          f"**{R['target_test_accuracy_ge_085_met']}**.", "",
          "## intent_v2 over all records", "",
          f"- {a['records']:,} records; Unclassified {a['unclassified_pct']}% (Phase 1 rules: "
          f"{R['models']['keyword_rules_phase1']['unclassified_pct_all_records']}%). Target < 15% met: "
          f"**{a['target_unclassified_lt_15pct_met']}**.",
          f"- Agreement with the rule label where the rules classified: {a['agreement_with_rules_on_rule_classified_pct']}%.",
          f"- Rule-Unclassified records now classified: {a['rule_unclassified_now_classified']:,}.",
          "- Output: `out/intent_v2.csv` (`intent_v2`, `intent_conf`, `intent_source`).", "",
          f"## Per-class test scores (`{best}`)", "", "| Class | P | R | F1 | support |", "|---|---|---|---|---|"]
    for c, v in R["models"][best]["test"]["per_class"].items():
        L.append(f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    t = R["models"][best]["test_on_rule_unclassified_records"]
    rules_acc = R["models"]["keyword_rules_phase1"]["test"]["accuracy"]
    devc, silc = R["gold_label_counts"]["dev"], R["silver_class_counts"]
    gap_cls = max((c for c in devc if c != UNK),
                  key=lambda c: (devc[c] / R["gold"]["dev"]) / (silc.get(c, 0.5) / R["silver_train_records"]))
    silver_n, dev_n = silc.get(gap_cls, 0), devc[gap_cls]
    L += ["", f"On test records the rules left Unclassified (n = {t['n']}): accuracy {t['accuracy']:.4f}, "
          f"macro-F1 {t['macro_f1']:.4f}.", "",
          "Confusion matrices (rows gold, columns predicted, label order as in `LABELS`) are in `results.json`.", "",
          "## Known limitations", "",
          f"- Silver training labels come from the keyword rules, so the models partly learn the rules' errors. The provisional gold labels differ from the rules a lot (rules test accuracy {rules_acc:.4f}), and gold-frequent classes are rare in silver ({gap_cls}: {silver_n:,} of {R['silver_train_records']:,} silver records vs {dev_n} of {R['gold']['dev']} dev labels).",
          f"- The confidence threshold chosen on dev maps low-confidence predictions to Unclassified (`{best}` threshold {R['models'][best].get('threshold_chosen_on_dev')}; {R['models'][best]['test']['unclassified_pred_pct']}% of test predictions are Unclassified).",
          "- Gold classes are imbalanced; classes with support < 10 on test give unstable per-class scores.",
          "- DistilBERT is trained on a class-capped subsample on CPU (see `results.json` for size and time)."]
    (TASK / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
