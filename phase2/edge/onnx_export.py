#!/usr/bin/env python3
"""Edge path: export the query encoder (all-MiniLM-L6-v2) and the cross-encoder to ONNX, quantise to int8
(onnxruntime dynamic quantisation), and compare with the PyTorch models on the 20 DEV queries only:
FAISS top-50 overlap, cross-encoder top-5 overlap, dev P@5 (qrels_dev), and CPU latency per query.
The same ONNX files can be served by ONNX Runtime NPU execution providers (e.g. QNN, OpenVINO); that path is not
measured here (no NPU on this machine). Models go to onnx/models/ (not committed). Writes onnx/results.json, REPORT.md.
Run:  python phase2/edge/onnx_export.py --phase1 PATH
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TASK = Path(__file__).resolve().parent
P2 = TASK.parent
OUT = TASK / "onnx"
sys.path.insert(0, str(P2.parent))
sys.path.insert(0, str(P2 / "generation"))
from phase1 import phase1_root  # noqa: E402
from generate import TOPK, load_module  # noqa: E402

ENC = "sentence-transformers/all-MiniLM-L6-v2"


def export(model, tok, path, n_in):
    x = tok(["a b c", "d e"], ["f g", "h"] if n_in == 2 else None, padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {k: {0: "b", 1: "t"} for k in names}
    torch.onnx.export(model, tuple(x[k] for k in names), str(path), input_names=names, output_names=["out"],
                      dynamic_axes={**dyn, "out": {0: "b"}}, opset_version=17, dynamo=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1")
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="test: one parity run after the dev result is written; reuses the exported models")
    split = ap.parse_args().split
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    repo = phase1_root()
    cwd = Path.cwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    rag = load_module("rag_query_engine", repo / "src" / "rag" / "rag_query_engine.py")
    rag.initialize_resources(verbose=False)
    bm25_mod = load_module("bm25_baseline", repo / "src" / "rag" / "bm25_baseline.py")
    os.chdir(cwd)
    t3 = load_module("p2_retrieval_run", P2 / "retrieval" / "run.py")
    t3.set_seeds()
    torch.set_num_threads(os.cpu_count())
    md = OUT / "models"
    md.mkdir(parents=True, exist_ok=True)

    class Enc(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = AutoModel.from_pretrained(ENC).eval()

        def forward(s, ids, mask, tt):
            h = s.m(input_ids=ids, attention_mask=mask, token_type_ids=tt).last_hidden_state
            m = mask.unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(e, dim=-1)

    class CE(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = AutoModelForSequenceClassification.from_pretrained(t3.CE_MODEL, revision=t3.CE_REVISION).eval()

        def forward(s, ids, mask, tt):
            return s.m(input_ids=ids, attention_mask=mask, token_type_ids=tt).logits[:, 0]

    etok, ctok = AutoTokenizer.from_pretrained(ENC), AutoTokenizer.from_pretrained(t3.CE_MODEL, revision=t3.CE_REVISION)
    enc, cem = Enc(), CE()
    sizes = {}
    for name, m, tk, n in [("encoder", enc, etok, 1), ("cross_encoder", cem, ctok, 2)]:
        fp = md / f"{name}_fp32.onnx"
        if split == "dev" or not (md / f"{name}_int8.onnx").exists():
            export(m, tk, fp, n)
            quantize_dynamic(str(fp), str(md / f"{name}_int8.onnx"), weight_type=QuantType.QInt8)
        sizes[name] = {k: round((md / f"{name}_{k}.onnx").stat().st_size / 2**20, 1) for k in ["fp32", "int8"]}
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    sess = {f"{n}_int8": ort.InferenceSession(str(md / f"{n}_int8.onnx"), so, providers=["CPUExecutionProvider"])
            for n in ["encoder", "cross_encoder"]}

    def run(s, x):
        return s.run(None, {k: x[k].astype(np.int64) for k in ["input_ids", "attention_mask", "token_type_ids"]})[0]

    meta = pd.read_csv(repo / "data" / "processed" / "asrs_faiss_meta.csv", low_memory=False)
    meta["incident_id"] = meta.incident_id.astype(str)
    text_of = meta.set_index("incident_id")["summary"].astype(str)
    bm25, _ = bm25_mod.build_bm25(meta, "summary")
    q = pd.read_csv(P2 / "gold_eval_rebuild" / "splits" / f"{split}_queries.csv")
    qr = pd.read_csv(P2 / "retrieval" / "eval" / f"qrels_{split}.csv")
    rel = set(zip(qr[qr.relevance >= 1].query_id, qr[qr.relevance >= 1].incident_id.astype(str)))
    rows = []
    for r in q.itertuples():
        qt = str(r.query_text)
        b = [str(x["incident_id"]) for x in bm25_mod.bm25_retrieve(bm25, qt, meta, "incident_id", t3.DEPTH)]
        t0 = time.perf_counter()
        with torch.no_grad():
            x = etok([qt], return_tensors="pt")
            et = enc(x["input_ids"], x["attention_mask"], x["token_type_ids"]).numpy()
        t1 = time.perf_counter()
        eo = run(sess["encoder_int8"], etok([qt], return_tensors="np"))
        t2 = time.perf_counter()
        ft = [str(meta.incident_id.iat[i]) for i in rag.index.search(et.astype(np.float32), t3.DEPTH)[1][0] if i >= 0]
        fo = [str(meta.incident_id.iat[i]) for i in rag.index.search(eo.astype(np.float32), t3.DEPTH)[1][0] if i >= 0]
        union = list(dict.fromkeys(b + ft))
        pairs = ctok([qt] * len(union), [text_of[i] for i in union], padding=True, truncation=True,
                     max_length=512, return_tensors="pt")
        t3_ = time.perf_counter()
        with torch.no_grad():
            lt = cem(pairs["input_ids"], pairs["attention_mask"], pairs["token_type_ids"]).numpy()
        t4 = time.perf_counter()
        lo = run(sess["cross_encoder_int8"], {k: v.numpy() for k, v in pairs.items()})
        t5 = time.perf_counter()
        top = lambda lg: [union[j] for j in sorted(range(len(union)), key=lambda j: (-float(lg[j]), j))[:TOPK]]
        ct, co = top(lt), top(lo)
        rows.append({"query_id": r.query_id, "union": len(union), "faiss50_overlap": len(set(ft) & set(fo)) / t3.DEPTH,
                     "ce_top5_overlap": len(set(ct) & set(co)) / TOPK,
                     "p5_torch": sum((r.query_id, i) in rel for i in ct) / TOPK,
                     "p5_int8": sum((r.query_id, i) in rel for i in co) / TOPK,
                     "enc_ms_torch": (t1 - t0) * 1000, "enc_ms_int8": (t2 - t1) * 1000,
                     "ce_ms_torch": (t4 - t3_) * 1000, "ce_ms_int8": (t5 - t4) * 1000})
        print(rows[-1]["query_id"], rows[-1]["ce_top5_overlap"], round(rows[-1]["ce_ms_int8"]), flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"{split}_parity.csv", index=False)
    R = {"split": split, "queries": len(D), "cpu_count": os.cpu_count(), "onnxruntime": ort.__version__,
         "model_mb": sizes, "faiss_top50_overlap_mean": round(D.faiss50_overlap.mean(), 4),
         "ce_top5_overlap_mean": round(D.ce_top5_overlap.mean(), 4),
         f"{split}_p5_torch_fp32": round(D.p5_torch.mean(), 4), f"{split}_p5_onnx_int8": round(D.p5_int8.mean(), 4),
         **{f"{c}_p50": round(float(D[c].median()), 1) for c in ["enc_ms_torch", "enc_ms_int8", "ce_ms_torch", "ce_ms_int8"]},
         "npu": "not measured: no NPU/DPU on this machine; ONNX Runtime QNN / OpenVINO execution providers load the same files"}
    (OUT / ("results.json" if split == "dev" else "results_test.json")).write_text(json.dumps(R, indent=2) + "\n",
                                                                                     encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
