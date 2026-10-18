#!/usr/bin/env python3
"""UK CAA CAP 413 (Edition 24) Chapter 8 "Emergency Phraseology": extract example transmissions and count
emergency phrases. Input: data/phraseology_sources/CAP413_edition24.pdf (supplied by the user; not in git).
Chapter pages = pages whose footer reads "Chapter 8: Emergency Phraseology". Prose starts left of x=150 pt;
example transmissions are set from x~163 pt in two columns (split at x=340 pt); a vertical gap > 22 pt or a prose
line ends a transmission. Page 230 also sets a prose table in those columns, so a block counts as a transmission
only if it contains a callsign or station token (RT regex below); dropped blocks stay in the CSV with kept=False. A transmission is "emergency" if it matches extract.EMERGENCY (same regex as the ICAO
count). CAP 413 is a CAA document, so it is NOT added to the ICAO matching corpus (CAA phrases would then be
matched against CAA text, which would make the CAA-unique rate circular); it is counted separately.
Writes out/sources/cap413_ch8.csv, results_cap413.json, REPORT_cap413.md.
Run:  python phase2/phraseology/cap413.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(ROOT / "phase2"))
from extract import EMERGENCY, GNSS  # noqa: E402

PDF = ROOT / "data" / "phraseology_sources" / "CAP413_edition24.pdf"
FOOTER = "Chapter 8: Emergency Phraseology"
PROSE_X, COL_X, GAP = 150, 340, 22
RT = re.compile(r"\bG-\s?[A-Z]{2,4}\b|\bBIGJET\b|\bTower\b|\bCentre\b|\(Callsign\)|All stations|Attention all|"
                r"Training Fix|\bPRACTICE PAN\b|^MAYDAY MAYDAY|\bRunway \d+")


def sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import pdfplumber
    rows, pages = [], []
    with pdfplumber.open(PDF) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            words = page.extract_words()
            text = " ".join(w["text"] for w in words if w["top"] > page.height - 60)
            if FOOTER not in text:
                continue
            pages.append(pno)
            lines = defaultdict(list)
            for w in words:
                if 50 < w["top"] < page.height - 60:
                    lines[round(w["top"])].append(w)
            section, heading = None, None
            open_ = {0: None, 1: None}
            for top in sorted(lines):
                ws = sorted(lines[top], key=lambda w: w["x0"])
                if ws[0]["x0"] < PROSE_X:
                    line = " ".join(w["text"] for w in ws)
                    m = re.match(r"^(8\.\d+)\s", line)
                    section = m.group(1) if m else section
                    if ws[0]["x0"] < 80 and not m:
                        heading = line
                    open_ = {0: None, 1: None}
                    continue
                for col in (0, 1):
                    cw = [w for w in ws if (w["x0"] >= COL_X) == bool(col)]
                    if not cw:
                        continue
                    cur = open_[col]
                    if cur is not None and top - cur["bottom"] <= GAP:
                        cur["parts"].append(" ".join(w["text"] for w in cw))
                        cur["bottom"] = top
                    else:
                        cur = {"page": pno, "section": section, "heading": heading, "column": col,
                               "parts": [" ".join(w["text"] for w in cw)], "bottom": top}
                        rows.append(cur)
                        open_[col] = cur
    df = pd.DataFrame([{"source": "UK CAA CAP 413 Ed. 24 Ch. 8", "page": r["page"], "section": r["section"],
                        "heading": r["heading"], "speaker_column": "first" if r["column"] == 0 else "reply",
                        "phrase": re.sub(r"\s+", " ", " ".join(r["parts"])).strip()} for r in rows])
    df = df[df.phrase.str.len() > 2].assign(kept=lambda d: d.phrase.str.contains(RT))
    dropped = d0 = df[~df.kept]
    df = df[df.kept]
    df["is_emergency"] = df.phrase.str.contains(EMERGENCY)
    df["is_gnss"] = df.phrase.str.contains(GNSS)
    (TASK / "out" / "sources").mkdir(parents=True, exist_ok=True)
    pd.concat([df, d0]).to_csv(TASK / "out" / "sources" / "cap413_ch8.csv", index=False)
    norm = df.phrase.str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.strip()
    icao = json.loads((TASK / "results_crossmap.json").read_text(encoding="utf-8"))["emergency"]["icao_rt_rows"]
    R = {"pdf": PDF.name, "pdf_sha256": sha(PDF), "chapter_pages": [pages[0], pages[-1]] if pages else None,
         "blocks_dropped_no_callsign_or_station": len(dropped), "transmissions": len(df), "distinct_transmissions": int(norm.nunique()),
         "emergency_regex_rows": int(df.is_emergency.sum()),
         "emergency_regex_distinct": int(norm[df.is_emergency].nunique()),
         "all_ch8_distinct": int(norm.nunique()), "gnss_rows": int(df.is_gnss.sum()),
         "icao_emergency_rows_unchanged": icao,
         "icao_plus_cap413_emergency_rows": icao + int(norm[df.is_emergency].nunique()),
         "target_ge_25_as_defined_icao_only": icao >= 25}
    (TASK / "results_cap413.json").write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
    L = ["# CAP 413 Chapter 8 (UK CAA) emergency phraseology", "",
         f"From `cap413.py`; `{PDF.name}` (SHA-256 `{R['pdf_sha256'][:12]}…`), chapter pages {R['chapter_pages']}.", "",
         f"- Example transmissions extracted: {R['transmissions']} ({R['distinct_transmissions']} distinct); "
         f"{R['blocks_dropped_no_callsign_or_station']} text blocks without a callsign or station dropped (listed in the "
         "CSV with kept=False; they include a prose table and one real phrase, 'Ballistic recovery system deployed').",
         f"- Matching the emergency regex used for the ICAO count: {R['emergency_regex_rows']} rows, "
         f"{R['emergency_regex_distinct']} distinct. The other Ch. 8 examples are emergency-context exchanges "
         "(e.g. ejecting, stop transmitting) that do not contain the regex words.",
         f"- ICAO emergency radiotelephony rows (plan target >= 25, ICAO sources): {icao}, unchanged, "
         f"{'met' if icao >= 25 else 'MISSED'}.",
         f"- ICAO + CAP 413 (regex, distinct): {R['icao_plus_cap413_emergency_rows']}. This is a different, wider count "
         "than the target as written and is reported separately.", "", "## Limits", "",
         "- Layout-based extraction (x positions); a transmission split across a page break counts as two.",
         "- CAP 413 is not in the ICAO matching corpus, so the crossmap and CAA-unique rate are unchanged."]
    (TASK / "REPORT_cap413.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(df[df.is_emergency].phrase.head(40).to_string())


if __name__ == "__main__":
    main()
