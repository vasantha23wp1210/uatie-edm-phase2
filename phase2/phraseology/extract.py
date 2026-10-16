#!/usr/bin/env python3
"""Task 4, step 1: extract structured phraseology from the PDFs in data/phraseology_sources/ (no downloads).

Writes out/sources/doc4444_ch12.csv (ICAO Doc 4444 Chapter 12, sections 12.3-12.7) with
source, section, section_title, category, speaker, marker, phrase, is_emergency, is_gnss.
Speaker: Doc 4444 marks pilot transmissions with '*'; everything else is recorded as ATC, and the raw
marker is kept so other legend symbols can be checked.
Run:  python phase2/phraseology/extract.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
SRC = ROOT / "data" / "phraseology_sources"
OUT = TASK / "out" / "sources"
EMERGENCY = re.compile(r"MAYDAY|PAN ?PAN|EMERGENCY|DISTRESS|URGENCY|SEVEN-SEVEN-ZERO-ZERO|\b7700\b", re.I)
GNSS = re.compile(r"\bGNSS\b|\bRNAV\b|\bRNP\b|\bGPS\b|\bRAIM\b|\bPBN\b|\bSBAS\b|\bGBAS\b|\bGLS\b", re.I)
SEC = re.compile(r"^(12\.[3-7](?:\.\d+){0,2})\s+(.*)$")
ITEM = re.compile(r"([*†‡]{0,2})([a-z])\)\s+")
SKIP = re.compile(r"^(<<<PAGE|\* Denotes|Note\b|Note \d|PANS-ATM|Chapter 12|\d+/\d+/\d+|12-\d+)")


def pdf_lines(path):
    lines = []
    for i, p in enumerate(PdfReader(path).pages, 1):
        lines.append(f"<<<PAGE {i}>>>")
        lines += (p.extract_text() or "").splitlines()
    return [re.sub(r"\s+", " ", l).strip() for l in lines]


def upper_ratio(s):
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", s)
    letters = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s*(;\s*or|;|\.)\s*$", "", s).strip()


def doc4444(path):
    """Doc 4444 Ch. 12 is a two-column table: circumstances on the left, phraseologies on the right.
    The phrase column starts at the x0 of the item letters (a), b), ...) on each page."""
    import pdfplumber
    ITEMW = re.compile(r"^([*†‡]{0,2})([a-z]{1,2})\)$")
    rows, cur, sec, title, cat, titles, col = [], None, "", "", "", {}, 264.0
    started = False

    def flush():
        nonlocal cur
        if cur and cur["phrase"].strip():
            rows.append(cur)
        cur = None

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[250:]:
            words = page.extract_words()
            lines = {}
            for w in words:
                lines.setdefault(round(w["top"]), []).append(w)
            items_x = [w["x0"] for w in words if ITEMW.match(w["text"])]
            if items_x:
                col = min(items_x) - 2
            for top in sorted(lines):
                ws = lines[top]
                text = " ".join(w["text"] for w in ws)
                if not started:
                    started = text == "12.3.1 General"
                    if not started:
                        continue
                if text == "Chapter 13":
                    flush()
                    return finish(rows, titles)
                if (SKIP.match(text) or re.match(r"^\d+/\d+/\d+$|^12-\d+|Air Traffic Management|Circumstances", text)):
                    flush()
                    continue
                m = re.match(r"^(12\.[3-7](?:\.\d+){0,2})$", ws[0]["text"])
                left = [w["text"] for w in ws if w["x0"] < col]
                right = [w for w in ws if w["x0"] >= col]
                if m:
                    flush()
                    sec = m.group(1)
                    title = " ".join(left[1:])
                    titles[sec] = title
                    cat = titles.get(".".join(sec.split(".")[:3]), title)
                elif left and sec in titles:
                    titles[sec] += " " + " ".join(left)
                    title = titles[sec]
                if not right:
                    continue
                im = ITEMW.match(right[0]["text"])
                rtext = " ".join(w["text"] for w in (right[1:] if im else right))
                if im:
                    flush()
                    cur = dict(section=sec, section_title=title, category=cat, marker=im.group(1),
                               item=im.group(2), phrase=rtext)
                elif cur:
                    cur["phrase"] += " " + rtext
                else:
                    mk = re.match(r"^[*†‡]+", rtext)
                    cur = dict(section=sec, section_title=title, category=cat, marker=mk.group(0) if mk else "",
                               item="", phrase=rtext[mk.end():] if mk else rtext)
    flush()
    return finish(rows, titles)


def finish(rows, titles):
    df = pd.DataFrame(rows)
    def upper_head(t):
        out = []
        for tok in str(t).split():
            if tok.upper() != tok or not re.search(r"[A-Z]", tok):
                break
            out.append(tok)
        return " ".join(out)
    raw = dict(titles)
    titles = {k: upper_head(v) or v[:80] for k, v in raw.items()}
    df["section_title"] = df.section.map(titles).fillna("")
    df["category"] = df.section.map(lambda s: raw.get(".".join(s.split(".")[:3]), ""))
    df["phrase"] = df.phrase.map(clean)
    df = df[df.phrase.str.len() > 0]
    df.insert(0, "source", "ICAO Doc 4444 (PANS-ATM) Ch. 12")
    df["speaker"] = df.marker.map(lambda m: "Pilot" if "*" in m else ("Aircraft/tow vehicle" if "†" in m else "ATC"))
    txt = df.section_title + " " + df.phrase
    df["is_emergency"] = txt.str.contains(EMERGENCY)
    df["is_gnss"] = txt.str.contains(GNSS)
    return df


def doc9432(path):
    """Doc 9432 prints dialogues in two columns: controller lines right-aligned to x1 ~300, pilot lines
    left-aligned from x0 >= 303. Body text starts at x0 66/84 and is skipped."""
    import pdfplumber
    rows, chap, sec, cur = [], "", "", None
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            lines = {}
            for w in page.extract_words():
                lines.setdefault(round(w["top"]), []).append(w)
            prev_top = None
            for top in sorted(lines):
                ws = lines[top]
                text, x0 = " ".join(w["text"] for w in ws), ws[0]["x0"]
                head = re.match(r"^((?:1[0-2]|[1-9])\.\d{1,2}(?:\.\d+)*)\s+([A-Z][A-Z \-/,()&]+)$", text)
                if re.match(r"^Chapter \d+$", text):
                    chap = text
                if head:
                    sec = head.group(1)
                is_dlg = (not head and x0 > 85 and upper_ratio(text) >= 0.8
                          and not re.match(r"^\d+-\d+$|^Chapter \d+$", text))
                if not is_dlg:
                    if cur: rows.append(cur)
                    cur, prev_top = None, None
                    continue
                spk = "Pilot" if x0 >= 303 else "ATC"
                if cur and cur["speaker"] == spk and prev_top is not None and top - prev_top <= 16:
                    cur["phrase"] += " " + text
                else:
                    if cur: rows.append(cur)
                    cur = dict(source="ICAO Doc 9432 (Manual of Radiotelephony)", page=pno, chapter=chap,
                               section=sec, speaker=spk, phrase=text)
                prev_top = top
            if cur: rows.append(cur)
            cur = None
    df = pd.DataFrame(rows)
    df["phrase"] = df.phrase.map(clean)
    df = df[df.phrase.str.len() > 2]
    df["is_emergency"] = df.phrase.str.contains(EMERGENCY)
    df["is_gnss"] = df.phrase.str.contains(GNSS)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = doc4444(SRC / "doc_4444_pans_atm.pdf")
    d.to_csv(OUT / "doc4444_ch12.csv", index=False)
    print("doc4444 rows", len(d), "sections", d.section.nunique(), d.speaker.value_counts().to_dict(),
          "markers", d.marker.value_counts().to_dict(), "emergency", int(d.is_emergency.sum()), "gnss", int(d.is_gnss.sum()))
    print(d.groupby(d.section.str[:6]).size().to_dict())
    e = doc9432(SRC / "ICAO-Doc-9432.pdf")
    e.to_csv(OUT / "doc9432_dialogues.csv", index=False)
    print("doc9432 rows", len(e), e.speaker.value_counts().to_dict(), "emergency", int(e.is_emergency.sum()),
          "gnss", int(e.is_gnss.sum()))


if __name__ == "__main__":
    sys.exit(main())
