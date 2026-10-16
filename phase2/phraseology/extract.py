#!/usr/bin/env python3
"""Task 4, step 1: extract structured phraseology from the PDFs in data/phraseology_sources/ (no downloads).

Writes out/sources/doc4444_ch12.csv (ICAO Doc 4444 Chapter 12, sections 12.3-12.7),
doc4444_other.csv (Ch. 6 SID/STAR lists, Ch. 15 quoted phrases), doc4444_cpdlc.csv (Appendix 5 CPDLC, data link)
and doc9432_dialogues.csv. Ch. 12 columns:
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


def doc4444_other(path):
    """Phraseology printed outside Chapter 12: Chapter 6 SID/STAR clearance lists (6.3.2.4, 6.5.2.4) and
    quoted multi-word phrases in Chapter 15 (radiotelephony, so they join the crossmap corpus)."""
    r = PdfReader(str(path))
    rows = []
    for pno in (128, 132):
        L = (r.pages[pno - 1].extract_text() or "").splitlines()
        j = 0
        while j < len(L):
            m = re.match(r"^\s*([a-f])\)\s+((?:CLIMB|DESCEND)\b.*)$", L[j])
            if m:
                ph = m.group(2).strip()
                while not ph.endswith(":") and j + 1 < len(L) and upper_ratio(L[j + 1]) >= 0.8:
                    j += 1
                    ph += " " + L[j].strip()
                rows.append(dict(source="ICAO Doc 4444 (PANS-ATM) Ch. 6", page=pno, section="6.3.2.4" if pno == 128
                                 else "6.5.2.4", item=m.group(1), speaker="ATC", phrase=ph.rstrip(":").strip()))
            j += 1
    for pno in range(319, 348):
        t = r.pages[pno - 1].extract_text() or ""
        if re.search(r"(?m)^\s*Appendix \d", t):
            break
        for q in re.findall(r"[\u201c\"]([A-Z][A-Z0-9 ,()\-/]{3,})[\u201d\"]", t):
            if len(q.split()) >= 2:
                rows.append(dict(source="ICAO Doc 4444 (PANS-ATM) Ch. 15", page=pno, section="15", item="",
                                 speaker="Pilot", phrase=q.strip()))
    df = pd.DataFrame(rows)
    df["phrase"] = df.phrase.map(clean)
    df["is_emergency"] = df.phrase.str.contains(EMERGENCY)
    df["is_gnss"] = df.phrase.str.contains(GNSS)
    return df


def doc4444_cpdlc(path):
    """Appendix 5 CPDLC message set. Data link, not radiotelephony: kept out of the crossmap corpus and
    counted separately. Columns by x0: identifier < 125, display format 290-435."""
    import pdfplumber
    IDW = re.compile(r"^[A-Z]{3}[UD]-\d+$")
    rows, cur = [], None
    with pdfplumber.open(path) as pdf:
        for pno in range(438, 468):
            lines = {}
            for w in pdf.pages[pno - 1].extract_words():
                lines.setdefault(round(w["top"]), []).append(w)
            for top in sorted(lines):
                ws = lines[top]
                text = " ".join(w["text"] for w in ws)
                if IDW.match(ws[0]["text"]) and ws[0]["x0"] < 125:
                    if cur: rows.append(cur)
                    cur = dict(source="ICAO Doc 4444 (PANS-ATM) Appendix 5 CPDLC", page=pno, element=ws[0]["text"],
                               speaker="ATC (uplink)" if ws[0]["text"][3] == "U" else "Pilot (downlink)", phrase="")
                elif re.match(r"^(Table|Message|Format|element|identifier|\d+\.\s|Appendix)", text):
                    if cur: rows.append(cur)
                    cur = None
                    continue
                if cur:
                    fmt = [w["text"] for w in ws if 290 <= w["x0"] < 435]
                    cur["phrase"] = (cur["phrase"] + " " + " ".join(fmt)).strip()
            if cur: rows.append(cur)
            cur = None
    df = pd.DataFrame(rows)
    df = df[df.phrase.str.len() > 0].drop_duplicates("element")
    df["is_emergency"] = df.element.str.startswith("EMG") | df.phrase.str.contains(EMERGENCY)
    df["is_gnss"] = df.phrase.str.contains(GNSS)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = doc4444(SRC / "doc_4444_pans_atm.pdf")
    d.to_csv(OUT / "doc4444_ch12.csv", index=False)
    print("doc4444 rows", len(d), "sections", d.section.nunique(), d.speaker.value_counts().to_dict(),
          "markers", d.marker.value_counts().to_dict(), "emergency", int(d.is_emergency.sum()), "gnss", int(d.is_gnss.sum()))
    print(d.groupby(d.section.str[:6]).size().to_dict())
    o = doc4444_other(SRC / "doc_4444_pans_atm.pdf")
    o.to_csv(OUT / "doc4444_other.csv", index=False)
    print("doc4444 other rows", len(o), o.source.value_counts().to_dict(), "emergency", int(o.is_emergency.sum()),
          "gnss", int(o.is_gnss.sum()))
    c = doc4444_cpdlc(SRC / "doc_4444_pans_atm.pdf")
    c.to_csv(OUT / "doc4444_cpdlc.csv", index=False)
    print("doc4444 cpdlc rows", len(c), c.speaker.value_counts().to_dict(), "emergency", int(c.is_emergency.sum()),
          "gnss", int(c.is_gnss.sum()))
    e = doc9432(SRC / "ICAO-Doc-9432.pdf")
    e.to_csv(OUT / "doc9432_dialogues.csv", index=False)
    print("doc9432 rows", len(e), e.speaker.value_counts().to_dict(), "emergency", int(e.is_emergency.sum()),
          "gnss", int(e.is_gnss.sum()))


if __name__ == "__main__":
    sys.exit(main())
