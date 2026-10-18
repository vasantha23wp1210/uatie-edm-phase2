"""Role-aware resolution for slots with several values of one type (e.g. cleared FL150, stopped at FL130).
Assumption, stated in the report: a query names the instructed value before the deviation. For an ambiguous slot,
keep only values that appear before the first deviation cue and are not directly preceded by "from"; fill only if
exactly one value remains, otherwise leave the slot bracketed. Cues were written from dev items only."""
import re

DEVIATION = re.compile(r"\b(?:but|instead of|however|stopped at|levell?ed (?:off )?at|dialled|selected|lined up on|"
                       r"aligned with|turned onto|read ?back as|descended to|showed)\b", re.I)
AMBIG = re.compile(r"ambiguous \(([A-Z_]+):")


def resolve(values, text):
    cut = DEVIATION.search(text)
    end = cut.start() if cut else len(text)
    keep = []
    for v in values:
        i = text.upper().find(v.upper())
        if 0 <= i < end and not re.search(r"\bfrom\s+$", text[max(0, i - 6):i], re.I):
            keep.append(v)
    return keep


def fill(F, phrase, ents, text):
    filled, slots = F.fill(phrase, ents, text)
    for s in slots:
        m = AMBIG.match(s["status"])
        if not m:
            continue
        t = m.group(1)
        vals = sorted({v for v, lab in ents if lab == t})
        keep = resolve(vals, text)
        if len(keep) == 1:
            ents = [(v, lab) for v, lab in ents if lab != t or v == keep[0]]
    return F.fill(phrase, ents, text)
