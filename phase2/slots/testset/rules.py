"""Deterministic entity patterns added after the dev errors (FL tokens tagged AIRPORT or missed; callsigns tagged
AIRCRAFT_TYPE or missed). A pattern match replaces any NER entity that overlaps it; other NER entities are kept."""
import re

FLIGHT_LEVEL = re.compile(r"\b(?:FL ?\d{2,3}|flight level \d{2,3})\b", re.I)
NOT_TELEPHONY = {"runway", "rwy", "heading", "hdg", "flight", "level", "squawk", "code", "contact", "tower", "ground",
                 "approach", "center", "centre", "control", "departure", "gate", "taxiway", "ils", "vor", "sid",
                 "star", "final", "radial", "track", "frequency", "the", "at", "to", "via", "and", "on", "of", "fl"}
CALLSIGN = re.compile(r"\b(?:[A-Z]{3}\d{1,4}[A-Z]{0,2}|N\d{1,5}[A-Z]{0,2}|[A-Z][a-z]+ \d{1,4}[A-Z]{0,2})\b")


def entities(text, ner):
    found = []
    for m in FLIGHT_LEVEL.finditer(text):
        found.append((m.span(), m.group(0), "FLIGHT_LEVEL"))
    for m in CALLSIGN.finditer(text):
        word = m.group(0).split(" ")[0].lower()
        if " " in m.group(0) and word in NOT_TELEPHONY:
            continue
        if any(s <= m.start() < e for (s, e), _, _ in found):
            continue
        found.append((m.span(), m.group(0), "CALLSIGN"))
    keep = []
    for t, lab in ner:
        i = text.find(t)
        span = (i, i + len(t)) if i >= 0 else (-1, -1)
        if not any(s < span[1] and span[0] < e for (s, e), _, _ in found):
            keep.append((t, lab))
    return keep + [(t, lab) for _, t, lab in found]

# Added for the role-aware round (dev items only, incl. dev_multi.csv): runway designators, VHF frequencies,
# altitudes in feet (NER missed one of two altitudes in dev item D02).
RUNWAY = re.compile(r"(?<=\brunway )\d{2}[LRC]?\b|\b\d{2}[LRC]\b", re.I)
FREQ = re.compile(r"\b1[1-3]\d\.\d{1,3}\b")
ALT = re.compile(r"\b\d{1,2},?\d{3} ?(?:feet|ft)\b", re.I)


def entities_v2(text, ner):
    found = [(m.span(), m.group(0).upper(), "RUNWAY") for m in RUNWAY.finditer(text)]
    found += [(m.span(), m.group(0), "FREQUENCY") for m in FREQ.finditer(text)]
    found += [(m.span(), m.group(0), "ALTITUDE") for m in ALT.finditer(text)]
    base = entities(text, ner)
    keep = []
    for t, lab in base:
        i = text.find(t)
        if i >= 0 and any(s < i + len(t) and i < e for (s, e), _, _ in found):
            continue
        keep.append((t, lab))
    return keep + [(t, lab) for _, t, lab in found]
