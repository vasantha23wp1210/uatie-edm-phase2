#!/usr/bin/env python3
"""Writes the slot test set (items.csv): 60 ATC-style queries written by Devin with stated slot values, each paired
with a fixed crossmap phrase and the gold value of every slot in that phrase (empty = the query states no value,
so the slot must stay bracketed). Split 20 dev / 40 test by seed 42. Written and frozen before any scoring.
Run:  python phase2/slots/testset/build_items.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parents[1]
sys.path.insert(0, str(P2 / "slots"))
from fill import SLOT, is_slot  # noqa: E402

SEED, N_DEV = 42, 20
# (crossmap row, query text, gold value per slot of the phrase in order; None = not stated in the query)
ITEMS = [
    (2, "Speedbird 214 told to climb via the SID to FL120 but levelled at FL100", ["FL120"]),
    (2, "after departure we were cleared to climb via SID to 6000 feet", ["6000"]),
    (2, "climb via SID clearance was misunderstood by the crew", [None]),
    (2, "UAL881 was cleared from FL240 to FL280 on the SID and stopped at FL260", ["FL280"]),
    (111, "ATC told us to maintain FL330 until 1415Z because of crossing traffic", ["FL330", "1415"]),
    (111, "instructed to maintain 5000 ft until 0930 zulu then expect descent", ["5000", "0930"]),
    (111, "maintain flight level 90 until advised, no time given", ["FL90", None]),
    (46, "crew requested change to 128.725 to talk to company", ["128.725"]),
    (46, "pilot asked to change frequency but the controller was busy", [None]),
    (303, "after landing runway 28R contact ground 121.8", ["121.8"]),
    (303, "told to contact ground on 121.65 but we dialled 121.56", ["121.65"]),
    (303, "we were never told to contact ground", [None]),
    (49, "we were monitoring 121.5 when the call came", ["121.5"]),
    (49, "aircraft reported monitoring guard 121.5 and tower 118.3", [None]),
    (153, "cleared visual approach runway 27L but lined up on 27R", ["27L"]),
    (153, "N4521T cleared visual approach rwy 16 at night", ["16"]),
    (153, "the visual approach clearance was issued late", [None]),
    (146, "cleared ILS approach runway 09 at KBOS in heavy rain", ["09"]),
    (146, "Ryanair 45KP cleared approach runway 24, descending through 3000 ft", ["24"]),
    (146, "approach clearance given without a runway assignment", [None]),
    (65, "RVR runway 33 not reported during the low visibility approach", ["33"]),
    (418, "ILS runway 04 left localiser frequency is 110.3 according to the ATIS", ["04", "110.3"]),
    (418, "the localiser frequency for runway 22 was wrong on the chart", ["22", None]),
    (97, "Speedbird 123 cleared to London Heathrow via the filed route", ["SPEEDBIRD 123"]),
    (97, "DAL1182 cleared to KATL, flight planned route, climb FL330", ["DAL1182"]),
    (97, "the clearance was read back by the wrong aircraft", [None]),
    (317, "tower requested release of N123AB for departure runway 30", ["N123AB"]),
    (317, "request release of Easy 52LH from the other unit", ["EASY 52LH"]),
    (319, "we asked if American 2231 was released for climb", ["AMERICAN 2231"]),
    (318, "Lufthansa 400 released at 1520Z with a climb restriction", ["LUFTHANSA 400", "1520"]),
    (318, "SWA2290 released, no release time given", ["SWA2290", None]),
    (339, "for identification turn left heading 270 was issued twice", ["270"]),
    (339, "radar asked us to turn right heading 045 for identification", ["045"]),
    (346, "continue heading 090 after passing the VOR", ["090"]),
    (346, "pilot continued the assigned heading but drifted north", [None]),
    (348, "ATC said fly heading 180 vectors ILS runway 36", ["180"]),
    (348, "we were told to fly hdg 310 then descend to 4000 feet", ["310"]),
    (405, "VFR traffic requested 5 nm final for runway 26", ["5"]),
    (405, "pilot requested a long final for training", [None]),
    (425, "go-around initiated at 2 nm from touchdown after unstable approach", ["2"]),
    (425, "the aircraft was 4.5 nautical miles from touchdown when the gear warning sounded", ["4.5"]),
    (422, "approach instructions will be terminated at 3 nm from touchdown on the PAR", ["3"]),
    (413, "maintain 3000 feet until glide path interception, ILS 27", ["3000"]),
    (413, "we descended below 2500 ft before intercepting the glide path although told to maintain it", ["2500"]),
    (413, "glide path intercepted late, no altitude restriction was given", [None]),
    (470, "squawk 7700 selected after the engine fire warning", ["7700"]),
    (470, "ATC asked us to squawk 4521 but we set 4512", ["4521"]),
    (470, "transponder code was not set before departure", [None]),
    (469, "for departure squawk 6213 was in the PDC", ["6213"]),
    (471, "reset squawk 2000 after the transponder failure", ["2000"]),
    (27, "crew requested descent at 1245z to avoid turbulence", ["1245"]),
    (27, "we requested descent early because of a medical issue", [None]),
    (147, "cleared to commence approach at 0815 utc after holding", ["0815"]),
    (122, "expect clearance at 1700Z due to flow control", ["1700"]),
    (42, "contact London Control 132.6 now", [None, "132.6"]),
    (42, "told to contact Boston Center on 133.45 but no answer", [None, "133.45"]),
    (42, "contact departure was given without a frequency", [None, None]),
    (2, "climb via SID to FL070 then FL150 after the restriction was cancelled", [None]),
    (348, "fly heading 250, then turn right heading 280 to intercept", [None]),
    (470, "squawk 7600 for radio failure then squawk 7700", [None]),
]


def main():
    cm = pd.read_csv(P2 / "phraseology" / "out" / "CAA_ICAO_crossmap_v2.csv")
    rows = []
    for k, (row, q, gold) in enumerate(ITEMS, 1):
        phrase = str(cm.Phraseology.iat[row])
        names = [m.group(2) for m in SLOT.finditer(phrase) if is_slot(m.group(2))]
        assert len(names) == len(gold), (k, phrase, names, gold)
        rows.append({"item_id": f"S{k:03d}", "query_text": q, "phrase_row": row, "phraseology": phrase,
                     "slots": json.dumps(names), "gold": json.dumps(gold)})
    ids = [r["item_id"] for r in rows]
    random.Random(SEED).shuffle(ids)
    dev = set(ids[:N_DEV])
    d = pd.DataFrame(rows).assign(split=lambda x: x.item_id.map(lambda i: "dev" if i in dev else "test"))
    d.to_csv(HERE / "items.csv", index=False, encoding="utf-8")
    print(d.split.value_counts().to_dict(), sum(len(json.loads(g)) for g in d.gold), "slots")


if __name__ == "__main__":
    main()
