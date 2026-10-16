# Task 6: slot filling

## Method

- Phrase per query: TF-IDF (1-2 grams) over CAA Phraseology + Circumstances + Section_Title of `phase2/phraseology/out/CAA_ICAO_crossmap_v2.csv`; top-1 phrase.
- Entities: Task 2 spaCy model (`phase2/ner/out/model`) on the query text plus the top-1 incident summary from `ce`, the primary Task 3 retrieval system (fixed by the Phase 2 plan).
- A slot is filled only if its type is mapped and exactly one distinct value of that type was found. Unmapped, missing and ambiguous slots stay bracketed and are flagged in `out/slot_fills.csv`.
- Parentheses holding alternatives (`(or LEFT)`) or a plural suffix (`(S)`) are wording, not slots.
- Slot-type map: `(level)` -> FLIGHT_LEVEL/ALTITUDE; `(altitude)` -> ALTITUDE; `(frequency)` -> FREQUENCY; `(aircraft call sign)` -> CALLSIGN; `(aircraft type)` -> AIRCRAFT_TYPE; `(aerodrome)` -> AIRPORT; `(name of aerodrome)` -> AIRPORT; `(airport)` -> AIRPORT; `RUNWAY (number)` -> RUNWAY; `HEADING (three digits)` -> HEADING.
- Slots with no NER label, filled by fixed regular expressions over the same text (exactly one distinct match required): `(time)`, `(distance)`, `(speed)`, `(code)`, `(ssr code)`, `(direction)`. Every other slot name is unmapped.

## Numbers

| Status | Queries |
|---|---:|
| incomplete | 61 |
| no_slots | 37 |
| complete | 2 |

Complete among queries whose phrase has slots: 3.17%. Slots filled: 5 of 91; 0 of the 5 filled values occur in the query text, the rest come from the top-1 incident summary.

Across the whole crossmap, 243 of 625 slot occurrences (168 distinct names) have a mapped type (NER entity or fixed regex); the rest (significant point, reason, ...) are unmapped and stay bracketed.

| Slot type : outcome | Count |
|---|---:|
| AIRCRAFT_TYPE:filled | 1 |
| AIRCRAFT_TYPE:no_entity | 1 |
| ALTITUDE:no_entity | 1 |
| CALLSIGN:no_entity | 3 |
| FLIGHT_LEVEL|ALTITUDE:filled | 1 |
| FLIGHT_LEVEL|ALTITUDE:no_entity | 3 |
| FREQUENCY:no_entity | 6 |
| REGEX:code:no_entity | 2 |
| REGEX:distance:no_entity | 3 |
| REGEX:speed:no_entity | 1 |
| RUNWAY:filled | 3 |
| RUNWAY:no_entity | 12 |
| unmapped:unmapped | 54 |

## Failures and limits

- NER labels are provisional (LLM-judged (Claude), pending user review); so are these fills.
- CALLSIGN slots can never fill from ASRS text: ASRS de-identifies callsigns (Task 2).
- Phrase choice is a lexical TF-IDF match, not judged for correctness here.
- Entities from the top-1 incident describe that incident, not necessarily the user's situation.

## 15 examples (seed 42)

| Query | Phrase | Filled | Status |
|---|---|---|---|
| Q001 late runway change during approach | REQUEST CHANGE TO (frequency) | REQUEST CHANGE TO (frequency) | incomplete |
| Q010 runway reassignment caused confusion on approach | CLEARED APPROACH [RUNWAY (number)] | CLEARED APPROACH [RUNWAY (number)] | incomplete |
| Q017 runway change while on short final with traffic ahead | EXPEDITE CROSSING RUNWAY (number) TRAFFIC (aircraft type) (distance) (MILES) FINAL | EXPEDITE CROSSING RUNWAY (number) TRAFFIC (aircraft type) (distance) (MILES) FINAL | incomplete |
| Q020 ATC issued circle-to-land to different runway late in approach | RUNWAY (number) CLEARED TO LAND | RUNWAY runway 22l CLEARED TO LAND | complete |
| Q021 incorrect readback of amended clearance | RECLEARED (amended clearance details) [REST OF CLEARANCE UNCHANGED] | RECLEARED (amended clearance details) [REST OF CLEARANCE UNCHANGED] | incomplete |
| Q026 pilot used non standard phraseology requesting runway change | (standard departure name and number) DEPARTURE | (standard departure name and number) DEPARTURE | incomplete |
| Q039 missed handoff resulting in loss of contact | IF YOU READ [manoeuvre instructions or SQUAWK (code or IDENT)] | IF YOU READ [manoeuvre instructions or SQUAWK (code or IDENT)] | incomplete |
| Q054 unstable non-precision approach with high workload | VECTORING FOR PRECISION APPROACH RUNWAY (number) | VECTORING FOR PRECISION APPROACH RUNWAY (number) | incomplete |
| Q059 unstable approach during circle-to-land manoeuvre | RUNWAY (number) CLEARED TO LAND | RUNWAY (number) CLEARED TO LAND | incomplete |
| Q063 landing clearance issued with aircraft still on runway | RUNWAY (number) CLEARED TO LAND | RUNWAY (number) CLEARED TO LAND | incomplete |
| Q066 runway incursion during low visibility operations | VISIBILITY (distance) (units) [direction] | VISIBILITY (distance) (units) [direction] | incomplete |
| Q086 smoke in cabin requiring emergency landing | EMERGENCY DESCENT (intentions) | EMERGENCY DESCENT (intentions) | incomplete |
| Q089 electrical failure causing loss of some instruments | IF YOU READ [manoeuvre instructions or SQUAWK (code or IDENT)] | IF YOU READ [manoeuvre instructions or SQUAWK (code or IDENT)] | incomplete |
| Q095 PF/PM role confusion during approach | CLEARED APPROACH [RUNWAY (number)] | CLEARED APPROACH [RUNWAY (number)] | incomplete |
| Q096 inadequate briefing led to misunderstanding of plan | REVERT TO FLIGHT PLAN CALL SIGN (call sign) [AT (significant point)] | REVERT TO FLIGHT PLAN CALL SIGN (call sign) [AT (significant point)] | incomplete |
