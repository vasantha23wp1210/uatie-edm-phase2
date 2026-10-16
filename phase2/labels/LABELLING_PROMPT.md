# Labelling method — LLM-judged (Claude)

All three files were labelled by Claude (Claude Opus 5.5, claude.ai, 12–13 Oct 2026).
Every row is marked `LLM-judged (Claude)` in its notes column. No rows have been
human-reviewed yet; change a row's marker to `human` when you check or correct it.

Each row was read and judged individually. No keyword rules, embeddings or retriever
scores were used to assign labels. Scripts were used only to write the files and to
check that IDs, text and row order are unchanged.

The prompt given to Claude was the user's request to label/judge/correct the attached
files, plus the guidance below (forwarded from Devin): mark rows, keep every row in
order, do not edit ID or text columns, use 0/1/2 for retrieval relevance.

---

## 1. retrieval_to_judge.csv → retrieval_judged.csv (975 pairs, 100 queries)

Note: Claude did not have the exact wording of the earlier 0/1/2 instructions used
for the Q021–Q100 test labels. The scale below is the one Claude applied here; check
it matches the earlier instructions before mixing qrels.

Each narrative is judged against the query's `information_need` (the `query_text`
gives context):

- **2 — Relevant.** The narrative describes the situation the information need asks
  for: its main event and, where the need names one, the right phase or cause.
- **1 — Partially relevant.** The narrative shares a key element but misses another,
  e.g. a runway change during taxi when the need is a runway change on final; a
  readback error that is not about a runway assignment; general non-standard
  phraseology when the need is phraseology for speed control.
- **0 — Not relevant.** Shares only surface words (e.g. "go-around", "confusion",
  "frequency") or a different event entirely.

Result: 359 × 2, 282 × 1, 334 × 0. 19 queries have no pair graded 2 (Q004, Q008,
Q009, Q011, Q014, Q017, Q019, Q021, Q025, Q026, Q027, Q030, Q034, Q053, Q055, Q064,
Q068, Q074, Q100) — the candidate lists for these did not contain a narrative that
fully meets the need.

## 2. intent_to_label.csv → intent_labeled.csv (400 rows)

One label per summary from `allowed_labels.txt`. The labels have no official
definitions, so Claude applied these conventions:

- **Emergency_Event** — declared emergency, diversion or return caused by a failure;
  fire, smoke, fumes; engine failure; hijacking.
- **Maintenance_Error** — equipment malfunction or defect handled without emergency
  action, and maintenance mistakes. (Covers plain malfunctions, not only human
  maintenance error.)
- **Airspace_Violation** — NMAC, TCAS RA, loss of separation, TFR / Class B
  incursions, drone or balloon conflicts.
- **ATC_Operational_Error** — the report attributes the event to the controller.
- **Procedural_Non_Compliance** — altitude, heading, SID/STAR deviations, missed
  restrictions, low-altitude alerts, SOP, hazmat or checklist lapses.
- **Crew_Resource_Management** — the stated cause is fatigue, distraction, workload or
  crew coordination.
- **Communication_Failure** — lost comms, radio/frequency interference, misheard or
  misinterpreted clearances.
- **Weather_Deviation** — weather, turbulence, wake turbulence, wind/crosswind
  excursions, icing, braking action, volcanic ash.
- **Runway_Incursion** — entering or crossing a runway without clearance, or a vehicle
  on the runway.
- **Ground_Incident** — taxi, ramp, pushback and parking events.
- **IMC_Spatial_Disorientation** — VFR into IMC, disorientation, visual illusions.
- **Unclassified** — no good fit: 5G / GPS interference, chart errors, passenger
  mask issues, lasers, UAS-only events.

About 35 borderline rows have a short explanation in `intent_ambiguity_notes.csv`
(kept out of `notes` so that column holds only the marker).

Distribution: Procedural_Non_Compliance 71, Maintenance_Error 68, Emergency_Event 57,
Airspace_Violation 32, Unclassified 30, Ground_Incident 29, Weather_Deviation 25,
Crew_Resource_Management 25, ATC_Operational_Error 22, Communication_Failure 17,
Runway_Incursion 16, IMC_Spatial_Disorientation 8.

Caveat for the report: if these labels are used for the 201 test rows, the ≥85%
accuracy target measures agreement with Claude, not with a human annotator.

## 3. ner_to_correct.csv → ner_corrected.csv (260 sentences)

`corrected` is the pre-annotation with errors fixed, in the same `[span](LABEL)`
format and using only the existing labels (AIRCRAFT_TYPE, AIRPORT, RUNWAY,
ALTITUDE, FLIGHT_LEVEL, FREQUENCY, HEADING). Removing the brackets from `corrected`
gives back `text` exactly; this was checked for every row. 107 rows changed.

Conventions applied:

- **AIRCRAFT_TYPE** spans the full designator including variant suffix (`b767-400`,
  `b737-800`, `b787-9`, `pa32t`). Missed types were added (`l39`, `lr45`,
  `cessna 172`, `boeing 737`, `emb-145`, `ce-560xl`, `citabria`, `mooney m20` …).
  Generic words (`gulfstream`, `emb jet`, `cessna twin`) are not tagged.
- **AIRPORT** includes bare airport codes used as a place (`lax`, `teb`, `dfw`,
  `rjaa`, `mmmx`, anonymised `zzz` / `uuu`). Not tagged: ATC facilities (`zhu`,
  `sct`, `n90`, `roa tracon`, `pdx tower`), airspace (`slc class b`,
  `msy mode c veil`), navaids, and generic phrases (`vfr arpt`, `dep arpt` — removed).
- **FREQUENCY** spans on FAR section numbers (`far 121.563` etc.) were removed.
- **ALTITUDE** spans that were really distances along a runway were removed
  (S020, S126). Altitudes written without a unit were added (S238 `27;000`,
  `29;500`); a range is one span (S250 `700-1000 ft`).
- **HEADING** added where a heading is stated without the word "heading"
  (S062 `328 degrees`, S214 `170`, `270`, S259 `280 degree heading`).
- **RUNWAY** added for bare designators (`28l`, `19l`, `28r`, `17l/35r`); spans
  extended to cover the full designator (`runways 28l and r`, `rwy 1-19`).
