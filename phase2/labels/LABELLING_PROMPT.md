# Labelling method — LLM-judged (Claude)

This document covers five tasks, run with Claude (Claude Opus 5.5, claude.ai) from 12 to 16 Oct 2026.
Sections 1, 2, 3 and 5 are dataset labels made by Claude: every row's notes column starts with the
marker `LLM-judged (Claude)`, sometimes followed by a longer note. The 400 intent rows of section 2
were relabelled on 16 Oct under the section 5 rules (marker `LLM-judged (Claude), section 5 relabel`;
the first pass is kept as `intent_labelled_claude_v1.csv`). No dataset row has been human-reviewed
yet; change a row's marker to `human` when you check or correct it.
Section 4 is different: it rates generated answers, not dataset rows. The ratings are the user's;
Claude reviewed them and rewrote the notes.

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

## 4. answers_to_rate.csv → answers_rated.csv (30 generated answers)

Instruction given: read `query_text`, then the `summary`, `key_hazards` and
`recommended_phraseology` columns, and rate the answer with a whole number:

- **5** — accurate and useful for the query.
- **4** — accurate, but there are small gaps.
- **3** — partly useful, or vague.
- **2** — mostly unhelpful, or has a factual error.
- **1** — wrong or misleading.

`notes` (optional): note any claim judged wrong.

Only those three columns were scored. Risk level and citations were not; where
they matter, notes mention them as "not scored". A recommended phrase that does
not apply to the query was treated as reducing usefulness (usually 3); a phrase
that would actively mislead (e.g. ROGER for an ambiguous clearance, CLEARED TO
LAND for a vehicle on the runway) was treated as misleading (2). Every note starts
with `LLM-judged (Claude).` and ends with the citation counts, marked not scored.

Citations were checked against `asrs_for_fusion.csv`. The `ASRS_XXXXXXX` IDs are
not `event_id` values or plain row numbers; they map to rows by a drifting offset
(about +1,993). Claude rebuilt the mapping from the 1,292 IDs whose text was already
known from the other files (leave-one-out check: 1,291 / 1,292 correct) and used it
to resolve all 144 citations, each to a single report. Each cited report was graded
2 (on-topic), 1 (partial) or 0 (off-topic) for its query; see
`answers_citations_checked.csv`.

Result: 1 × 1, 9 × 2, 18 × 3, 2 × 4, none at 5 (mean 2.7).
Citations: 57 on-topic, 48 partial, 39 off-topic.

## 5. to_label_batch1–4.csv → labeled_batch1–4.csv (2,000 rows, T0001–T2000)

Same task, label set and conventions as section 2 (`intent_to_label.csv`). Each
summary was read and given one label from `allowed_labels.txt`; `notes` holds only
the marker `LLM-judged (Claude)`. No separate ambiguity file was produced for these
batches. None of the 2,000 record IDs overlap the earlier 400.

Applied consistently with section 2 where the summary leaves room for doubt:

- A failure followed by a declared emergency, diversion, return or engine shutdown →
  Emergency_Event; the same failure with the flight continuing normally →
  Maintenance_Error.
- Loss of control on landing or taxi (ground loops, excursions) with no weather cause
  → Ground_Incident; with gusts or crosswind → Weather_Deviation.
- A cause stated as distraction, fatigue, workload or crew coordination →
  Crew_Resource_Management, even when the outcome was an altitude or track deviation.
- Hazmat / dangerous-goods paperwork and loading errors, MEL and checklist lapses →
  Procedural_Non_Compliance.
- UAS flown into controlled airspace or a TFR → Airspace_Violation; other UAS-only
  events, GPS jamming, chart/database errors, COVID / mask issues, lasers and
  accident-database records with no clear fit → Unclassified.

Distribution: Maintenance_Error 347, Emergency_Event 309, Procedural_Non_Compliance
308, Unclassified 213, Ground_Incident 158, Airspace_Violation 149,
Crew_Resource_Management 121, Communication_Failure 102, Weather_Deviation 100,
ATC_Operational_Error 99, Runway_Incursion 78, IMC_Spatial_Disorientation 16.

Checks: every row present in original order; `label_id`, `record_id`, `source` and
`summary` unchanged; every label in the allowed set; random spot-checks confirmed
labels are aligned with their rows.
