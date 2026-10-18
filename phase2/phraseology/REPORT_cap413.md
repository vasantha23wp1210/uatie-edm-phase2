# CAP 413 Chapter 8 (UK CAA) emergency phraseology

From `cap413.py`; `CAP413_edition24.pdf` (SHA-256 `576b1c98b7f8…`), chapter pages [227, 236].

- Example transmissions extracted: 23 (23 distinct); 9 text blocks without a callsign or station dropped (listed in the CSV with kept=False; they include a prose table and one real phrase, 'Ballistic recovery system deployed').
- Matching the emergency regex used for the ICAO count: 13 rows, 13 distinct. The other Ch. 8 examples are emergency-context exchanges (e.g. ejecting, stop transmitting) that do not contain the regex words.
- ICAO emergency radiotelephony rows (plan target >= 25, ICAO sources): 19, unchanged, MISSED.
- ICAO + CAP 413 (regex, distinct): 32. This is a different, wider count than the target as written and is reported separately.

## Limits

- Layout-based extraction (x positions); a transmission split across a page break counts as two.
- CAP 413 is not in the ICAO matching corpus, so the crossmap and CAA-unique rate are unchanged.
