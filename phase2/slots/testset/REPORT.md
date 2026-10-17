# Slot filling on a slot test set (queries that state values)

The 100 gold retrieval queries state almost no slot values (Task 6: 0 of 91 filled, 0 wrong fills; the slots stay bracketed), so they cannot measure slot filling. This set measures it directly.

## Method

- `items.csv`: 60 ATC-style queries **written by Devin** (not ASRS text), each paired with a fixed crossmap phrase and the gold value of every slot in it (empty = the query states none; the slot must stay bracketed). Includes queries with two values of one type. Split 20 dev / 40 test (seed 42), frozen before scoring (SHA-256 `7d681ea0fc8e5d3b...`).
- Phrase is fixed per item, so this isolates slot filling from phrase matching.
- `ner`: the Task 6 filler unchanged (spaCy NER on the query + fixed regexes; fill only if exactly one value).
- `ner_rules`: adds two deterministic patterns written from the 4 dev errors only (`rules.py`): flight levels (`FL350`, `flight level 90`) and callsigns (`UAL881`, `N123AB`, `Speedbird 123`); a pattern match replaces an overlapping NER entity.
- Normalisation for comparison (fixed in `evaluate.py`): upper case, unit words and punctuation dropped.

## Numbers

| Method | Split | Slots with a value | Correct | Wrong fills | Missed | Correctly left empty | Precision | Recall | Items fully correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ner` | dev | 17 | 13 | 1 | 3 | 8 | 0.9286 | 0.7647 | 16 of 20 |
| `ner` | test | 31 | 21 | 0 | 10 | 14 | 1.0 | 0.6774 | 30 of 40 |
| `ner_rules` | dev | 17 | 16 | 0 | 1 | 8 | 1.0 | 0.9412 | 19 of 20 |
| `ner_rules` | test | 31 | 27 | 0 | 4 | 14 | 1.0 | 0.871 | 36 of 40 |

Test, chosen method: 27 of 31 stated values filled correctly, 0 wrong fills, 36 of 40 items fully correct.

## Failures and limits

Test items still wrong:

- S001 `Speedbird 214 told to climb via the SID to FL120 but levelled at FL100`: gold ["FL120"], got [null] (["ambiguous (FLIGHT_LEVEL: 2 values)"]).
- S008 `crew requested change to 128.725 to talk to company`: gold ["128.725"], got [null] (["no_entity"]).
- S011 `told to contact ground on 121.65 but we dialled 121.56`: gold ["121.65"], got [null] (["ambiguous (FREQUENCY: 2 values)"]).
- S015 `cleared visual approach runway 27L but lined up on 27R`: gold ["27L"], got [null] (["no_entity"]).

- Two values of one type (e.g. cleared level and level reached) are left bracketed by design: the filler never guesses, so recall drops but no wrong value is filled.
- The same author wrote the queries and the rules, and the set is small (40 test items); the numbers show the filler works when values are stated, not how often real queries state them.
- Runway and some frequency forms are missed when the NER model does not tag them; no rule was added for them because no dev item failed on them.
- The Task 6 numbers on the 100 gold queries are unchanged; `rules.py` is not used by `fill.py`.

## Multi-value round (dev only; frozen before a fresh test set)

- `dev_multi.csv`: 12 new dev items written by Devin before the rules (10 with two or three values of one type, plus runway/frequency formats and one empty item), SHA-256 `f80c4d6b1208f8d2...`.
- `ner_rules_roles`: `rules.entities_v2` adds runway designators (`27L`), VHF frequencies (`128.725`) and altitudes in feet; `roles.py` resolves a slot with several values by keeping values stated before a deviation cue (but, stopped at, dialled, instead of, ...) and not directly after "from", and fills only if exactly one remains. Assumption: the query states the instructed value before the deviation.
- The altitude pattern was added after `roles` gave one wrong fill on dev item D02 (NER missed `4000 feet`).

| Method | Items | Slots with a value | Correct | Wrong fills | Missed | Precision | Recall | Items fully correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ner_rules` | dev | 17 | 16 | 0 | 1 | 1.0 | 0.9412 | 19 of 20 |
| `ner_rules` | dev_multi | 13 | 3 | 1 | 9 | 0.75 | 0.2308 | 2 of 12 |
| `ner_rules_roles` | dev | 17 | 17 | 0 | 0 | 1.0 | 1.0 | 20 of 20 |
| `ner_rules_roles` | dev_multi | 13 | 13 | 0 | 0 | 1.0 | 1.0 | 12 of 12 |

These are dev numbers from items written by the rule author, so they are optimistic by construction. The 40-item test set is not re-scored with these rules because its failures were already seen. The files are hashed in `freeze.json`; `fresh_eval.py` scores a second-author set once and refuses to run if any hashed file changed. No fresh test result exists yet.
