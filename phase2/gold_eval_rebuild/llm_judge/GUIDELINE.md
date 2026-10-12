# Relevance judging guideline (fixed before any judging)

Applies to the LLM judge (Devin) and to the human spot-check. Each pair is judged only from
`query_text`, `information_need` and `narrative`. No system, rank, score or other label is visible.

- **2 = highly relevant.** The narrative describes the event the query and information need ask for, as a central part of the report.
- **1 = partially relevant.** It is clearly related but does not fully meet the need: the right hazard type in a different
  flight phase or context, the event mentioned only in passing, or a closely related event (e.g. a clearance confusion
  but not specifically a readback error).
- **0 = not relevant.** The topic is different, or the narrative only shares surface words with the query.

Rules: judge each pair independently. If a short narrative is ambiguous, prefer the lower grade.
Do not reward matching keywords that lack the matching meaning.
