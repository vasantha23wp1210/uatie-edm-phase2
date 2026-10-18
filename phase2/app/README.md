# Task 8: Streamlit app (Phase 1 / Phase 2 toggle)

Needs the outputs of Tasks 1-7 and a running Ollama with `llama3.2:3b`.

Run: `streamlit run phase2/app/app.py -- --phase1 ../uatie-edm-phase1`

Phase 2 mode shows the dev-selected retrieval (FAISS; cross-encoder rerank optional), `intent_v2` per result, NER entities, the filled phraseology and the generated answer with clickable citations.
