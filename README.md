# Appliance Knowledge Assistant — Dual-Mode RAG Chatbot (Prototype)

A working prototype of a Retrieval-Augmented Generation (RAG) chatbot for an
electronics company, built for an internship pitch. It has two modes:

- **Customer Support** — answers installation/troubleshooting questions using
  product manuals only.
- **Internal Engineering** (password-gated demo) — answers component/spec
  questions using internal engineering standards only.

Both modes are strictly grounded: the bot only answers from the retrieved
document excerpts, cites its source, and explicitly says "I don't know" when
the answer isn't in the documents, instead of guessing.

## How it works (pipeline)

```
data/*.txt/.pdf/.docx
        │
        ▼
   ingest.py        -> loads + cleans + splits documents into overlapping chunks
        │
        ▼
   vectorstore.py    -> TF-IDF vectorizes chunks, builds a searchable index
        │
        ▼
   build_index.py    -> runs the above once, saves index/*.pkl
        │
        ▼
   rag_pipeline.py   -> at query time: retrieve top-k relevant chunks
                         -> if best match score < threshold: say "not found" (no guessing)
                         -> else: send chunks + strict system prompt to the LLM
                         -> LLM must answer ONLY from context + cite source
        │
        ▼
   app.py (Streamlit UI) -> chat interface, mode switcher, source viewer
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional — works with zero config too)
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY or OPENAI_API_KEY if you want full
# generated answers. Leave MODEL_PROVIDER=none to run retrieval-only
# (returns the best-matching manual excerpt, no API key needed).

# 3. Build the search index from the sample documents in data/
python build_index.py

# 4. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

Default demo password for "Internal Engineering" mode: `employee123`
(change it in `.env` via `INTERNAL_MODE_PASSWORD`).

## Trying it with your own documents

Drop `.txt`, `.pdf`, or `.docx` files into:
- `data/customer_docs/` — for anything customers should be able to ask about (manuals, FAQs)
- `data/internal_docs/` — for internal-only content (standards, specs)

Then re-run `python build_index.py` to rebuild the search index.

## Anti-hallucination guardrails (the core "no misinformation" feature)

1. **Relevance threshold** — if nothing in the documents is actually
   relevant to the question, the bot refuses to answer instead of guessing
   (`MIN_RELEVANCE_SCORE` in `src/config.py`).
2. **Strict system prompt** — instructs the model to answer only from the
   provided context and to say "I don't have this information" when unsure
   (see `SYSTEM_PROMPT` in `src/rag_pipeline.py`).
3. **Mandatory citations** — every generated answer must name its source
   document, so a human can verify it.
4. **Separate indexes per mode** — the customer bot physically cannot see
   internal engineering documents and vice versa (two separate `.pkl`
   index files, loaded independently).
5. **Retrieval-only fallback** — if no LLM key is configured at all, the app
   still works safely: it shows the raw matching excerpt instead of
   generating text, so it's never possible to "hallucinate" even in this mode.

## Upgrading retrieval (semantic search)

This prototype uses TF-IDF (keyword-based) retrieval via scikit-learn so it
runs instantly with no downloads or GPU. For production, swap
`src/vectorstore.py`'s TF-IDF vectorizer for a real embedding model, e.g.:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
```

paired with a proper vector index (FAISS, Chroma, or Pinecone) for
similarity search. This gives true semantic matching (understanding
paraphrased questions), not just keyword overlap. The rest of the pipeline
(`rag_pipeline.py`, `app.py`) doesn't need to change — only the internals of
`VectorStore.build_index()` and `VectorStore.query()`.

## Project structure

```
rag_chatbot/
├── app.py                  # Streamlit UI (main entry point)
├── build_index.py          # Script to (re)build search indexes
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            # All settings (model provider, thresholds, paths)
│   ├── ingest.py             # Document loading + chunking
│   ├── vectorstore.py        # TF-IDF index build/query
│   └── rag_pipeline.py       # Retrieval + guarded LLM generation
├── data/
│   ├── customer_docs/        # Sample product manuals (fan, TV)
│   └── internal_docs/        # Sample engineering standards
└── index/                    # Generated .pkl search indexes (created by build_index.py)
```

## Notes for the pitch / demo

- This is a fully working prototype, not just a mockup — try asking:
  - Customer mode: *"my fan is making a clicking noise"* or *"how do I wall-mount the TV"*
  - Internal mode: *"which capacitor should I use for a fan motor circuit"* or *"what's the fuse sizing rule"*
  - Try an off-topic question in either mode (e.g. *"what's the weather today"*) to see the "I don't know" guardrail in action — this is the anti-hallucination behavior in practice.
- The `INTERNAL_MODE_PASSWORD` gate is a placeholder for demo purposes only —
  a real deployment would use the company's SSO/employee login system.
