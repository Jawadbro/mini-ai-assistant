# Mini AI Assistant

A mini AI assistant supporting **knowledge ingestion (RAG)**, **chat**, **session
memory**, and **tool calling**, built with FastAPI, OpenAI, and ChromaDB.

Built for the Studio Butterfly AI Developer take-home assignment.

- Architecture diagram: [`diagrams/architecture.md`](diagrams/architecture.md)
- Demo frontend: served at `/ui` once the server is running

---

## Features

| Requirement | Implementation |
|---|---|
| Knowledge ingestion | `POST /ingest` — PDF/TXT/MD → chunk → OpenAI embeddings → FAISS |
| Chat | `POST /chat` — answers from the knowledge base, tools, or directly |
| Context memory | In-process per-`session_id` conversation history, fed back into every LLM call |
| Tool calling | `get_order_status` (orders.json) and `search_product` (products.json) via OpenAI native function calling |
| Pipeline | Single-agent-with-tools: one LLM decides retrieval vs. tool vs. direct response |

---

## Project structure

```
mini-ai-assistant/
├── app/
│   ├── main.py                  # FastAPI app + routes
│   ├── config.py                # env vars, paths, tunables
│   ├── ingestion/
│   │   ├── loader.py            # PDF / TXT / MD -> raw text
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter
│   │   └── embedder.py          # OpenAI embeddings + ChromaDB persistence
│   ├── retrieval/
│   │   └── retriever.py         # similarity search + relevance threshold
│   ├── storage/
│   │   └── vector_store.py      # FAISS index + JSON metadata sidecar
│   ├── memory/
│   │   └── session_memory.py    # in-process session -> turns store
│   ├── tools/
│   │   ├── order_status.py      # Tool 1: order lookup
│   │   └── product_search.py    # Tool 2: product lookup
│   ├── agent/
│   │   ├── planner.py           # orchestration: routing + execution
│   │   └── prompts.py           # system prompt + templates
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── data/
│   ├── orders.json
│   ├── products.json
│   └── vectorstore/              # FAISS index + metadata.json (created at runtime)
├── frontend/
│   └── index.html                # minimal chat + upload + trace-panel UI
├── diagrams/
│   └── architecture.md           # Mermaid pipeline diagrams
├── tests/
│   ├── test_tools.py
│   └── test_chunker.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, an OpenAI API key.

```bash
git clone <your-repo-url>
cd mini-ai-assistant

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs (Swagger UI): http://localhost:8000/docs
- Demo UI: http://localhost:8000/ui
- Health check: http://localhost:8000/health

---

## Using it

### 1. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/your-doc.pdf"
```

Response:
```json
{
  "doc_id": "a1b2c3d4e5f6",
  "filename": "your-doc.pdf",
  "num_chunks": 14,
  "message": "Ingested 'your-doc.pdf' into the knowledge base as 14 chunks."
}
```

### 2. Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "Where is my order ORD001?"}'
```

Response includes which route was taken and what tool/retrieval data was used:
```json
{
  "session_id": "demo-1",
  "reply": "Order ORD001 has been shipped and is estimated to arrive on 2026-07-02.",
  "route": "tool_call",
  "sources": [],
  "tool_calls": [
    {
      "tool_name": "get_order_status",
      "tool_input": {"order_id": "ORD001"},
      "tool_output": {"order_id": "ORD001", "status": "Shipped", "estimated_delivery": "2026-07-02"}
    }
  ]
}
```

### 3. Memory example

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo-2", "message": "My name is John."}'

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo-2", "message": "What is my name?"}'
# -> "Your name is John."
```

### 4. Inspect / clear a session

```bash
curl http://localhost:8000/chat/demo-2
curl -X DELETE http://localhost:8000/chat/demo-2
```

---

## Design notes

### Ingestion pipeline
`loader.py` dispatches on file extension (`pypdf` for PDFs, plain read for
`.txt`/`.md`). `chunker.py` uses LangChain's `RecursiveCharacterTextSplitter`
(800 chars, 120 overlap) which tries paragraph → sentence → word boundaries
before falling back to a hard cut, so chunks stay semantically coherent.
`embedder.py` batch-embeds chunks with `text-embedding-3-small` and hands
them to `app/storage/vector_store.py`, which stores the vectors in a FAISS
`IndexFlatIP` index and keeps a parallel JSON sidecar (`metadata.json`) with
`source`/`doc_id`/`chunk_id`/original text for every vector position, so
answers can always be traced back to their origin. Both files persist to
`data/vectorstore/` and reload automatically on server restart.

FAISS was chosen over ChromaDB specifically because `faiss-cpu` ships
prebuilt wheels for all current CPython versions/platforms — Chroma's HNSW
dependency (`chroma-hnswlib`) frequently has no prebuilt wheel for newer
Python versions and falls back to compiling from source, which fails on any
machine without a C++ build toolchain (a real issue encountered when
building this project on Windows).

### Retrieval approach
Vectors are L2-normalized before insertion and search, so FAISS's inner
product search (`IndexFlatIP`) is mathematically equivalent to cosine
similarity — scores returned are in `[-1, 1]`, higher meaning more similar.
Results below `MIN_RELEVANT_SIMILARITY` (default `0.25`) are discarded —
this is what makes the "I couldn't find that information in the uploaded
documents" fallback trigger correctly on both an empty knowledge base and
an irrelevant query, instead of forcing the LLM to answer from whatever the
nearest (but unrelated) chunk happens to be.

### Memory implementation
A simple `session_id -> [{role, content}, ...]` store, kept in process
memory (`app/memory/session_memory.py`), capped at the most recent
`MAX_HISTORY_TURNS` turns. On every chat call the full recent history is
passed to the LLM alongside the new message — reference resolution ("my
name", "cheaper options", "that order") is handled by the LLM reading the
history naturally, rather than hand-written coreference logic. This is
swappable for Redis/Postgres behind the same two functions
(`add_turn`, `get_history`) without touching the rest of the codebase.

### Tool-calling strategy
**Single-agent-with-tools**: one OpenAI chat completion call is given all
three callable functions every turn —
`get_order_status`, `search_product`, and `search_knowledge_base` (RAG
retrieval is itself exposed as a tool, not special-cased outside the
function-calling loop). The model decides which, if any, to call based on
the message + conversation history. If it calls one or more tools, we
execute them, feed the JSON results back as `tool` messages, and make a
second completion call to produce the final grounded answer. If it calls
none, its first response is used directly (covers greetings, chit-chat, and
memory-only questions like "what's my name?").

This was chosen over a separate intent-classifier step because native
function calling is more robust to phrasing variation ("where's my
package ORD001" vs. "status of ORD001") and keeps routing + argument
extraction in one place instead of two.

### Prompt design
The system prompt (`app/agent/prompts.py`) explicitly:
- Names all three tools and when to use each.
- Forbids fabricating order/product data or document facts — the model
  must call a tool/retrieval function rather than guessing.
- Hard-codes the exact required fallback string for "not found in
  knowledge base."
- Instructs the model to resolve references using conversation history
  rather than asking the user to repeat themselves.

### Error handling
- Unsupported file extensions, empty/unparseable documents, and empty
  chunk sets are rejected with clear `400`/`422` responses at ingestion time.
- Embedding/vector-store failures return `502` rather than crashing.
- Missing `OPENAI_API_KEY` is caught before any LLM call and surfaced as a
  clear `500` rather than an opaque network error.
- Tool functions themselves never raise on bad input (missing/invalid
  order ID or product name) — they return a structured `{"error": ...}`
  payload that the LLM is instructed to relay honestly instead of
  papering over.

---

## Running tests

Offline unit tests (no API key / network needed) for the tools and chunker:

```bash
python tests/test_tools.py
python tests/test_chunker.py
```

---

## Deploying

A `Dockerfile` and `render.yaml` are included for deploying to
[Render](https://render.com) (free tier, no credit card needed to start).

### Option A: One-click blueprint (recommended)

1. Push this repo to GitHub.
2. In the Render dashboard: **New +** → **Blueprint** → connect your repo.
   Render reads `render.yaml` automatically and provisions the service.
3. When prompted, paste your `OPENAI_API_KEY` (it's marked `sync: false` in
   `render.yaml` so Render asks for it interactively rather than committing
   it to the repo).
4. Deploy. Once live, your app is at `https://<your-service-name>.onrender.com`
   — the demo UI is at `/ui`, same as locally.

### Option B: Manual web service

1. In Render: **New +** → **Web Service** → connect your repo.
2. Runtime: **Docker** (it will detect the `Dockerfile` automatically).
3. Under **Environment**, add:
   - `OPENAI_API_KEY` = your key
   - (optional) `CHAT_MODEL`, `EMBEDDING_MODEL`, etc. — see `.env.example`
4. Deploy.

### Important caveat: ephemeral storage

Render's free tier filesystem is **ephemeral** — every redeploy or restart
wipes `data/vectorstore/` and `data/uploads/`. This means:
- Session memory resets (expected — it's in-process by design, see README
  notes above).
- **Any documents you ingested are gone** and need to be re-uploaded after
  a restart/redeploy.

This is fine for demoing the assignment (ingest a doc, show it working,
that's the point), but isn't durable for real production use. To fix that
properly:
- Attach a [Render persistent disk](https://render.com/docs/disks) mounted
  at `/app/data/vectorstore`, **or**
- Swap the FAISS index for a hosted vector DB (Pinecone, Qdrant Cloud,
  or Chroma Cloud) so the knowledge base survives restarts independent of
  the app's own filesystem.

### Running the container locally first (sanity check before deploying)

```bash
docker build -t mini-ai-assistant .
docker run -p 8000:8000 --env-file .env mini-ai-assistant
```
Then visit http://localhost:8000/ui exactly as with the non-Docker setup.

### Alternatives to Render

- **Railway** — same Dockerfile works unchanged; connect the repo, add the
  same env vars, deploy.
- **Fly.io** — run `fly launch` in the repo root (it detects the
  Dockerfile), then `fly secrets set OPENAI_API_KEY=sk-...` before deploying.
- **Your own VPS** — `docker build` + `docker run` as above, put it behind
  nginx/Caddy with TLS if you want a real domain.

---

## Notes / possible extensions

- Swap `session_memory.py`'s in-process dict for Redis to survive restarts
  and scale across multiple server instances.
- Swap `search_product`'s substring match for embedding-based fuzzy match
  if the catalog grows large.
- Add per-document deletion — currently ingestion is additive only, and
  FAISS's flat index doesn't support in-place deletion (would need an
  `IndexIDMap` + rebuild strategy, or periodic index compaction).
- Swap the flat FAISS index for an `IndexIVFFlat`/HNSW variant if the
  knowledge base grows large enough that brute-force search becomes slow.
