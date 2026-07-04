# Architecture / Pipeline Diagram

## 1. Ingestion pipeline

```mermaid
flowchart LR
    A[User uploads<br/>PDF / TXT / MD] --> B["POST /ingest"]
    B --> C[Loader<br/>pypdf / plain text]
    C --> D[Chunker<br/>RecursiveCharacterTextSplitter<br/>800 chars, 120 overlap]
    D --> E[Embedder<br/>OpenAI text-embedding-3-small]
    E --> F[(FAISS IndexFlatIP<br/>+ JSON metadata sidecar)]
    F --> G[IngestResponse<br/>doc_id, num_chunks]
```

## 2. Chat pipeline (single-agent-with-tools)

```mermaid
flowchart TD
    U[User message] --> P["POST /chat"]
    P --> M[Load session memory<br/>last N turns]
    M --> L1[LLM call #1<br/>system prompt + history + message<br/>+ 3 tool schemas]

    L1 -->|tool_call: get_order_status| T1[Tool: Order Status<br/>orders.json lookup]
    L1 -->|tool_call: search_product| T2[Tool: Product Search<br/>products.json lookup]
    L1 -->|tool_call: search_knowledge_base| T3[Retrieval<br/>embed query -> FAISS<br/>cosine similarity search + threshold filter]
    L1 -->|no tool call| D1[Direct response<br/>from choice.content]

    T1 --> L2[LLM call #2<br/>feed tool result back<br/>generate final grounded reply]
    T2 --> L2
    T3 --> L2

    L2 --> R[Final reply]
    D1 --> R
    R --> SM[Persist turn to<br/>session memory]
    SM --> Resp[ChatResponse<br/>reply, route, sources, tool_calls]
```

## 3. Routing decision

The routing decision is made **inside a single OpenAI chat completion call**
using native function calling (`tool_choice="auto"`), rather than a separate
hand-written classifier step:

| User intent | Tool the LLM calls | Executed by |
|---|---|---|
| "Where is my order ORD001?" | `get_order_status` | `app/tools/order_status.py` |
| "Do you have a wireless mouse?" | `search_product` | `app/tools/product_search.py` |
| "What does the uploaded doc say about X?" | `search_knowledge_base` | `app/retrieval/retriever.py` |
| "Hi", "thanks", "what's my name?" (already in history) | *(none)* | Direct LLM response using conversation memory |

If `search_knowledge_base` finds no chunk above the similarity threshold, the
tool result tells the LLM there is no relevant context, and the system prompt
instructs it to reply with the exact required fallback string.
