"""
Similarity search over the ingested knowledge base (FAISS-backed).

Applies a minimum-similarity threshold so that when the store is empty or
the best matches simply aren't relevant to the query, we surface that as
"no relevant context" rather than forcing the LLM to answer from noise.
"""
from dataclasses import dataclass
from typing import List

from app.ingestion.embedder import embed_texts
from app.storage.vector_store import get_store
from app.config import TOP_K, MIN_RELEVANT_SIMILARITY

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the uploaded knowledge base documents for information "
            "relevant to the user's question. Use this for any question "
            "that isn't specifically about an order status or product "
            "catalog lookup, e.g. general questions about uploaded "
            "reference material, policies, specs, or documentation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, ideally the user's question rephrased for retrieval.",
                }
            },
            "required": ["query"],
        },
    },
}


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: int
    similarity: float


def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    store = get_store()
    if store.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    results = store.search(query_embedding, top_k)

    chunks: List[RetrievedChunk] = []
    for meta, similarity in results:
        if similarity >= MIN_RELEVANT_SIMILARITY:
            chunks.append(
                RetrievedChunk(
                    text=meta["text"],
                    source=meta.get("source", "unknown"),
                    chunk_id=meta.get("chunk_id", -1),
                    similarity=similarity,
                )
            )
    return chunks


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks into a prompt-ready context block with citations."""
    blocks = []
    for c in chunks:
        blocks.append(f"[Source: {c.source}, chunk {c.chunk_id}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)
