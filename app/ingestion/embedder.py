"""
Embeds chunks with the OpenAI embeddings API and persists them via the
FAISS-backed VectorStore (see app/storage/vector_store.py).
"""
import uuid
from typing import List

from openai import OpenAI

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY
from app.storage.vector_store import get_store

_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of strings via the OpenAI embeddings endpoint."""
    if not texts:
        return []
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # response.data is returned in the same order as the input list
    return [item.embedding for item in response.data]


def store_chunks(chunks: List[str], source: str, doc_id: str) -> int:
    """Embed and persist chunks for one document. Returns count stored."""
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    metadatas = [
        {"source": source, "doc_id": doc_id, "chunk_id": i, "text": chunks[i]}
        for i in range(len(chunks))
    ]

    get_store().add(embeddings, metadatas)
    return len(chunks)


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]
