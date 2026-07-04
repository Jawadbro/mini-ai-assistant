"""
FAISS-backed vector store with a JSON sidecar for metadata/text.

FAISS itself only stores raw vectors and returns integer positions — it has
no concept of metadata. So we keep a parallel list of metadata dicts
(source, doc_id, chunk_id, original text) indexed by the same position
FAISS assigns each vector, and persist both to disk together.

Chosen over ChromaDB for this project specifically because faiss-cpu ships
prebuilt wheels for all current CPython versions/platforms (no compiler
needed), whereas chroma's HNSW dependency (chroma-hnswlib) frequently has
no prebuilt wheel for newer Python versions and falls back to compiling
from source, which fails on machines without a C++ toolchain.

Uses IndexFlatIP (inner product) over L2-normalized vectors, which is
mathematically equivalent to cosine similarity search — scores returned
are in [-1, 1], higher is more similar.
"""
import json
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np

from app.config import VECTORSTORE_DIR, EMBEDDING_DIM

_lock = threading.Lock()


class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM, base_dir: Path = None):
        self.dim = dim
        self.base_dir = base_dir if base_dir is not None else VECTORSTORE_DIR
        self.index_path = self.base_dir / "index.faiss"
        self.metadata_path = self.base_dir / "metadata.json"
        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(self.dim)
            self._metadata = []

    def _persist(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f)

    def add(self, embeddings: List[List[float]], metadatas: List[dict]) -> None:
        if not embeddings:
            return
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        with _lock:
            self._index.add(vectors)
            self._metadata.extend(metadatas)
            self._persist()

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[dict, float]]:
        """Returns up to top_k (metadata, cosine_similarity) pairs, best first."""
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._metadata[idx], float(score)))
        return results

    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0


_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    """Process-wide singleton so the index is loaded from disk once."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store
