"""
Unit tests for the FAISS-backed vector store — no OpenAI/network calls
needed, since we exercise it directly with synthetic vectors.
"""
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.storage.vector_store import VectorStore


def _unit_vector(dim: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def _temp_store(dim: int = 8) -> tuple:
    tmp_dir = Path(tempfile.mkdtemp(prefix="vstore_test_"))
    store = VectorStore(dim=dim, base_dir=tmp_dir)
    return store, tmp_dir


def test_empty_store_search_returns_nothing():
    store, tmp_dir = _temp_store()
    try:
        assert store.count() == 0
        assert store.search(_unit_vector(8, 1), top_k=3) == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_add_and_search_finds_exact_match():
    store, tmp_dir = _temp_store()
    try:
        v1 = _unit_vector(8, 1)
        v2 = _unit_vector(8, 2)
        store.add([v1, v2], [{"text": "doc one"}, {"text": "doc two"}])
        assert store.count() == 2

        results = store.search(v1, top_k=1)
        assert len(results) == 1
        meta, score = results[0]
        assert meta["text"] == "doc one"
        assert score > 0.99  # exact same vector -> ~1.0 cosine similarity
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_respects_top_k():
    store, tmp_dir = _temp_store()
    try:
        vecs = [_unit_vector(8, i) for i in range(5)]
        metas = [{"text": f"doc {i}"} for i in range(5)]
        store.add(vecs, metas)

        results = store.search(vecs[0], top_k=2)
        assert len(results) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_persistence_across_instances():
    tmp_dir = Path(tempfile.mkdtemp(prefix="vstore_test_"))
    try:
        store1 = VectorStore(dim=8, base_dir=tmp_dir)
        v1 = _unit_vector(8, 42)
        store1.add([v1], [{"text": "persisted doc"}])

        # A fresh instance pointed at the same dir should load what was saved
        store2 = VectorStore(dim=8, base_dir=tmp_dir)
        assert store2.count() == 1
        results = store2.search(v1, top_k=1)
        assert results[0][0]["text"] == "persisted doc"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
