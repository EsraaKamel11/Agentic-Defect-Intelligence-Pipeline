"""Unit tests for rag/vector_store.py — in-memory fallback."""
import numpy as np
import pytest

from adip.rag.vector_store import VectorStore


@pytest.fixture
def store():
    """Create a fresh in-memory vector store."""
    s = VectorStore()
    s._use_qdrant = False  # Force in-memory
    return s


class TestVectorStoreMemory:
    def test_upsert_and_count(self, store):
        vecs = np.random.randn(3, 768).astype(np.float32)
        ids = ["a", "b", "c"]
        payloads = [{"content": f"doc {i}"} for i in range(3)]
        inserted = store.upsert(ids, vecs, payloads, [f"doc {i}" for i in range(3)])
        assert inserted == 3
        assert store.count() == 3

    def test_deduplication(self, store):
        vecs = np.random.randn(2, 768).astype(np.float32)
        ids = ["a", "b"]
        payloads = [{"content": "same"}, {"content": "same"}]
        inserted = store.upsert(ids, vecs, payloads, ["same", "same"])
        assert inserted == 1  # Second is duplicate

    def test_search(self, store):
        vecs = np.random.randn(5, 768).astype(np.float32)
        ids = [str(i) for i in range(5)]
        payloads = [{"content": f"doc {i}", "component": "auth"} for i in range(5)]
        store.upsert(ids, vecs, payloads, [f"doc {i}" for i in range(5)])

        query = vecs[0]  # Search for first vector
        results = store.search_hybrid(query, top_k=3)
        assert len(results) <= 3
        assert results[0][0] == "0"  # Should match itself

    def test_get_embeddings_batch(self, store):
        vecs = np.random.randn(3, 768).astype(np.float32)
        ids = ["x", "y", "z"]
        store.upsert(ids, vecs, [{} for _ in range(3)], ["a", "b", "c"])

        ret_ids, ret_vecs = store.get_embeddings_batch()
        assert len(ret_ids) == 3
        assert ret_vecs.shape == (3, 768)

    def test_empty_search(self, store):
        query = np.random.randn(768).astype(np.float32)
        results = store.search_hybrid(query)
        assert results == []
