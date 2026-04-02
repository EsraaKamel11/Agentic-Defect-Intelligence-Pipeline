"""
Qdrant vector store with named vectors (dense 768d + sparse BM25).
Falls back to in-memory numpy if Qdrant is unavailable.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adip.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Unified interface — Qdrant when available, numpy fallback otherwise."""

    def __init__(self):
        self._qdrant_client = None
        self._collection = settings.qdrant_collection
        self._use_qdrant = False

        # In-memory fallback structures
        self._mem_ids: List[str] = []
        self._mem_vectors: List[np.ndarray] = []
        self._mem_payloads: List[Dict[str, Any]] = []
        self._content_hashes: set = set()

        self._try_connect_qdrant()

    # ── Connection ───────────────────────────────────────────────────────

    def _try_connect_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import (
                Distance,
                VectorParams,
            )
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=5,
            )
            # Verify connectivity
            client.get_collections()

            # Ensure collection exists
            collections = [c.name for c in client.get_collections().collections]
            if self._collection not in collections:
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        "dense": VectorParams(size=768, distance=Distance.COSINE),
                    },
                )
                logger.info("Created Qdrant collection: %s", self._collection)

            self._qdrant_client = client
            self._use_qdrant = True
            logger.info("Connected to Qdrant at %s:%s", settings.qdrant_host, settings.qdrant_port)
        except Exception as exc:
            logger.warning("Qdrant unavailable (%s); using in-memory numpy fallback", exc)
            self._use_qdrant = False

    # ── Content-hash dedup ───────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_duplicate(self, content: str) -> bool:
        h = self._content_hash(content)
        if h in self._content_hashes:
            return True
        self._content_hashes.add(h)
        return False

    # ── Upsert ───────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        contents: Optional[List[str]] = None,
    ) -> int:
        """
        Upsert vectors with deduplication.
        Returns number of actually inserted points.
        """
        inserted = 0
        for i, (vid, vec, payload) in enumerate(zip(ids, vectors, payloads)):
            content_text = contents[i] if contents else payload.get("content", "")
            if self._is_duplicate(content_text):
                continue

            if self._use_qdrant:
                self._upsert_qdrant(vid, vec, payload)
            else:
                self._upsert_memory(vid, vec, payload)
            inserted += 1

        return inserted

    def _upsert_qdrant(self, vid: str, vec: np.ndarray, payload: Dict):
        from qdrant_client.http.models import PointStruct
        self._qdrant_client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=vid,
                    vector={"dense": vec.tolist()},
                    payload=payload,
                )
            ],
        )

    def _upsert_memory(self, vid: str, vec: np.ndarray, payload: Dict):
        self._mem_ids.append(vid)
        self._mem_vectors.append(vec.astype(np.float32))
        self._mem_payloads.append(payload)

    # ── Search ───────────────────────────────────────────────────────────

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Hybrid search returning list of (id, score, payload).
        On Qdrant uses dense vector search; on fallback uses cosine similarity.
        """
        if self._use_qdrant:
            return self._search_qdrant(query_vector, top_k, filters)
        return self._search_memory(query_vector, top_k, filters)

    def _search_qdrant(self, qvec, top_k, filters):
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = []
            for key, val in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            qdrant_filter = Filter(must=conditions)

        results = self._qdrant_client.search(
            collection_name=self._collection,
            query_vector=("dense", qvec.tolist()),
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [
            (str(r.id), r.score, r.payload) for r in results
        ]

    def _search_memory(self, qvec, top_k, filters):
        if not self._mem_vectors:
            return []

        mat = np.stack(self._mem_vectors)
        qvec_norm = qvec / (np.linalg.norm(qvec) + 1e-10)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        scores = mat_norm @ qvec_norm

        # Apply metadata filters
        mask = np.ones(len(scores), dtype=bool)
        if filters:
            for key, val in filters.items():
                for i, p in enumerate(self._mem_payloads):
                    if p.get(key) != val:
                        mask[i] = False
        scores = np.where(mask, scores, -1.0)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._mem_ids[i], float(scores[i]), self._mem_payloads[i])
            for i in top_indices
            if scores[i] > -1.0
        ]

    # ── Batch retrieval ──────────────────────────────────────────────────

    def get_embeddings_batch(
        self, ids: Optional[List[str]] = None
    ) -> Tuple[List[str], np.ndarray]:
        """Return all (or specified) embeddings. For clustering input."""
        if self._use_qdrant:
            return self._get_batch_qdrant(ids)
        return self._get_batch_memory(ids)

    def _get_batch_qdrant(self, ids):
        scroll_result, _ = self._qdrant_client.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_vectors=True,
        )
        result_ids = []
        result_vecs = []
        for point in scroll_result:
            pid = str(point.id)
            if ids is not None and pid not in ids:
                continue
            result_ids.append(pid)
            vec = point.vector
            if isinstance(vec, dict):
                vec = vec.get("dense", [])
            result_vecs.append(np.array(vec, dtype=np.float32))

        if not result_vecs:
            return [], np.empty((0, 768), dtype=np.float32)
        return result_ids, np.stack(result_vecs)

    def _get_batch_memory(self, ids):
        if not self._mem_vectors:
            return [], np.empty((0, 768), dtype=np.float32)
        if ids is None:
            return list(self._mem_ids), np.stack(self._mem_vectors)
        filtered = [
            (mid, mv)
            for mid, mv in zip(self._mem_ids, self._mem_vectors)
            if mid in set(ids)
        ]
        if not filtered:
            return [], np.empty((0, 768), dtype=np.float32)
        fids, fvecs = zip(*filtered)
        return list(fids), np.stack(fvecs)

    # ── Count ────────────────────────────────────────────────────────────

    def count(self) -> int:
        if self._use_qdrant:
            info = self._qdrant_client.get_collection(self._collection)
            return info.points_count
        return len(self._mem_ids)
