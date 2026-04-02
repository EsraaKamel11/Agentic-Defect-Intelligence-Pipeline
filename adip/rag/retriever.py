"""
Hybrid retrieval with Reciprocal Rank Fusion + cross-encoder reranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from adip.rag.embedder import bm25_scores, embed_single
from adip.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Loaded cross-encoder reranker")
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s); skipping reranking", exc)
            _cross_encoder = "UNAVAILABLE"
    return _cross_encoder


@dataclass
class RetrievalResult:
    id: str
    score: float
    payload: Dict[str, Any]
    text: str = ""


class HybridRetriever:
    """BM25 + dense search with RRF merging and optional cross-encoder reranking."""

    def __init__(self, store: VectorStore, rrf_k: int = 60):
        self.store = store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        """Full hybrid retrieval pipeline."""
        # 1. Dense search
        qvec = embed_single(query)
        dense_results = self.store.search_hybrid(qvec, top_k=top_k * 2, filters=filters)

        # 2. BM25 scores (if fitted)
        sparse = bm25_scores(query)

        # 3. Reciprocal Rank Fusion
        merged = self._rrf_merge(dense_results, sparse, top_k * 2)

        # 4. Cross-encoder reranking
        if rerank and merged:
            merged = self._rerank(query, merged, top_k)
        else:
            merged = merged[:top_k]

        return merged

    def _rrf_merge(
        self,
        dense_results: list,
        sparse_scores: np.ndarray,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion of dense and sparse results."""
        score_map: Dict[str, float] = {}
        payload_map: Dict[str, Dict] = {}
        k = self.rrf_k

        # Dense ranks
        for rank, (rid, score, payload) in enumerate(dense_results):
            rrf_score = 1.0 / (k + rank + 1)
            score_map[rid] = score_map.get(rid, 0.0) + rrf_score
            payload_map[rid] = payload

        # Sparse ranks (if available)
        if sparse_scores.size > 0:
            sorted_indices = np.argsort(sparse_scores)[::-1]
            for rank, idx in enumerate(sorted_indices[:top_k]):
                if sparse_scores[idx] <= 0:
                    break
                # Try to match by index to existing IDs
                if idx < len(self.store._mem_ids):
                    rid = self.store._mem_ids[idx]
                    rrf_score = 1.0 / (k + rank + 1)
                    score_map[rid] = score_map.get(rid, 0.0) + rrf_score
                    if rid not in payload_map and idx < len(self.store._mem_payloads):
                        payload_map[rid] = self.store._mem_payloads[idx]

        # Sort by fused score
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(
                id=rid,
                score=score,
                payload=payload_map.get(rid, {}),
                text=payload_map.get(rid, {}).get("content", ""),
            )
            for rid, score in ranked
        ]

    def _rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Cross-encoder reranking."""
        encoder = _get_cross_encoder()
        if encoder == "UNAVAILABLE" or not candidates:
            return candidates[:top_k]

        pairs = [(query, c.text or c.payload.get("content", "")) for c in candidates]
        scores = encoder.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand.score = float(score)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
