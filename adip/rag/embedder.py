"""
Local embedding using sentence-transformers all-mpnet-base-v2.
Also generates BM25 sparse vectors for hybrid search.
Never calls an external embedding API.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from adip.config.settings import settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded globals ──────────────────────────────────────────────────
_dense_model = None
_bm25_model = None
_bm25_corpus_tokens: List[List[str]] = []


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _dense_model = SentenceTransformer(settings.embedding_model)
            logger.info("Loaded dense embedding model: %s", settings.embedding_model)
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s); using random embeddings", exc)
            _dense_model = "FALLBACK"
    return _dense_model


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def embed_dense(texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
    """
    Embed a list of texts using all-mpnet-base-v2.
    Returns (N, 768) float32 array.
    """
    bs = batch_size or settings.embedding_batch_size
    model = _get_dense_model()
    if model == "FALLBACK":
        # Deterministic-ish random fallback for mock mode
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(texts), 768).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)
    return model.encode(texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)


def embed_single(text: str) -> np.ndarray:
    """Convenience: embed one text, return (768,) vector."""
    return embed_dense([text])[0]


# ── BM25 sparse vectors ─────────────────────────────────────────────────

def fit_bm25(corpus: List[str]) -> None:
    """Fit BM25 on a corpus of texts. Call once before sparse queries."""
    global _bm25_model, _bm25_corpus_tokens
    try:
        from rank_bm25 import BM25Okapi
        _bm25_corpus_tokens = [_tokenize(doc) for doc in corpus]
        _bm25_model = BM25Okapi(_bm25_corpus_tokens)
        logger.info("BM25 fitted on %d documents", len(corpus))
    except ImportError:
        logger.warning("rank_bm25 not installed; sparse search disabled")
        _bm25_model = None


def bm25_scores(query: str) -> np.ndarray:
    """Return BM25 scores for query against fitted corpus. Shape (N,)."""
    if _bm25_model is None:
        return np.array([])
    tokens = _tokenize(query)
    return _bm25_model.get_scores(tokens)


def sparse_vector(text: str, vocab: Optional[Dict[str, int]] = None) -> Dict[int, float]:
    """
    Generate a sparse vector dict {token_index: tf} for Qdrant named sparse vector.
    Uses a simple term-frequency approach.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    if vocab is None:
        # Auto-assign indices by hash
        return {hash(t) % (2**31): count / len(tokens) for t, count in tf.items()}
    return {vocab[t]: count / len(tokens) for t, count in tf.items() if t in vocab}
