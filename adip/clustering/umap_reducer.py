"""
UMAP dimensionality reduction: 768d → 50d.
Falls back to PCA if umap-learn unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from adip.config.settings import settings

logger = logging.getLogger(__name__)

_reducer = None
_reducer_type: Optional[str] = None


def _init_reducer(n_samples: int):
    global _reducer, _reducer_type
    n_components = min(settings.umap_n_components, n_samples - 1) if n_samples > 1 else 1

    try:
        import umap
        _reducer = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            random_state=42,
        )
        _reducer_type = "umap"
        logger.info("Using UMAP reducer → %dd", n_components)
    except ImportError:
        from sklearn.decomposition import PCA
        _reducer = PCA(n_components=n_components, random_state=42)
        _reducer_type = "pca"
        logger.warning("umap-learn not available; falling back to PCA → %dd", n_components)


def fit_transform(embeddings: np.ndarray) -> np.ndarray:
    """
    Fit reducer on embeddings and return reduced array.
    Input shape: (N, 768). Output shape: (N, 50) or less.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return embeddings

    _init_reducer(n_samples)
    try:
        reduced = _reducer.fit_transform(embeddings)
    except (TypeError, ValueError, Exception) as exc:
        # UMAP spectral initialization can fail on small/sparse datasets
        logger.warning("UMAP fit_transform failed (%s); falling back to PCA", exc)
        _force_pca(n_samples)
        reduced = _reducer.fit_transform(embeddings)

    logger.info(
        "Reduced %d vectors from %dd to %dd via %s",
        n_samples, embeddings.shape[1], reduced.shape[1], _reducer_type,
    )
    return reduced.astype(np.float32)


def _force_pca(n_samples: int):
    """Switch to PCA as a fallback."""
    global _reducer, _reducer_type
    from sklearn.decomposition import PCA
    n_components = min(settings.umap_n_components, n_samples - 1)
    _reducer = PCA(n_components=n_components, random_state=42)
    _reducer_type = "pca"
    logger.info("Switched to PCA reducer → %dd", n_components)


def transform(embeddings: np.ndarray) -> np.ndarray:
    """Transform new embeddings using already-fitted reducer."""
    if _reducer is None:
        logger.warning("Reducer not fitted; returning raw embeddings")
        return embeddings
    return _reducer.transform(embeddings).astype(np.float32)
