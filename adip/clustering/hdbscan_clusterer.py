"""
HDBSCAN clustering — no K required, handles variable density and noise.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from adip.config.settings import settings

logger = logging.getLogger(__name__)


def cluster(embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Cluster reduced embeddings via HDBSCAN.

    Returns:
        labels: (N,) int array — cluster IDs, -1 = noise
        n_clusters: number of clusters found (excluding noise)
    """
    n_samples = embeddings.shape[0]
    min_size = min(settings.min_cluster_size, max(2, n_samples // 3))

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(1, min(2, n_samples // 5)),
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=1,
        )
        labels = clusterer.fit_predict(embeddings)
    except (ImportError, TypeError) as exc:
        if isinstance(exc, ImportError):
            logger.warning("hdbscan not installed; falling back to sklearn AgglomerativeClustering")
        else:
            logger.warning("HDBSCAN failed (%s); falling back to AgglomerativeClustering", exc)
        from sklearn.cluster import AgglomerativeClustering
        n_clust = max(2, n_samples // 5) if n_samples >= 10 else None
        agg = AgglomerativeClustering(
            n_clusters=n_clust,
            distance_threshold=1.5 if n_clust is None else None,
            metric="euclidean",
            linkage="average",
        )
        labels = agg.fit_predict(embeddings)

    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    logger.info(
        "HDBSCAN: %d clusters, %d noise points out of %d samples",
        n_clusters, noise_count, n_samples,
    )
    return labels, n_clusters


def compute_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[int, np.ndarray]:
    """Compute mean centroid per cluster (excluding noise -1)."""
    centroids = {}
    unique_labels = set(labels)
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        centroids[int(lbl)] = embeddings[mask].mean(axis=0)
    return centroids
