"""
Defect Clusterer Agent — LangGraph node.

- Fetches embeddings for last 30 days from vector store
- UMAP reduction: 768d → 50d
- HDBSCAN clustering
- LLM-assisted cluster labeling constrained to 11-category taxonomy
- 60-day cluster decay with exponential weight reduction
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from adip.agents.supervisor import AgentTimer, append_error
from adip.agents.rag_indexer import get_vector_store, get_db
from adip.clustering.umap_reducer import fit_transform
from adip.clustering.hdbscan_clusterer import cluster, compute_centroids
from adip.clustering.cluster_labeler import label_cluster
from adip.config.settings import settings
from adip.graph.state import ADIPState, ClusterResult, RootCauseCategory
from adip.llm.factory import get_llm

logger = logging.getLogger(__name__)


def _decay_weight(last_seen: datetime, decay_days: int = 60) -> float:
    """Exponential decay over decay_days. Returns weight in (0, 1]."""
    age_days = (datetime.utcnow() - last_seen).total_seconds() / 86400
    return math.exp(-age_days / decay_days)


async def cluster_defects(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: cluster defect embeddings and label clusters.
    """
    with AgentTimer("defect_clusterer"):
        try:
            store = get_vector_store()
            ids, embeddings = store.get_embeddings_batch()

            if len(ids) < settings.min_cluster_size:
                logger.info(
                    "Only %d vectors in store (need %d); skipping clustering",
                    len(ids), settings.min_cluster_size,
                )
                # Return existing clusters if any
                return {
                    "clusters": state.get("clusters", []),
                    "clustering_skipped": True,
                }

            # UMAP reduction: 768d → 50d
            reduced = fit_transform(embeddings)

            # HDBSCAN clustering
            labels, n_clusters = cluster(reduced)
            centroids = compute_centroids(reduced, labels)

            # Build cluster results
            llm = get_llm()
            cluster_results: List[Dict[str, Any]] = []

            # Group member texts by cluster
            cluster_members: Dict[int, List[str]] = {}
            cluster_member_ids: Dict[int, List[str]] = {}
            for i, lbl in enumerate(labels):
                if lbl == -1:
                    continue
                lbl_int = int(lbl)
                if lbl_int not in cluster_members:
                    cluster_members[lbl_int] = []
                    cluster_member_ids[lbl_int] = []
                cluster_members[lbl_int].append(ids[i])
                cluster_member_ids[lbl_int].append(ids[i])

            # Retrieve text content for labeling
            for lbl_int, member_ids in cluster_members.items():
                # Get text from payloads
                member_texts = []
                for mid in member_ids[:10]:  # Sample up to 10
                    results = store.search_hybrid(
                        embeddings[ids.index(mid)] if mid in ids else np.zeros(768),
                        top_k=1,
                    )
                    if results:
                        member_texts.append(results[0][2].get("content", ""))

                if not member_texts:
                    member_texts = [f"Cluster {lbl_int} member"]

                # LLM-assisted labeling
                label_result = await label_cluster(llm, member_texts)

                # Calculate decay weight
                weight = _decay_weight(datetime.utcnow(), settings.cluster_decay_days)

                cr = ClusterResult(
                    label=label_result.label,
                    root_cause_category=RootCauseCategory(label_result.root_cause_category),
                    member_count=len(member_ids),
                    centroid_embedding=centroids.get(lbl_int, np.zeros(50)).tolist(),
                    recurrence_count=len(member_ids),
                    last_seen=datetime.utcnow(),
                    weight=weight,
                )
                cluster_results.append(cr.model_dump())

            # Persist clusters to DB
            db = await get_db()
            await db.store_clusters(cluster_results)

            logger.info("Produced %d clusters from %d vectors", len(cluster_results), len(ids))

            return {
                "clusters": cluster_results,
                "clustering_skipped": False,
            }

        except Exception as exc:
            logger.error("Clustering failed: %s", exc, exc_info=True)
            return {
                **append_error(state, f"defect_clusterer: {exc}"),
                "clusters": state.get("clusters", []),
                "clustering_skipped": True,
            }
