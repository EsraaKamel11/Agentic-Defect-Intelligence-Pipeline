"""
Risk Scorer Agent — LangGraph node.

Five-feature formula with Bayesian-updated weights.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List

from adip.agents.supervisor import AgentTimer, append_error
from adip.graph.state import ADIPState, ClusterResult, DefectEvent, DefectSeverity
from adip.scoring.risk_formula import compute_risk_score
from adip.scoring.bayesian_updater import get_current_weights

logger = logging.getLogger(__name__)


def _severity_weight(severity: str) -> float:
    """Map severity to a 0-1 weight for cluster_severity_weight."""
    return {"P0": 1.0, "P1": 0.75, "P2": 0.5, "P3": 0.25}.get(severity, 0.25)


def _recency_score(events: List[Dict], days: int = 30) -> float:
    """More recent defects → higher recency score."""
    if not events:
        return 0.0
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)
    recent = 0
    for e in events:
        try:
            ts = datetime.fromisoformat(str(e.get("timestamp", "")))
            if ts >= cutoff:
                recent += 1
        except (ValueError, TypeError):
            pass
    return min(recent / max(len(events), 1), 1.0)


async def score_risks(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: compute risk scores for all files with defects.
    """
    with AgentTimer("risk_scorer"):
        try:
            events = state.get("defect_events", [])
            clusters = state.get("clusters", [])
            weights = get_current_weights()

            # Aggregate defects per file
            file_events: Dict[str, List[Dict]] = defaultdict(list)
            file_components: Dict[str, str] = {}
            for event in events:
                fp = event.get("file_path")
                if fp:
                    file_events[fp].append(event)
                    file_components[fp] = event.get("component", "unknown")

            # If no file-level events, create scores from cluster data
            if not file_events and clusters:
                for c in clusters:
                    # Use cluster label as pseudo-file
                    fp = f"cluster:{c.get('cluster_id', 'unknown')}"
                    file_events[fp] = [{"severity": "P1", "timestamp": c.get("last_seen", "")}]
                    file_components[fp] = c.get("label", "unknown")

            # Build severity weight from clusters
            cluster_severity_map: Dict[str, float] = {}
            for c in clusters:
                sev_weight = c.get("weight", 0.5)
                # Assign to all files in cluster
                for fp in file_events:
                    current = cluster_severity_map.get(fp, 0.0)
                    cluster_severity_map[fp] = max(current, sev_weight)

            # Score each file
            risk_scores: List[Dict[str, Any]] = []
            for fp, events_list in file_events.items():
                freq_30d = len(events_list)
                # Mock git churn — in production, read from git log
                git_churn = random.uniform(5, 80)
                # Mock coverage gap — in production, read from coverage reports
                coverage_gap = random.uniform(0.1, 0.9)
                cluster_sev = cluster_severity_map.get(fp, 0.3)
                recency = _recency_score(events_list)

                # Max severity across events for this file
                max_sev = max(
                    (_severity_weight(e.get("severity", "P3")) for e in events_list),
                    default=0.25,
                )
                cluster_sev = max(cluster_sev, max_sev * 0.5)

                score = compute_risk_score(
                    file_path=fp,
                    component=file_components.get(fp, "unknown"),
                    defect_frequency_30d=freq_30d,
                    git_churn=git_churn,
                    coverage_gap=coverage_gap,
                    cluster_severity_weight=cluster_sev,
                    recency_decay=recency,
                    weights=weights,
                )
                risk_scores.append(score.model_dump())

            # Sort by risk score descending
            risk_scores.sort(key=lambda s: s["risk_score"], reverse=True)

            logger.info("Scored %d files, top risk: %.3f",
                        len(risk_scores),
                        risk_scores[0]["risk_score"] if risk_scores else 0)

            return {"risk_scores": risk_scores}

        except Exception as exc:
            logger.error("Risk scoring failed: %s", exc, exc_info=True)
            return {
                **append_error(state, f"risk_scorer: {exc}"),
                "risk_scores": [],
            }
