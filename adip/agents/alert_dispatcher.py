"""
Alert Dispatcher Agent — LangGraph node.

- Creates Jira ticket for HOLD/CONDITIONAL recommendations
- Posts Slack alert
- Falls back to mock/logging if no external services
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from adip.agents.supervisor import AgentTimer, append_error
from adip.config.settings import settings
from adip.graph.state import ADIPState, NotificationResult, ReleaseRecommendation

logger = logging.getLogger(__name__)


async def alert_dispatch(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: dispatch alerts based on release recommendation.
    """
    with AgentTimer("alert_dispatcher"):
        try:
            report = state.get("risk_report", {})
            recommendation = state.get(
                "release_recommendation",
                report.get("release_recommendation", "PROCEED"),
            )
            notifications: List[Dict[str, Any]] = []

            if recommendation in (
                ReleaseRecommendation.HOLD,
                ReleaseRecommendation.CONDITIONAL,
                "HOLD",
                "CONDITIONAL",
            ):
                # Jira ticket
                jira_result = await _create_jira_ticket(report, recommendation)
                notifications.append(jira_result.model_dump())

                # Slack alert
                slack_result = await _post_slack_alert(report, recommendation)
                notifications.append(slack_result.model_dump())

            else:
                # PROCEED — just log, no alerts
                logger.info("Recommendation is PROCEED; no alerts dispatched")
                notifications.append(
                    NotificationResult(
                        channel="log", success=True,
                        detail="PROCEED — no alerts needed",
                    ).model_dump()
                )

            return {"notifications": notifications}

        except Exception as exc:
            logger.error("Alert dispatch failed: %s", exc, exc_info=True)
            return {
                **append_error(state, f"alert_dispatcher: {exc}"),
                "notifications": [],
            }


async def _create_jira_ticket(report: Dict, recommendation: str) -> NotificationResult:
    """Create a Jira ticket or mock it."""
    summary = report.get("executive_summary", "Risk report requires attention")
    high_risk = report.get("high_risk_files", [])
    top_files = ", ".join(f.get("file_path", "?") for f in high_risk[:5])

    title = f"[ADIP] {recommendation} — {len(high_risk)} high-risk files"
    body = (
        f"Release Recommendation: {recommendation}\n\n"
        f"Executive Summary:\n{summary}\n\n"
        f"Top Risk Files: {top_files}\n\n"
        f"Actions: {report.get('recommended_actions', [])}"
    )

    if settings.jira_base_url and settings.jira_api_token:
        try:
            import aiohttp
            url = f"{settings.jira_base_url}/rest/api/2/issue"
            payload = {
                "fields": {
                    "project": {"key": settings.jira_project_key},
                    "summary": title,
                    "description": body,
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": "Highest" if recommendation == "HOLD" else "High"},
                }
            }
            headers = {
                "Authorization": f"Bearer {settings.jira_api_token}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    data = await resp.json()
                    return NotificationResult(
                        channel="jira", success=resp.status < 300,
                        detail=f"Created {data.get('key', 'unknown')}",
                    )
        except Exception as exc:
            logger.error("Jira ticket creation failed: %s", exc)
            return NotificationResult(
                channel="jira", success=False, detail=str(exc)
            )

    # Mock Jira
    logger.info("[MOCK JIRA] %s\n%s", title, body[:200])
    return NotificationResult(
        channel="jira", success=True,
        detail=f"[MOCK] {title}",
    )


async def _post_slack_alert(report: Dict, recommendation: str) -> NotificationResult:
    """Post Slack alert or mock it."""
    summary = report.get("executive_summary", "Risk report generated")
    emoji = ":red_circle:" if recommendation == "HOLD" else ":large_orange_diamond:"

    message = (
        f"{emoji} *ADIP Risk Alert: {recommendation}*\n\n"
        f"{summary}\n\n"
        f"High-risk files: {len(report.get('high_risk_files', []))}"
    )

    if settings.slack_webhook_url:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.slack_webhook_url,
                    json={"text": message},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return NotificationResult(
                        channel="slack", success=resp.status == 200,
                        detail=f"Status {resp.status}",
                    )
        except Exception as exc:
            logger.error("Slack alert failed: %s", exc)
            return NotificationResult(
                channel="slack", success=False, detail=str(exc)
            )

    # Mock Slack
    logger.info("[MOCK SLACK] %s", message[:300])
    return NotificationResult(
        channel="slack", success=True,
        detail=f"[MOCK] Slack alert sent for {recommendation}",
    )
