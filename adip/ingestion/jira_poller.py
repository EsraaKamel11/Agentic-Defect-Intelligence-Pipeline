"""
Jira poller — fetches new defect tickets every 15 minutes.
Falls back to mock data if no Jira credentials.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from adip.config.settings import settings

logger = logging.getLogger(__name__)


class JiraPoller:
    """Poll Jira REST API for defect tickets or generate mock data."""

    def __init__(self):
        self._last_poll: datetime = datetime.utcnow() - timedelta(days=1)
        self._use_jira = bool(settings.jira_base_url and settings.jira_api_token)
        if self._use_jira:
            logger.info("Jira poller configured: %s", settings.jira_base_url)
        else:
            logger.info("Jira credentials not configured; using mock data")

    async def poll(self) -> List[Dict[str, Any]]:
        """Fetch new tickets since last poll."""
        if self._use_jira:
            return await self._poll_jira()
        return self._generate_mock_tickets()

    async def _poll_jira(self) -> List[Dict[str, Any]]:
        """Real Jira API call."""
        try:
            import aiohttp
            jql = (
                f"project = {settings.jira_project_key} "
                f"AND type = Bug "
                f"AND updated >= '{self._last_poll.strftime('%Y-%m-%d %H:%M')}' "
                f"ORDER BY updated DESC"
            )
            url = f"{settings.jira_base_url}/rest/api/2/search"
            headers = {
                "Authorization": f"Bearer {settings.jira_api_token}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params={"jql": jql, "maxResults": 50}
                ) as resp:
                    data = await resp.json()
                    self._last_poll = datetime.utcnow()
                    return self._normalize_jira_issues(data.get("issues", []))
        except Exception as exc:
            logger.error("Jira poll failed: %s", exc)
            return self._generate_mock_tickets()

    def _normalize_jira_issues(self, issues: list) -> List[Dict[str, Any]]:
        results = []
        for issue in issues:
            fields = issue.get("fields", {})
            results.append({
                "source": "jira",
                "raw_content": (
                    f"[{issue['key']}] {fields.get('summary', '')}\n\n"
                    f"{fields.get('description', '')}"
                ),
                "component": (
                    fields.get("components", [{}])[0].get("name", "unknown")
                    if fields.get("components")
                    else "unknown"
                ),
                "severity": self._map_priority(fields.get("priority", {}).get("name", "")),
                "timestamp": fields.get("updated", datetime.utcnow().isoformat()),
            })
        return results

    @staticmethod
    def _map_priority(jira_priority: str) -> str:
        mapping = {
            "Highest": "P0", "High": "P1",
            "Medium": "P2", "Low": "P3", "Lowest": "P3",
        }
        return mapping.get(jira_priority, "P2")

    def _generate_mock_tickets(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate mock Jira-style defect tickets."""
        from adip.ingestion.kafka_consumer import _mock_jira
        tickets = [_mock_jira() for _ in range(count)]
        self._last_poll = datetime.utcnow()
        logger.info("Generated %d mock Jira tickets", len(tickets))
        return tickets

    @property
    def seconds_since_last_poll(self) -> float:
        return (datetime.utcnow() - self._last_poll).total_seconds()
