"""
Normalize raw events from any source into DefectEvent objects.
Auto-detects source type and extracts stack traces, file paths, severity.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from adip.graph.state import DefectEvent, DefectSeverity

logger = logging.getLogger(__name__)

_STACK_TRACE_RE = re.compile(
    r"(Traceback.*?(?:\n\s+.*)+\n\w+(?:Error|Exception).*)",
    re.DOTALL,
)
_FILE_PATH_RE = re.compile(r'["\']?([\w/\\]+\.(?:py|js|ts|java|go|rs))["\']?')
_SEVERITY_RE = re.compile(r"\b(P0|P1|P2|P3|CRITICAL|HIGH|MEDIUM|LOW)\b", re.IGNORECASE)


def normalize_event(raw: Dict[str, Any]) -> DefectEvent:
    """Convert a raw event dict from any source to a DefectEvent."""
    source = _detect_source(raw)
    raw_content = raw.get("raw_content", str(raw))

    # Extract structured fields
    stack_trace = _extract_stack_trace(raw_content)
    file_path = raw.get("file_path") or _extract_file_path(raw_content)
    severity = _parse_severity(raw.get("severity", ""))
    component = raw.get("component", "unknown")
    timestamp = _parse_timestamp(raw.get("timestamp"))

    # Build normalized content
    normalized = _build_normalized_content(raw_content, source)

    return DefectEvent(
        id=raw.get("id", str(uuid.uuid4())),
        source=source,
        raw_content=raw_content,
        normalized_content=normalized,
        component=component,
        file_path=file_path,
        severity=severity,
        stack_trace=stack_trace,
        timestamp=timestamp,
    )


def normalize_batch(raws: List[Dict[str, Any]]) -> List[DefectEvent]:
    """Normalize a batch of raw events."""
    events = []
    for raw in raws:
        try:
            events.append(normalize_event(raw))
        except Exception as exc:
            logger.error("Failed to normalize event: %s — %s", exc, raw.get("source", "?"))
    return events


def _detect_source(raw: Dict[str, Any]) -> str:
    """Auto-detect the event source."""
    explicit = raw.get("source", "").lower()
    if explicit in ("cicd", "jira", "sentry", "datadog"):
        return explicit

    content = raw.get("raw_content", "")
    if "Traceback" in content or "Sentry" in content:
        return "sentry"
    if "Build #" in content or "PASSED" in content or "FAILED" in content:
        return "cicd"
    if "[ALERT]" in content or "Monitor" in content:
        return "datadog"
    if "DEFECT-" in content or "Steps to reproduce" in content:
        return "jira"
    return "unknown"


def _extract_stack_trace(text: str) -> Optional[str]:
    match = _STACK_TRACE_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_file_path(text: str) -> Optional[str]:
    match = _FILE_PATH_RE.search(text)
    return match.group(1) if match else None


def _parse_severity(sev: str) -> DefectSeverity:
    sev_upper = sev.upper().strip()
    severity_map = {
        "P0": DefectSeverity.P0, "CRITICAL": DefectSeverity.P0,
        "P1": DefectSeverity.P1, "HIGH": DefectSeverity.P1,
        "P2": DefectSeverity.P2, "MEDIUM": DefectSeverity.P2,
        "P3": DefectSeverity.P3, "LOW": DefectSeverity.P3,
    }
    return severity_map.get(sev_upper, DefectSeverity.P3)


def _parse_timestamp(ts: Any) -> datetime:
    if ts is None:
        return datetime.utcnow()
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00").replace("+00:00", ""))
    except (ValueError, TypeError):
        return datetime.utcnow()


def _build_normalized_content(raw: str, source: str) -> str:
    """Clean and structure raw content for embedding."""
    # Remove excessive whitespace
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    cleaned = "\n".join(lines)

    # Prefix with source for context
    return f"[{source.upper()}] {cleaned}"
