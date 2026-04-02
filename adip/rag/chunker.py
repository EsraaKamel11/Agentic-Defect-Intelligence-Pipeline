"""
Structure-aware chunking for defect events.

- Stack traces: error message + top-5 frames as one atomic unit
- Jira tickets: recursive character split, 800 token max
- CI logs: split on PASSED/FAILED test boundaries
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextChunk:
    """A single chunk with provenance metadata."""
    text: str
    source_type: str          # stack_trace | jira | cicd | generic
    source_id: str = ""       # original DefectEvent.id
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


# Rough token estimate: 1 token ≈ 4 chars
_CHARS_PER_TOKEN = 4
_JIRA_MAX_TOKENS = 800
_JIRA_MAX_CHARS = _JIRA_MAX_TOKENS * _CHARS_PER_TOKEN

# Regex helpers
_FRAME_PATTERN = re.compile(
    r"^\s*(File |at |Traceback|Caused by|\.{3}\s*\d+ more)",
    re.MULTILINE,
)
_TEST_BOUNDARY = re.compile(
    r"^.*\b(PASSED|FAILED|ERROR|OK)\b.*$", re.MULTILINE
)


def chunk_stack_trace(text: str, source_id: str = "") -> List[TextChunk]:
    """Extract error message + top-5 frames as a single atomic chunk."""
    lines = text.strip().splitlines()
    error_line: Optional[str] = None
    frames: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _FRAME_PATTERN.match(line):
            frames.append(stripped)
        elif error_line is None and stripped:
            error_line = stripped

    top_frames = frames[:5]
    combined = (error_line or "") + "\n" + "\n".join(top_frames)
    return [
        TextChunk(
            text=combined.strip(),
            source_type="stack_trace",
            source_id=source_id,
            chunk_index=0,
            metadata={"frame_count": len(top_frames)},
        )
    ]


def _recursive_char_split(text: str, max_chars: int) -> List[str]:
    """Split text recursively on paragraph → sentence → word boundaries."""
    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: List[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)
            if all(len(c) <= max_chars for c in chunks):
                return chunks
    # Hard split as last resort
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_jira_ticket(text: str, source_id: str = "") -> List[TextChunk]:
    """Recursive character split at 800 token max."""
    parts = _recursive_char_split(text, _JIRA_MAX_CHARS)
    return [
        TextChunk(
            text=p,
            source_type="jira",
            source_id=source_id,
            chunk_index=i,
            metadata={"total_chunks": len(parts)},
        )
        for i, p in enumerate(parts)
    ]


def chunk_cicd_log(text: str, source_id: str = "") -> List[TextChunk]:
    """Split CI/CD logs on PASSED/FAILED test boundaries."""
    boundaries = list(_TEST_BOUNDARY.finditer(text))
    if not boundaries:
        return [
            TextChunk(
                text=text.strip(),
                source_type="cicd",
                source_id=source_id,
                chunk_index=0,
            )
        ]

    chunks: List[TextChunk] = []
    prev_end = 0
    for idx, match in enumerate(boundaries):
        # Include everything up to and including this boundary line
        end = match.end()
        segment = text[prev_end:end].strip()
        if segment:
            chunks.append(
                TextChunk(
                    text=segment,
                    source_type="cicd",
                    source_id=source_id,
                    chunk_index=idx,
                    metadata={"boundary": match.group().strip()},
                )
            )
        prev_end = end

    # Trailing content after last boundary
    tail = text[prev_end:].strip()
    if tail:
        chunks.append(
            TextChunk(
                text=tail,
                source_type="cicd",
                source_id=source_id,
                chunk_index=len(chunks),
            )
        )
    return chunks


def chunk_event(
    text: str,
    source_type: str,
    source_id: str = "",
) -> List[TextChunk]:
    """Dispatch to the appropriate chunking strategy."""
    if source_type == "stack_trace" or _looks_like_stack_trace(text):
        return chunk_stack_trace(text, source_id)
    elif source_type == "jira":
        return chunk_jira_ticket(text, source_id)
    elif source_type in ("cicd", "datadog"):
        return chunk_cicd_log(text, source_id)
    else:
        # Generic fallback — treat as Jira-style recursive split
        return chunk_jira_ticket(text, source_id)


def _looks_like_stack_trace(text: str) -> bool:
    indicators = ["Traceback", "at ", "File ", "Exception", "Error:"]
    return sum(1 for ind in indicators if ind in text) >= 2
