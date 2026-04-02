"""Unit tests for rag/chunker.py."""
from adip.rag.chunker import (
    chunk_cicd_log,
    chunk_event,
    chunk_jira_ticket,
    chunk_stack_trace,
)


class TestStackTraceChunking:
    def test_extracts_error_and_frames(self):
        trace = (
            "Traceback (most recent call last):\n"
            '  File "src/auth.py", line 42, in login\n'
            "    token = get_token()\n"
            '  File "src/auth.py", line 10, in get_token\n'
            "    return cache[key]\n"
            "KeyError: 'session_token'"
        )
        chunks = chunk_stack_trace(trace, source_id="evt1")
        assert len(chunks) == 1
        assert chunks[0].source_type == "stack_trace"
        assert "KeyError" in chunks[0].text or "File" in chunks[0].text

    def test_single_atomic_chunk(self):
        chunks = chunk_stack_trace("Error: something\n  at line 1\n  at line 2")
        assert len(chunks) == 1


class TestJiraChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_jira_ticket("Short bug report", source_id="j1")
        assert len(chunks) == 1
        assert chunks[0].source_type == "jira"

    def test_long_text_split(self):
        long_text = "A" * 5000  # Well over 800 tokens
        chunks = chunk_jira_ticket(long_text)
        assert len(chunks) > 1


class TestCICDChunking:
    def test_splits_on_boundaries(self):
        log = (
            "Running test_auth...\nAssertionError\nFAILED\n"
            "Running test_payment...\nOK\nPASSED\n"
            "Running test_users...\nTimeoutError\nFAILED\n"
        )
        chunks = chunk_cicd_log(log)
        assert len(chunks) >= 2

    def test_no_boundaries_single_chunk(self):
        chunks = chunk_cicd_log("Just some log output with no test markers")
        assert len(chunks) == 1


class TestChunkEvent:
    def test_auto_detect_stack_trace(self):
        text = "Traceback (most recent call last):\n  File 'x.py', line 1\nException: boom"
        chunks = chunk_event(text, "sentry", "e1")
        assert chunks[0].source_type == "stack_trace"

    def test_jira_source(self):
        chunks = chunk_event("Bug report text", "jira", "e2")
        assert chunks[0].source_type == "jira"
