"""Unit tests for ingestion/normalizer.py."""
from adip.graph.state import DefectSeverity
from adip.ingestion.normalizer import normalize_event, normalize_batch


class TestNormalizer:
    def test_cicd_event(self):
        raw = {
            "source": "cicd",
            "raw_content": "Build #1234 FAILED\nFile: src/auth.py\nAssertionError",
            "severity": "P1",
            "component": "auth-service",
        }
        event = normalize_event(raw)
        assert event.source == "cicd"
        assert event.severity == DefectSeverity.P1

    def test_auto_detect_sentry(self):
        raw = {
            "raw_content": "Traceback (most recent call last):\n  File 'x.py'\nSentryError",
        }
        event = normalize_event(raw)
        assert event.source == "sentry"

    def test_extract_file_path(self):
        raw = {
            "source": "cicd",
            "raw_content": 'File "src/payment/charge.py", line 42',
        }
        event = normalize_event(raw)
        assert event.file_path == "src/payment/charge.py"

    def test_batch_normalize(self):
        raws = [
            {"source": "jira", "raw_content": "Bug 1"},
            {"source": "cicd", "raw_content": "Build failed"},
        ]
        events = normalize_batch(raws)
        assert len(events) == 2

    def test_severity_mapping(self):
        for sev, expected in [("P0", DefectSeverity.P0), ("CRITICAL", DefectSeverity.P0),
                               ("HIGH", DefectSeverity.P1), ("LOW", DefectSeverity.P3)]:
            event = normalize_event({"source": "jira", "raw_content": "x", "severity": sev})
            assert event.severity == expected
