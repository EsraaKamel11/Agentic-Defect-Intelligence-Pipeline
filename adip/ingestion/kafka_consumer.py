"""
Kafka consumer for CI/CD and production event streams.
Falls back to a mock event generator if Kafka is unavailable.
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List

from adip.config.settings import settings

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """Consume defect events from Kafka or generate mock events."""

    def __init__(self):
        self._consumer = None
        self._use_kafka = False
        self._try_connect()

    def _try_connect(self):
        if settings.mock_mode:
            logger.info("Mock mode — skipping Kafka connection")
            return
        try:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                *settings.kafka_topics,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group,
                auto_offset_reset="latest",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=5000,
            )
            self._use_kafka = True
            logger.info("Connected to Kafka at %s", settings.kafka_bootstrap_servers)
        except Exception as exc:
            logger.warning("Kafka unavailable (%s); using mock event generator", exc)

    def consume_batch(self, max_events: int = 50) -> List[Dict[str, Any]]:
        """Pull a batch of events from Kafka or generate mocks."""
        if self._use_kafka and self._consumer:
            return self._consume_kafka(max_events)
        return self._generate_mock_events(max_events)

    def _consume_kafka(self, max_events: int) -> List[Dict[str, Any]]:
        events = []
        try:
            raw = self._consumer.poll(timeout_ms=5000, max_records=max_events)
            for tp, messages in raw.items():
                for msg in messages:
                    events.append(msg.value)
        except Exception as exc:
            logger.error("Kafka poll error: %s", exc)
        return events

    def _generate_mock_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic mock defect events from multiple sources."""
        events = []
        for _ in range(count):
            source = random.choice(["cicd", "jira", "sentry", "datadog"])
            event = _MOCK_GENERATORS[source]()
            events.append(event)
        logger.info("Generated %d mock events", len(events))
        return events

    def close(self):
        if self._consumer:
            self._consumer.close()


# ── Mock event generators ────────────────────────────────────────────────

_COMPONENTS = ["auth-service", "payment-api", "user-service", "gateway", "notification-service"]
_FILES = [
    "src/auth/login.py", "src/auth/token.py", "src/payment/charge.py",
    "src/payment/refund.py", "src/users/profile.py", "src/gateway/router.py",
    "src/notify/email.py", "src/notify/sms.py", "src/db/connection.py",
    "src/cache/redis_client.py",
]
_SEVERITIES = ["P0", "P1", "P2", "P3"]


def _mock_cicd() -> Dict[str, Any]:
    file = random.choice(_FILES)
    return {
        "source": "cicd",
        "raw_content": (
            f"Build #{ random.randint(1000, 9999)} FAILED\n"
            f"Test: test_{file.split('/')[-1].replace('.py', '')}\n"
            f"File: {file}\n"
            f"Error: AssertionError: expected 200 got 500\n"
            f"Traceback (most recent call last):\n"
            f'  File "{file}", line {random.randint(10, 200)}, in test_handler\n'
            f"    response = client.post('/api/endpoint')\n"
            f"  File \"src/core/http.py\", line 42, in post\n"
            f"    raise HTTPError(status_code=500)\n"
            f"AssertionError: expected 200 got 500\n"
            f"FAILED"
        ),
        "component": random.choice(_COMPONENTS),
        "file_path": file,
        "severity": random.choice(_SEVERITIES),
        "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(0, 48))).isoformat(),
    }


def _mock_jira() -> Dict[str, Any]:
    titles = [
        "NullPointerException in auth middleware on token refresh",
        "Race condition in payment processing causes double charge",
        "Memory leak in connection pool after 24h uptime",
        "Schema mismatch after DB migration v3.2.1",
        "Timeout on user profile API when cache is cold",
        "Config error: wrong Redis endpoint in staging",
        "Dependency failure: upstream auth service returns 503",
        "Input validation bypass allows negative payment amount",
    ]
    return {
        "source": "jira",
        "raw_content": (
            f"[{settings.jira_project_key}-{random.randint(100, 999)}] "
            f"{random.choice(titles)}\n\n"
            f"Description: This issue was observed in production after the latest deployment. "
            f"Multiple users reported the problem. Stack trace attached.\n\n"
            f"Steps to reproduce:\n1. Login as admin\n2. Trigger token refresh\n3. Observe 500 error\n\n"
            f"Impact: ~{random.randint(10, 5000)} users affected"
        ),
        "component": random.choice(_COMPONENTS),
        "file_path": random.choice(_FILES),
        "severity": random.choice(_SEVERITIES),
        "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(0, 72))).isoformat(),
    }


def _mock_sentry() -> Dict[str, Any]:
    errors = [
        ("NullReferenceError", "Cannot read property 'token' of null"),
        ("ConnectionError", "Connection pool exhausted after 100 retries"),
        ("TimeoutError", "Request to /api/users timed out after 30s"),
        ("ValidationError", "Field 'amount' must be positive, got -50"),
        ("AuthenticationError", "JWT signature verification failed"),
    ]
    err_type, err_msg = random.choice(errors)
    file = random.choice(_FILES)
    return {
        "source": "sentry",
        "raw_content": (
            f"{err_type}: {err_msg}\n"
            f"Traceback (most recent call last):\n"
            f'  File "{file}", line {random.randint(10, 300)}, in handle_request\n'
            f"    result = process(data)\n"
            f'  File "src/core/processor.py", line {random.randint(10, 100)}, in process\n'
            f"    return transform(data.payload)\n"
            f'  File "src/core/transform.py", line {random.randint(10, 50)}, in transform\n'
            f"    raise {err_type}('{err_msg}')\n"
            f"{err_type}: {err_msg}"
        ),
        "component": random.choice(_COMPONENTS),
        "file_path": file,
        "severity": random.choice(["P0", "P1", "P2"]),
        "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(1, 600))).isoformat(),
    }


def _mock_datadog() -> Dict[str, Any]:
    monitors = [
        "High error rate on auth-service (>5% 5xx)",
        "Memory usage > 90% on payment-api pod",
        "Latency p99 > 2s on gateway",
        "Connection pool saturation on user-service",
    ]
    return {
        "source": "datadog",
        "raw_content": (
            f"[ALERT] {random.choice(monitors)}\n"
            f"Monitor triggered at {datetime.utcnow().isoformat()}\n"
            f"Threshold: exceeded for >5 minutes\n"
            f"Tags: env:production, service:{random.choice(_COMPONENTS)}\n"
            f"Status: CRITICAL\n"
            f"FAILED health check"
        ),
        "component": random.choice(_COMPONENTS),
        "severity": random.choice(["P0", "P1"]),
        "timestamp": datetime.utcnow().isoformat(),
    }


_MOCK_GENERATORS = {
    "cicd": _mock_cicd,
    "jira": _mock_jira,
    "sentry": _mock_sentry,
    "datadog": _mock_datadog,
}
