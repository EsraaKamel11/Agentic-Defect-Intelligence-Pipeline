"""
Persistence layer — PostgreSQL with JSONB, falls back to SQLite.

Tables: defect_events, clusters, risk_scores, reports, test_directives, weight_history
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class _DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from adip.config.settings import settings

logger = logging.getLogger(__name__)


class DefectDB:
    """Unified DB interface — PostgreSQL preferred, SQLite fallback."""

    def __init__(self):
        self._pg_pool = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._use_pg = False

    async def initialize(self):
        """Connect and create tables."""
        if await self._try_postgres():
            self._use_pg = True
        else:
            self._init_sqlite()
        await self._create_tables()

    async def _try_postgres(self) -> bool:
        if settings.mock_mode:
            return False
        try:
            import asyncpg
            self._pg_pool = await asyncpg.create_pool(
                settings.database_url, min_size=2, max_size=10, timeout=5
            )
            logger.info("Connected to PostgreSQL: %s", settings.database_url)
            return True
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); falling back to SQLite", exc)
            return False

    def _init_sqlite(self):
        db_path = Path(settings.sqlite_fallback_path)
        self._sqlite_conn = sqlite3.connect(str(db_path))
        self._sqlite_conn.row_factory = sqlite3.Row
        self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
        logger.info("Using SQLite fallback: %s", db_path)

    async def _create_tables(self):
        ddl = [
            """CREATE TABLE IF NOT EXISTS defect_events (
                id TEXT PRIMARY KEY,
                source TEXT,
                raw_content TEXT,
                normalized_content TEXT,
                component TEXT,
                file_path TEXT,
                severity TEXT,
                stack_trace TEXT,
                timestamp TEXT,
                embedding_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                label TEXT,
                root_cause_category TEXT,
                member_count INTEGER,
                recurrence_count INTEGER,
                last_seen TEXT,
                weight REAL,
                data_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                component TEXT,
                risk_score REAL,
                risk_tier TEXT,
                data_json TEXT,
                run_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                trigger_type TEXT,
                release_recommendation TEXT,
                executive_summary TEXT,
                data_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS test_directives (
                directive_id TEXT PRIMARY KEY,
                target_file TEXT,
                test_type TEXT,
                priority TEXT,
                scenarios_json TEXT,
                cluster_id TEXT,
                published_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weights_json TEXT,
                update_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
        ]
        if self._use_pg:
            async with self._pg_pool.acquire() as conn:
                for stmt in ddl:
                    # Convert SQLite-isms to PostgreSQL
                    pg_stmt = stmt.replace("AUTOINCREMENT", "GENERATED ALWAYS AS IDENTITY")
                    pg_stmt = pg_stmt.replace("INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                                              "SERIAL PRIMARY KEY")
                    await conn.execute(pg_stmt)
        else:
            for stmt in ddl:
                self._sqlite_conn.execute(stmt)
            self._sqlite_conn.commit()

    # ── Generic helpers ──────────────────────────────────────────────────

    async def _execute(self, sql: str, params: tuple = ()):
        if self._use_pg:
            async with self._pg_pool.acquire() as conn:
                await conn.execute(sql, *params)
        else:
            self._sqlite_conn.execute(sql, params)
            self._sqlite_conn.commit()

    async def _fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        if self._use_pg:
            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
                return [dict(r) for r in rows]
        else:
            cur = self._sqlite_conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    async def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        if self._use_pg:
            async with self._pg_pool.acquire() as conn:
                row = await conn.fetchrow(sql, *params)
                return dict(row) if row else None
        else:
            cur = self._sqlite_conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    # ── Defect Events ────────────────────────────────────────────────────

    async def store_defect_event(self, event: Dict[str, Any]):
        await self._execute(
            """INSERT OR REPLACE INTO defect_events
               (id, source, raw_content, normalized_content, component, file_path,
                severity, stack_trace, timestamp, embedding_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event["id"], event["source"], event.get("raw_content", ""),
                event.get("normalized_content", ""), event.get("component", ""),
                event.get("file_path"), event.get("severity", "P3"),
                event.get("stack_trace"), event.get("timestamp", ""),
                event.get("embedding_id"),
            ),
        )

    async def store_defect_events_batch(self, events: List[Dict[str, Any]]):
        for event in events:
            await self.store_defect_event(event)

    async def get_defect_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        return await self._fetch_all(
            "SELECT * FROM defect_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        )

    # ── Clusters ─────────────────────────────────────────────────────────

    async def store_clusters(self, clusters: List[Dict[str, Any]]):
        for c in clusters:
            await self._execute(
                """INSERT OR REPLACE INTO clusters
                   (cluster_id, label, root_cause_category, member_count,
                    recurrence_count, last_seen, weight, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    c["cluster_id"], c["label"], c.get("root_cause_category", "UNKNOWN"),
                    c.get("member_count", 0), c.get("recurrence_count", 0),
                    c.get("last_seen", ""), c.get("weight", 1.0),
                    json.dumps(c, cls=_DateTimeEncoder),
                ),
            )

    async def get_clusters(self) -> List[Dict[str, Any]]:
        rows = await self._fetch_all("SELECT * FROM clusters ORDER BY weight DESC")
        return [
            {**dict(r), **(json.loads(r["data_json"]) if r.get("data_json") else {})}
            for r in rows
        ]

    # ── Risk Scores ──────────────────────────────────────────────────────

    async def store_risk_scores(self, scores: List[Dict[str, Any]], run_id: str):
        for s in scores:
            await self._execute(
                """INSERT INTO risk_scores (file_path, component, risk_score, risk_tier, data_json, run_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    s["file_path"], s.get("component", ""), s["risk_score"],
                    s.get("risk_tier", "LOW"), json.dumps(s, cls=_DateTimeEncoder), run_id,
                ),
            )

    async def get_latest_risk_scores(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if run_id:
            return await self._fetch_all(
                "SELECT * FROM risk_scores WHERE run_id = ? ORDER BY risk_score DESC", (run_id,)
            )
        return await self._fetch_all(
            "SELECT * FROM risk_scores ORDER BY created_at DESC LIMIT 50"
        )

    # ── Reports ──────────────────────────────────────────────────────────

    async def store_report(self, report: Dict[str, Any]):
        await self._execute(
            """INSERT OR REPLACE INTO reports
               (report_id, trigger_type, release_recommendation, executive_summary, data_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                report["report_id"], report.get("trigger_type", ""),
                report.get("release_recommendation", "PROCEED"),
                report.get("executive_summary", ""), json.dumps(report, cls=_DateTimeEncoder),
            ),
        )

    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        row = await self._fetch_one(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT 1"
        )
        if row and row.get("data_json"):
            return json.loads(row["data_json"])
        return row

    # ── Test Directives ──────────────────────────────────────────────────

    async def store_test_directives(self, directives: List[Dict[str, Any]]):
        for d in directives:
            await self._execute(
                """INSERT OR REPLACE INTO test_directives
                   (directive_id, target_file, test_type, priority, scenarios_json, cluster_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    d["directive_id"], d["target_file"], d["test_type"],
                    d.get("priority", "medium"),
                    json.dumps(d.get("scenarios", [])),
                    d.get("cluster_id"),
                ),
            )

    async def store_test_outcome(self, outcome: Dict[str, Any]):
        """Store a test outcome (uses weight_history table for simplicity)."""
        pass  # Outcomes feed directly into bayesian_updater

    # ── Weight History ───────────────────────────────────────────────────

    async def store_weight_history(
        self, weights: Dict[str, float], update_count: int, timestamp: datetime
    ):
        await self._execute(
            "INSERT INTO weight_history (weights_json, update_count) VALUES (?, ?)",
            (json.dumps(weights), update_count),
        )

    async def get_latest_weights(self) -> Optional[Dict[str, Any]]:
        row = await self._fetch_one(
            "SELECT * FROM weight_history ORDER BY id DESC LIMIT 1"
        )
        if row and row.get("weights_json"):
            return {
                "weights": json.loads(row["weights_json"]),
                "update_count": row.get("update_count", 0),
            }
        return None

    # ── Cleanup ──────────────────────────────────────────────────────────

    async def close(self):
        if self._pg_pool:
            await self._pg_pool.close()
        if self._sqlite_conn:
            self._sqlite_conn.close()
