"""
FastAPI application — REST API for ADIP pipeline.

Endpoints:
- POST /run          — trigger pipeline run
- GET  /status       — last run status
- GET  /reports      — list reports
- GET  /health       — health check
- POST /webhook/pr   — PR webhook trigger
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from adip.agents.supervisor import init_run
from adip.graph.state import TriggerType

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ADIP — Agentic Defect Intelligence Pipeline",
    version="1.0.0",
    description="Multi-agent AI pipeline for defect risk assessment",
)

# Module-level state
_last_result: Dict[str, Any] = {}
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from adip.graph.graph import compile_graph, get_checkpointer
        checkpointer = get_checkpointer()
        _pipeline = compile_graph(checkpointer)
    return _pipeline


class RunRequest(BaseModel):
    trigger_type: str = "STREAM_EVENT"
    raw_events: list = []


class PRWebhookPayload(BaseModel):
    action: str = "opened"
    pr_number: int = 0
    repo: str = ""
    changed_files: list = []


@app.get("/health")
async def health():
    return {"status": "ok", "service": "adip"}


@app.post("/run")
async def trigger_run(req: RunRequest, background_tasks: BackgroundTasks):
    """Trigger a pipeline run."""
    global _last_result
    state = init_run(req.trigger_type)
    if req.raw_events:
        state["raw_events"] = req.raw_events

    pipeline = _get_pipeline()

    async def _run():
        global _last_result
        try:
            config = {"configurable": {"thread_id": state["run_id"]}}
            result = await pipeline.ainvoke(state, config=config)
            _last_result = result
            logger.info("Pipeline run %s completed", state["run_id"])
        except Exception as exc:
            logger.error("Pipeline run failed: %s", exc)
            _last_result = {"error": str(exc), "run_id": state["run_id"]}

    background_tasks.add_task(_run)
    return {"run_id": state["run_id"], "status": "started", "trigger": req.trigger_type}


@app.get("/status")
async def get_status():
    """Get last run result."""
    if not _last_result:
        return {"status": "no runs yet"}
    return {
        "run_id": _last_result.get("run_id", ""),
        "recommendation": _last_result.get("release_recommendation", ""),
        "errors": _last_result.get("errors", []),
        "clusters": len(_last_result.get("clusters", [])),
        "risk_scores": len(_last_result.get("risk_scores", [])),
        "directives": len(_last_result.get("test_directives", [])),
    }


@app.get("/reports")
async def get_reports():
    """Get the latest risk report."""
    report = _last_result.get("risk_report")
    if not report:
        return {"reports": []}
    return {"reports": [report]}


@app.post("/webhook/pr")
async def pr_webhook(payload: PRWebhookPayload, background_tasks: BackgroundTasks):
    """PR webhook — triggers on-demand pipeline run (<3 min target)."""
    global _last_result
    state = init_run(TriggerType.PR_WEBHOOK)
    state["raw_events"] = [
        {
            "source": "cicd",
            "raw_content": (
                f"PR #{payload.pr_number} {payload.action} on {payload.repo}\n"
                f"Changed files: {', '.join(payload.changed_files)}"
            ),
            "component": payload.repo,
            "severity": "P2",
            "file_path": payload.changed_files[0] if payload.changed_files else None,
        }
    ]

    pipeline = _get_pipeline()

    async def _run():
        global _last_result
        try:
            config = {"configurable": {"thread_id": state["run_id"]}}
            result = await pipeline.ainvoke(state, config=config)
            _last_result = result
        except Exception as exc:
            logger.error("PR webhook run failed: %s", exc)
            _last_result = {"error": str(exc)}

    background_tasks.add_task(_run)
    return {
        "run_id": state["run_id"],
        "status": "started",
        "trigger": "PR_WEBHOOK",
        "pr": payload.pr_number,
    }
