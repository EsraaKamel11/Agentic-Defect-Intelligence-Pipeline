"""
ADIP main entry point.

Usage:
    python -m adip.main                      # Normal mode
    python -m adip.main --mock               # Full pipeline with zero external deps
    python -m adip.main --mode batch         # Force batch mode
    python -m adip.main --mode streaming     # Streaming mode
    python -m adip.main --api                # Start FastAPI server
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("adip")


def parse_args():
    parser = argparse.ArgumentParser(description="ADIP — Agentic Defect Intelligence Pipeline")
    parser.add_argument("--mock", action="store_true", help="Run with all mock fallbacks")
    parser.add_argument("--mode", choices=["streaming", "batch", "on_demand"], default="streaming")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    return parser.parse_args()


async def run_pipeline(mode: str):
    """Execute a single pipeline run."""
    from adip.agents.supervisor import init_run
    from adip.graph.graph import compile_graph, get_checkpointer
    from adip.graph.state import TriggerType
    from adip.observability.langsmith_config import configure_langsmith
    from adip.observability.metrics import start_metrics_server

    configure_langsmith()

    # Map mode to trigger type
    trigger_map = {
        "streaming": TriggerType.STREAM_EVENT,
        "batch": TriggerType.SCHEDULED_BATCH,
        "on_demand": TriggerType.PR_WEBHOOK,
    }
    trigger = trigger_map.get(mode, TriggerType.STREAM_EVENT)

    # Build graph
    checkpointer = get_checkpointer()
    pipeline = compile_graph(checkpointer)

    # Initialize state
    state = init_run(trigger)
    config = {"configurable": {"thread_id": state["run_id"]}}

    logger.info("=" * 60)
    logger.info("ADIP Pipeline — mode=%s, run_id=%s", mode, state["run_id"])
    logger.info("=" * 60)

    start = time.monotonic()

    # Execute
    try:
        result = await pipeline.ainvoke(state, config=config)
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        return

    elapsed = time.monotonic() - start

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.2fs", elapsed)
    logger.info("  Trigger:         %s", result.get("trigger_type", "?"))
    logger.info("  Events ingested: %d", len(result.get("defect_events", [])))
    logger.info("  Indexed:         %d", result.get("indexed_count", 0))
    logger.info("  Clusters:        %d", len(result.get("clusters", [])))
    logger.info("  Risk scores:     %d", len(result.get("risk_scores", [])))
    logger.info("  Recommendation:  %s", result.get("release_recommendation", "?"))
    logger.info("  Directives:      %d", len(result.get("test_directives", [])))
    logger.info("  Notifications:   %d", len(result.get("notifications", [])))
    logger.info("  Errors:          %s", result.get("errors", []))
    logger.info("=" * 60)

    # Print executive summary
    report = result.get("risk_report", {})
    if report:
        logger.info("\nEXECUTIVE SUMMARY:")
        logger.info(report.get("executive_summary", "(none)"))
        logger.info("\nRECOMMENDED ACTIONS:")
        for action in report.get("recommended_actions", []):
            logger.info("  • %s", action)

    # Check HITL
    if result.get("human_review_required"):
        logger.warning("\n⚠ HUMAN REVIEW REQUIRED — recommendation is HOLD")


async def run_batch_loop():
    """Run batch mode on a 6-hour loop."""
    from adip.config.settings import settings
    interval = settings.batch_interval_hours * 3600

    while True:
        logger.info("Starting scheduled batch run...")
        await run_pipeline("batch")
        logger.info("Next batch in %d hours", settings.batch_interval_hours)
        await asyncio.sleep(interval)


def start_api(port: int):
    """Start FastAPI server."""
    import uvicorn
    from adip.api.app import app
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main():
    args = parse_args()

    # Set mock mode
    if args.mock:
        from adip.config.settings import settings
        settings.mock_mode = True
        logger.info("🔧 Mock mode enabled — all external services will use fallbacks")

    if args.api:
        start_api(args.port)
        return

    if args.mode == "batch":
        asyncio.run(run_batch_loop())
    else:
        asyncio.run(run_pipeline(args.mode))


if __name__ == "__main__":
    main()
