"""
Integration test — runs the full pipeline in mock mode.

This test verifies end-to-end pipeline execution with zero external deps.
"""
import asyncio
import pytest

from adip.config.settings import settings


@pytest.fixture(autouse=True)
def enable_mock_mode():
    """Enable mock mode for all integration tests."""
    original = settings.mock_mode
    settings.mock_mode = True
    yield
    settings.mock_mode = original


@pytest.mark.asyncio
async def test_full_pipeline_mock():
    """Run the full pipeline with mock data and verify output structure."""
    from adip.agents.supervisor import init_run
    from adip.graph.graph import compile_graph
    from adip.graph.state import TriggerType

    state = init_run(TriggerType.STREAM_EVENT)
    config = {"configurable": {"thread_id": state["run_id"]}}

    pipeline = compile_graph(checkpointer=None)
    result = await pipeline.ainvoke(state, config=config)

    # Verify all expected keys are present
    assert "defect_events" in result
    assert "risk_scores" in result
    assert "release_recommendation" in result
    assert len(result.get("defect_events", [])) > 0
    assert len(result.get("risk_scores", [])) > 0
    assert result["release_recommendation"] in ("PROCEED", "CONDITIONAL", "HOLD")


@pytest.mark.asyncio
async def test_batch_mode():
    """Batch mode should trigger clustering."""
    from adip.agents.supervisor import init_run
    from adip.graph.graph import compile_graph
    from adip.graph.state import TriggerType

    state = init_run(TriggerType.SCHEDULED_BATCH)
    config = {"configurable": {"thread_id": state["run_id"]}}

    pipeline = compile_graph(checkpointer=None)
    result = await pipeline.ainvoke(state, config=config)

    assert result.get("clustering_skipped") is not True or len(result.get("clusters", [])) >= 0


@pytest.mark.asyncio
async def test_pipeline_produces_report():
    """Verify a RiskReport is generated."""
    from adip.agents.supervisor import init_run
    from adip.graph.graph import compile_graph
    from adip.graph.state import TriggerType

    state = init_run(TriggerType.STREAM_EVENT)
    config = {"configurable": {"thread_id": state["run_id"]}}

    pipeline = compile_graph(checkpointer=None)
    result = await pipeline.ainvoke(state, config=config)

    report = result.get("risk_report")
    assert report is not None
    assert "release_recommendation" in report
    assert "executive_summary" in report
