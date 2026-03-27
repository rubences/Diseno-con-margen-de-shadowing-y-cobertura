"""LangGraph workflow — three-node directed state graph.

Nodes
-----
1. ``validate_input``   – Validates and normalises the input with Pydantic.
2. ``run_crew``         – Calls the CrewAI researcher/writer crew.
3. ``format_output``    – Parses the crew text output into a Pydantic model
                          and stores it in ``GraphState["final_output"]``.

The graph is compiled once (module-level ``app``) and can be reused across
multiple invocations.

Usage::

    from diseno_multiagente.graphs.basic_graph import build_graph

    app = build_graph()
    result = app.invoke({
        "topic": "LTE coverage design",
        "context": "industrial warehouses",
        "max_words": 300,
    })
    print(result["final_output"])
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from diseno_multiagente.core.models import GraphState, ResearchInput, ResearchOutput
from diseno_multiagente.crews.basic_crew import build_crew

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------


def validate_input_node(state: GraphState) -> dict[str, Any]:
    """Node 1 — Validate and normalise the workflow input.

    Raises ``pydantic.ValidationError`` if required fields are missing or
    invalid; this surfaces naturally as a graph execution error.
    """
    research_input = ResearchInput(
        topic=state.get("topic", ""),
        context=state.get("context") or None,
        max_words=state.get("max_words", 300),
    )
    logger.info(  # noqa: E501 — long but intentional for debugging
        "validate_input: topic=%r max_words=%d",
        research_input.topic,
        research_input.max_words,
    )
    return {
        "topic": research_input.topic,
        "context": research_input.context or "",
        "max_words": research_input.max_words,
        "status": "validated",
    }


def run_crew_node(state: GraphState) -> dict[str, Any]:
    """Node 2 — Execute the CrewAI researcher/writer crew.

    Passes ``topic`` and ``context`` to the crew and stores the raw text
    output in ``state["crew_result"]``.
    """
    topic: str = state["topic"]
    context: str = state.get("context", "")
    logger.info("run_crew: starting crew for topic=%r", topic)

    crew = build_crew(topic=topic, context=context)
    result = crew.kickoff()
    raw_text: str = getattr(result, "raw", str(result))

    logger.info("run_crew: crew finished, output length=%d chars", len(raw_text))
    return {
        "crew_result": raw_text,
        "status": "crew_completed",
    }


def format_output_node(state: GraphState) -> dict[str, Any]:
    """Node 3 — Parse and validate the crew output with Pydantic.

    Converts the raw text into a ``ResearchOutput`` instance and serialises
    it to a plain dict stored in ``state["final_output"]``.
    """
    raw: str = state.get("crew_result", "")
    topic: str = state["topic"]

    output: ResearchOutput = ResearchOutput.from_raw_text(topic=topic, raw=raw)
    logger.info(
        "format_output: findings=%d word_count=%d",
        len(output.key_findings),
        output.word_count,
    )
    return {
        "final_output": output.model_dump(),
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Build and compile the LangGraph ``StateGraph``.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph application ready to ``.invoke()`` or ``.stream()``.
    """
    builder: StateGraph = StateGraph(GraphState)

    builder.add_node("validate_input", validate_input_node)
    builder.add_node("run_crew", run_crew_node)
    builder.add_node("format_output", format_output_node)

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "run_crew")
    builder.add_edge("run_crew", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile()
