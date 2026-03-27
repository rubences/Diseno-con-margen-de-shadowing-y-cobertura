"""Pydantic models for inputs, graph state, and outputs.

These models serve as the data contracts throughout the LangGraph
workflow and CrewAI crew executions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Constants used in output parsing
# ---------------------------------------------------------------------------

_MAX_FALLBACK_FINDINGS: int = 5   # max bullet lines to use when no bullets found
_MAX_SUMMARY_CHARS: int = 300     # fallback character limit when building a summary

# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class ResearchInput(BaseModel):
    """Validated input for a research workflow."""

    topic: str = Field(..., min_length=3, description="Topic to research")
    context: str | None = Field(None, description="Additional background context")
    max_words: int = Field(300, ge=50, le=2000, description="Target word count for output")


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class ResearchOutput(BaseModel):
    """Validated output produced by the research crew and returned by the graph."""

    topic: str = Field(..., description="The researched topic")
    summary: str = Field(..., description="Concise narrative summary")
    key_findings: list[str] = Field(
        ..., min_length=1, description="Bullet-point key findings"
    )
    word_count: int = Field(..., ge=0, description="Approximate word count of the summary")

    @classmethod
    def from_raw_text(cls, topic: str, raw: str) -> ResearchOutput:
        """Build a validated ResearchOutput from raw crew text output."""
        lines = raw.splitlines()
        findings = [
            ln.lstrip("-• ").strip()
            for ln in lines
            if ln.strip().startswith(("-", "•", "*"))
        ]
        # Fall back: split non-empty lines as individual findings
        if not findings:
            all_lines = [ln.strip() for ln in lines if ln.strip()]
            findings = all_lines[:_MAX_FALLBACK_FINDINGS] or [raw[:_MAX_SUMMARY_CHARS]]

        # First non-bullet line (or the whole text) becomes the summary
        non_bullets = [
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.strip().startswith(("-", "•", "*"))
        ]
        summary = " ".join(non_bullets[:3]) if non_bullets else raw[:_MAX_SUMMARY_CHARS]

        return cls(
            topic=topic,
            summary=summary,
            key_findings=findings,
            word_count=len(raw.split()),
        )


# ---------------------------------------------------------------------------
# LangGraph state (TypedDict — one dict updated across nodes)
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """Shared mutable state flowing through all LangGraph nodes."""

    # Set at graph entry
    topic: str
    context: str
    max_words: int

    # Populated by the crew node
    crew_result: str

    # Populated by the finalise node
    final_output: dict  # serialised ResearchOutput

    # Lifecycle tracking
    status: str
    error: str
