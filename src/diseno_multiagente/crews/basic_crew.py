"""CrewAI multi-agent crew — Researcher + Writer.

The crew takes a *topic* and produces a structured research report:

* **Researcher agent** — gathers key facts and findings.
* **Writer agent**    — turns the findings into a polished summary.

The output is a plain-text report; callers (e.g. the LangGraph node)
are responsible for parsing it with Pydantic (see ``ResearchOutput``).

Usage::

    from diseno_multiagente.crews.basic_crew import build_crew

    crew = build_crew(topic="LTE coverage design", context="industrial sites")
    result = crew.kickoff()
    print(result.raw)
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI


def _get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured from environment variables."""
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def build_crew(topic: str, context: str = "") -> Crew:
    """Build and return a CrewAI ``Crew`` ready to be kicked off.

    Parameters
    ----------
    topic:
        The subject to research and write about.
    context:
        Optional background information that guides the agents.

    Returns
    -------
    Crew
        A configured CrewAI crew (call ``.kickoff()`` to run it).
    """
    llm = _get_llm()
    context_note = f"\nAdditional context: {context}" if context else ""

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    researcher = Agent(
        role="Senior Researcher",
        goal=(
            f"Research '{topic}' thoroughly and compile a list of factual, "
            "well-sourced key findings."
        ),
        backstory=(
            "You are an experienced researcher with broad expertise. "
            "You always back your findings with clear evidence and present "
            "them in concise bullet-point form."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal=(
            f"Transform research findings about '{topic}' into a clear, "
            "structured written report."
        ),
        backstory=(
            "You are a skilled technical writer who excels at distilling "
            "complex research into readable, well-organised summaries."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    research_task = Task(
        description=(
            f"Research the following topic: '{topic}'.{context_note}\n\n"
            "Your output must be a bullet-point list (each line starting with '- ') "
            "containing at least 5 distinct, factual findings. "
            "Be specific and informative."
        ),
        expected_output=(
            "A bullet-point list of at least 5 key findings about the topic, "
            "each beginning with '- '."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            f"Using the research findings provided, write a structured report about '{topic}'.\n\n"
            "Format your output as follows:\n"
            "1. A concise summary paragraph (2–4 sentences).\n"
            "2. A section titled 'Key Findings:' with each finding on its own line "
            "starting with '- '.\n\n"
            "Keep the total output under 400 words."
        ),
        expected_output=(
            "A structured report with a summary paragraph followed by "
            "a 'Key Findings:' bullet-point section."
        ),
        agent=writer,
        context=[research_task],
    )

    # ------------------------------------------------------------------
    # Crew
    # ------------------------------------------------------------------

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )
