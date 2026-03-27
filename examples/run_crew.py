"""examples/run_crew.py — End-to-end CrewAI workflow example.

Runs the two-agent CrewAI crew (Researcher + Writer) directly, without
going through LangGraph, and prints the Pydantic-validated output.

Prerequisites
-------------
1. Install the package::

       pip install -e .

2. Copy ``.env.example`` to ``.env`` and fill in your API keys.

3. Run::

       python examples/run_crew.py
       # or
       python -m examples.run_crew
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from diseno_multiagente.core.models import ResearchOutput  # noqa: E402
from diseno_multiagente.crews.basic_crew import build_crew  # noqa: E402
from diseno_multiagente.observability.langsmith import (  # noqa: E402
    configure_langsmith,
    tracing_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Configure LangSmith tracing
    cfg = configure_langsmith()
    logger.info("LangSmith config: %s", cfg)

    status = tracing_status()
    if status["tracing_enabled"] == "true":
        print(
            f"\n🔭  LangSmith tracing ON  »  project='{status['project']}'  "
            f"endpoint={status['endpoint']}\n"
        )
    else:
        print("\n🔭  LangSmith tracing OFF  (set LANGCHAIN_TRACING_V2=true to enable)\n")

    # 2. Build and run the crew
    topic = "shadowing margin calculation for cellular networks"
    context = "Focus on log-normal shadowing model and 95th-percentile coverage."

    print("=" * 60)
    print("Starting CrewAI crew")
    print(f"  topic   : {topic}")
    print(f"  context : {context}")
    print("=" * 60 + "\n")

    crew = build_crew(topic=topic, context=context)
    crew_result = crew.kickoff()

    raw_text: str = getattr(crew_result, "raw", str(crew_result))

    # 3. Validate output with Pydantic
    output: ResearchOutput = ResearchOutput.from_raw_text(topic=topic, raw=raw_text)

    # 4. Display results
    print("\n" + "=" * 60)
    print("Crew completed — Pydantic-validated output:")
    print("=" * 60)
    print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))

    if status["tracing_enabled"] == "true":
        print(
            f"\n✅  Traces visible in LangSmith UI under project '{status['project']}'.\n"
            "    Visit https://smith.langchain.com to inspect the run.\n"
        )


if __name__ == "__main__":
    main()
