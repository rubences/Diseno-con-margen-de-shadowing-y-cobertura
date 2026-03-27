"""examples/run_graph.py — End-to-end LangGraph workflow example.

Runs the three-node LangGraph research workflow and prints the
Pydantic-validated output to the console.

Prerequisites
-------------
1. Install the package::

       pip install -e .

2. Copy ``.env.example`` to ``.env`` and fill in your API keys.

3. Run::

       python examples/run_graph.py
       # or
       python -m examples.run_graph
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

from diseno_multiagente.graphs.basic_graph import build_graph  # noqa: E402
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
    # 1. Configure LangSmith tracing (reads from .env / environment)
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

    # 2. Build the compiled graph
    app = build_graph()

    # 3. Define the initial state
    initial_state = {
        "topic": "LTE coverage design with shadowing margins",
        "context": (
            "Industrial warehouses and loading docks; "
            "95% edge coverage requirement; frequency 1800 MHz."
        ),
        "max_words": 350,
    }

    print("=" * 60)
    print("Starting LangGraph workflow")
    print(f"  topic   : {initial_state['topic']}")
    print(f"  context : {initial_state['context']}")
    print("=" * 60 + "\n")

    # 4. Run the graph
    final_state = app.invoke(initial_state)

    # 5. Display results
    print("\n" + "=" * 60)
    print("Workflow completed — Pydantic-validated output:")
    print("=" * 60)
    output = final_state.get("final_output", {})
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if tracing_status()["tracing_enabled"] == "true":
        print(
            f"\n✅  Traces visible in LangSmith UI under project '{status['project']}'.\n"
            "    Visit https://smith.langchain.com to inspect the run.\n"
        )


if __name__ == "__main__":
    main()
