# Diseño con Margen de Shadowing y Cobertura

A Python project that combines **LTE coverage design** tooling with a
**multi-agent AI architecture** built on LangGraph, CrewAI, Pydantic, and LangSmith.

---

## Table of Contents

1. [Project structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Environment variables](#environment-variables)
5. [Running the examples](#running-the-examples)
6. [LangSmith observability](#langsmith-observability)
7. [LTE coverage script](#lte-coverage-script)

---

## Project structure

```
.
├── pyproject.toml                        # PEP 621 project + dependencies
├── .env.example                          # Environment variable template
├── diseno_shadowing_cobertura.py         # Original LTE coverage script
├── diseno_shadowing_cobertura.ipynb      # Jupyter notebook version
│
├── src/
│   └── diseno_multiagente/
│       ├── core/
│       │   └── models.py                 # Pydantic models (input / state / output)
│       ├── graphs/
│       │   └── basic_graph.py            # LangGraph 3-node workflow
│       ├── crews/
│       │   └── basic_crew.py             # CrewAI crew (Researcher + Writer)
│       └── observability/
│           └── langsmith.py              # LangSmith tracing helper
│
└── examples/
    ├── run_graph.py                      # Runs the LangGraph workflow end-to-end
    └── run_crew.py                       # Runs the CrewAI workflow end-to-end
```

---

## Requirements

- Python 3.11 or later
- An [OpenAI API key](https://platform.openai.com/api-keys) (for LLM calls)
- *(Optional)* A [LangSmith API key](https://smith.langchain.com) for tracing

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rubences/Diseno-con-margen-de-shadowing-y-cobertura.git
cd Diseno-con-margen-de-shadowing-y-cobertura

# 2. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the package and all dependencies
pip install -e .
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# then edit .env with your actual keys
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for LLM calls |
| `OPENAI_MODEL` | ❌ | Model name (default: `gpt-4o-mini`) |
| `LANGCHAIN_API_KEY` | ❌ | LangSmith API key (for tracing) |
| `LANGCHAIN_TRACING_V2` | ❌ | Set `true` to enable tracing (default: `false`) |
| `LANGCHAIN_PROJECT` | ❌ | LangSmith project name (default: `diseno-multiagente`) |
| `LANGCHAIN_ENDPOINT` | ❌ | LangSmith endpoint (default: `https://api.smith.langchain.com`) |

---

## Running the examples

### LangGraph workflow (3-node directed graph)

```bash
python examples/run_graph.py
# or
python -m examples.run_graph
```

The graph executes three nodes in sequence:

1. **`validate_input`** — validates the topic and parameters with Pydantic.
2. **`run_crew`** — invokes the CrewAI researcher/writer crew via an LLM.
3. **`format_output`** — parses the raw output into a `ResearchOutput` Pydantic model.

### CrewAI multi-agent crew

```bash
python examples/run_crew.py
# or
python -m examples.run_crew
```

Two agents collaborate sequentially:

- **Senior Researcher** — compiles bullet-point findings.
- **Technical Writer** — produces a structured report from those findings.

Both examples print the Pydantic-validated JSON output to the console.

---

## LangSmith observability

When `LANGCHAIN_TRACING_V2=true` is set, every LLM call, agent step, and
graph node execution is traced automatically.

To verify in the UI:

1. Open <https://smith.langchain.com> and log in.
2. Navigate to **Projects** → `diseno-multiagente` (or your chosen project name).
3. You will see a run for each `python examples/run_*.py` execution with:
   - Latency per node / agent step
   - Token counts
   - Input/output for each LLM call
   - Error details (if any)

---

## LTE coverage script

The original simulation script is unchanged:

```bash
python diseno_shadowing_cobertura.py
```

It calculates shadowing margins and area coverage for LTE macro-cells and
saves several figures (`fig_*.png`) to the working directory.
