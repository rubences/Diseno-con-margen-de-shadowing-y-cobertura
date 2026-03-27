"""LangSmith observability helper.

Reads configuration from environment variables and activates LangSmith
tracing for all LangChain / LangGraph calls made in the same process.

Required environment variables (when tracing is desired):
    LANGCHAIN_API_KEY      – Your LangSmith API key.
    LANGCHAIN_PROJECT      – Project name shown in the LangSmith UI
                             (defaults to "diseno-multiagente").

Optional:
    LANGCHAIN_TRACING_V2   – Set to "true" to enable tracing (default: "false").
    LANGCHAIN_ENDPOINT     – API endpoint (default: https://api.smith.langchain.com).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_langsmith(
    *,
    project: str | None = None,
    enable_tracing: bool | None = None,
) -> dict[str, str]:
    """Configure LangSmith tracing from environment variables.

    Parameters
    ----------
    project:
        Override the project name. Falls back to the
        ``LANGCHAIN_PROJECT`` env var, then ``"diseno-multiagente"``.
    enable_tracing:
        Explicit toggle. When *None* (default) the value is read from
        the ``LANGCHAIN_TRACING_V2`` env var.

    Returns
    -------
    dict
        Active configuration keys (safe to log — no secrets).
    """
    # Resolve values --------------------------------------------------------
    resolved_project = (
        project
        or os.environ.get("LANGCHAIN_PROJECT")
        or "diseno-multiagente"
    )

    if enable_tracing is None:
        enable_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    endpoint = os.environ.get(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    api_key = os.environ.get("LANGCHAIN_API_KEY", "")

    # Apply to os.environ so LangChain picks them up automatically ----------
    os.environ["LANGCHAIN_PROJECT"] = resolved_project
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enable_tracing else "false"

    # Emit a useful log message ---------------------------------------------
    if enable_tracing:
        if not api_key:
            logger.warning(
                "LangSmith tracing is enabled but LANGCHAIN_API_KEY is not set. "
                "Traces will NOT be sent to LangSmith."
            )
        else:
            logger.info(
                "LangSmith tracing enabled — project=%r endpoint=%s",
                resolved_project,
                endpoint,
            )
    else:
        logger.info("LangSmith tracing is disabled.")

    return {
        "LANGCHAIN_PROJECT": resolved_project,
        "LANGCHAIN_ENDPOINT": endpoint,
        "LANGCHAIN_TRACING_V2": "true" if enable_tracing else "false",
        "api_key_set": "true" if api_key else "false",
    }


def tracing_status() -> dict[str, str]:
    """Return the current LangSmith tracing status (no secrets)."""
    return {
        "project": os.environ.get("LANGCHAIN_PROJECT", "(not set)"),
        "endpoint": os.environ.get("LANGCHAIN_ENDPOINT", "(not set)"),
        "tracing_enabled": os.environ.get("LANGCHAIN_TRACING_V2", "false"),
        "api_key_set": str(bool(os.environ.get("LANGCHAIN_API_KEY"))).lower(),
    }
