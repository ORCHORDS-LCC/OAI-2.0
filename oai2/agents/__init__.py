"""Multi-agent orchestration scaffold + live agent loop.

Public surface
--------------

- :class:`AgentLoop` — multi-step tool-use loop driven by any
  :class:`~oai2.runtime.InferenceRuntime` (defaults to the env-selected
  gateway runtime; falls back to :class:`PlaceholderRuntime` in CI).
  Optionally consults a :class:`~oai2.knowledge.KnowledgeStore` to
  inject a compact evidence package into the model context per #22/#23.
- :class:`AgentRun` / :class:`AgentStep` — result shapes.
- :func:`default_tool_definitions` — the canonical six tools
  (``Read`` / ``Edit`` / ``Write`` / ``Bash`` / ``Glob`` / ``Grep``).
- :func:`default_system_prompt` — host-side system prompt.
- :class:`AgentSpec` / :class:`Orchestrator` / :class:`OrchestratorContext`
  — SWARM-shaped scaffold (still PROPOSED; live spawn is future work).
- :func:`oai2.agents.learning.extract_lesson` / :func:`record_failure` —
  verified-lesson (#87) and negative-memory (#88) bridges into the
  canonical :class:`KnowledgeStore` abstraction.
"""

from __future__ import annotations

from ..tools.registry import (
    default_tool_definitions,
    execute_tool,
    to_openai_wire,
)
from .agent_loop import (
    AgentLoop,
    AgentRun,
    AgentStep,
    build_default_gateway,
    build_default_runtime,
    default_dispatch_policy,
    default_system_prompt,
)
from .learning import extract_lesson, record_failure, sanitize_text
from .orchestration import AgentSpec, Orchestrator, OrchestratorContext

__all__ = [
    "AgentLoop",
    "AgentRun",
    "AgentStep",
    "AgentSpec",
    "Orchestrator",
    "OrchestratorContext",
    "build_default_gateway",
    "build_default_runtime",
    "default_dispatch_policy",
    "default_system_prompt",
    "default_tool_definitions",
    "execute_tool",
    "extract_lesson",
    "record_failure",
    "sanitize_text",
    "to_openai_wire",
]
