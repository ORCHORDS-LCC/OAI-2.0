"""Tool dispatch, policy pipeline, and the canonical tool registry.

Public surface
--------------

- :class:`DispatchPolicy` / :class:`ToolDispatcher` — the 6-gate policy
  pipeline from ``TOOL_CALLING.md``.
- :func:`default_tool_definitions` — the canonical six tools
  (``Read`` / ``Edit`` / ``Write`` / ``Bash`` / ``Glob`` / ``Grep``).
- :func:`execute_tool` — local execution handler for the six tools.
- :func:`to_openai_wire` — convert internal tool definitions to the
  OpenAI chat-completions wire shape.
"""

from __future__ import annotations

from .dispatch import (
    DispatchDecision,
    DispatchPolicy,
    DispatchStage,
    ToolDispatcher,
    default_dispatcher,
)
from .registry import (
    default_tool_definitions,
    execute_tool,
    to_openai_wire,
)

__all__ = [
    "DispatchDecision",
    "DispatchPolicy",
    "DispatchStage",
    "ToolDispatcher",
    "default_dispatcher",
    "default_tool_definitions",
    "execute_tool",
    "to_openai_wire",
]
