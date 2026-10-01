"""Tool dispatch and policy pipeline.

Status: PROPOSED. Implements the validation gates listed in
``TOOL_CALLING.md``: tool exists → arguments validate → capability
allowed → resource in scope → budget respected → high-impact approval.
"""

from __future__ import annotations

from .dispatch import (
    DispatchDecision,
    DispatchPolicy,
    DispatchStage,
    ToolDispatcher,
    default_dispatcher,
)

__all__ = [
    "DispatchDecision",
    "DispatchPolicy",
    "DispatchStage",
    "ToolDispatcher",
    "default_dispatcher",
]
