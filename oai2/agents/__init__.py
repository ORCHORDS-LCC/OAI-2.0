"""Multi-agent orchestration scaffold.

Status: PROPOSED. Defines the public orchestration shapes
(:class:`Orchestrator`, :class:`AgentSpec`). A live implementation will
spin up SWARM workers in parallel.
"""

from __future__ import annotations

from .orchestration import AgentSpec, Orchestrator, OrchestratorContext

__all__ = ["AgentSpec", "Orchestrator", "OrchestratorContext"]
