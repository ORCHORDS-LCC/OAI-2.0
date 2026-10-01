"""Agent orchestration tests."""

from __future__ import annotations

from oai2.agents import AgentSpec, Orchestrator, OrchestratorContext
from oai2.core import AgentId
from oai2.reasoning import SwarmRole


def test_orchestrator_filters_by_role() -> None:
    agents = (
        AgentSpec(id=AgentId("p"), role=SwarmRole.PLANNER, description="plan"),
        AgentSpec(id=AgentId("i"), role=SwarmRole.IMPLEMENTER, description="code"),
        AgentSpec(id=AgentId("c"), role=SwarmRole.CRITIC, description="judge"),
    )
    orch = Orchestrator(OrchestratorContext(agents=agents))
    assert [a.id for a in orch.by_role(SwarmRole.PLANNER)] == ["p"]
    assert [a.id for a in orch.by_role(SwarmRole.CRITIC)] == ["c"]
    assert orch.by_role(SwarmRole.VERIFIER) == ()


def test_agents_module_exports_from_agents_package() -> None:
    from oai2.agents import AgentSpec as ExportedAgentSpec
    from oai2.agents import Orchestrator as ExportedOrchestrator
    from oai2.agents import (
        OrchestratorContext as ExportedOrchestratorContext,
    )
    from oai2.agents.orchestration import (
        AgentSpec,
        Orchestrator,
        OrchestratorContext,
    )

    assert ExportedAgentSpec is AgentSpec
    assert ExportedOrchestrator is Orchestrator
    assert ExportedOrchestratorContext is OrchestratorContext
