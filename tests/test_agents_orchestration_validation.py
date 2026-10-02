"""Structural / validation tests for ``oai2.agents.orchestration``.

Pins the multi-agent orchestration scaffold (PROPOSED status):

- :class:`AgentSpec` -- Pydantic ``BaseModel`` carrying
  ``id`` / ``role`` / ``description`` / ``status`` for a single agent.
- :class:`OrchestratorContext` -- Pydantic ``BaseModel`` carrying
  ``agents`` / ``merge_strategy`` / ``status`` for an orchestration
  session.
- :class:`Orchestrator` -- thin wrapper that holds a context and
  exposes a :meth:`Orchestrator.by_role` selector.

Behavioural end-to-end coverage lives in ``tests/test_agents.py``;
this file pins the *shape* of the API and the invariants the live
SWARM worker dispatch will rely on.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from oai2.agents.orchestration import AgentSpec, Orchestrator, OrchestratorContext
from oai2.core import AgentId, Status
from oai2.reasoning import SwarmRole

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "agents" / "orchestration.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_orchestration_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "agents/orchestration.py is unexpectedly empty"


def test_orchestration_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_orchestration_module_docstring_mentions_orchestrator() -> None:
    """The docstring must reference the orchestrator role."""

    assert "orchestrator" in _MODULE_SOURCE.lower()


def test_orchestration_module_docstring_notes_proposed_status() -> None:
    """The module is PROPOSED — the docstring must say so."""

    assert "PROPOSED" in _MODULE_SOURCE


def test_orchestration_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_orchestration_module_imports_pydantic_basemodel() -> None:
    """``AgentSpec`` and ``OrchestratorContext`` are Pydantic ``BaseModel`` subclasses."""

    assert re.search(
        r"^from pydantic import ([^\n]*,\s*)*(BaseModel)(\s*,\s*|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "orchestration module should import BaseModel from pydantic"


def test_orchestration_module_imports_configdict_and_field() -> None:
    """``ConfigDict`` and ``Field`` are used to pin extra='forbid' and length bounds."""

    assert "ConfigDict" in _MODULE_SOURCE
    assert "Field" in _MODULE_SOURCE


def test_orchestration_module_imports_agentid_and_status_from_core() -> None:
    """``AgentId`` and ``Status`` are imported relatively from ``..core``."""

    assert re.search(
        r"^from \.\.core import ([^\n]*,\s*)*(AgentId|Status)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )


def test_orchestration_module_imports_swarmrole_from_reasoning() -> None:
    """``SwarmRole`` is imported relatively from ``..reasoning``."""

    assert "from ..reasoning import SwarmRole" in _MODULE_SOURCE


def test_orchestration_module_does_not_import_cloud_runtime_modules() -> None:
    """The orchestration module is pure; no cloud-runtime imports."""

    forbidden = ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"orchestration module must not import cloud runtime module {token!r}"
        )


def test_orchestration_module_does_not_hardcode_credentials() -> None:
    """No api_key= literals, no BEGIN PRIVATE KEY blocks, no sk-/ghp_ tokens."""

    forbidden_patterns = (
        re.compile(r"api_key\s*=\s*['\"]sk-"),
        re.compile(r"BEGIN PRIVATE KEY"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    )
    for pattern in forbidden_patterns:
        assert not pattern.search(_MODULE_SOURCE), (
            f"orchestration module must not contain credential marker {pattern.pattern!r}"
        )


def test_orchestration_module_has_no_print_or_pprint_calls() -> None:
    """Library code must not perform I/O on import."""

    assert not re.search(r"\bprint\s*\(", _MODULE_SOURCE)
    assert not re.search(r"\bpprint\s*\.", _MODULE_SOURCE)


def test_orchestration_module_has_no_subprocess_or_shell_invocation() -> None:
    """No subprocess / os.system / shell=True paths."""

    assert "import subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE
    assert "shell=True" not in _MODULE_SOURCE


def test_orchestration_module_has_no_network_client_imports() -> None:
    """No requests / urllib / httpx imports."""

    forbidden = ("import requests", "from requests ", "import urllib", "import httpx")
    for token in forbidden:
        assert token not in _MODULE_SOURCE


def test_orchestration_module_has_no_eval_or_exec() -> None:
    """No dynamic code execution."""

    assert not re.search(r"^\s*eval\s*\(", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^\s*exec\s*\(", _MODULE_SOURCE, re.MULTILINE)


def test_orchestration_module_has_no_wildcard_imports() -> None:
    """No ``from X import *``."""

    assert not re.search(r"^from\s+\S+\s+import\s+\*", _MODULE_SOURCE, re.MULTILINE)


def test_orchestration_module_has_no_todo_fixme_xxx_markers() -> None:
    """No TODO / FIXME / XXX markers in shipped source."""

    forbidden = ("TODO", "FIXME", "XXX")
    for token in forbidden:
        assert not re.search(rf"\b{token}\b", _MODULE_SOURCE)


def test_orchestration_module_has_no_os_environ_or_getenv() -> None:
    """The orchestration module is pure; no environment-variable access."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_orchestration_module_pins_two_basemodel_subclasses() -> None:
    """Exactly 2 Pydantic ``BaseModel`` subclasses: ``AgentSpec`` + ``OrchestratorContext``."""

    basemodel_count = len(
        re.findall(r"^class\s+\w+\s*\(\s*BaseModel\s*\)", _MODULE_SOURCE, re.MULTILINE)
    )
    assert basemodel_count == 2, f"Expected 2 BaseModel subclasses, found {basemodel_count}"


# ---------------------------------------------------------------------------
# 2. __all__ completeness
# ---------------------------------------------------------------------------


def test_orchestration_module_all_is_exactly_three_names() -> None:
    """``__all__`` must pin exactly 3 names."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = [n.strip().strip("'\"") for n in body.split(",") if n.strip()]
    assert len(names) == 3, f"Expected 3 names in __all__, got {len(names)}: {names}"


def test_orchestration_module_all_names_match_documented_surface() -> None:
    """The 3 names are AgentSpec, Orchestrator, OrchestratorContext."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = {n.strip().strip("'\"") for n in body.split(",") if n.strip()}
    assert names == {"AgentSpec", "Orchestrator", "OrchestratorContext"}, (
        f"__all__ set mismatch: {names}"
    )


def test_orchestration_module_all_names_are_importable() -> None:
    """Each name in __all__ must be importable from oai2.agents.orchestration."""

    from oai2.agents import orchestration as module

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = [n.strip().strip("'\"") for n in body.split(",") if n.strip()]
    for name in names:
        assert hasattr(module, name), f"{name!r} is not exported by the module"


def test_orchestration_module_symbols_are_re_exported_at_package_level() -> None:
    """The three names are also re-exported via oai2.agents."""

    from oai2 import agents as package

    assert package.AgentSpec is AgentSpec
    assert package.Orchestrator is Orchestrator
    assert package.OrchestratorContext is OrchestratorContext


# ---------------------------------------------------------------------------
# 3. AgentSpec
# ---------------------------------------------------------------------------


def test_agent_spec_field_set_is_pinned() -> None:
    """AgentSpec has the four documented fields."""

    fields = set(AgentSpec.model_fields.keys())
    assert fields == {"id", "role", "description", "status"}, f"AgentSpec fields mismatch: {fields}"


def test_agent_spec_forbids_extra_fields() -> None:
    """model_config = ConfigDict(extra='forbid') rejects unknown keys."""

    with pytest.raises(ValidationError):
        AgentSpec(  # type: ignore[call-arg]
            id=AgentId("a-1"),
            role=SwarmRole.PLANNER,
            description="x",
            unknown_field="bad",
        )


def test_agent_spec_id_accepts_agentid_newtype() -> None:
    """``id`` accepts and stores the ``AgentId`` newtype (runtime: plain str)."""

    spec = AgentSpec(
        id=AgentId("a-1"),
        role=SwarmRole.PLANNER,
        description="plan",
    )
    assert spec.id == AgentId("a-1")
    assert isinstance(spec.id, str)


def test_agent_spec_role_accepts_swarm_role_values() -> None:
    """``role`` accepts each documented SwarmRole value."""

    for role in SwarmRole:
        spec = AgentSpec(
            id=AgentId(f"a-{role.value}"),
            role=role,
            description=f"agent for {role.value}",
        )
        assert spec.role is role


def test_agent_spec_role_rejects_non_swarm_role() -> None:
    """``role`` rejects strings outside the SwarmRole enum."""

    with pytest.raises(ValidationError):
        AgentSpec(
            id=AgentId("a-1"),
            role="planner_typo",  # type: ignore[arg-type]
            description="x",
        )


def test_agent_spec_description_min_length_is_one() -> None:
    """Empty description is rejected (Field(min_length=1))."""

    with pytest.raises(ValidationError):
        AgentSpec(
            id=AgentId("a-1"),
            role=SwarmRole.PLANNER,
            description="",
        )


def test_agent_spec_description_max_length_is_1024() -> None:
    """Description above 1024 chars is rejected."""

    with pytest.raises(ValidationError):
        AgentSpec(
            id=AgentId("a-1"),
            role=SwarmRole.PLANNER,
            description="x" * 1025,
        )


def test_agent_spec_description_one_char_boundary_accepted() -> None:
    """Description of exactly 1 character is the smallest legal value."""

    spec = AgentSpec(
        id=AgentId("a-1"),
        role=SwarmRole.PLANNER,
        description="x",
    )
    assert spec.description == "x"


def test_agent_spec_description_1024_char_boundary_accepted() -> None:
    """Description of exactly 1024 characters is the largest legal value."""

    spec = AgentSpec(
        id=AgentId("a-1"),
        role=SwarmRole.PLANNER,
        description="x" * 1024,
    )
    assert len(spec.description) == 1024


def test_agent_spec_default_status_is_proposed() -> None:
    """``status`` defaults to ``Status.PROPOSED``."""

    spec = AgentSpec(
        id=AgentId("a-1"),
        role=SwarmRole.PLANNER,
        description="x",
    )
    assert spec.status is Status.PROPOSED


def test_agent_spec_accepts_each_status_value() -> None:
    """``status`` accepts any of the 4 documented Status values."""

    for status in Status:
        spec = AgentSpec(
            id=AgentId("a-1"),
            role=SwarmRole.PLANNER,
            description="x",
            status=status,
        )
        assert spec.status is status


# ---------------------------------------------------------------------------
# 4. OrchestratorContext
# ---------------------------------------------------------------------------


def test_orchestrator_context_field_set_is_pinned() -> None:
    """OrchestratorContext has the three documented fields."""

    fields = set(OrchestratorContext.model_fields.keys())
    assert fields == {"agents", "merge_strategy", "status"}, (
        f"OrchestratorContext fields mismatch: {fields}"
    )


def test_orchestrator_context_forbids_extra_fields() -> None:
    """model_config = ConfigDict(extra='forbid') rejects unknown keys."""

    with pytest.raises(ValidationError):
        OrchestratorContext(unknown="bad")  # type: ignore[call-arg]


def test_orchestrator_context_default_agents_is_empty_tuple() -> None:
    """``agents`` defaults to an empty tuple."""

    ctx = OrchestratorContext()
    assert ctx.agents == ()
    assert isinstance(ctx.agents, tuple)


def test_orchestrator_context_accepts_agents_tuple() -> None:
    """``agents`` accepts a tuple of AgentSpec instances."""

    a = AgentSpec(
        id=AgentId("a-1"),
        role=SwarmRole.PLANNER,
        description="plan",
    )
    b = AgentSpec(
        id=AgentId("a-2"),
        role=SwarmRole.IMPLEMENTER,
        description="code",
    )
    ctx = OrchestratorContext(agents=(a, b))
    assert len(ctx.agents) == 2
    assert ctx.agents[0] is a
    assert ctx.agents[1] is b


def test_orchestrator_context_accepts_agents_list() -> None:
    """``agents`` also accepts a list (Pydantic coerces to tuple)."""

    a = AgentSpec(
        id=AgentId("a-1"),
        role=SwarmRole.PLANNER,
        description="plan",
    )
    ctx = OrchestratorContext(agents=[a])  # type: ignore[arg-type]
    assert ctx.agents == (a,)


def test_orchestrator_context_agents_rejects_non_agent_spec() -> None:
    """Non-AgentSpec entries are rejected."""

    with pytest.raises(ValidationError):
        OrchestratorContext(agents=("not-an-agent-spec",))  # type: ignore[arg-type]


def test_orchestrator_context_default_merge_strategy_is_critic_pick() -> None:
    """``merge_strategy`` defaults to ``critic_pick``."""

    ctx = OrchestratorContext()
    assert ctx.merge_strategy == "critic_pick"


def test_orchestrator_context_accepts_custom_merge_strategy() -> None:
    """``merge_strategy`` accepts an alternative non-empty strategy name."""

    ctx = OrchestratorContext(merge_strategy="weighted_vote")
    assert ctx.merge_strategy == "weighted_vote"


def test_orchestrator_context_merge_strategy_rejects_empty_string() -> None:
    """``merge_strategy`` rejects empty string (min_length=1)."""

    with pytest.raises(ValidationError):
        OrchestratorContext(merge_strategy="")


def test_orchestrator_context_merge_strategy_rejects_above_64_chars() -> None:
    """``merge_strategy`` rejects strings above max_length=64."""

    with pytest.raises(ValidationError):
        OrchestratorContext(merge_strategy="x" * 65)


def test_orchestrator_context_merge_strategy_one_char_boundary_accepted() -> None:
    """``merge_strategy`` of exactly 1 character is the smallest legal value."""

    ctx = OrchestratorContext(merge_strategy="x")
    assert ctx.merge_strategy == "x"


def test_orchestrator_context_merge_strategy_64_char_boundary_accepted() -> None:
    """``merge_strategy`` of exactly 64 characters is the largest legal value."""

    ctx = OrchestratorContext(merge_strategy="x" * 64)
    assert len(ctx.merge_strategy) == 64


def test_orchestrator_context_default_status_is_proposed() -> None:
    """``status`` defaults to ``Status.PROPOSED``."""

    ctx = OrchestratorContext()
    assert ctx.status is Status.PROPOSED


def test_orchestrator_context_accepts_each_status_value() -> None:
    """``status`` accepts any of the 4 documented Status values."""

    for status in Status:
        ctx = OrchestratorContext(status=status)
        assert ctx.status is status


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_is_a_class() -> None:
    """``Orchestrator`` is a class (not a function or instance)."""

    assert inspect.isclass(Orchestrator)


def test_orchestrator_status_class_attribute_is_proposed() -> None:
    """The ``Orchestrator.STATUS`` class attribute is ``Status.PROPOSED``."""

    assert Orchestrator.STATUS is Status.PROPOSED


def test_orchestrator_init_signature_is_ctx_only() -> None:
    """``__init__`` takes a single ``ctx`` parameter (no extras)."""

    sig = inspect.signature(Orchestrator.__init__)
    params = list(sig.parameters.values())
    assert len(params) == 2, f"Expected self + ctx, got {len(params)}"
    assert params[0].name == "self"
    assert params[1].name == "ctx"


def test_orchestrator_init_stores_context_attribute() -> None:
    """The constructor stores the context on ``self.ctx``."""

    ctx = OrchestratorContext()
    orch = Orchestrator(ctx)
    assert orch.ctx is ctx


def test_orchestrator_by_role_is_callable() -> None:
    """``by_role`` is a regular method."""

    assert callable(Orchestrator.by_role)


def test_orchestrator_by_role_signature_is_role_only() -> None:
    """``by_role`` takes a single ``role`` parameter (besides self)."""

    sig = inspect.signature(Orchestrator.by_role)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "self"
    assert params[1].name == "role"


def test_orchestrator_by_role_empty_agents_returns_empty_tuple() -> None:
    """``by_role`` on an empty context returns an empty tuple."""

    orch = Orchestrator(OrchestratorContext())
    assert orch.by_role(SwarmRole.PLANNER) == ()


def test_orchestrator_by_role_returns_matching_agents() -> None:
    """``by_role`` returns the subset whose ``role`` matches."""

    planner = AgentSpec(
        id=AgentId("p"),
        role=SwarmRole.PLANNER,
        description="plan",
    )
    implementer = AgentSpec(
        id=AgentId("i"),
        role=SwarmRole.IMPLEMENTER,
        description="code",
    )
    critic = AgentSpec(
        id=AgentId("c"),
        role=SwarmRole.CRITIC,
        description="judge",
    )
    orch = Orchestrator(OrchestratorContext(agents=(planner, implementer, critic)))
    assert orch.by_role(SwarmRole.PLANNER) == (planner,)
    assert orch.by_role(SwarmRole.IMPLEMENTER) == (implementer,)
    assert orch.by_role(SwarmRole.CRITIC) == (critic,)
    assert orch.by_role(SwarmRole.VERIFIER) == ()


def test_orchestrator_by_role_returns_multiple_agents_with_same_role() -> None:
    """``by_role`` returns ALL agents matching the role, in declaration order."""

    p1 = AgentSpec(
        id=AgentId("p1"),
        role=SwarmRole.PLANNER,
        description="plan 1",
    )
    p2 = AgentSpec(
        id=AgentId("p2"),
        role=SwarmRole.PLANNER,
        description="plan 2",
    )
    orch = Orchestrator(OrchestratorContext(agents=(p1, p2)))
    result = orch.by_role(SwarmRole.PLANNER)
    assert len(result) == 2
    assert result[0] is p1
    assert result[1] is p2


def test_orchestrator_by_role_return_type_is_tuple() -> None:
    """``by_role`` always returns a tuple (not a list / generator)."""

    orch = Orchestrator(OrchestratorContext())
    result = orch.by_role(SwarmRole.PLANNER)
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# 6. Public-safety (boundary) pin
# ---------------------------------------------------------------------------


def test_orchestration_module_does_not_reexport_pydantic_internals() -> None:
    """Pydantic internals must not be in ``__all__``."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    forbidden = ("BaseModel", "ConfigDict", "Field")
    for name in forbidden:
        assert name not in body, f"Pydantic internal {name!r} must not be in __all__"


def test_orchestration_module_does_not_reexport_core_internals() -> None:
    """Core imports (AgentId, Status) must not be in ``__all__``."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    forbidden = ("AgentId", "Status", "SwarmRole")
    for name in forbidden:
        assert name not in body, f"Imported type {name!r} must not be in __all__"


def test_orchestration_module_pin_one_class_attribute_per_pydantic_model() -> None:
    """Each Pydantic ``BaseModel`` subclass has exactly one ``model_config`` block."""

    config_count = len(
        re.findall(r"^\s+model_config\s*=\s*ConfigDict", _MODULE_SOURCE, re.MULTILINE)
    )
    assert config_count == 2, f"Expected 2 model_config blocks, found {config_count}"


def test_orchestration_module_pin_one_class_status_constant() -> None:
    """Exactly one class-level ``STATUS`` constant — on ``Orchestrator``."""

    status_constants = re.findall(r"^\s+STATUS\s*=\s*Status\.", _MODULE_SOURCE, re.MULTILINE)
    assert len(status_constants) == 1, (
        f"Expected 1 STATUS class constant, found {len(status_constants)}"
    )
