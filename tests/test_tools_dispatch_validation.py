"""Structural / validation tests for ``oai2.tools.dispatch``.

Pin the tool-dispatch gate pipeline: the :class:`DispatchStage`
:class:`StrEnum`, the :class:`DispatchDecision` frozen-slotted dataclass,
the :class:`DispatchPolicy` Pydantic v2 :class:`BaseModel` (with its
``extra="forbid"`` / ``frozen=True`` configuration), the
:class:`ToolDispatcher` six-gate ``check`` method, and the
``default_dispatcher`` permissive factory.

Behavioural end-to-end coverage lives in ``tests/test_tools_dispatch.py``;
this file pins the *shape* of the API and the invariants the call
sequencer relies on.
"""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from oai2.core import Status, ToolId
from oai2.protocols import (
    ToolArgument,
    ToolCall,
    ToolDefinition,
    ToolPolicy,
    ToolResult,
)
from oai2.tools import (
    DispatchDecision,
    DispatchPolicy,
    DispatchStage,
    ToolDispatcher,
    default_dispatcher,
)
from oai2.tools.dispatch import (
    DispatchDecision as DispatchDecisionFromModule,
)
from oai2.tools.dispatch import (
    DispatchPolicy as DispatchPolicyFromModule,
)
from oai2.tools.dispatch import (
    DispatchStage as DispatchStageFromModule,
)
from oai2.tools.dispatch import (
    ToolDispatcher as ToolDispatcherFromModule,
)
from oai2.tools.dispatch import (
    default_dispatcher as default_dispatcher_from_module,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "tools" / "dispatch.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Module docstring + import surface
# ---------------------------------------------------------------------------


def test_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "tools/dispatch.py is unexpectedly empty"


def test_module_has_docstring() -> None:
    """The module ships an overview of the dispatch pipeline."""

    assert _MODULE_SOURCE.startswith('"""')
    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert first_para, "module docstring is empty"


def test_module_docstring_mentions_dispatch_decision() -> None:
    """The overview mentions ``DispatchDecision``."""

    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert "DispatchDecision" in first_para


def test_module_uses_future_annotations() -> None:
    """``from __future__ import annotations`` is present (UP006-clean)."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_module_imports_use_relative_core_and_protocols() -> None:
    """``Status`` is from ``..core`` and ``ToolCall``/``ToolDefinition``/``ToolResult``
    are from ``..protocols``; no absolute ``from oai2`` imports."""

    assert "from ..core import Status" in _MODULE_SOURCE
    # ``ToolCall`` / ``ToolDefinition`` / ``ToolResult`` are co-imported on
    # one line: ``from ..protocols import ToolCall, ToolDefinition, ToolResult``.
    proto_imports = re.findall(
        r"from \.+\.protocols\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
    )
    assert proto_imports, "missing relative protocols import line"
    imported_names = " ".join(proto_imports)
    for name in ("ToolCall", "ToolDefinition", "ToolResult"):
        assert name in imported_names, f"missing relative import: {name}"
    # No accidental absolute imports of core / protocols.
    assert not re.search(r"^import oai2\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.core\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.protocols\b", _MODULE_SOURCE, re.MULTILINE)


def test_module_imports_pydantic_symbols() -> None:
    """The module pulls the Pydantic symbols it needs (``BaseModel``,
    ``ConfigDict``, ``Field``)."""

    for symbol in ("BaseModel", "ConfigDict", "Field"):
        assert symbol in _MODULE_SOURCE, f"missing Pydantic import: {symbol}"


def test_module_imports_dataclass_and_strenum() -> None:
    """The module uses stdlib ``dataclass`` and ``enum.StrEnum``."""

    assert "from dataclasses import dataclass" in _MODULE_SOURCE
    assert "from enum import StrEnum" in _MODULE_SOURCE


def test_module_has_no_wildcard_imports() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


# ---------------------------------------------------------------------------
# __all__ completeness + package re-export
# ---------------------------------------------------------------------------


def test_dunder_all_lists_exactly_five_public_names() -> None:
    """The module's public surface is exactly 5 names, no more, no less."""

    import oai2.tools.dispatch as mod

    assert isinstance(mod.__all__, list)
    assert set(mod.__all__) == {
        "DispatchDecision",
        "DispatchPolicy",
        "DispatchStage",
        "ToolDispatcher",
        "default_dispatcher",
    }
    assert len(mod.__all__) == 5


def test_each_all_name_is_importable_from_module() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    import oai2.tools.dispatch as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ name missing: {name}"


def test_public_names_are_reexported_from_tools_package() -> None:
    """Top-level ``oai2.tools`` re-exports the 5 dispatch public names."""

    import oai2.tools as pkg

    for name in (
        "DispatchDecision",
        "DispatchPolicy",
        "DispatchStage",
        "ToolDispatcher",
        "default_dispatcher",
    ):
        assert name in pkg.__all__, f"package re-export missing: {name}"


def test_package_re_export_preserves_identity() -> None:
    """Package re-exports point at the *same* objects as the module."""

    import oai2.tools as pkg
    import oai2.tools.dispatch as mod

    assert pkg.DispatchDecision is mod.DispatchDecision
    assert pkg.DispatchPolicy is mod.DispatchPolicy
    assert pkg.DispatchStage is mod.DispatchStage
    assert pkg.ToolDispatcher is mod.ToolDispatcher
    assert pkg.default_dispatcher is mod.default_dispatcher


def test_module_imports_match_top_level() -> None:
    """``from oai2.tools.dispatch import X`` matches the top-level re-export."""

    assert DispatchDecision is DispatchDecisionFromModule
    assert DispatchPolicy is DispatchPolicyFromModule
    assert DispatchStage is DispatchStageFromModule
    assert ToolDispatcher is ToolDispatcherFromModule
    assert default_dispatcher is default_dispatcher_from_module


# ---------------------------------------------------------------------------
# DispatchStage StrEnum
# ---------------------------------------------------------------------------


def test_dispatch_stage_subclasses_strenum() -> None:
    """``DispatchStage`` subclasses ``enum.StrEnum`` (and ``str``)."""

    from enum import StrEnum

    assert issubclass(DispatchStage, StrEnum)
    assert issubclass(DispatchStage, str)


def test_dispatch_stage_has_six_members() -> None:
    """``DispatchStage`` has exactly 6 members — no more, no less."""

    assert len(DispatchStage) == 6


def test_dispatch_stage_members_and_wire_strings() -> None:
    """Every member's wire string is the lowercase ``value`` attribute."""

    expected = {
        "EXECUTE": "execute",
        "REPAIR": "repair_arguments",
        "CHOOSE_ALT": "choose_alternative",
        "RETRY": "retry",
        "REPLAN": "replan",
        "DENY": "deny",
    }
    actual = {member.name: str(member.value) for member in DispatchStage}
    assert actual == expected


def test_dispatch_stage_member_names_are_unique() -> None:
    """No duplicate member names — ``DispatchStage`` is a valid enum."""

    names = [member.name for member in DispatchStage]
    assert len(names) == len(set(names))


def test_dispatch_stage_wire_strings_are_unique() -> None:
    """No duplicate wire-string values — the enum is a true bijection."""

    values = [str(member.value) for member in DispatchStage]
    assert len(values) == len(set(values))


def test_dispatch_stage_lookup_by_wire_string() -> None:
    """``DispatchStage("execute")`` resolves to ``EXECUTE`` (and likewise for each)."""

    for member in DispatchStage:
        assert DispatchStage(str(member.value)) is member


def test_dispatch_stage_rejects_unknown_wire_string() -> None:
    """An unknown wire string raises ``ValueError`` — no silent fallback."""

    with pytest.raises(ValueError):
        DispatchStage("bogus")


def test_dispatch_stage_string_membership() -> None:
    """``DispatchStage.EXECUTE`` survives ``str`` equality in both directions."""

    assert str(DispatchStage.EXECUTE) == "execute"
    assert DispatchStage.EXECUTE == "execute"
    assert "execute" == DispatchStage.EXECUTE


# ---------------------------------------------------------------------------
# DispatchDecision frozen-slotted dataclass
# ---------------------------------------------------------------------------


def test_dispatch_decision_is_a_dataclass() -> None:
    """``DispatchDecision`` is a stdlib ``@dataclass``."""

    assert is_dataclass(DispatchDecision)


def test_dispatch_decision_is_frozen() -> None:
    """``DispatchDecision`` is frozen — assignment raises ``FrozenInstanceError``."""

    d = DispatchDecision(stage=DispatchStage.EXECUTE, reason="ok")
    with pytest.raises(FrozenInstanceError):
        d.reason = "tampered"  # type: ignore[misc]


def test_dispatch_decision_has_slots() -> None:
    """``DispatchDecision`` is slotted — no ``__dict__``."""

    d = DispatchDecision(stage=DispatchStage.EXECUTE, reason="ok")
    with pytest.raises(AttributeError):
        _ = d.__dict__


def test_dispatch_decision_fields() -> None:
    """``DispatchDecision`` has exactly ``(stage, reason, result)`` fields."""

    field_names = {f.name for f in fields(DispatchDecision)}
    assert field_names == {"stage", "reason", "result"}


def test_dispatch_decision_result_default_is_none() -> None:
    """``result`` defaults to ``None`` when omitted."""

    d = DispatchDecision(stage=DispatchStage.EXECUTE, reason="ok")
    assert d.result is None


def test_dispatch_decision_accepts_tool_result() -> None:
    """``result`` accepts a ``ToolResult`` payload."""

    payload = ToolResult(call_id="c1", ok=True, output="done", elapsed_ms=12.0)
    d = DispatchDecision(
        stage=DispatchStage.EXECUTE,
        reason="ok",
        result=payload,
    )
    assert d.result is payload
    assert d.result.ok is True


def test_dispatch_decision_repr_is_deterministic() -> None:
    """The dataclass ``repr`` is stable (dataclasses-generates one)."""

    d = DispatchDecision(stage=DispatchStage.DENY, reason="nope")
    rendered = repr(d)
    assert "DispatchDecision" in rendered
    assert "DENY" in rendered


# ---------------------------------------------------------------------------
# DispatchPolicy Pydantic BaseModel
# ---------------------------------------------------------------------------


def test_dispatch_policy_is_a_pydantic_basemodel() -> None:
    """``DispatchPolicy`` subclasses :class:`pydantic.BaseModel`."""

    assert issubclass(DispatchPolicy, BaseModel)


def test_dispatch_policy_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        DispatchPolicy(unknown_field="oops")  # type: ignore[call-arg]


def test_dispatch_policy_is_frozen() -> None:
    """``DispatchPolicy`` is frozen — assignment raises ``ValidationError``."""

    p = DispatchPolicy()
    with pytest.raises(ValidationError):
        p.budget_calls = 7


def test_dispatch_policy_allow_capabilities_default_is_empty_frozenset() -> None:
    """``allow_capabilities`` defaults to an empty ``frozenset``."""

    p = DispatchPolicy()
    assert isinstance(p.allow_capabilities, frozenset)
    assert p.allow_capabilities == frozenset()


def test_dispatch_policy_deny_capabilities_default_is_empty_frozenset() -> None:
    """``deny_capabilities`` defaults to an empty ``frozenset``."""

    p = DispatchPolicy()
    assert isinstance(p.deny_capabilities, frozenset)
    assert p.deny_capabilities == frozenset()


def test_dispatch_policy_resource_scopes_default_is_empty_frozenset() -> None:
    """``resource_scopes`` defaults to an empty ``frozenset``."""

    p = DispatchPolicy()
    assert isinstance(p.resource_scopes, frozenset)
    assert p.resource_scopes == frozenset()


def test_dispatch_policy_high_impact_approved_default_is_false() -> None:
    """``high_impact_approved`` defaults to ``False``."""

    p = DispatchPolicy()
    assert p.high_impact_approved is False


def test_dispatch_policy_budget_calls_default_is_64() -> None:
    """``budget_calls`` defaults to ``64``."""

    p = DispatchPolicy()
    assert p.budget_calls == 64


def test_dispatch_policy_status_default_is_proposed() -> None:
    """``status`` defaults to :attr:`Status.PROPOSED`."""

    p = DispatchPolicy()
    assert p.status is Status.PROPOSED


def test_dispatch_policy_accepts_custom_capability_sets() -> None:
    """``allow_capabilities`` / ``deny_capabilities`` accept arbitrary strings."""

    p = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read", "fs.write"}),
        deny_capabilities=frozenset({"net.publish"}),
    )
    assert p.allow_capabilities == frozenset({"fs.read", "fs.write"})
    assert p.deny_capabilities == frozenset({"net.publish"})


def test_dispatch_policy_accepts_resource_scopes() -> None:
    """``resource_scopes`` accepts a non-empty ``frozenset``."""

    p = DispatchPolicy(resource_scopes=frozenset({"/tmp", "/var/log"}))
    assert p.resource_scopes == frozenset({"/tmp", "/var/log"})


def test_dispatch_policy_accepts_high_impact_approved_true() -> None:
    """``high_impact_approved`` accepts ``True``."""

    p = DispatchPolicy(high_impact_approved=True)
    assert p.high_impact_approved is True


def test_dispatch_policy_accepts_custom_budget_calls() -> None:
    """``budget_calls`` accepts a custom int."""

    p = DispatchPolicy(budget_calls=128)
    assert p.budget_calls == 128


def test_dispatch_policy_accepts_status_enum() -> None:
    """``status`` accepts any :class:`Status` enum value."""

    p = DispatchPolicy(status=Status.EXPERIMENTAL)
    assert p.status is Status.EXPERIMENTAL


def test_dispatch_policy_fields() -> None:
    """``DispatchPolicy`` exposes exactly the 6 documented fields."""

    assert set(DispatchPolicy.model_fields.keys()) == {
        "allow_capabilities",
        "deny_capabilities",
        "high_impact_approved",
        "resource_scopes",
        "budget_calls",
        "status",
    }


# ---------------------------------------------------------------------------
# ToolDispatcher class
# ---------------------------------------------------------------------------


def test_tool_dispatcher_is_a_class() -> None:
    """``ToolDispatcher`` is a class (not a function or module)."""

    import inspect

    assert inspect.isclass(ToolDispatcher)


def test_tool_dispatcher_status_class_attribute() -> None:
    """``ToolDispatcher.STATUS`` is :attr:`Status.PROPOSED`."""

    assert ToolDispatcher.STATUS is Status.PROPOSED


def test_tool_dispatcher_init_with_empty_registry() -> None:
    """An empty tuple registry produces a dispatcher with no tools."""

    d = ToolDispatcher(registry=(), policy=DispatchPolicy())
    assert d.registry == {}
    assert d.policy.status is Status.PROPOSED


def test_tool_dispatcher_init_builds_id_indexed_registry() -> None:
    """The registry is keyed by ``ToolDefinition.id`` — not by ``name``."""

    td1 = ToolDefinition(
        id=ToolId("read"),
        name="read_file",
        description="read",
        arguments=(ToolArgument(name="path", type="path"),),
        capability="fs.read",
    )
    td2 = ToolDefinition(
        id=ToolId("write"),
        name="write_file",
        description="write",
        arguments=(ToolArgument(name="path", type="path"),),
        capability="fs.write",
    )
    d = ToolDispatcher(registry=(td1, td2), policy=DispatchPolicy())
    assert set(d.registry.keys()) == {"read", "write"}
    assert d.registry[ToolId("read")] is td1
    assert d.registry[ToolId("write")] is td2


def test_tool_dispatcher_init_with_later_tools_overwrites_earlier() -> None:
    """If two tools share an id, the later one wins (dict assignment)."""

    first = ToolDefinition(
        id=ToolId("dup"),
        name="first",
        description="first",
        arguments=(ToolArgument(name="x", type="string"),),
        capability="first.cap",
    )
    second = ToolDefinition(
        id=ToolId("dup"),
        name="second",
        description="second",
        arguments=(ToolArgument(name="x", type="string"),),
        capability="second.cap",
    )
    d = ToolDispatcher(registry=(first, second), policy=DispatchPolicy())
    assert d.registry[ToolId("dup")] is second


def test_tool_dispatcher_init_stores_policy() -> None:
    """``policy`` is stored as a ``DispatchPolicy`` instance on the dispatcher."""

    p = DispatchPolicy(budget_calls=8)
    d = ToolDispatcher(registry=(), policy=p)
    assert d.policy is p


def test_tool_dispatcher_check_is_callable() -> None:
    """``ToolDispatcher.check`` is a callable method."""

    d = ToolDispatcher(registry=(), policy=DispatchPolicy())
    assert callable(d.check)


def test_tool_dispatcher_check_returns_dispatch_decision() -> None:
    """``check`` returns a :class:`DispatchDecision` instance."""

    td = ToolDefinition(
        id=ToolId("read"),
        name="read",
        description="read",
        arguments=(ToolArgument(name="path", type="path"),),
        capability="fs.read",
    )
    d = ToolDispatcher(registry=(td,), policy=DispatchPolicy())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert isinstance(out, DispatchDecision)
    assert out.stage is DispatchStage.EXECUTE


def test_tool_dispatcher_check_signature_has_keyword_only_calls_used() -> None:
    """``check(call, *, calls_used)`` — ``calls_used`` is keyword-only."""

    import inspect

    sig = inspect.signature(ToolDispatcher.check)
    params = list(sig.parameters.values())
    # self, call, calls_used
    assert [p.name for p in params] == ["self", "call", "calls_used"]
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# ToolDispatcher gate-by-gate behaviour
# ---------------------------------------------------------------------------


def _read_def(scoped: bool = False) -> ToolDefinition:
    return ToolDefinition(
        id=ToolId("read"),
        name="read",
        description="read",
        arguments=(ToolArgument(name="path", type="path"),),
        capability="fs.read",
        scoped=scoped,
    )


def _high_impact_def() -> ToolDefinition:
    return ToolDefinition(
        id=ToolId("publish"),
        name="publish",
        description="publish",
        arguments=(ToolArgument(name="message", type="string"),),
        capability="net.publish",
        high_impact=True,
    )


def test_gate1_unknown_tool_returns_replan() -> None:
    """Gate 1: tool id not in registry → ``DispatchStage.REPLAN``."""

    d = ToolDispatcher(registry=(_read_def(),), policy=DispatchPolicy())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("nope"), arguments={}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.REPLAN
    assert "unknown tool" in out.reason


def test_gate2_unknown_arguments_returns_repair() -> None:
    """Gate 2: arguments not a subset of the declared schema → ``REPAIR``."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"fs.read"})),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x", "bogus": 1}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.REPAIR
    assert "unknown arguments" in out.reason


def test_gate2_exact_subset_of_arguments_returns_execute() -> None:
    """Gate 2: passing exactly the declared argument names is allowed."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"fs.read"})),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate3_denied_capability_returns_deny() -> None:
    """Gate 3: capability present in ``deny_capabilities`` → ``DENY``."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(deny_capabilities=frozenset({"fs.read"})),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY


def test_gate3_capability_not_in_allow_list_returns_choose_alt() -> None:
    """Gate 3: non-empty ``allow_capabilities`` and capability not in it → ``CHOOSE_ALT``."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"fs.write"})),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.CHOOSE_ALT


def test_gate3_empty_allow_list_admits_any_capability() -> None:
    """Gate 3: empty ``allow_capabilities`` admits any non-denied capability."""

    d = ToolDispatcher(registry=(_read_def(),), policy=DispatchPolicy())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate4_resource_out_of_scope_returns_deny() -> None:
    """Gate 4: tool marked ``scoped`` and the path not in ``resource_scopes`` → ``DENY``."""

    d = ToolDispatcher(
        registry=(_read_def(scoped=True),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            resource_scopes=frozenset({"/allowed"}),
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/secret"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY
    assert "resource out of scope" in out.reason


def test_gate4_resource_in_scope_passes() -> None:
    """Gate 4: tool marked ``scoped`` and the path inside ``resource_scopes`` → ``EXECUTE``."""

    d = ToolDispatcher(
        registry=(_read_def(scoped=True),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            resource_scopes=frozenset({"/x"}),
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate4_unscoped_tool_skips_scope_check() -> None:
    """Gate 4: tool NOT marked ``scoped`` skips the scope check entirely."""

    d = ToolDispatcher(
        registry=(_read_def(scoped=False),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            resource_scopes=frozenset(),  # would deny everything if checked
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/anywhere"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate4_scope_must_be_declared_in_arguments_schema() -> None:
    """Gate 4 reads ``scope`` from ``call.arguments``, so any non-declared
    argument (including ``scope`` itself) is caught by Gate 2 first and
    returns ``REPAIR`` with an "unknown arguments" reason."""

    # The source reads ``call.arguments.get("scope") or call.arguments.get("path")``
    # in gate 4, but Gate 2 short-circuits when ``scope`` is provided without
    # being declared in the tool schema. This documents the gate ordering.
    d = ToolDispatcher(
        registry=(_read_def(scoped=True),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            resource_scopes=frozenset({"/explicit"}),
        ),
    )
    out = d.check(
        ToolCall(
            id="c1",
            tool_id=ToolId("read"),
            arguments={"path": "/secret", "scope": "/explicit"},
        ),
        calls_used=0,
    )
    assert out.stage is DispatchStage.REPAIR
    assert "unknown arguments" in out.reason


def test_gate4_path_arg_matches_when_scope_not_provided() -> None:
    """Gate 4 falls back to the ``path`` arg when ``scope`` is not in args."""

    d = ToolDispatcher(
        registry=(_read_def(scoped=True),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            resource_scopes=frozenset({"/x"}),
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate5_budget_exceeded_returns_retry() -> None:
    """Gate 5: ``calls_used + 1 > budget_calls`` → ``RETRY``."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            budget_calls=4,
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=4,
    )
    assert out.stage is DispatchStage.RETRY
    assert "budget exceeded" in out.reason


def test_gate5_budget_equal_admits() -> None:
    """Gate 5: ``calls_used + 1 == budget_calls`` is still within budget."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            budget_calls=4,
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=3,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate6_high_impact_without_approval_returns_deny() -> None:
    """Gate 6: high-impact tool with neither host nor per-call approval → ``DENY``."""

    d = ToolDispatcher(
        registry=(_high_impact_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"net.publish"})),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("publish"), arguments={"message": "hi"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY
    assert "high-impact" in out.reason


def test_gate6_host_high_impact_approved_passes() -> None:
    """Gate 6: host policy ``high_impact_approved=True`` admits the call."""

    d = ToolDispatcher(
        registry=(_high_impact_def(),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"net.publish"}),
            high_impact_approved=True,
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("publish"), arguments={"message": "hi"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate6_per_call_high_impact_approved_passes() -> None:
    """Gate 6: per-call ``policy.high_impact_approved=True`` admits it."""

    d = ToolDispatcher(
        registry=(_high_impact_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"net.publish"})),
    )
    out = d.check(
        ToolCall(
            id="c1",
            tool_id=ToolId("publish"),
            arguments={"message": "hi"},
            policy=ToolPolicy(high_impact_approved=True),
        ),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_gate7_per_call_deny_capability_returns_deny() -> None:
    """Gate 7: per-call ``deny_capabilities`` overrides the host policy."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"fs.read"})),
    )
    out = d.check(
        ToolCall(
            id="c1",
            tool_id=ToolId("read"),
            arguments={"path": "/x"},
            policy=ToolPolicy(deny_capabilities=frozenset({"fs.read"})),
        ),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY
    assert "per-call deny" in out.reason


def test_gate8_per_call_not_in_allow_returns_choose_alt() -> None:
    """Gate 8: per-call ``allow_capabilities`` excludes capability → ``CHOOSE_ALT``."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(allow_capabilities=frozenset({"fs.read"})),
    )
    out = d.check(
        ToolCall(
            id="c1",
            tool_id=ToolId("read"),
            arguments={"path": "/x"},
            policy=ToolPolicy(allow_capabilities=frozenset({"fs.write"})),
        ),
        calls_used=0,
    )
    assert out.stage is DispatchStage.CHOOSE_ALT
    assert "per-call not in allow" in out.reason


def test_all_gates_pass_returns_execute() -> None:
    """All gates pass → ``DispatchStage.EXECUTE`` with reason ``all gates passed``."""

    d = ToolDispatcher(
        registry=(_read_def(),),
        policy=DispatchPolicy(
            allow_capabilities=frozenset({"fs.read"}),
            resource_scopes=frozenset({"/x"}),
        ),
    )
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE
    assert out.reason == "all gates passed"
    assert out.result is None


# ---------------------------------------------------------------------------
# default_dispatcher helper
# ---------------------------------------------------------------------------


def test_default_dispatcher_is_callable() -> None:
    """``default_dispatcher`` is a callable factory."""

    import inspect

    assert callable(default_dispatcher)
    assert inspect.isfunction(default_dispatcher) or inspect.isbuiltin(default_dispatcher)


def test_default_dispatcher_returns_tool_dispatcher() -> None:
    """``default_dispatcher`` returns a :class:`ToolDispatcher`."""

    d = default_dispatcher()
    assert isinstance(d, ToolDispatcher)


def test_default_dispatcher_registry_is_empty() -> None:
    """The default dispatcher has an empty registry mapping."""

    d = default_dispatcher()
    assert d.registry == {}


def test_default_dispatcher_policy_is_default_dispatch_policy() -> None:
    """The default dispatcher uses the default :class:`DispatchPolicy` (all-falsy)."""

    d = default_dispatcher()
    assert isinstance(d.policy, DispatchPolicy)
    assert d.policy.budget_calls == 64
    assert d.policy.allow_capabilities == frozenset()
    assert d.policy.deny_capabilities == frozenset()
    assert d.policy.high_impact_approved is False
    assert d.policy.status is Status.PROPOSED


# ---------------------------------------------------------------------------
# Public-safety boundary
# ---------------------------------------------------------------------------


def test_module_source_has_no_cloud_sdk_imports() -> None:
    """No cloud SDK / orchestration / network dependency surface in the source."""

    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "kubernetes",
        "docker",
        "fabric",  # SSH/automation; not part of a pure dispatcher
    )
    for marker in forbidden:
        assert marker not in _MODULE_SOURCE, f"forbidden import marker: {marker}"


def test_module_source_has_no_hardcoded_secrets() -> None:
    """No literal API-key / private-key / token strings."""

    forbidden_patterns = (
        r"sk-[A-Za-z0-9_-]{8,}",  # generic API-key style
        r"api_key\s*=\s*['\"]sk-",
        r"BEGIN PRIVATE KEY",
        r"BEGIN RSA PRIVATE KEY",
        r"AWS_SECRET_ACCESS_KEY",
    )
    for pat in forbidden_patterns:
        assert not re.search(pat, _MODULE_SOURCE), f"secret pattern: {pat}"


def test_module_source_has_no_print_or_pprint() -> None:
    """No stdout printing — the dispatcher is silent."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_module_source_has_no_subprocess_or_os_system() -> None:
    """No subprocess / ``os.system`` invocations — the dispatcher is pure."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_module_source_has_no_network_requests() -> None:
    """No HTTP client surface — the dispatcher does not talk to the network."""

    forbidden = ("requests.", "urllib.request", "httpx.", "aiohttp.")
    for marker in forbidden:
        assert marker not in _MODULE_SOURCE, f"network marker: {marker}"


def test_module_source_has_no_eval_or_exec() -> None:
    """No dynamic code execution."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_module_source_has_no_env_access() -> None:
    """No ``os.environ`` / ``os.getenv`` — the dispatcher is deterministic."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_module_source_has_no_todo_or_fixme() -> None:
    """No ``# TODO`` / ``# FIXME`` / ``# XXX`` markers."""

    forbidden = ("# TODO", "# FIXME", "# XXX")
    for marker in forbidden:
        assert marker not in _MODULE_SOURCE, f"todo marker: {marker}"


def test_module_source_pins_six_dispatch_stage_members() -> None:
    """The source declares exactly 6 ``DispatchStage`` member lines."""

    # Each member occupies its own ``MEMBER = "value"`` line; allow
    # leading whitespace inside the class body.
    matches = re.findall(r"^\s{4}([A-Z_]+)\s*=\s*\"", _MODULE_SOURCE, re.MULTILINE)
    # EXECUTE, REPAIR, CHOOSE_ALT, RETRY, REPLAN, DENY
    assert "EXECUTE" in matches
    assert "REPAIR" in matches
    assert "CHOOSE_ALT" in matches
    assert "RETRY" in matches
    assert "REPLAN" in matches
    assert "DENY" in matches
    assert (
        len(
            [
                m
                for m in matches
                if m in {"EXECUTE", "REPAIR", "CHOOSE_ALT", "RETRY", "REPLAN", "DENY"}
            ]
        )
        == 6
    )


def test_module_source_pins_two_frozen_markers() -> None:
    """The source pins exactly two ``frozen=True`` markers — one for the
    :class:`DispatchDecision` frozen-slotted dataclass and one for the
    :class:`DispatchPolicy` Pydantic model_config."""

    # @dataclass(slots=True, frozen=True)   <- DispatchDecision (1)
    # model_config = ConfigDict(extra="forbid", frozen=True)   <- DispatchPolicy (2)
    assert _MODULE_SOURCE.count("frozen=True") == 2


def test_module_source_pins_one_frozen_slotted_decision() -> None:
    """The source pins exactly one ``@dataclass(slots=True, frozen=True)``
    (DispatchDecision)."""

    assert _MODULE_SOURCE.count("@dataclass(slots=True, frozen=True)") == 1


def test_module_source_pins_one_pydantic_basemodel_subclass_for_policy() -> None:
    """``DispatchPolicy(BaseModel)`` appears exactly once."""

    assert _MODULE_SOURCE.count("DispatchPolicy(BaseModel)") == 1


def test_module_source_pins_no_extra_pydantic_basemodel_subclasses() -> None:
    """The module declares only one Pydantic ``BaseModel`` subclass (``DispatchPolicy``)."""

    # Count the ``class X(BaseModel):`` occurrences in the source.
    matches = re.findall(r"^class\s+\w+\(BaseModel\)\s*:", _MODULE_SOURCE, re.MULTILINE)
    assert len(matches) == 1
    assert matches[0] == "class DispatchPolicy(BaseModel):"
