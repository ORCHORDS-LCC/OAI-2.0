"""Tool dispatch policy tests — exercises the six gates from TOOL_CALLING.md."""

from __future__ import annotations

from oai2.core import ToolId
from oai2.protocols import ToolArgument, ToolCall, ToolDefinition, ToolPolicy
from oai2.tools import DispatchPolicy, DispatchStage, ToolDispatcher


def _read_def() -> ToolDefinition:
    return ToolDefinition(
        id=ToolId("read"),
        name="read",
        description="read",
        arguments=(ToolArgument(name="path", type="path"),),
        capability="fs.read",
        scoped=True,
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


def _dispatcher(allow: set[str], deny: set[str], scopes: set[str]) -> ToolDispatcher:
    return ToolDispatcher(
        registry=(_read_def(), _high_impact_def()),
        policy=DispatchPolicy(
            allow_capabilities=frozenset(allow),
            deny_capabilities=frozenset(deny),
            resource_scopes=frozenset(scopes),
            budget_calls=4,
            high_impact_approved=False,
        ),
    )


def test_unknown_tool_is_replan() -> None:
    d = _dispatcher(set(), set(), set())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("nope"), arguments={}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.REPLAN


def test_unknown_arguments_is_repair() -> None:
    d = _dispatcher({"fs.read"}, set(), {"x"})
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "x", "bad": 1}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.REPAIR


def test_denied_capability_is_deny() -> None:
    d = _dispatcher(set(), {"fs.read"}, set())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY


def test_capability_not_in_allow_list_is_choose_alt() -> None:
    d = _dispatcher({"fs.write"}, set(), set())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.CHOOSE_ALT


def test_resource_out_of_scope_is_deny() -> None:
    d = _dispatcher({"fs.read"}, set(), {"/allowed"})
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/secret"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY


def test_budget_exceeded_is_retry() -> None:
    d = _dispatcher({"fs.read"}, set(), {"/x"})
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=4,
    )
    assert out.stage is DispatchStage.RETRY


def test_high_impact_needs_approval() -> None:
    d = _dispatcher({"net.publish"}, set(), set())
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("publish"), arguments={"message": "hi"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.DENY


def test_high_impact_with_per_call_approval_executes() -> None:
    d = _dispatcher({"net.publish"}, set(), set())
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


def test_all_gates_pass_executes() -> None:
    d = _dispatcher({"fs.read"}, set(), {"/x"})
    out = d.check(
        ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"}),
        calls_used=0,
    )
    assert out.stage is DispatchStage.EXECUTE


def test_tools_module_exports_from_tools_package() -> None:
    from oai2.tools import DispatchDecision as ExportedDecision
    from oai2.tools import DispatchPolicy as ExportedPolicy
    from oai2.tools import DispatchStage as ExportedStage
    from oai2.tools import ToolDispatcher as ExportedDispatcher
    from oai2.tools import default_dispatcher as ExportedDefaultDispatcher
    from oai2.tools.dispatch import (
        DispatchDecision,
        DispatchPolicy,
        DispatchStage,
        ToolDispatcher,
        default_dispatcher,
    )

    assert ExportedDecision is DispatchDecision
    assert ExportedPolicy is DispatchPolicy
    assert ExportedStage is DispatchStage
    assert ExportedDispatcher is ToolDispatcher
    assert ExportedDefaultDispatcher is default_dispatcher
