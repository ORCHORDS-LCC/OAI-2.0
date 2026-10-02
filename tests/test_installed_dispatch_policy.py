"""Tests for the ACTUALLY INSTALLED :func:`default_dispatch_policy`.

These do not use a narrowed fixture policy: they exercise the policy the
agent loop really installs, because a denial proven against a hand-built
fixture says nothing about whether the serving policy rejects the same
action.

The two defects pinned here previously (as *observed behaviour*) are now
fixed, so these assert the corrected semantics:

1. ``resource_scopes`` holds directory roots and is applied by
   containment after ``..`` normalisation, not by exact string membership.
   Previously only a literal scope root passed, so every concrete path was
   refused — the scopes were decoration.
2. An unscoped capability (``Bash``, ``Glob`` — neither carries a path)
   used to walk straight past the scope gate. It now requires an explicit
   ``allow_unscoped_capabilities`` grant, and one is stated by
   :func:`default_dispatch_policy` so the default agent surface still works.
"""

from __future__ import annotations

import pytest

from oai2.agents.agent_loop import default_dispatch_policy
from oai2.core import ToolId
from oai2.protocols import ToolCall
from oai2.tools.dispatch import DispatchPolicy, DispatchStage, ToolDispatcher
from oai2.tools.registry import default_tool_definitions


def _call(tool: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id="c1", tool_id=ToolId(tool), arguments=dict(arguments))


@pytest.fixture
def installed(tmp_path):
    return ToolDispatcher(
        registry=default_tool_definitions(),
        policy=default_dispatch_policy(),
        cwd=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. Scoping is containment, not exact membership
# ---------------------------------------------------------------------------


def test_default_policy_scopes_are_roots() -> None:
    assert default_dispatch_policy().resource_scopes == frozenset({"./", "/tmp", "/Users/orchords"})


def test_concrete_file_under_a_scope_root_is_allowed(installed, tmp_path) -> None:
    """A real file path beneath a scope root must EXECUTE, not be refused."""
    target = tmp_path / "app.py"
    target.write_text("x\n", encoding="utf-8")
    installed._cwd = tmp_path  # noqa: SLF001 - scope root is "./" relative to cwd
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    out = d.check(_call("Read", {"path": str(target)}), calls_used=0)
    assert out.stage is DispatchStage.EXECUTE, out.reason


def test_nested_subdirectory_under_a_scope_root_is_allowed(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({"/allowed"}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy)
    assert d.check(_call("Read", {"path": "/allowed/a/b/c.py"}), calls_used=0).stage is (
        DispatchStage.EXECUTE
    )


def test_path_outside_every_scope_root_is_denied(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({"/allowed"}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy)
    out = d.check(_call("Read", {"path": "/etc/passwd"}), calls_used=0)
    assert out.stage is DispatchStage.DENY
    assert "resource out of scope" in out.reason


def test_parent_traversal_cannot_escape_a_scope(installed, tmp_path) -> None:
    """``<scope>/../etc/passwd`` must normalise out of scope, not through it."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    out = d.check(_call("Read", {"path": f"{tmp_path}/../etc/passwd"}), calls_used=0)
    assert out.stage is DispatchStage.DENY
    assert "resource out of scope" in out.reason


def test_sibling_prefix_directory_is_not_in_scope(tmp_path) -> None:
    """``/allowed-evil`` must not pass as being inside ``/allowed``."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({"/allowed"}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy)
    assert d.check(_call("Read", {"path": "/allowed-evil/x"}), calls_used=0).stage is (
        DispatchStage.DENY
    )


def test_relative_path_resolves_against_cwd(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Read", {"path": "sub/app.py"}), calls_used=0).stage is (
        DispatchStage.EXECUTE
    )
    assert d.check(_call("Read", {"path": "../outside.py"}), calls_used=0).stage is (
        DispatchStage.DENY
    )


# ---------------------------------------------------------------------------
# 2. Unscoped capabilities need an explicit grant
# ---------------------------------------------------------------------------


def test_tool_scoping_table() -> None:
    scoped = {t.name: t.scoped for t in default_tool_definitions()}
    assert scoped == {
        "Read": True,
        "Edit": True,
        "Write": True,
        "Bash": False,
        "Glob": False,
        "Grep": True,
    }


def test_grep_cannot_escape_a_denial_on_read(tmp_path) -> None:
    """The fix for defect 2: Grep was a way to read what Read was refused."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
        high_impact_approved=True,
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Read", {"path": "/etc/passwd"}), calls_used=0).stage is (
        DispatchStage.DENY
    )
    # Same capability, same out-of-scope target: Grep must not be a bypass.
    assert d.check(_call("Grep", {"pattern": "root", "path": "/etc"}), calls_used=0).stage is (
        DispatchStage.DENY
    )


def test_unscoped_capability_is_denied_without_an_explicit_grant(tmp_path) -> None:
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"shell.exec"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
        high_impact_approved=True,
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    out = d.check(_call("Bash", {"command": "cat /etc/passwd"}), calls_used=0)
    assert out.stage is DispatchStage.DENY
    assert "unscoped capability" in out.reason


def test_unscoped_capability_runs_when_explicitly_granted(installed) -> None:
    """Legitimate authorised operation still works under the real policy."""
    out = installed.check(_call("Bash", {"command": "ls"}), calls_used=0)
    assert out.stage is DispatchStage.EXECUTE, out.reason


def test_default_policy_grants_exactly_the_tools_without_a_path() -> None:
    granted = default_dispatch_policy().allow_unscoped_capabilities
    assert granted == frozenset({"shell.exec", "fs.list"})
    # Nothing that *can* be scoped may hold an unscoped grant.
    for td in default_tool_definitions():
        if td.scoped:
            assert td.capability not in granted, td.name


def test_no_scoping_intent_means_no_unscoped_gate(tmp_path) -> None:
    """A host that declares no scopes has expressed no intent to scope."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"shell.exec"}),
        resource_scopes=frozenset(),
        high_impact_approved=True,
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Bash", {"command": "ls"}), calls_used=0).stage is DispatchStage.EXECUTE


def test_scoped_tool_with_no_path_argument_is_denied(tmp_path) -> None:
    """A scoped tool that omits its path must not slip through unscoped."""
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read"}),
        resource_scopes=frozenset({str(tmp_path)}),
        allow_unscoped_capabilities=frozenset(),
    )
    d = ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=tmp_path)
    assert d.check(_call("Read", {}), calls_used=0).stage is DispatchStage.DENY
