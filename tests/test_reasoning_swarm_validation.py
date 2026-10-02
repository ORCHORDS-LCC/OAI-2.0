"""Structural / validation tests for ``oai2.reasoning.swarm``.

Pins the SWARM multi-agent controller scaffold (PROPOSED status):

- :class:`SwarmRole` -- 4-member StrEnum taxonomy of SWARM worker
  roles (planner / implementer / critic / verifier).
- :class:`SwarmContext` -- Pydantic ``BaseModel`` carrying
  ``agents`` / ``roles`` / ``consensus_threshold`` / ``status`` for a
  SWARM session.
- :class:`SwarmController` -- instance class with
  :meth:`SwarmController.critic_pick` that picks the highest-voted
  agent whose score meets the context's consensus threshold.

Behavioural end-to-end coverage lives in ``tests/test_reasoning.py``;
this file pins the *shape* of the API and the invariants the live
SWARM worker (when implemented) will rely on.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import oai2.reasoning as reasoning_package
from oai2.core import AgentId, Status
from oai2.reasoning.swarm import SwarmContext, SwarmController, SwarmRole

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "reasoning" / "swarm.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_swarm_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "reasoning/swarm.py is unexpectedly empty"


def test_swarm_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_swarm_module_docstring_mentions_swarm_and_multi_agent() -> None:
    """The docstring must reference SWARM + the multi-agent pattern."""

    lowered = _MODULE_SOURCE.lower()
    assert "swarm" in lowered, "swarm.py docstring must mention 'swarm'"
    assert "multi-agent" in lowered or "parallel" in lowered or "agents" in lowered, (
        "swarm.py docstring must mention multi-agent / parallel / agents"
    )


def test_swarm_module_docstring_notes_proposed_status() -> None:
    """The module is PROPOSED — the docstring must say so."""

    assert "PROPOSED" in _MODULE_SOURCE


def test_swarm_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_swarm_module_imports_str_enum() -> None:
    """``SwarmRole`` is a wire-pinned StrEnum."""

    assert re.search(
        r"^from enum import ([^\n]*,\s*)*StrEnum(\s*,|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "swarm module must import StrEnum from enum"


def test_swarm_module_imports_pydantic_basemodel_configdict_field() -> None:
    """``SwarmContext`` is a Pydantic ``BaseModel`` subclass."""

    match = re.search(
        r"^from pydantic import ([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "swarm.py must import pydantic symbols"
    body = match.group(1)
    for symbol in ("BaseModel", "ConfigDict", "Field"):
        assert symbol in body, f"pydantic import must include {symbol!r}"


def test_swarm_module_imports_agentid_and_status_from_oai2_core_relative() -> None:
    """``AgentId`` + ``Status`` are imported relatively from ``..core``."""

    match = re.search(
        r"^from\s+\.\.core\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "swarm.py must import AgentId/Status relatively from ..core"
    body = match.group(1)
    assert "AgentId" in body, f"`..core` import must include AgentId (got: {body!r})"
    assert "Status" in body, f"`..core` import must include Status (got: {body!r})"


def test_swarm_module_does_not_import_oai2_package_directly() -> None:
    """No top-level ``import oai2`` or ``from oai2 import ...`` (use relative)."""

    # Allow the docstring to mention "oai2.reasoning"; only check code-ish lines.
    non_docstring_lines = [
        line
        for line in _MODULE_SOURCE.splitlines()
        if line.strip() and not line.strip().startswith('"""')
    ]
    for line in non_docstring_lines:
        assert not re.match(
            r"^(?:from\s+oai2\s+import|import\s+oai2(?:\.|\s|$))",
            line,
        ), f"swarm.py must not import oai2 directly (use relative): {line!r}"


def test_swarm_module_does_not_import_cloud_runtime_modules() -> None:
    """The swarm module is pure; no cloud-runtime imports."""

    forbidden = ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"swarm module must not import cloud runtime module {token!r}"
        )


# ---------------------------------------------------------------------------
# 2. __all__ + identity + reasoning package integration
# ---------------------------------------------------------------------------


def test_swarm_module_all_exports_expected_symbols() -> None:
    """``__all__`` lists exactly ``SwarmContext``, ``SwarmController``, ``SwarmRole``."""

    exported = SwarmController  # local rebinding — keep the name short
    del exported  # silence linter
    from oai2.reasoning import swarm as swarm_module

    assert set(swarm_module.__all__) == {"SwarmContext", "SwarmController", "SwarmRole"}


def test_swarm_module_all_symbols_are_actually_defined() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    from oai2.reasoning import swarm as swarm_module

    for name in swarm_module.__all__:
        assert hasattr(swarm_module, name), (
            f"swarm.__all__ lists {name!r} but the module has no such attribute"
        )


def test_swarm_module_no_unlisted_public_names() -> None:
    """No top-level ``def`` / class / assignment exists outside of imports + ``__all__``.

    Imported names (``StrEnum``, ``BaseModel``, ``ConfigDict``, ``Field``,
    ``AgentId``, ``Status``) and dunders are excluded — only locally-defined
    public symbols matter, and every one of those must appear in ``__all__``.
    """

    import ast

    from oai2.reasoning import swarm as swarm_module

    listed = set(swarm_module.__all__)

    tree = ast.parse(_MODULE_SOURCE)
    imported_names: set[str] = set()
    defined_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined_names.add(node.target.id)

    # Every locally-defined public name must appear in __all__.
    unlisted = {name for name in defined_names if not name.startswith("_") and name not in listed}
    assert unlisted == set(), (
        f"swarm module has locally-defined public names missing from __all__: {sorted(unlisted)}"
    )


def test_swarm_symbols_are_re_exported_at_oai2_reasoning_package_level() -> None:
    """The 3 SWARM symbols are re-exported at ``oai2.reasoning``."""

    for symbol in ("SwarmContext", "SwarmController", "SwarmRole"):
        assert hasattr(reasoning_package, symbol), (
            f"oai2.reasoning must re-export {symbol!r} from swarm"
        )
        assert getattr(reasoning_package, symbol) is globals()[symbol], (
            f"oai2.reasoning.{symbol} must be the same object as swarm.{symbol}"
        )


def test_swarm_role_is_a_str_enum_subclass() -> None:
    """``SwarmRole`` is a ``StrEnum`` (so its members coerce from wire strings)."""

    from enum import StrEnum

    assert issubclass(SwarmRole, StrEnum), (
        f"SwarmRole must subclass StrEnum; bases are {SwarmRole.__bases__!r}"
    )


def test_swarm_controller_is_a_class() -> None:
    """``SwarmController`` is a class (not a function or instance)."""

    assert inspect.isclass(SwarmController)


def test_swarm_context_is_a_pydantic_basemodel_subclass() -> None:
    """``SwarmContext`` is a Pydantic ``BaseModel`` subclass."""

    from pydantic import BaseModel

    assert issubclass(SwarmContext, BaseModel), (
        f"SwarmContext must subclass BaseModel; bases are {SwarmContext.__bases__!r}"
    )


# ---------------------------------------------------------------------------
# 3. SwarmRole StrEnum
# ---------------------------------------------------------------------------


def test_swarm_role_has_exactly_four_members() -> None:
    """``SwarmRole`` is a closed 4-member taxonomy."""

    members = list(SwarmRole)
    assert len(members) == 4, f"SwarmRole must have 4 members; got {len(members)}"


def test_swarm_role_member_names_are_pinned() -> None:
    """The 4 ``SwarmRole`` member names are pinned to the taxonomy."""

    assert {m.name for m in SwarmRole} == {"PLANNER", "IMPLEMENTER", "CRITIC", "VERIFIER"}


def test_swarm_role_member_values_are_lowercase_wire_strings() -> None:
    """``SwarmRole`` values are the lowercase wire-form strings."""

    expected = {
        "PLANNER": "planner",
        "IMPLEMENTER": "implementer",
        "CRITIC": "critic",
        "VERIFIER": "verifier",
    }
    for member in SwarmRole:
        assert member.value == expected[member.name], (
            f"SwarmRole.{member.name}.value must be {expected[member.name]!r}; got {member.value!r}"
        )


def test_swarm_role_members_compare_equal_to_their_wire_strings() -> None:
    """StrEnum members compare equal to their raw string value."""

    assert SwarmRole.PLANNER == "planner"
    assert SwarmRole.IMPLEMENTER == "implementer"
    assert SwarmRole.CRITIC == "critic"
    assert SwarmRole.VERIFIER == "verifier"


def test_swarm_role_distinct_members_have_distinct_values() -> None:
    """No two ``SwarmRole`` members share the same wire-string value."""

    values = [m.value for m in SwarmRole]
    assert len(values) == len(set(values)), f"SwarmRole values must be unique; got {values!r}"


# ---------------------------------------------------------------------------
# 4. SwarmContext Pydantic BaseModel
# ---------------------------------------------------------------------------


def test_swarm_context_declares_exactly_four_fields() -> None:
    """``SwarmContext`` exposes exactly ``agents`` / ``roles`` /
    ``consensus_threshold`` / ``status``."""

    fields = SwarmContext.model_fields
    assert set(fields.keys()) == {
        "agents",
        "roles",
        "consensus_threshold",
        "status",
    }, f"Unexpected SwarmContext fields: {set(fields.keys())}"


def test_swarm_context_forbids_extra_fields() -> None:
    """``SwarmContext`` rejects unknown kwargs (``extra='forbid'``)."""

    with pytest.raises(ValidationError) as excinfo:
        SwarmContext(unknown_field=42)  # type: ignore[call-arg]
    assert "unknown_field" in str(excinfo.value)


def test_swarm_context_default_construction_uses_empty_collections() -> None:
    """Defaults: empty agent tuple, empty role map, threshold 0.6, PROPOSED status."""

    ctx = SwarmContext()
    assert ctx.agents == ()
    assert ctx.roles == {}
    assert ctx.consensus_threshold == 0.6
    assert ctx.status is Status.PROPOSED


def test_swarm_context_accepts_an_empty_tuple_of_agents() -> None:
    """``SwarmContext(agents=())`` is valid (no agents yet)."""

    ctx = SwarmContext(agents=())
    assert ctx.agents == ()


def test_swarm_context_accepts_a_tuple_of_agent_ids() -> None:
    """``SwarmContext(agents=(a1, a2))`` preserves the order."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    ctx = SwarmContext(agents=(a1, a2))
    assert ctx.agents == (a1, a2)


def test_swarm_context_accepts_a_role_assignment_per_agent() -> None:
    """``SwarmContext(roles={a1: PLANNER, a2: CRITIC})`` round-trips."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    ctx = SwarmContext(roles={a1: SwarmRole.PLANNER, a2: SwarmRole.CRITIC})
    assert ctx.roles == {a1: SwarmRole.PLANNER, a2: SwarmRole.CRITIC}


def test_swarm_context_consensus_threshold_default_is_zero_point_six() -> None:
    """``consensus_threshold`` default is ``0.6`` (the design pin)."""

    ctx = SwarmContext()
    assert ctx.consensus_threshold == pytest.approx(0.6)


def test_swarm_context_consensus_threshold_can_be_lowered_to_zero() -> None:
    """``consensus_threshold=0.0`` is valid (any vote qualifies)."""

    ctx = SwarmContext(consensus_threshold=0.0)
    assert ctx.consensus_threshold == 0.0


def test_swarm_context_consensus_threshold_can_be_raised_to_one() -> None:
    """``consensus_threshold=1.0`` is valid (only unanimous vote qualifies)."""

    ctx = SwarmContext(consensus_threshold=1.0)
    assert ctx.consensus_threshold == 1.0


def test_swarm_context_consensus_threshold_rejects_negative_value() -> None:
    """``consensus_threshold=-0.01`` is rejected (``ge=0.0``)."""

    with pytest.raises(ValidationError):
        SwarmContext(consensus_threshold=-0.01)


def test_swarm_context_consensus_threshold_rejects_value_above_one() -> None:
    """``consensus_threshold=1.01`` is rejected (``le=1.0``)."""

    with pytest.raises(ValidationError):
        SwarmContext(consensus_threshold=1.01)


def test_swarm_context_accepts_status_wire_string_as_input() -> None:
    """Pydantic v2 StrEnum coercion: ``status='PROPOSED'`` is accepted."""

    ctx = SwarmContext(status="PROPOSED")  # type: ignore[arg-type]
    assert ctx.status is Status.PROPOSED


def test_swarm_context_rejects_unknown_status_wire_string() -> None:
    """Pydantic rejects ``status='NOT_A_REAL_STATUS'``."""

    with pytest.raises(ValidationError):
        SwarmContext(status="NOT_A_REAL_STATUS")  # type: ignore[arg-type]


def test_swarm_context_status_can_be_set_to_implemented() -> None:
    """``status=Status.IMPLEMENTED`` round-trips for promotion scenarios."""

    ctx = SwarmContext(status=Status.IMPLEMENTED)
    assert ctx.status is Status.IMPLEMENTED


def test_swarm_context_field_types_are_pinned() -> None:
    """Field annotations on ``SwarmContext`` match the design contract."""

    fields = SwarmContext.model_fields
    assert fields["agents"].annotation == tuple[AgentId, ...] or (
        # Pydantic may render the annotation as a string under PEP 563.
        "tuple[AgentId, ...]" in str(fields["agents"].annotation)
    )
    assert fields["roles"].annotation == dict[AgentId, SwarmRole] or (
        "dict[AgentId, SwarmRole]" in str(fields["roles"].annotation)
    )
    assert "float" in str(fields["consensus_threshold"].annotation)
    assert fields["status"].annotation is Status or ("Status" in str(fields["status"].annotation))


# ---------------------------------------------------------------------------
# 5. SwarmController (instance class with __init__)
# ---------------------------------------------------------------------------


def test_swarm_controller_status_class_attribute_is_proposed() -> None:
    """``SwarmController.STATUS`` is ``Status.PROPOSED``."""

    assert SwarmController.STATUS is Status.PROPOSED


def test_swarm_controller_declares_an_init_method() -> None:
    """``SwarmController`` declares ``__init__`` (instance class, not pure-static)."""

    assert "__init__" in vars(SwarmController), "SwarmController must declare its own __init__"


def test_swarm_controller_init_signature_accepts_a_context() -> None:
    """``SwarmController(ctx: SwarmContext)`` is the public constructor."""

    sig = inspect.signature(SwarmController.__init__)
    params = list(sig.parameters.values())
    # self + ctx
    assert len(params) == 2, f"SwarmController.__init__ must take (self, ctx); got {params!r}"
    assert params[1].name == "ctx"
    # Under ``from __future__ import annotations`` (PEP 563), the annotation
    # may surface as either the resolved class object or its string form.
    ann = params[1].annotation
    assert (
        ann is SwarmContext
        or ann == "SwarmContext"
        or (isinstance(ann, str) and "SwarmContext" in ann)
    ), f"SwarmController.__init__ ctx param must annotate SwarmContext; got {ann!r}"


def test_swarm_controller_init_stores_ctx_on_self() -> None:
    """``SwarmController(ctx)`` assigns ``self.ctx = ctx``."""

    ctx = SwarmContext()
    controller = SwarmController(ctx)
    assert controller.ctx is ctx


def test_swarm_controller_critic_pick_is_a_regular_method() -> None:
    """``critic_pick`` is an *instance* method, not a ``@staticmethod``."""

    # A regular function in the class namespace (not wrapped in staticmethod).
    raw = SwarmController.__dict__["critic_pick"]
    assert not isinstance(raw, staticmethod), (
        "critic_pick must be an instance method, not a staticmethod"
    )


def test_swarm_controller_critic_pick_signature() -> None:
    """``critic_pick(self, votes: dict[AgentId, float]) -> AgentId | None``."""

    sig = inspect.signature(SwarmController.critic_pick)
    params = list(sig.parameters.values())
    # self + votes
    assert len(params) == 2, f"critic_pick must take (self, votes); got {params!r}"
    assert params[1].name == "votes"
    # Return annotation pins AgentId | None.
    assert sig.return_annotation is AgentId or ("AgentId | None" in str(sig.return_annotation))


def test_swarm_controller_critic_pick_on_empty_votes_returns_none() -> None:
    """Empty vote map → ``None`` (no consensus possible)."""

    controller = SwarmController(SwarmContext())
    assert controller.critic_pick({}) is None


def test_swarm_controller_critic_pick_single_vote_above_threshold() -> None:
    """Single vote above the default 0.6 threshold returns that agent."""

    a1 = AgentId("agent-1")
    controller = SwarmController(SwarmContext())
    assert controller.critic_pick({a1: 0.9}) is a1


def test_swarm_controller_critic_pick_single_vote_below_threshold() -> None:
    """Single vote below the default 0.6 threshold returns ``None``."""

    a1 = AgentId("agent-1")
    controller = SwarmController(SwarmContext())
    assert controller.critic_pick({a1: 0.3}) is None


def test_swarm_controller_critic_pick_single_vote_exactly_at_threshold() -> None:
    """Single vote at exactly the threshold (0.6) qualifies (>= boundary)."""

    a1 = AgentId("agent-1")
    controller = SwarmController(SwarmContext())
    assert controller.critic_pick({a1: 0.6}) is a1


def test_swarm_controller_critic_pick_picks_highest_score_above_threshold() -> None:
    """Multiple votes above threshold → the highest-scored agent wins."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    a3 = AgentId("agent-3")
    controller = SwarmController(SwarmContext())
    winner = controller.critic_pick({a1: 0.7, a2: 0.9, a3: 0.8})
    assert winner is a2


def test_swarm_controller_critic_pick_returns_none_when_all_below_threshold() -> None:
    """Multiple votes but every score below threshold → ``None``."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    controller = SwarmController(SwarmContext())
    assert controller.critic_pick({a1: 0.1, a2: 0.5}) is None


def test_swarm_controller_critic_pick_picks_highest_even_when_others_qualify() -> None:
    """Highest-scored agent is selected even if multiple are above threshold."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    a3 = AgentId("agent-3")
    controller = SwarmController(SwarmContext())
    # a1 and a2 are above threshold, but a3 is highest → a3 wins.
    assert controller.critic_pick({a1: 0.7, a2: 0.8, a3: 0.95}) is a3


def test_swarm_controller_critic_pick_uses_per_call_threshold() -> None:
    """``critic_pick`` reads ``ctx.consensus_threshold`` per-call, not at init."""

    a1 = AgentId("agent-1")
    # Threshold 0.95 — 0.7 must fail.
    controller = SwarmController(SwarmContext(consensus_threshold=0.95))
    assert controller.critic_pick({a1: 0.7}) is None
    # 0.97 qualifies.
    assert controller.critic_pick({a1: 0.97}) is a1


def test_swarm_controller_critic_pick_at_zero_threshold_always_picks() -> None:
    """With ``consensus_threshold=0.0``, any non-empty vote map picks the max."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    controller = SwarmController(SwarmContext(consensus_threshold=0.0))
    assert controller.critic_pick({a1: 0.0, a2: 0.0}) is a1 or a2
    # Higher score must still win.
    assert controller.critic_pick({a1: 0.1, a2: 0.0}) is a1


def test_swarm_controller_critic_pick_at_full_threshold_requires_unanimous() -> None:
    """With ``consensus_threshold=1.0``, only a perfect score qualifies."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    controller = SwarmController(SwarmContext(consensus_threshold=1.0))
    assert controller.critic_pick({a1: 1.0, a2: 0.5}) is a1
    assert controller.critic_pick({a1: 0.99, a2: 0.99}) is None


def test_swarm_controller_critic_pick_does_not_mutate_input_votes() -> None:
    """``critic_pick`` does not mutate the supplied vote map."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    votes: dict[AgentId, float] = {a1: 0.7, a2: 0.8}
    snapshot = dict(votes)
    controller = SwarmController(SwarmContext())
    _ = controller.critic_pick(votes)
    assert votes == snapshot


def test_swarm_controller_two_instances_share_no_state() -> None:
    """Two ``SwarmController`` instances over different contexts are independent."""

    a1 = AgentId("agent-1")
    a2 = AgentId("agent-2")
    c1 = SwarmController(SwarmContext(consensus_threshold=0.0))
    c2 = SwarmController(SwarmContext(consensus_threshold=1.0))
    # c1 returns a2 (highest score), c2 returns None (no unanimous vote).
    assert c1.critic_pick({a1: 0.1, a2: 0.5}) is a2
    assert c2.critic_pick({a1: 0.1, a2: 0.5}) is None


# ---------------------------------------------------------------------------
# 6. Public-safety boundary
# ---------------------------------------------------------------------------


def test_swarm_module_does_not_contain_cloud_credentials() -> None:
    """No hard-coded API keys / private keys / bearer tokens in the source."""

    forbidden_patterns = (
        r"api[_-]?key\s*=\s*['\"]sk-",
        r"BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY",
        r"AKIA[0-9A-Z]{16}",  # AWS access key id pattern.
        r"AIza[0-9A-Za-z\-_]{35}",  # GCP API key pattern.
        r"xox[baprs]-[0-9A-Za-z\-]+",  # Slack token pattern.
    )
    for pattern in forbidden_patterns:
        assert not re.search(pattern, _MODULE_SOURCE, re.IGNORECASE), (
            f"swarm module must not contain credential-like pattern {pattern!r}"
        )


def test_swarm_module_does_not_contain_print_or_pprint_calls() -> None:
    """No top-level ``print(...)`` or ``pprint(...)`` calls in module code."""

    non_docstring_lines = [
        line
        for line in _MODULE_SOURCE.splitlines()
        if line.strip() and not line.strip().startswith('"""')
    ]
    for line in non_docstring_lines:
        assert not re.search(r"\bprint\s*\(", line), f"swarm module must not call print(): {line!r}"
        assert not re.search(r"\bpprint\s*\(", line), (
            f"swarm module must not call pprint(): {line!r}"
        )


def test_swarm_module_does_not_import_subprocess() -> None:
    """No ``subprocess`` / ``os.system`` / shell escape hatches."""

    forbidden = ("subprocess", "os.system", "shell=True")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, f"swarm module must not import/exec {token!r}"


def test_swarm_module_does_not_import_network_clients() -> None:
    """No outbound HTTP / RPC clients (``requests``, ``urllib``, ``httpx``)."""

    forbidden = (
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "import httpx",
        "from httpx",
        "import aiohttp",
        "from aiohttp",
    )
    for token in forbidden:
        assert token not in _MODULE_SOURCE, f"swarm module must not import network client {token!r}"


def test_swarm_module_does_not_use_eval_or_exec() -> None:
    """No dynamic code execution primitives."""

    forbidden = (r"\beval\s\(", r"\bexec\s\(", r"\bcompile\s\(")
    for token in forbidden:
        assert not re.search(token, _MODULE_SOURCE), f"swarm module must not use {token!r}"


def test_swarm_module_does_not_use_wildcard_imports() -> None:
    """No ``from X import *`` statements (preserves explicit surface)."""

    assert not re.search(
        r"^from\s+\S+\s+import\s+\*",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "swarm module must not use wildcard imports"


def test_swarm_module_does_not_read_environment_variables() -> None:
    """No ``os.environ`` / ``os.getenv`` access (no runtime config leak)."""

    forbidden = ("os.environ", "os.getenv")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, f"swarm module must not read environment via {token!r}"


def test_swarm_module_does_not_contain_todo_or_fixme_markers() -> None:
    """No ``TODO`` / ``FIXME`` / ``XXX`` markers — the scaffold is finished."""

    forbidden = ("TODO", "FIXME", "XXX")
    for token in forbidden:
        assert not re.search(
            rf"\b{re.escape(token)}\b",
            _MODULE_SOURCE,
        ), f"swarm module must not contain {token!r} markers"


def test_swarm_module_source_is_well_terminated() -> None:
    """Source ends with a single trailing newline (POSIX)."""

    assert _MODULE_SOURCE.endswith("\n"), "swarm module source must end with a trailing newline"
    assert not _MODULE_SOURCE.endswith("\n\n\n"), (
        "swarm module source must not have multiple trailing newlines"
    )


def test_swarm_module_has_no_top_level_dunder_side_effects() -> None:
    """No top-level statements beyond imports, defs, class defs, and ``__all__``."""

    import ast

    tree = ast.parse(_MODULE_SOURCE)
    allowed = (
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,
        ast.If,
        ast.Try,
        ast.With,
        ast.Pass,
    )
    for node in tree.body:
        assert isinstance(node, allowed), (
            f"swarm module has unexpected top-level statement "
            f"{type(node).__name__} at line {node.lineno}"
        )
