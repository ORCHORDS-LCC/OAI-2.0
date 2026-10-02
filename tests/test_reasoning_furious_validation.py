"""Structural / validation tests for ``oai2.reasoning.furious``.

Pins the FURIOUS fast-path controller scaffold (PROPOSED status):

- :class:`FuriousState` -- 8-member StrEnum state machine for the
  single-agent, low-budget profile (IDLE / PLANNING / ACTING / OBSERVING /
  CONTINUE / ESCALATE / HANDOFF / DONE).
- :class:`FuriousDecision` -- frozen-slotted dataclass carrying
  ``next_state`` / ``reason`` / optional ``escalate_to`` ("DEEP" or "SWARM").
- :class:`FuriousContext` -- Pydantic ``BaseModel`` with ``extra="forbid"``
  carrying ``budget_remaining`` / ``last_action`` / ``last_observation`` /
  ``consecutive_failures`` / ``status``.
- :class:`FuriousController` -- pure-function state machine with
  :meth:`FuriousController.decide` that maps
  ``(FuriousContext, current) -> FuriousDecision`` per the design
  transition table.

Behavioural end-to-end coverage lives in ``tests/test_reasoning.py``;
this file pins the *shape* of the API and the invariants the live
FURIOUS worker (when implemented) will rely on.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

import oai2.reasoning as reasoning_package
from oai2.core import Status
from oai2.reasoning.furious import (
    FuriousContext,
    FuriousController,
    FuriousDecision,
    FuriousState,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "reasoning" / "furious.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_furious_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "reasoning/furious.py is unexpectedly empty"


def test_furious_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_furious_module_docstring_mentions_furious_and_fast_path() -> None:
    """The docstring must reference FURIOUS + the fast-path / single-agent pattern."""

    lowered = _MODULE_SOURCE.lower()
    assert "furious" in lowered, "furious.py docstring must mention 'furious'"
    assert "fast" in lowered or "single-agent" in lowered or "low-budget" in lowered, (
        "furious.py docstring must mention fast / single-agent / low-budget"
    )


def test_furious_module_docstring_notes_proposed_status() -> None:
    """The module is PROPOSED — the docstring must say so."""

    assert "PROPOSED" in _MODULE_SOURCE


def test_furious_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_furious_module_imports_str_enum() -> None:
    """``FuriousState`` is a wire-pinned StrEnum."""

    assert re.search(
        r"^from enum import ([^\n]*,\s*)*StrEnum(\s*,|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "furious module must import StrEnum from enum"


def test_furious_module_imports_dataclass_decorator() -> None:
    """``FuriousDecision`` is a ``@dataclass``."""

    assert re.search(
        r"^from dataclasses import ([^\n]*,\s*)*dataclass(\s*,|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "furious module must import dataclass decorator"


def test_furious_module_imports_pydantic_basemodel_configdict_field() -> None:
    """``FuriousContext`` is a Pydantic ``BaseModel`` subclass."""

    match = re.search(
        r"^from pydantic import ([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "furious.py must import pydantic symbols"
    body = match.group(1)
    for symbol in ("BaseModel", "ConfigDict", "Field"):
        assert symbol in body, f"pydantic import must include {symbol!r}"


def test_furious_module_imports_status_from_oai2_core_relative() -> None:
    """``Status`` is imported relatively from ``..core``."""

    match = re.search(
        r"^from\s+\.\.core\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "furious.py must import Status relatively from ..core"
    body = match.group(1)
    assert "Status" in body, f"`..core` import must include Status (got: {body!r})"


def test_furious_module_does_not_import_oai2_package_directly() -> None:
    """No top-level ``import oai2`` or ``from oai2 import ...`` (use relative)."""

    non_docstring_lines = [
        line
        for line in _MODULE_SOURCE.splitlines()
        if line.strip() and not line.strip().startswith('"""')
    ]
    for line in non_docstring_lines:
        assert not re.match(
            r"^(?:from\s+oai2\s+import|import\s+oai2(?:\.|\s|$))",
            line,
        ), f"furious.py must not import oai2 directly (use relative): {line!r}"


def test_furious_module_does_not_import_cloud_runtime_modules() -> None:
    """The furious module is pure; no cloud-runtime imports."""

    forbidden = ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"furious module must not import cloud runtime module {token!r}"
        )


# ---------------------------------------------------------------------------
# 2. __all__ + identity + reasoning package integration
# ---------------------------------------------------------------------------


def test_furious_module_all_exports_expected_symbols() -> None:
    """``__all__`` lists the 4 public symbols."""

    from oai2.reasoning import furious as furious_module

    assert set(furious_module.__all__) == {
        "FuriousController",
        "FuriousContext",
        "FuriousDecision",
        "FuriousState",
    }


def test_furious_module_all_symbols_are_actually_defined() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    from oai2.reasoning import furious as furious_module

    for name in furious_module.__all__:
        assert hasattr(furious_module, name), (
            f"furious.__all__ lists {name!r} but the module has no such attribute"
        )


def test_furious_module_no_unlisted_public_names() -> None:
    """No top-level ``def`` / class / assignment exists outside imports + ``__all__``.

    Imported names (``dataclass``, ``StrEnum``, ``BaseModel``, ``ConfigDict``,
    ``Field``, ``Status``) and dunders are excluded — only locally-defined
    public symbols matter, and every one of those must appear in ``__all__``.
    """

    import ast

    from oai2.reasoning import furious as furious_module

    listed = set(furious_module.__all__)

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
        f"furious module has locally-defined public names missing from __all__: {sorted(unlisted)}"
    )


def test_furious_symbols_are_re_exported_at_oai2_reasoning_package_level() -> None:
    """The 4 FURIOUS symbols are re-exported at ``oai2.reasoning``."""

    for symbol in ("FuriousController", "FuriousContext", "FuriousDecision", "FuriousState"):
        assert hasattr(reasoning_package, symbol), (
            f"oai2.reasoning must re-export {symbol!r} from furious"
        )
        assert getattr(reasoning_package, symbol) is globals()[symbol], (
            f"oai2.reasoning.{symbol} must be the same object as furious.{symbol}"
        )


def test_furious_state_is_a_str_enum_subclass() -> None:
    """``FuriousState`` is a ``StrEnum``."""

    from enum import StrEnum

    assert issubclass(FuriousState, StrEnum), (
        f"FuriousState must subclass StrEnum; bases are {FuriousState.__bases__!r}"
    )


def test_furious_controller_is_a_class() -> None:
    """``FuriousController`` is a class (not a function or instance)."""

    assert inspect.isclass(FuriousController)


def test_furious_context_is_a_pydantic_basemodel_subclass() -> None:
    """``FuriousContext`` is a Pydantic ``BaseModel`` subclass."""

    from pydantic import BaseModel

    assert issubclass(FuriousContext, BaseModel), (
        f"FuriousContext must subclass BaseModel; bases are {FuriousContext.__bases__!r}"
    )


def test_furious_decision_is_a_dataclass() -> None:
    """``FuriousDecision`` is a ``@dataclass``."""

    assert is_dataclass(FuriousDecision), "FuriousDecision must be decorated with @dataclass"


# ---------------------------------------------------------------------------
# 3. FuriousState StrEnum
# ---------------------------------------------------------------------------


def test_furious_state_has_exactly_eight_members() -> None:
    """``FuriousState`` is an 8-member state machine taxonomy."""

    members = list(FuriousState)
    assert len(members) == 8, f"FuriousState must have 8 members; got {len(members)}"


def test_furious_state_member_names_are_pinned() -> None:
    """The 8 ``FuriousState`` member names are pinned to the design taxonomy."""

    assert {m.name for m in FuriousState} == {
        "IDLE",
        "PLANNING",
        "ACTING",
        "OBSERVING",
        "CONTINUE",
        "ESCALATE",
        "HANDOFF",
        "DONE",
    }


def test_furious_state_member_values_are_lowercase_wire_strings() -> None:
    """``FuriousState`` values are the lowercase wire-form strings."""

    expected = {
        "IDLE": "idle",
        "PLANNING": "planning",
        "ACTING": "acting",
        "OBSERVING": "observing",
        "CONTINUE": "continue",
        "ESCALATE": "escalate",
        "HANDOFF": "handoff",
        "DONE": "done",
    }
    for member in FuriousState:
        assert member.value == expected[member.name], (
            f"FuriousState.{member.name}.value must be {expected[member.name]!r}; "
            f"got {member.value!r}"
        )


def test_furious_state_members_compare_equal_to_their_wire_strings() -> None:
    """StrEnum members compare equal to their raw string value."""

    assert FuriousState.IDLE == "idle"
    assert FuriousState.PLANNING == "planning"
    assert FuriousState.ACTING == "acting"
    assert FuriousState.OBSERVING == "observing"
    assert FuriousState.CONTINUE == "continue"
    assert FuriousState.ESCALATE == "escalate"
    assert FuriousState.HANDOFF == "handoff"
    assert FuriousState.DONE == "done"


def test_furious_state_distinct_members_have_distinct_values() -> None:
    """No two ``FuriousState`` members share the same wire-string value."""

    values = [m.value for m in FuriousState]
    assert len(values) == len(set(values)), f"FuriousState values must be unique; got {values!r}"


# ---------------------------------------------------------------------------
# 4. FuriousDecision dataclass
# ---------------------------------------------------------------------------


def test_furious_decision_is_frozen() -> None:
    """``FuriousDecision`` is a frozen dataclass (no mutation after construction)."""

    decision = FuriousDecision(next_state=FuriousState.PLANNING, reason="start")
    with pytest.raises((AttributeError, Exception)) as excinfo:
        decision.reason = "tampered"  # type: ignore[misc]
    # Frozen dataclasses raise FrozenInstanceError (a subclass of AttributeError).
    assert (
        "frozen" in str(excinfo.value).lower()
        or "FrozenInstanceError"
        in type(
            excinfo.value,
        ).__name__
    )


def test_furious_decision_is_slotted() -> None:
    """``FuriousDecision`` declares ``slots=True`` (no ``__dict__``)."""

    decision = FuriousDecision(next_state=FuriousState.PLANNING, reason="start")
    assert not hasattr(decision, "__dict__"), "FuriousDecision(slots=True) must not expose __dict__"


def test_furious_decision_declares_exactly_three_fields() -> None:
    """``FuriousDecision`` exposes exactly ``next_state`` / ``reason`` / ``escalate_to``."""

    field_names = {f.name for f in fields(FuriousDecision)}
    assert field_names == {"next_state", "reason", "escalate_to"}


def test_furious_decision_next_state_and_reason_are_required() -> None:
    """``next_state`` and ``reason`` are mandatory positional args."""

    decision = FuriousDecision(next_state=FuriousState.PLANNING, reason="start")
    assert decision.next_state is FuriousState.PLANNING
    assert decision.reason == "start"


def test_furious_decision_escalate_to_defaults_to_none() -> None:
    """``escalate_to`` defaults to ``None`` (no escalation target by default)."""

    decision = FuriousDecision(next_state=FuriousState.PLANNING, reason="start")
    assert decision.escalate_to is None


def test_furious_decision_accepts_escalate_to_target() -> None:
    """``escalate_to`` accepts the design wire-target strings (``"DEEP"``, ``"SWARM"``)."""

    deep = FuriousDecision(
        next_state=FuriousState.ESCALATE,
        reason="escalating",
        escalate_to="DEEP",
    )
    swarm = FuriousDecision(
        next_state=FuriousState.HANDOFF,
        reason="handing off",
        escalate_to="SWARM",
    )
    assert deep.escalate_to == "DEEP"
    assert swarm.escalate_to == "SWARM"


# ---------------------------------------------------------------------------
# 5. FuriousContext Pydantic BaseModel
# ---------------------------------------------------------------------------


def test_furious_context_declares_exactly_five_fields() -> None:
    """``FuriousContext`` exposes exactly ``budget_remaining`` / ``last_action`` /
    ``last_observation`` / ``consecutive_failures`` / ``status``."""

    fields_ = FuriousContext.model_fields
    assert set(fields_.keys()) == {
        "budget_remaining",
        "last_action",
        "last_observation",
        "consecutive_failures",
        "status",
    }, f"Unexpected FuriousContext fields: {set(fields_.keys())}"


def test_furious_context_forbids_extra_fields() -> None:
    """``FuriousContext`` rejects unknown kwargs (``extra='forbid'``)."""

    with pytest.raises(ValidationError) as excinfo:
        FuriousContext(unknown_field=42)  # type: ignore[call-arg]
    assert "unknown_field" in str(excinfo.value)


def test_furious_context_default_construction_is_a_fresh_session() -> None:
    """Defaults: budget=1.0, no last action, no last observation, 0 consecutive failures, PROPOSED."""

    ctx = FuriousContext()
    assert ctx.budget_remaining == 1.0
    assert ctx.last_action is None
    assert ctx.last_observation is None
    assert ctx.consecutive_failures == 0
    assert ctx.status is Status.PROPOSED


def test_furious_context_accepts_string_last_action_and_observation() -> None:
    """``last_action`` / ``last_observation`` accept ``str`` payloads."""

    ctx = FuriousContext(last_action="ls -la", last_observation="3 files")
    assert ctx.last_action == "ls -la"
    assert ctx.last_observation == "3 files"


def test_furious_context_consecutive_failures_defaults_to_zero() -> None:
    """``consecutive_failures`` default is ``0``."""

    ctx = FuriousContext()
    assert ctx.consecutive_failures == 0


def test_furious_context_consecutive_failures_accepts_zero_and_above() -> None:
    """``consecutive_failures`` is bounded ``ge=0`` (0, 1, 5 all valid)."""

    for n in (0, 1, 3, 5, 100):
        ctx = FuriousContext(consecutive_failures=n)
        assert ctx.consecutive_failures == n


def test_furious_context_consecutive_failures_rejects_negative() -> None:
    """``consecutive_failures=-1`` is rejected (``ge=0``)."""

    with pytest.raises(ValidationError):
        FuriousContext(consecutive_failures=-1)


def test_furious_context_rejects_fractional_floats_for_consecutive_failures() -> None:
    """``consecutive_failures`` rejects fractional floats via ``int_from_float``."""

    with pytest.raises(ValidationError) as excinfo:
        FuriousContext(consecutive_failures=2.5)  # type: ignore[arg-type]
    # Pydantic v2 surfaces this as `int_from_float` in the error string.
    assert "int_from_float" in str(excinfo.value)


def test_furious_context_budget_remaining_accepts_zero_and_negative() -> None:
    """``budget_remaining`` accepts ``0.0`` and negative values (no lower bound)."""

    for v in (-1.0, -100.0, 0.0, 0.5):
        ctx = FuriousContext(budget_remaining=v)
        assert ctx.budget_remaining == v


def test_furious_context_accepts_status_wire_string_as_input() -> None:
    """Pydantic v2 StrEnum coercion: ``status='PROPOSED'`` is accepted."""

    ctx = FuriousContext(status="PROPOSED")  # type: ignore[arg-type]
    assert ctx.status is Status.PROPOSED


def test_furious_context_rejects_unknown_status_wire_string() -> None:
    """Pydantic rejects ``status='NOT_A_REAL_STATUS'``."""

    with pytest.raises(ValidationError):
        FuriousContext(status="NOT_A_REAL_STATUS")  # type: ignore[arg-type]


def test_furious_context_status_can_be_set_to_implemented() -> None:
    """``status=Status.IMPLEMENTED`` round-trips for promotion scenarios."""

    ctx = FuriousContext(status=Status.IMPLEMENTED)
    assert ctx.status is Status.IMPLEMENTED


def test_furious_context_model_dump_round_trips_every_field() -> None:
    """``model_dump()`` round-trips a fully populated FuriousContext."""

    ctx = FuriousContext(
        budget_remaining=0.42,
        last_action="ls -la",
        last_observation="3 files",
        consecutive_failures=2,
        status=Status.EXPERIMENTAL,
    )
    dumped = ctx.model_dump()
    assert dumped["budget_remaining"] == 0.42
    assert dumped["last_action"] == "ls -la"
    assert dumped["last_observation"] == "3 files"
    assert dumped["consecutive_failures"] == 2
    assert dumped["status"] is Status.EXPERIMENTAL


# ---------------------------------------------------------------------------
# 6. FuriousController state machine
# ---------------------------------------------------------------------------


def test_furious_controller_status_class_attribute_is_proposed() -> None:
    """``FuriousController.STATUS`` is ``Status.PROPOSED``."""

    assert FuriousController.STATUS is Status.PROPOSED


def test_furious_controller_does_not_declare_init() -> None:
    """``FuriousController`` does NOT declare ``__init__`` (pure-static helper)."""

    assert "__init__" not in vars(FuriousController), (
        "FuriousController must not declare __init__ — it is a pure-static helper"
    )


def test_furious_controller_decide_is_staticmethod() -> None:
    """``decide`` is declared ``@staticmethod`` (no ``self``)."""

    assert isinstance(
        FuriousController.__dict__["decide"],
        staticmethod,
    )


def test_furious_controller_decide_signature() -> None:
    """``decide(ctx: FuriousContext, *, current: FuriousState) -> FuriousDecision``."""

    sig = inspect.signature(FuriousController.decide)
    params = list(sig.parameters.values())
    # ctx + keyword-only current.
    assert len(params) == 2, f"decide must take (ctx, *, current); got {params!r}"
    assert params[0].name == "ctx"
    assert params[1].name == "current"
    # `current` is keyword-only (the `*` separator is present).
    assert params[1].kind == inspect.Parameter.KEYWORD_ONLY, (
        f"`current` must be keyword-only; got kind={params[1].kind!r}"
    )


def test_furious_controller_decide_current_must_be_passed_as_keyword() -> None:
    """``decide(ctx, current="IDLE")`` — ``current`` cannot be positional."""

    ctx = FuriousContext()
    # Positional call must fail (TypeError: missing keyword-only argument).
    with pytest.raises(TypeError):
        FuriousController.decide(ctx, FuriousState.IDLE)  # type: ignore[call-arg]


def test_furious_controller_decide_is_callable_without_instantiation() -> None:
    """``decide`` is callable on the class itself (no instance needed)."""

    decision = FuriousController.decide(
        FuriousContext(),
        current=FuriousState.IDLE,
    )
    assert isinstance(decision, FuriousDecision)


def test_furious_controller_decide_exhausts_budget_first() -> None:
    """``budget_remaining <= 0.0`` short-circuits before the state transition."""

    decision = FuriousController.decide(
        FuriousContext(budget_remaining=0.0),
        current=FuriousState.IDLE,
    )
    assert decision.next_state is FuriousState.DONE
    assert decision.reason == "budget exhausted"
    assert decision.escalate_to is None


def test_furious_controller_decide_handles_negative_budget() -> None:
    """Negative budget also short-circuits to ``DONE`` (``budget exhausted``)."""

    decision = FuriousController.decide(
        FuriousContext(budget_remaining=-1.0),
        current=FuriousState.PLANNING,
    )
    assert decision.next_state is FuriousState.DONE
    assert decision.reason == "budget exhausted"


def test_furious_controller_decide_escalates_after_three_failures() -> None:
    """``consecutive_failures >= 3`` escalates to ``DEEP``."""

    decision = FuriousController.decide(
        FuriousContext(consecutive_failures=3),
        current=FuriousState.IDLE,
    )
    assert decision.next_state is FuriousState.ESCALATE
    assert decision.escalate_to == "DEEP"
    assert "3 consecutive failures" in decision.reason


def test_furious_controller_decide_escalates_at_exactly_three_failures() -> None:
    """The boundary ``consecutive_failures == 3`` escalates (>= boundary inclusive)."""

    decision = FuriousController.decide(
        FuriousContext(consecutive_failures=3),
        current=FuriousState.PLANNING,
    )
    assert decision.next_state is FuriousState.ESCALATE
    assert decision.escalate_to == "DEEP"


def test_furious_controller_decide_does_not_escalate_at_two_failures() -> None:
    """``consecutive_failures == 2`` falls through to the match branch."""

    decision = FuriousController.decide(
        FuriousContext(consecutive_failures=2),
        current=FuriousState.IDLE,
    )
    assert decision.next_state is FuriousState.PLANNING
    assert decision.escalate_to is None


def test_furious_controller_decide_escalation_short_circuits_before_state_transition() -> None:
    """Escalation happens even when ``current`` would normally loop."""

    # Without escalation, CONTINUE → PLANNING. With escalation, it goes ESCALATE.
    decision = FuriousController.decide(
        FuriousContext(consecutive_failures=5),
        current=FuriousState.CONTINUE,
    )
    assert decision.next_state is FuriousState.ESCALATE
    assert decision.escalate_to == "DEEP"


# Full happy-path transition table.
def test_furious_controller_decide_idle_to_planning() -> None:
    """``IDLE → PLANNING`` with reason ``"start"``."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.IDLE)
    assert decision.next_state is FuriousState.PLANNING
    assert decision.reason == "start"
    assert decision.escalate_to is None


def test_furious_controller_decide_planning_to_acting() -> None:
    """``PLANNING → ACTING`` with reason ``"plan ready"``."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.PLANNING)
    assert decision.next_state is FuriousState.ACTING
    assert decision.reason == "plan ready"


def test_furious_controller_decide_acting_to_observing() -> None:
    """``ACTING → OBSERVING`` with reason ``"action sent"``."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.ACTING)
    assert decision.next_state is FuriousState.OBSERVING
    assert decision.reason == "action sent"


def test_furious_controller_decide_observing_to_continue() -> None:
    """``OBSERVING → CONTINUE`` with reason ``"observed"``."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.OBSERVING)
    assert decision.next_state is FuriousState.CONTINUE
    assert decision.reason == "observed"


def test_furious_controller_decide_continue_loops_to_planning() -> None:
    """``CONTINUE → PLANNING`` (the steady-state loop)."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.CONTINUE)
    assert decision.next_state is FuriousState.PLANNING
    assert decision.reason == "loop"


def test_furious_controller_decide_done_is_terminal() -> None:
    """``DONE`` (current) → ``DONE`` (next) with reason ``"terminal"``."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.DONE)
    assert decision.next_state is FuriousState.DONE
    assert decision.reason == "terminal"


def test_furious_controller_decide_escalate_is_terminal() -> None:
    """``ESCALATE`` (current) hits the default branch → ``DONE`` (``"terminal"``)."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.ESCALATE)
    assert decision.next_state is FuriousState.DONE
    assert decision.reason == "terminal"


def test_furious_controller_decide_handoff_is_terminal() -> None:
    """``HANDOFF`` (current) hits the default branch → ``DONE`` (``"terminal"``)."""

    decision = FuriousController.decide(FuriousContext(), current=FuriousState.HANDOFF)
    assert decision.next_state is FuriousState.DONE
    assert decision.reason == "terminal"


def test_furious_controller_decide_eight_state_transition_contracts() -> None:
    """All 8 ``FuriousState`` values drive the design transition table."""

    expected_transitions = {
        FuriousState.IDLE: (FuriousState.PLANNING, "start"),
        FuriousState.PLANNING: (FuriousState.ACTING, "plan ready"),
        FuriousState.ACTING: (FuriousState.OBSERVING, "action sent"),
        FuriousState.OBSERVING: (FuriousState.CONTINUE, "observed"),
        FuriousState.CONTINUE: (FuriousState.PLANNING, "loop"),
        FuriousState.ESCALATE: (FuriousState.DONE, "terminal"),
        FuriousState.HANDOFF: (FuriousState.DONE, "terminal"),
        FuriousState.DONE: (FuriousState.DONE, "terminal"),
    }
    for current, (want_next, want_reason) in expected_transitions.items():
        decision = FuriousController.decide(FuriousContext(), current=current)
        assert decision.next_state is want_next, (
            f"current={current.name}: expected next={want_next.name}, "
            f"got {decision.next_state.name}"
        )
        assert decision.reason == want_reason, (
            f"current={current.name}: expected reason={want_reason!r}, got {decision.reason!r}"
        )


def test_furious_controller_decide_uses_last_action_and_observation_fields() -> None:
    """``last_action`` / ``last_observation`` exist on context (not used in transitions,
    but the contract pins their presence so a future worker can read them)."""

    ctx = FuriousContext(last_action="ls", last_observation="ok")
    decision = FuriousController.decide(ctx, current=FuriousState.IDLE)
    # Sanity: the transition still fires.
    assert decision.next_state is FuriousState.PLANNING


def test_furious_controller_decide_is_pure_no_state_mutation() -> None:
    """Two calls with identical inputs return identical Decision instances."""

    ctx = FuriousContext()
    d1 = FuriousController.decide(ctx, current=FuriousState.IDLE)
    d2 = FuriousController.decide(ctx, current=FuriousState.IDLE)
    assert d1.next_state is d2.next_state
    assert d1.reason == d2.reason
    assert d1.escalate_to == d2.escalate_to


# ---------------------------------------------------------------------------
# 7. Public-safety boundary
# ---------------------------------------------------------------------------


def test_furious_module_does_not_contain_cloud_credentials() -> None:
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
            f"furious module must not contain credential-like pattern {pattern!r}"
        )


def test_furious_module_does_not_contain_print_or_pprint_calls() -> None:
    """No top-level ``print(...)`` or ``pprint(...)`` calls in module code."""

    non_docstring_lines = [
        line
        for line in _MODULE_SOURCE.splitlines()
        if line.strip() and not line.strip().startswith('"""')
    ]
    for line in non_docstring_lines:
        assert not re.search(r"\bprint\s*\(", line), (
            f"furious module must not call print(): {line!r}"
        )
        assert not re.search(r"\bpprint\s*\(", line), (
            f"furious module must not call pprint(): {line!r}"
        )


def test_furious_module_does_not_import_subprocess() -> None:
    """No ``subprocess`` / ``os.system`` / shell escape hatches."""

    forbidden = ("subprocess", "os.system", "shell=True")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, f"furious module must not import/exec {token!r}"


def test_furious_module_does_not_import_network_clients() -> None:
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
        assert token not in _MODULE_SOURCE, (
            f"furious module must not import network client {token!r}"
        )


def test_furious_module_does_not_use_eval_or_exec() -> None:
    """No dynamic code execution primitives."""

    forbidden = (r"\beval\s\(", r"\bexec\s\(", r"\bcompile\s\(")
    for token in forbidden:
        assert not re.search(token, _MODULE_SOURCE), f"furious module must not use {token!r}"


def test_furious_module_does_not_use_wildcard_imports() -> None:
    """No ``from X import *`` statements (preserves explicit surface)."""

    assert not re.search(
        r"^from\s+\S+\s+import\s+\*",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "furious module must not use wildcard imports"


def test_furious_module_does_not_read_environment_variables() -> None:
    """No ``os.environ`` / ``os.getenv`` access (no runtime config leak)."""

    forbidden = ("os.environ", "os.getenv")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"furious module must not read environment via {token!r}"
        )


def test_furious_module_does_not_contain_todo_or_fixme_markers() -> None:
    """No ``TODO`` / ``FIXME`` / ``XXX`` markers — the scaffold is finished."""

    forbidden = ("TODO", "FIXME", "XXX")
    for token in forbidden:
        assert not re.search(
            rf"\b{re.escape(token)}\b",
            _MODULE_SOURCE,
        ), f"furious module must not contain {token!r} markers"


def test_furious_module_source_is_well_terminated() -> None:
    """Source ends with a single trailing newline (POSIX)."""

    assert _MODULE_SOURCE.endswith("\n"), "furious module source must end with a trailing newline"
    assert not _MODULE_SOURCE.endswith("\n\n\n"), (
        "furious module source must not have multiple trailing newlines"
    )


def test_furious_module_has_no_top_level_dunder_side_effects() -> None:
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
        ast.Match,
    )
    for node in tree.body:
        assert isinstance(node, allowed), (
            f"furious module has unexpected top-level statement "
            f"{type(node).__name__} at line {node.lineno}"
        )
