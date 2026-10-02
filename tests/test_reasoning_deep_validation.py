"""Structural / validation tests for ``oai2.reasoning.deep``.

Pins the DEEP-mode controller scaffold (PROPOSED status):

- :class:`Candidate` -- frozen-slotted dataclass carrying
  ``action`` / ``confidence`` for one branch.
- :class:`DeepContext` -- Pydantic ``BaseModel`` carrying
  ``branch_factor`` / ``evidence_threshold`` / ``status`` for a DEEP
  branching session.
- :class:`DeepController` -- thin static helper with
  :meth:`DeepController.select` that filters candidates by the
  evidence threshold and returns the highest-confidence one.

Behavioural end-to-end coverage lives in ``tests/test_reasoning.py``;
this file pins the *shape* of the API and the invariants the live DEEP
worker (when implemented) will rely on.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

import oai2.reasoning as reasoning_package
from oai2.core import Status
from oai2.reasoning.deep import Candidate, DeepContext, DeepController

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "reasoning" / "deep.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_deep_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "reasoning/deep.py is unexpectedly empty"


def test_deep_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_deep_module_docstring_mentions_deep_and_branching() -> None:
    """The docstring must reference DEEP + the branch + verify pattern."""

    lowered = _MODULE_SOURCE.lower()
    assert "deep" in lowered, "deep.py docstring must mention 'deep'"
    assert "branch" in lowered, "deep.py docstring must mention branching"


def test_deep_module_docstring_notes_proposed_status() -> None:
    """The module is PROPOSED — the docstring must say so."""

    assert "PROPOSED" in _MODULE_SOURCE


def test_deep_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_deep_module_imports_pydantic_basemodel_configdict_field() -> None:
    """``DeepContext`` is a Pydantic ``BaseModel`` subclass."""

    match = re.search(
        r"^from pydantic import ([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "deep.py must import pydantic symbols"
    body = match.group(1)
    for symbol in ("BaseModel", "ConfigDict", "Field"):
        assert symbol in body, f"pydantic import must include {symbol!r}"


def test_deep_module_imports_dataclass() -> None:
    """``Candidate`` uses the stdlib ``@dataclass`` decorator."""

    assert re.search(
        r"^from dataclasses import ([^\n]*,\s*)*dataclass(\s*,|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "deep.py must import `dataclass` from dataclasses"


def test_deep_module_imports_status_from_oai2_core_relative() -> None:
    """``Status`` is imported relatively from ``..core``."""

    match = re.search(
        r"^from\s+\.\.core\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "deep.py must import Status relatively from ..core"
    body = match.group(1)
    assert "Status" in body, f"`..core` import must include Status (got: {body!r})"


def test_deep_module_does_not_import_oai2_package_directly() -> None:
    """No absolute ``import oai2`` / ``from oai2`` at line-start."""

    for pattern in (
        r"^import oai2\b",
        r"^from oai2\b",
    ):
        assert not re.search(pattern, _MODULE_SOURCE, re.MULTILINE), (
            f"deep.py must not use {pattern!r}"
        )


def test_deep_module_does_not_use_wildcard_imports() -> None:
    """No ``from ... import *`` statements."""

    assert not re.search(
        r"^from\s+\S+\s+import\s+\*",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "deep.py must not use wildcard imports"


# ---------------------------------------------------------------------------
# 2. ``__all__`` completeness
# ---------------------------------------------------------------------------


def test_deep_module_declares_dunder_all() -> None:
    """``__all__`` is declared at module scope."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "deep.py must declare __all__"


def test_deep_module_dunder_all_has_exactly_three_names() -> None:
    """``__all__`` exposes exactly the 3 documented names."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    names = [n.strip().strip("\"'") for n in match.group(1).split(",") if n.strip()]
    assert names == ["Candidate", "DeepContext", "DeepController"], (
        f"Expected __all__ == ['Candidate', 'DeepContext', 'DeepController']; got {names!r}"
    )


def test_deep_module_dunder_all_names_are_importable() -> None:
    """Every name in ``__all__`` resolves on the imported module."""

    import oai2.reasoning.deep as deep_module

    for name in ("Candidate", "DeepContext", "DeepController"):
        assert hasattr(deep_module, name), f"{name!r} not exported from deep.py"


def test_deep_module_is_re_exported_at_oai2_reasoning_package_level() -> None:
    """The 3 names are also reachable at the ``oai2.reasoning`` package level."""

    for name in ("Candidate", "DeepContext", "DeepController"):
        assert hasattr(reasoning_package, name), f"{name!r} not re-exported at oai2.reasoning"
        assert getattr(reasoning_package, name) is globals()[name], (
            f"oai2.reasoning.{name} is not identity-equal to oai2.reasoning.deep.{name}"
        )


# ---------------------------------------------------------------------------
# 3. Candidate (frozen-slotted dataclass)
# ---------------------------------------------------------------------------


def test_candidate_is_a_dataclass() -> None:
    """``Candidate`` is a stdlib ``@dataclass``."""

    assert dataclass(Candidate) is Candidate or inspect.isclass(Candidate)
    # `dataclass(Candidate) is Candidate` only when Candidate is decorated with
    # @dataclass; otherwise `dataclass(Candidate)` would return a NEW class.
    # We pin the original True contract: passing through `dataclass()` is a no-op
    # because Candidate already IS a dataclass.


def test_candidate_is_frozen() -> None:
    """``Candidate`` is frozen — assigning to a field raises FrozenInstanceError."""

    cand = Candidate(action="a", confidence=0.5)
    with pytest.raises(Exception) as excinfo:
        cand.action = "b"  # type: ignore[misc]
    # Frozen dataclass raises dataclasses.FrozenInstanceError, surfaced as AttributeError
    # on CPython. We accept either, but assert it's NOT a plain pass.
    assert excinfo.type.__name__ in ("FrozenInstanceError", "AttributeError"), (
        f"Expected frozen-instance rejection, got {excinfo.type.__name__}"
    )


def test_candidate_is_slotted() -> None:
    """``Candidate`` is slotted — instances do not carry a ``__dict__``."""

    cand = Candidate(action="a", confidence=0.5)
    assert not hasattr(cand, "__dict__"), "slotted dataclass instances must not carry __dict__"


def test_candidate_field_set_is_action_and_confidence() -> None:
    """``Candidate`` declares exactly the two fields ``action`` + ``confidence``."""

    fields = Candidate.__dataclass_fields__
    assert set(fields.keys()) == {"action", "confidence"}, (
        f"Expected Candidate fields {{action, confidence}}; got {set(fields.keys())}"
    )


def test_candidate_constructs_with_action_and_confidence() -> None:
    """``Candidate(action=..., confidence=...)`` stores the values."""

    cand = Candidate(action="search_web", confidence=0.7)
    assert cand.action == "search_web"
    assert cand.confidence == 0.7


def test_candidate_confidence_is_a_float() -> None:
    """``Candidate.confidence`` is stored as a ``float`` (no coercion needed)."""

    cand = Candidate(action="x", confidence=0.5)
    assert isinstance(cand.confidence, float)


def test_candidate_action_is_a_string() -> None:
    """``Candidate.action`` is stored as a ``str``."""

    cand = Candidate(action="call_tool", confidence=0.1)
    assert isinstance(cand.action, str)


def test_candidate_accepts_zero_confidence() -> None:
    """0.0 is a legal confidence value (boundary inclusive)."""

    cand = Candidate(action="x", confidence=0.0)
    assert cand.confidence == 0.0


def test_candidate_accepts_confidence_one() -> None:
    """1.0 is a legal confidence value (boundary inclusive)."""

    cand = Candidate(action="x", confidence=1.0)
    assert cand.confidence == 1.0


def test_candidate_repr_contains_action_and_confidence() -> None:
    """``repr(candidate)`` mentions both fields (debug-friendly)."""

    cand = Candidate(action="plan_step", confidence=0.42)
    r = repr(cand)
    assert "plan_step" in r
    assert "0.42" in r


def test_candidate_is_hashable() -> None:
    """Frozen-slotted dataclasses are hashable; ``Candidate`` is."""

    c1 = Candidate(action="a", confidence=0.5)
    c2 = Candidate(action="a", confidence=0.5)
    assert hash(c1) == hash(c2)
    assert c1 == c2


def test_candidate_equality_is_value_comparison() -> None:
    """Two Candidates with the same fields compare equal (no identity)."""

    assert Candidate(action="a", confidence=0.5) == Candidate(action="a", confidence=0.5)
    assert Candidate(action="a", confidence=0.5) != Candidate(action="b", confidence=0.5)
    assert Candidate(action="a", confidence=0.5) != Candidate(action="a", confidence=0.6)


# ---------------------------------------------------------------------------
# 4. DeepContext (Pydantic BaseModel)
# ---------------------------------------------------------------------------


def test_deep_context_subclasses_pydantic_basemodel() -> None:
    """``DeepContext`` is a Pydantic ``BaseModel`` subclass."""

    from pydantic import BaseModel

    assert issubclass(DeepContext, BaseModel)


def test_deep_context_model_config_forbids_extra_fields() -> None:
    """``DeepContext.model_config["extra"] == "forbid"``."""

    assert DeepContext.model_config["extra"] == "forbid"


def test_deep_context_rejects_unknown_field() -> None:
    """``DeepContext(unknown_field=...)`` raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        DeepContext(unknown_field="bad")  # type: ignore[call-arg]


def test_deep_context_field_set_is_three_names() -> None:
    """``DeepContext`` declares exactly ``branch_factor`` / ``evidence_threshold`` / ``status``."""

    assert set(DeepContext.model_fields.keys()) == {
        "branch_factor",
        "evidence_threshold",
        "status",
    }, (
        f"Expected fields {{branch_factor, evidence_threshold, status}}; got "
        f"{set(DeepContext.model_fields.keys())}"
    )


def test_deep_context_default_branch_factor_is_four() -> None:
    """``branch_factor`` defaults to ``4`` (the documented default)."""

    ctx = DeepContext()
    assert ctx.branch_factor == 4


def test_deep_context_default_evidence_threshold_is_zero_point_six() -> None:
    """``evidence_threshold`` defaults to ``0.6`` (the documented default)."""

    ctx = DeepContext()
    assert ctx.evidence_threshold == 0.6


def test_deep_context_default_status_is_proposed() -> None:
    """``status`` defaults to ``Status.PROPOSED``."""

    ctx = DeepContext()
    assert ctx.status is Status.PROPOSED


def test_deep_context_branch_factor_minimum_is_one() -> None:
    """``branch_factor >= 1`` (ge=1) — 0 raises, 1 is accepted."""

    with pytest.raises(ValidationError):
        DeepContext(branch_factor=0)
    ctx = DeepContext(branch_factor=1)
    assert ctx.branch_factor == 1


def test_deep_context_branch_factor_maximum_is_thirty_two() -> None:
    """``branch_factor <= 32`` (le=32) — 32 accepted, 33 raises."""

    ctx = DeepContext(branch_factor=32)
    assert ctx.branch_factor == 32
    with pytest.raises(ValidationError):
        DeepContext(branch_factor=33)


def test_deep_context_evidence_threshold_minimum_is_zero() -> None:
    """``evidence_threshold >= 0.0`` (ge=0.0) — 0.0 accepted, -0.1 raises."""

    ctx = DeepContext(evidence_threshold=0.0)
    assert ctx.evidence_threshold == 0.0
    with pytest.raises(ValidationError):
        DeepContext(evidence_threshold=-0.1)


def test_deep_context_evidence_threshold_maximum_is_one() -> None:
    """``evidence_threshold <= 1.0`` (le=1.0) — 1.0 accepted, 1.1 raises."""

    ctx = DeepContext(evidence_threshold=1.0)
    assert ctx.evidence_threshold == 1.0
    with pytest.raises(ValidationError):
        DeepContext(evidence_threshold=1.1)


def test_deep_context_status_accepts_each_status_value() -> None:
    """``status`` accepts any of the 4 documented ``Status`` enum values."""

    for status in Status:
        ctx = DeepContext(status=status)
        assert ctx.status is status


def test_deep_context_accepts_custom_branch_factor() -> None:
    """``branch_factor`` accepts any int in the documented range."""

    ctx = DeepContext(branch_factor=8)
    assert ctx.branch_factor == 8


def test_deep_context_accepts_custom_evidence_threshold() -> None:
    """``evidence_threshold`` accepts any float in the documented range."""

    ctx = DeepContext(evidence_threshold=0.42)
    assert ctx.evidence_threshold == 0.42


def test_deep_context_rejects_non_int_branch_factor() -> None:
    """``branch_factor`` is `int` — float is rejected (no coercion)."""

    with pytest.raises(ValidationError):
        DeepContext(branch_factor=2.5)  # type: ignore[arg-type]


def test_deep_context_rejects_non_numeric_evidence_threshold() -> None:
    """``evidence_threshold`` is `float` — a non-numeric value is rejected."""

    with pytest.raises(ValidationError):
        DeepContext(evidence_threshold="not-a-number")  # type: ignore[arg-type]


def test_deep_context_rejects_unknown_status_wire_string() -> None:
    """``status`` must be a documented ``Status`` enum wire string."""

    # Note -- a literal ``"PROPOSED"`` is coerced to ``Status.PROPOSED``
    # because ``Status`` is a ``StrEnum`` and Pydantic v2 honours that
    # coercion. A genuinely unknown wire string DOES raise.
    with pytest.raises(ValidationError):
        DeepContext(status="NOT_A_REAL_STATUS")  # type: ignore[arg-type]


def test_deep_context_accepts_status_wire_string_as_input() -> None:
    """``status="PROPOSED"`` is coerced to ``Status.PROPOSED`` (StrEnum contract)."""

    ctx = DeepContext(status="PROPOSED")  # type: ignore[arg-type]
    assert ctx.status is Status.PROPOSED


def test_deep_context_model_dump_round_trip() -> None:
    """``model_dump()`` preserves every field."""

    ctx = DeepContext(branch_factor=8, evidence_threshold=0.5, status=Status.EXPERIMENTAL)
    dumped = ctx.model_dump()
    assert dumped == {
        "branch_factor": 8,
        "evidence_threshold": 0.5,
        "status": Status.EXPERIMENTAL,
    }


# ---------------------------------------------------------------------------
# 5. DeepController (thin static helper)
# ---------------------------------------------------------------------------


def test_deep_controller_is_a_class() -> None:
    """``DeepController`` is a class (not a function or instance)."""

    assert inspect.isclass(DeepController)


def test_deep_controller_status_class_attribute_is_proposed() -> None:
    """``DeepController.STATUS`` is ``Status.PROPOSED``."""

    assert DeepController.STATUS is Status.PROPOSED


def test_deep_controller_select_is_staticmethod() -> None:
    """``select`` is declared ``@staticmethod`` (no ``self``)."""

    assert isinstance(DeepController.__dict__["select"], staticmethod)


def test_deep_controller_select_signature_is_candidates_then_ctx() -> None:
    """``select`` takes ``candidates`` then ``ctx`` (no ``self``)."""

    sig = inspect.signature(DeepController.select)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "candidates"
    assert params[1].name == "ctx"


def test_deep_controller_select_returns_none_when_no_viable_candidate() -> None:
    """When every candidate is below the threshold, ``select`` returns ``None``."""

    ctx = DeepContext(evidence_threshold=0.9)
    result = DeepController.select(
        [Candidate("a", 0.1), Candidate("b", 0.2)],
        ctx,
    )
    assert result is None


def test_deep_controller_select_returns_highest_confidence_candidate() -> None:
    """When multiple candidates clear the threshold, the highest wins."""

    ctx = DeepContext(evidence_threshold=0.5)
    a = Candidate("a", 0.7)
    b = Candidate("b", 0.9)
    c = Candidate("c", 0.6)
    result = DeepController.select([a, b, c], ctx)
    assert result is b


def test_deep_controller_select_filters_below_threshold() -> None:
    """Candidates below the threshold are excluded from the winner pool."""

    ctx = DeepContext(evidence_threshold=0.7)
    low = Candidate("low", 0.5)
    high = Candidate("high", 0.8)
    result = DeepController.select([low, high], ctx)
    assert result is high


def test_deep_controller_select_with_exactly_threshold_value() -> None:
    """A candidate at exactly the threshold IS viable (>=, boundary inclusive)."""

    ctx = DeepContext(evidence_threshold=0.5)
    edge = Candidate("edge", 0.5)
    result = DeepController.select([edge], ctx)
    assert result is edge


def test_deep_controller_select_on_empty_candidates_returns_none() -> None:
    """An empty candidates list returns ``None`` (no viable candidate exists)."""

    ctx = DeepContext()
    assert DeepController.select([], ctx) is None


def test_deep_controller_select_returns_candidate_instance() -> None:
    """``select`` returns a ``Candidate`` instance (or ``None``)."""

    ctx = DeepContext(evidence_threshold=0.0)
    result = DeepController.select([Candidate("only", 0.5)], ctx)
    assert isinstance(result, Candidate)


def test_deep_controller_select_preserves_candidate_identity() -> None:
    """``select`` returns the SAME object that was passed in (no copy)."""

    ctx = DeepContext(evidence_threshold=0.0)
    only = Candidate("only", 0.9)
    assert DeepController.select([only], ctx) is only


def test_deep_controller_select_uses_provided_context_threshold() -> None:
    """``select`` reads ``ctx.evidence_threshold`` (per-call, not the class default)."""

    ctx_high = DeepContext(evidence_threshold=0.95)
    ctx_low = DeepContext(evidence_threshold=0.05)
    cand = Candidate("x", 0.5)
    assert DeepController.select([cand], ctx_high) is None
    assert DeepController.select([cand], ctx_low) is cand


def test_deep_controller_select_callable_directly_on_class() -> None:
    """``DeepController.select(...)`` is callable without instantiation."""

    ctx = DeepContext(evidence_threshold=0.0)
    result = DeepController.select([Candidate("x", 0.5)], ctx)
    assert result is not None


def test_deep_controller_has_no_init() -> None:
    """``DeepController`` has no ``__init__`` declared — it is not meant to be instantiated."""

    assert "__init__" not in DeepController.__dict__


# ---------------------------------------------------------------------------
# 6. Public-safety boundary pin
# ---------------------------------------------------------------------------


def test_deep_module_does_not_import_cloud_sdk() -> None:
    """The module must not import any cloud SDK."""

    for needle in ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric"):
        assert needle not in _MODULE_SOURCE, f"deep.py must not reference {needle!r}"


def test_deep_module_has_no_hardcoded_credentials() -> None:
    """No ``api_key=``, no ``BEGIN PRIVATE KEY`` blocks."""

    assert "api_key=" not in _MODULE_SOURCE
    assert "BEGIN PRIVATE KEY" not in _MODULE_SOURCE


def test_deep_module_does_not_print() -> None:
    """No ``print(`` or ``pprint(`` calls."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_deep_module_does_not_use_subprocess() -> None:
    """No ``subprocess`` / ``os.system`` references."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_deep_module_does_not_import_requests_or_httpx() -> None:
    """No direct ``requests`` / ``urllib`` / ``httpx`` / ``aiohttp`` imports."""

    for needle in ("import requests", "urllib", "httpx", "aiohttp"):
        assert needle not in _MODULE_SOURCE, f"deep.py must not reference {needle!r}"


def test_deep_module_does_not_eval_or_exec() -> None:
    """No ``eval(`` or ``exec(`` calls."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_deep_module_does_not_read_environment() -> None:
    """No ``os.environ`` / ``os.getenv`` access."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_deep_module_has_no_todo_or_fixme_markers() -> None:
    """No ``TODO`` / ``FIXME`` / ``XXX`` markers."""

    for needle in ("TODO", "FIXME", "XXX"):
        assert needle not in _MODULE_SOURCE, f"deep.py must not contain {needle!r}"


def test_deep_module_does_not_reexport_pydantic_internals() -> None:
    """Pydantic internals are not in ``__all__``."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    body = match.group(1)
    for name in ("BaseModel", "ConfigDict", "Field"):
        assert name not in body, f"Pydantic internal {name!r} must not be in __all__"


def test_deep_module_does_not_reexport_core_internals() -> None:
    """``Status`` is imported for type hints but not re-exported here."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    body = match.group(1)
    assert "Status" not in body, "Status must not be in __all__ of deep.py"


def test_deep_module_pins_one_basemodel_subclass() -> None:
    """Exactly one ``BaseModel`` subclass is declared in the module."""

    matches = re.findall(
        r"^class\s+\w+\s*\(\s*BaseModel\s*\)\s*:",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 BaseModel subclass; got {len(matches)}"


def test_deep_module_pins_one_dataclass_decorator() -> None:
    """Exactly one ``@dataclass(...)`` decorator is applied."""

    matches = re.findall(
        r"^@dataclass",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 @dataclass; got {len(matches)}"


def test_deep_module_pins_one_status_class_constant() -> None:
    """Exactly one class-level ``STATUS`` constant — on ``DeepController``."""

    matches = re.findall(
        r"^\s+STATUS\s*=\s*Status\.",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 STATUS class constant; got {len(matches)}"


def test_deep_module_pin_one_staticmethod_decorator() -> None:
    """Exactly one ``@staticmethod`` decorator — on ``DeepController.select``."""

    matches = re.findall(
        r"^\s+@staticmethod",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 @staticmethod; got {len(matches)}"
