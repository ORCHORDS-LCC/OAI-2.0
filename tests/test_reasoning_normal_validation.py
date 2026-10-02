"""Structural / validation tests for ``oai2.reasoning.normal``.

Pins the NORMAL-mode controller scaffold (PROPOSED status):

- :class:`NormalContext` -- Pydantic ``BaseModel`` carrying
  ``active_experts`` / ``speculative`` / ``multi_token_prediction`` /
  ``status`` for a NORMAL decoding session.
- :class:`NormalController` -- thin static helper with
  :meth:`NormalController.choose_decode_strategy` that selects one of
  four named decode strategies from the boolean flags.

Behavioural end-to-end coverage lives in ``tests/test_reasoning.py``;
this file pins the *shape* of the API and the invariants the live NORMAL
worker (when implemented) will rely on.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import oai2.reasoning as reasoning_package
from oai2.core import Status
from oai2.reasoning.normal import NormalContext, NormalController

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "reasoning" / "normal.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_normal_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "reasoning/normal.py is unexpectedly empty"


def test_normal_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_normal_module_docstring_mentions_normal_and_coding() -> None:
    """The docstring must reference NORMAL + the coding/broader-expert pattern."""

    lowered = _MODULE_SOURCE.lower()
    assert "normal" in lowered, "normal.py docstring must mention 'normal'"
    assert "coding" in lowered or "expert" in lowered, (
        "normal.py docstring must mention 'coding' or 'expert'"
    )


def test_normal_module_docstring_notes_proposed_status() -> None:
    """The module is PROPOSED — the docstring must say so."""

    assert "PROPOSED" in _MODULE_SOURCE


def test_normal_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_normal_module_imports_pydantic_basemodel_configdict_field() -> None:
    """``NormalContext`` is a Pydantic ``BaseModel`` subclass."""

    match = re.search(
        r"^from pydantic import ([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "normal.py must import pydantic symbols"
    body = match.group(1)
    for symbol in ("BaseModel", "ConfigDict", "Field"):
        assert symbol in body, f"pydantic import must include {symbol!r}"


def test_normal_module_imports_status_from_oai2_core_relative() -> None:
    """``Status`` is imported relatively from ``..core``."""

    match = re.search(
        r"^from\s+\.\.core\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "normal.py must import Status relatively from ..core"
    body = match.group(1)
    assert "Status" in body, f"`..core` import must include Status (got: {body!r})"


def test_normal_module_does_not_import_oai2_package_directly() -> None:
    """No absolute ``import oai2`` / ``from oai2`` at line-start."""

    for pattern in (
        r"^import oai2\b",
        r"^from oai2\b",
    ):
        assert not re.search(pattern, _MODULE_SOURCE, re.MULTILINE), (
            f"normal.py must not use {pattern!r}"
        )


def test_normal_module_does_not_use_wildcard_imports() -> None:
    """No ``from ... import *`` statements."""

    assert not re.search(
        r"^from\s+\S+\s+import\s+\*",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "normal.py must not use wildcard imports"


# ---------------------------------------------------------------------------
# 2. ``__all__`` completeness
# ---------------------------------------------------------------------------


def test_normal_module_declares_dunder_all() -> None:
    """``__all__`` is declared at module scope."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "normal.py must declare __all__"


def test_normal_module_dunder_all_has_exactly_two_names() -> None:
    """``__all__`` exposes exactly the 2 documented names."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    names = [n.strip().strip("\"'") for n in match.group(1).split(",") if n.strip()]
    assert names == ["NormalContext", "NormalController"], (
        f"Expected __all__ == ['NormalContext', 'NormalController']; got {names!r}"
    )


def test_normal_module_dunder_all_names_are_importable() -> None:
    """Every name in ``__all__`` resolves on the imported module."""

    import oai2.reasoning.normal as normal_module

    for name in ("NormalContext", "NormalController"):
        assert hasattr(normal_module, name), f"{name!r} not exported from normal.py"


def test_normal_module_is_re_exported_at_oai2_reasoning_package_level() -> None:
    """The 2 names are also reachable at the ``oai2.reasoning`` package level."""

    for name in ("NormalContext", "NormalController"):
        assert hasattr(reasoning_package, name), f"{name!r} not re-exported at oai2.reasoning"
        assert getattr(reasoning_package, name) is globals()[name], (
            f"oai2.reasoning.{name} is not identity-equal to oai2.reasoning.normal.{name}"
        )


# ---------------------------------------------------------------------------
# 3. NormalContext (Pydantic BaseModel)
# ---------------------------------------------------------------------------


def test_normal_context_subclasses_pydantic_basemodel() -> None:
    """``NormalContext`` is a Pydantic ``BaseModel`` subclass."""

    from pydantic import BaseModel

    assert issubclass(NormalContext, BaseModel)


def test_normal_context_model_config_forbids_extra_fields() -> None:
    """``NormalContext.model_config["extra"] == "forbid"``."""

    assert NormalContext.model_config["extra"] == "forbid"


def test_normal_context_rejects_unknown_field() -> None:
    """``NormalContext(unknown_field=...)`` raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        NormalContext(unknown_field="bad")  # type: ignore[call-arg]


def test_normal_context_field_set_is_four_names() -> None:
    """``NormalContext`` declares exactly the 4 documented fields."""

    assert set(NormalContext.model_fields.keys()) == {
        "active_experts",
        "speculative",
        "multi_token_prediction",
        "status",
    }, (
        f"Expected fields {{active_experts, speculative, multi_token_prediction, status}}; "
        f"got {set(NormalContext.model_fields.keys())}"
    )


def test_normal_context_default_active_experts_is_eight() -> None:
    """``active_experts`` defaults to ``8`` (the documented default)."""

    ctx = NormalContext()
    assert ctx.active_experts == 8


def test_normal_context_default_speculative_is_true() -> None:
    """``speculative`` defaults to ``True`` (speculative decoding on by default)."""

    ctx = NormalContext()
    assert ctx.speculative is True


def test_normal_context_default_multi_token_prediction_is_true() -> None:
    """``multi_token_prediction`` defaults to ``True`` (MTP on by default)."""

    ctx = NormalContext()
    assert ctx.multi_token_prediction is True


def test_normal_context_default_status_is_proposed() -> None:
    """``status`` defaults to ``Status.PROPOSED``."""

    ctx = NormalContext()
    assert ctx.status is Status.PROPOSED


def test_normal_context_active_experts_minimum_is_one() -> None:
    """``active_experts >= 1`` (ge=1) — 0 raises, 1 is accepted."""

    with pytest.raises(ValidationError):
        NormalContext(active_experts=0)
    ctx = NormalContext(active_experts=1)
    assert ctx.active_experts == 1


def test_normal_context_active_experts_maximum_is_sixty_four() -> None:
    """``active_experts <= 64`` (le=64) — 64 accepted, 65 raises."""

    ctx = NormalContext(active_experts=64)
    assert ctx.active_experts == 64
    with pytest.raises(ValidationError):
        NormalContext(active_experts=65)


def test_normal_context_speculative_accepts_false() -> None:
    """``speculative=False`` is accepted (turns off speculative decoding)."""

    ctx = NormalContext(speculative=False)
    assert ctx.speculative is False


def test_normal_context_multi_token_prediction_accepts_false() -> None:
    """``multi_token_prediction=False`` is accepted (turns off MTP)."""

    ctx = NormalContext(multi_token_prediction=False)
    assert ctx.multi_token_prediction is False


def test_normal_context_status_accepts_each_status_value() -> None:
    """``status`` accepts any of the 4 documented ``Status`` enum values."""

    for status in Status:
        ctx = NormalContext(status=status)
        assert ctx.status is status


def test_normal_context_accepts_custom_active_experts() -> None:
    """``active_experts`` accepts any int in the documented range."""

    ctx = NormalContext(active_experts=16)
    assert ctx.active_experts == 16


def test_normal_context_rejects_non_int_active_experts() -> None:
    """``active_experts`` is `int` — a fractional float is rejected."""

    with pytest.raises(ValidationError):
        NormalContext(active_experts=2.5)  # type: ignore[arg-type]


def test_normal_context_rejects_unknown_status_wire_string() -> None:
    """``status`` must be a documented ``Status`` enum wire string."""

    # Note -- a literal ``"PROPOSED"`` is coerced to ``Status.PROPOSED``
    # because ``Status`` is a ``StrEnum``. A genuinely unknown wire string
    # DOES raise.
    with pytest.raises(ValidationError):
        NormalContext(status="NOT_A_REAL_STATUS")  # type: ignore[arg-type]


def test_normal_context_accepts_status_wire_string_as_input() -> None:
    """``status="PROPOSED"`` is coerced to ``Status.PROPOSED`` (StrEnum contract)."""

    ctx = NormalContext(status="PROPOSED")  # type: ignore[arg-type]
    assert ctx.status is Status.PROPOSED


def test_normal_context_model_dump_round_trip() -> None:
    """``model_dump()`` preserves every field."""

    ctx = NormalContext(
        active_experts=16,
        speculative=False,
        multi_token_prediction=False,
        status=Status.EXPERIMENTAL,
    )
    dumped = ctx.model_dump()
    assert dumped == {
        "active_experts": 16,
        "speculative": False,
        "multi_token_prediction": False,
        "status": Status.EXPERIMENTAL,
    }


# ---------------------------------------------------------------------------
# 4. NormalController (thin static helper)
# ---------------------------------------------------------------------------


def test_normal_controller_is_a_class() -> None:
    """``NormalController`` is a class (not a function or instance)."""

    assert inspect.isclass(NormalController)


def test_normal_controller_status_class_attribute_is_proposed() -> None:
    """``NormalController.STATUS`` is ``Status.PROPOSED``."""

    assert NormalController.STATUS is Status.PROPOSED


def test_normal_controller_choose_decode_strategy_is_staticmethod() -> None:
    """``choose_decode_strategy`` is declared ``@staticmethod`` (no ``self``)."""

    assert isinstance(
        NormalController.__dict__["choose_decode_strategy"],
        staticmethod,
    )


def test_normal_controller_choose_decode_strategy_signature_is_ctx_only() -> None:
    """``choose_decode_strategy`` takes a single ``ctx`` parameter (no ``self``)."""

    sig = inspect.signature(NormalController.choose_decode_strategy)
    params = list(sig.parameters.values())
    assert len(params) == 1
    assert params[0].name == "ctx"


def test_normal_controller_choose_decode_strategy_returns_str() -> None:
    """``choose_decode_strategy`` returns a ``str`` (one of four named strategies)."""

    ctx = NormalContext()
    result = NormalController.choose_decode_strategy(ctx)
    assert isinstance(result, str)


def test_normal_controller_choose_decode_strategy_both_true_returns_speculative_mtp() -> None:
    """``speculative=True`` AND ``multi_token_prediction=True`` → ``speculative_mtp``."""

    ctx = NormalContext(speculative=True, multi_token_prediction=True)
    assert NormalController.choose_decode_strategy(ctx) == "speculative_mtp"


def test_normal_controller_choose_decode_strategy_speculative_only() -> None:
    """``speculative=True`` AND ``multi_token_prediction=False`` → ``speculative``."""

    ctx = NormalContext(speculative=True, multi_token_prediction=False)
    assert NormalController.choose_decode_strategy(ctx) == "speculative"


def test_normal_controller_choose_decode_strategy_mtp_only() -> None:
    """``speculative=False`` AND ``multi_token_prediction=True`` → ``multi_token_prediction``."""

    ctx = NormalContext(speculative=False, multi_token_prediction=True)
    assert NormalController.choose_decode_strategy(ctx) == "multi_token_prediction"


def test_normal_controller_choose_decode_strategy_both_false_returns_vanilla() -> None:
    """``speculative=False`` AND ``multi_token_prediction=False`` → ``vanilla``."""

    ctx = NormalContext(speculative=False, multi_token_prediction=False)
    assert NormalController.choose_decode_strategy(ctx) == "vanilla"


def test_normal_controller_choose_decode_strategy_default_context() -> None:
    """Default context (both flags True) → ``speculative_mtp``."""

    ctx = NormalContext()
    assert NormalController.choose_decode_strategy(ctx) == "speculative_mtp"


def test_normal_controller_choose_decode_strategy_returns_one_of_four_values() -> None:
    """The function returns exactly one of the four documented strategy names."""

    expected = {"speculative_mtp", "speculative", "multi_token_prediction", "vanilla"}
    for spec in (True, False):
        for mtp in (True, False):
            ctx = NormalContext(speculative=spec, multi_token_prediction=mtp)
            result = NormalController.choose_decode_strategy(ctx)
            assert result in expected, f"Unexpected strategy for spec={spec}, mtp={mtp}: {result!r}"


def test_normal_controller_choose_decode_strategy_priority_order() -> None:
    """``speculative_mtp`` wins over ``speculative`` when both flags are True.

    The branch order in the source is:
        1. ``if ctx.speculative and ctx.multi_token_prediction: return "speculative_mtp"``
        2. ``if ctx.speculative: return "speculative"``
        3. ``if ctx.multi_token_prediction: return "multi_token_prediction"``
        4. ``return "vanilla"``
    """

    # Both True → first branch wins (speculative_mtp).
    ctx = NormalContext(speculative=True, multi_token_prediction=True)
    assert NormalController.choose_decode_strategy(ctx) == "speculative_mtp"
    # Only speculative → second branch wins.
    ctx = NormalContext(speculative=True, multi_token_prediction=False)
    assert NormalController.choose_decode_strategy(ctx) == "speculative"


def test_normal_controller_has_no_init() -> None:
    """``NormalController`` has no ``__init__`` declared — it is not meant to be instantiated."""

    assert "__init__" not in NormalController.__dict__


def test_normal_controller_callable_directly_on_class() -> None:
    """``NormalController.choose_decode_strategy(...)`` works without instantiation."""

    ctx = NormalContext()
    result = NormalController.choose_decode_strategy(ctx)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. Public-safety boundary pin
# ---------------------------------------------------------------------------


def test_normal_module_does_not_import_cloud_sdk() -> None:
    """The module must not import any cloud SDK."""

    for needle in ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric"):
        assert needle not in _MODULE_SOURCE, f"normal.py must not reference {needle!r}"


def test_normal_module_has_no_hardcoded_credentials() -> None:
    """No ``api_key=``, no ``BEGIN PRIVATE KEY`` blocks."""

    assert "api_key=" not in _MODULE_SOURCE
    assert "BEGIN PRIVATE KEY" not in _MODULE_SOURCE


def test_normal_module_does_not_print() -> None:
    """No ``print(`` or ``pprint(`` calls."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_normal_module_does_not_use_subprocess() -> None:
    """No ``subprocess`` / ``os.system`` references."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_normal_module_does_not_import_requests_or_httpx() -> None:
    """No direct ``requests`` / ``urllib`` / ``httpx`` / ``aiohttp`` imports."""

    for needle in ("import requests", "urllib", "httpx", "aiohttp"):
        assert needle not in _MODULE_SOURCE, f"normal.py must not reference {needle!r}"


def test_normal_module_does_not_eval_or_exec() -> None:
    """No ``eval(`` or ``exec(`` calls."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_normal_module_does_not_read_environment() -> None:
    """No ``os.environ`` / ``os.getenv`` access."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_normal_module_has_no_todo_or_fixme_markers() -> None:
    """No ``TODO`` / ``FIXME`` / ``XXX`` markers."""

    for needle in ("TODO", "FIXME", "XXX"):
        assert needle not in _MODULE_SOURCE, f"normal.py must not contain {needle!r}"


def test_normal_module_does_not_reexport_pydantic_internals() -> None:
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


def test_normal_module_does_not_reexport_core_internals() -> None:
    """``Status`` is imported for type hints but not re-exported here."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    body = match.group(1)
    assert "Status" not in body, "Status must not be in __all__ of normal.py"


def test_normal_module_pins_one_basemodel_subclass() -> None:
    """Exactly one ``BaseModel`` subclass is declared in the module."""

    matches = re.findall(
        r"^class\s+\w+\s*\(\s*BaseModel\s*\)\s*:",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 BaseModel subclass; got {len(matches)}"


def test_normal_module_pins_one_status_class_constant() -> None:
    """Exactly one class-level ``STATUS`` constant — on ``NormalController``."""

    matches = re.findall(
        r"^\s+STATUS\s*=\s*Status\.",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 STATUS class constant; got {len(matches)}"


def test_normal_module_pin_one_staticmethod_decorator() -> None:
    """Exactly one ``@staticmethod`` decorator — on ``NormalController.choose_decode_strategy``."""

    matches = re.findall(
        r"^\s+@staticmethod",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 @staticmethod; got {len(matches)}"
