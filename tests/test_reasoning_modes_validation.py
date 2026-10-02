"""Structural / validation tests for ``oai2.reasoning.modes``.

Pins the routing surface used by :mod:`oai2.runtime.admission` and the
controller subclasses (FURIOUS / NORMAL / DEEP / SWARM):

- :class:`ReasoningMode` -- the 4-member wire-pinned StrEnum
  taxonomy of active-compute profiles from ``ARCHITECTURE_TARGET.md``.
- :func:`choose_mode` -- the cheap-signal routing heuristic.

Behavioural end-to-end coverage lives in
``tests/test_reasoning.py``; this file pins the *shape* of the API and
the invariants the admission layer relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from oai2.reasoning.modes import ReasoningMode, choose_mode

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "reasoning" / "modes.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_modes_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "reasoning/modes.py is unexpectedly empty"


def test_modes_module_has_docstring() -> None:
    """The module exposes a public-style one-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_modes_module_docstring_mentions_enum_and_routing() -> None:
    """The docstring must reference both the enum and the routing function."""

    assert "enum" in _MODULE_SOURCE.lower()
    assert "routing" in _MODULE_SOURCE.lower() or "choose_mode" in _MODULE_SOURCE


def test_modes_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_modes_module_imports_str_enum() -> None:
    """``ReasoningMode`` is a wire-pinned StrEnum."""

    assert re.search(
        r"^from enum import ([^\n]*,\s*)*StrEnum(\s*,|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "modes module must import StrEnum from enum"


def test_modes_module_does_not_import_cloud_runtime_modules() -> None:
    """The modes module is pure; no cloud-runtime imports."""

    forbidden = ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"modes module must not import cloud runtime module {token!r}"
        )


def test_modes_module_does_not_hardcode_credentials() -> None:
    """No api_key= literals, no BEGIN PRIVATE KEY blocks, no sk-/ghp_ tokens."""

    forbidden_patterns = (
        re.compile(r"api_key\s*=\s*['\"]sk-"),
        re.compile(r"BEGIN PRIVATE KEY"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    )
    for pattern in forbidden_patterns:
        assert not pattern.search(_MODULE_SOURCE), (
            f"modes module must not contain credential marker {pattern.pattern!r}"
        )


def test_modes_module_has_no_print_or_pprint_calls() -> None:
    """Library code must not perform I/O on import."""

    assert not re.search(r"\bprint\s*\(", _MODULE_SOURCE), "modes module must not call print()"
    assert not re.search(r"\bpprint\s*\.", _MODULE_SOURCE), "modes module must not call pprint.*"


def test_modes_module_has_no_subprocess_or_shell_invocation() -> None:
    """No subprocess / os.system / shell=True paths in modes module."""

    assert "import subprocess" not in _MODULE_SOURCE, "modes module must not import subprocess"
    assert "os.system" not in _MODULE_SOURCE, "modes module must not call os.system"
    assert "shell=True" not in _MODULE_SOURCE, "modes module must not pass shell=True"


def test_modes_module_has_no_network_client_imports() -> None:
    """No requests / urllib / httpx imports."""

    forbidden = ("import requests", "from requests ", "import urllib", "import httpx")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, f"modes module must not import network client {token!r}"


def test_modes_module_has_no_eval_or_exec() -> None:
    """No dynamic code execution."""

    assert not re.search(r"^\s*eval\s*\(", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^\s*exec\s*\(", _MODULE_SOURCE, re.MULTILINE)


def test_modes_module_has_no_wildcard_imports() -> None:
    """No ``from X import *``; public surface is pinned by ``__all__``."""

    assert not re.search(r"^from\s+\S+\s+import\s+\*", _MODULE_SOURCE, re.MULTILINE), (
        "modes module must not use wildcard imports"
    )


def test_modes_module_has_no_todo_fixme_xxx_markers() -> None:
    """No TODO / FIXME / XXX markers in shipped source."""

    forbidden = ("TODO", "FIXME", "XXX")
    for token in forbidden:
        assert not re.search(rf"\b{token}\b", _MODULE_SOURCE), (
            f"modes module must not contain {token!r} marker"
        )


def test_modes_module_has_no_os_environ_or_getenv() -> None:
    """The modes module is pure; no environment-variable access."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_modes_module_exposes_one_str_enum_and_one_routing_function() -> None:
    """Exactly one StrEnum subclass and exactly one top-level def with choose_mode."""

    strenum_count = len(
        re.findall(r"^class\s+\w+\s*\(\s*StrEnum\s*\)", _MODULE_SOURCE, re.MULTILINE)
    )
    assert strenum_count == 1, (
        f"modes module must have exactly 1 StrEnum subclass, found {strenum_count}"
    )
    choose_count = len(re.findall(r"^def\s+choose_mode\s*\(", _MODULE_SOURCE, re.MULTILINE))
    assert choose_count == 1, (
        f"modes module must declare choose_mode exactly once, found {choose_count}"
    )


# ---------------------------------------------------------------------------
# 2. __all__ completeness
# ---------------------------------------------------------------------------


def test_modes_module_all_is_exactly_two_names() -> None:
    """``__all__`` must pin exactly 2 names (ReasoningMode + choose_mode)."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None, "modes module must declare __all__"
    body = match.group(1)
    names = [n.strip().strip("'\"") for n in body.split(",") if n.strip()]
    assert len(names) == 2, f"Expected exactly 2 names in __all__, got {len(names)}: {names}"


def test_modes_module_all_names_match_documented_surface() -> None:
    """The 2 names are ReasoningMode + choose_mode."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = {n.strip().strip("'\"") for n in body.split(",") if n.strip()}
    assert names == {"ReasoningMode", "choose_mode"}, f"__all__ set mismatch: {names}"


def test_modes_module_all_names_are_importable() -> None:
    """Each name in __all__ must be importable from oai2.reasoning.modes."""

    from oai2.reasoning import modes as module

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = [n.strip().strip("'\"") for n in body.split(",") if n.strip()]
    for name in names:
        assert hasattr(module, name), f"{name!r} is not exported by the module"


def test_modes_module_symbols_are_re_exported_at_package_level() -> None:
    """The two modes names are also re-exported via oai2.reasoning."""

    from oai2 import reasoning

    assert reasoning.ReasoningMode is ReasoningMode
    assert reasoning.choose_mode is choose_mode


# ---------------------------------------------------------------------------
# 3. ReasoningMode (StrEnum)
# ---------------------------------------------------------------------------


def test_reasoning_mode_enum_value_set_is_pinned() -> None:
    """Each ReasoningMode member has an uppercase wire-pinned string value."""

    expected = {
        ReasoningMode.FURIOUS: "FURIOUS",
        ReasoningMode.NORMAL: "NORMAL",
        ReasoningMode.DEEP: "DEEP",
        ReasoningMode.SWARM: "SWARM",
    }
    actual = {member: member.value for member in ReasoningMode}
    assert actual == expected


def test_reasoning_mode_member_count_is_four() -> None:
    """The taxonomy must contain exactly 4 members (the ARCHITECTURE_TARGET.md
    set: FURIOUS / NORMAL / DEEP / SWARM)."""

    assert len(list(ReasoningMode)) == 4


def test_reasoning_mode_subclasses_str_enum() -> None:
    """ReasoningMode is a StrEnum subclass so members are wire-comparable to str."""

    from enum import StrEnum

    assert issubclass(ReasoningMode, StrEnum)


def test_reasoning_mode_value_round_trips() -> None:
    """A wire string lifts back to the same member."""

    for member in ReasoningMode:
        assert ReasoningMode(member.value) is member


def test_reasoning_mode_member_names_match_wire_values() -> None:
    """The wire value equals the member name (FURIOUS == 'FURIOUS')."""

    for member in ReasoningMode:
        assert member.value == member.name, (
            f"Wire value {member.value!r} != member name {member.name!r}"
        )


def test_reasoning_mode_compares_equal_to_str() -> None:
    """ReasoningMode members are equal to their wire-string value (StrEnum)."""

    assert ReasoningMode.FURIOUS == "FURIOUS"
    assert ReasoningMode.NORMAL == "NORMAL"
    assert ReasoningMode.DEEP == "DEEP"
    assert ReasoningMode.SWARM == "SWARM"


def test_reasoning_mode_unknown_value_raises_value_error() -> None:
    """An unknown wire string raises ValueError (no silent fallback)."""

    with pytest.raises(ValueError):
        ReasoningMode("TURBO")


# ---------------------------------------------------------------------------
# 4. choose_mode routing function
# ---------------------------------------------------------------------------


def test_choose_mode_is_a_regular_function() -> None:
    """``choose_mode`` is a module-level function (not a class / lambda)."""

    import inspect

    assert inspect.isfunction(choose_mode)
    assert inspect.isroutine(choose_mode)


def test_choose_mode_signature_is_keyword_only() -> None:
    """All three parameters are keyword-only (the function is documented to be
    called with named arguments)."""

    import inspect

    sig = inspect.signature(choose_mode)
    params = sig.parameters
    assert set(params.keys()) == {"task_size", "evidence_pressure", "budget_remaining"}
    for name, param in params.items():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"choose_mode.{name} must be keyword-only, got {param.kind}"
        )


def test_choose_mode_has_no_positional_or_var_args() -> None:
    """No positional-only, VAR_POSITIONAL, or VAR_KEYWORD parameters."""

    import inspect

    sig = inspect.signature(choose_mode)
    for name, param in sig.parameters.items():
        assert param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ), f"choose_mode.{name} must not accept positional or *args/**kwargs"


def test_choose_mode_returns_reasoning_mode_instance() -> None:
    """The return type is always one of the 4 ReasoningMode members."""

    result = choose_mode(task_size=2, evidence_pressure=0.1, budget_remaining=1.0)
    assert isinstance(result, ReasoningMode)


# -- threshold table --------------------------------------------------------
#
#   budget_remaining <= 0.0                                       -> FURIOUS
#   evidence_pressure >= 0.85 or task_size >= 16                 -> DEEP
#   task_size >= 8 and budget_remaining >= 0.6                   -> SWARM
#   task_size >= 3 or evidence_pressure >= 0.3                  -> NORMAL
#   otherwise                                                     -> FURIOUS
#


def test_choose_mode_zero_budget_short_circuits_to_furious() -> None:
    """Budget exhausted is last-resort cheap: FURIOUS regardless of pressure."""

    assert (
        choose_mode(task_size=20, evidence_pressure=0.99, budget_remaining=0.0)
        is ReasoningMode.FURIOUS
    )


def test_choose_mode_negative_budget_short_circuits_to_furious() -> None:
    """Negative budget also short-circuits to FURIOUS (budget <= 0.0)."""

    assert (
        choose_mode(task_size=20, evidence_pressure=0.99, budget_remaining=-0.1)
        is ReasoningMode.FURIOUS
    )


def test_choose_mode_high_evidence_pressure_routes_to_deep() -> None:
    """evidence_pressure >= 0.85 → DEEP."""

    assert (
        choose_mode(task_size=2, evidence_pressure=0.85, budget_remaining=1.0) is ReasoningMode.DEEP
    )
    assert (
        choose_mode(task_size=2, evidence_pressure=0.9, budget_remaining=1.0) is ReasoningMode.DEEP
    )


def test_choose_mode_very_large_task_routes_to_deep() -> None:
    """task_size >= 16 → DEEP (regardless of evidence_pressure below 0.85)."""

    assert (
        choose_mode(task_size=16, evidence_pressure=0.1, budget_remaining=1.0) is ReasoningMode.DEEP
    )
    assert (
        choose_mode(task_size=64, evidence_pressure=0.0, budget_remaining=1.0) is ReasoningMode.DEEP
    )


def test_choose_mode_evidence_takes_precedence_over_swarm_for_large_task() -> None:
    """When both DEEP and SWARM could fire, DEEP wins (it appears first in the
    if/elif chain). task_size >= 16 forces DEEP regardless of SWARM conditions."""

    # 0.85 evidence pressure triggers DEEP clause first
    assert (
        choose_mode(task_size=20, evidence_pressure=0.9, budget_remaining=0.95)
        is ReasoningMode.DEEP
    )


def test_choose_mode_swarm_routing_requires_budget_and_large_task() -> None:
    """SWARM needs task_size >= 8 AND budget_remaining >= 0.6."""

    assert (
        choose_mode(task_size=8, evidence_pressure=0.5, budget_remaining=0.6) is ReasoningMode.SWARM
    )
    assert (
        choose_mode(task_size=10, evidence_pressure=0.5, budget_remaining=0.9)
        is ReasoningMode.SWARM
    )


def test_choose_mode_swarm_not_picked_when_budget_below_threshold() -> None:
    """budget_remaining < 0.6 with task_size in [8, 15] skips SWARM."""

    assert (
        choose_mode(task_size=8, evidence_pressure=0.5, budget_remaining=0.59)
        is not ReasoningMode.SWARM
    )


def test_choose_mode_swarm_not_picked_when_task_below_threshold() -> None:
    """task_size < 8 skips SWARM even with high budget."""

    assert (
        choose_mode(task_size=7, evidence_pressure=0.5, budget_remaining=0.95)
        is not ReasoningMode.SWARM
    )


def test_choose_mode_normal_routing_for_mid_task_or_mid_pressure() -> None:
    """task_size >= 3 OR evidence_pressure >= 0.3 → NORMAL."""

    assert (
        choose_mode(task_size=3, evidence_pressure=0.1, budget_remaining=1.0)
        is ReasoningMode.NORMAL
    )
    assert (
        choose_mode(task_size=1, evidence_pressure=0.3, budget_remaining=1.0)
        is ReasoningMode.NORMAL
    )


def test_choose_mode_normal_routing_when_swarm_falls_through() -> None:
    """task_size=8 with budget < 0.6 skips SWARM but meets NORMAL (task_size >= 3)."""

    assert (
        choose_mode(task_size=8, evidence_pressure=0.1, budget_remaining=0.3)
        is ReasoningMode.NORMAL
    )


def test_choose_mode_furious_for_easy_routing() -> None:
    """Default (easy) routing: small task, low pressure, plenty of budget → FURIOUS."""

    assert (
        choose_mode(task_size=0, evidence_pressure=0.0, budget_remaining=1.0)
        is ReasoningMode.FURIOUS
    )
    assert (
        choose_mode(task_size=2, evidence_pressure=0.1, budget_remaining=1.0)
        is ReasoningMode.FURIOUS
    )


def test_choose_mode_furious_at_low_task_boundary() -> None:
    """task_size < 3 with evidence_pressure < 0.3 → FURIOUS."""

    assert (
        choose_mode(task_size=2, evidence_pressure=0.29, budget_remaining=1.0)
        is ReasoningMode.FURIOUS
    )


def test_choose_mode_deterministic_for_fixed_inputs() -> None:
    """Same inputs always produce the same mode (no randomness)."""

    a = choose_mode(task_size=5, evidence_pressure=0.5, budget_remaining=0.7)
    b = choose_mode(task_size=5, evidence_pressure=0.5, budget_remaining=0.7)
    assert a is b
    assert a is ReasoningMode.NORMAL


@pytest.mark.parametrize(
    "task_size,evidence_pressure,budget_remaining,expected",
    [
        # Zero / negative budget short-circuits to FURIOUS regardless of size.
        (0, 0.0, 0.0, ReasoningMode.FURIOUS),
        (10, 0.9, -0.1, ReasoningMode.FURIOUS),
        (50, 0.99, 0.0, ReasoningMode.FURIOUS),
        # High evidence pressure → DEEP.
        (0, 0.85, 1.0, ReasoningMode.DEEP),
        (1, 0.95, 1.0, ReasoningMode.DEEP),
        # Large task size → DEEP.
        (16, 0.0, 1.0, ReasoningMode.DEEP),
        (100, 0.0, 1.0, ReasoningMode.DEEP),
        # SWARM needs task >= 8 AND budget >= 0.6 (but task < 16, pressure < 0.85).
        (8, 0.5, 0.6, ReasoningMode.SWARM),
        (15, 0.5, 1.0, ReasoningMode.SWARM),
        (12, 0.84, 0.99, ReasoningMode.SWARM),
        # NORMAL needs task >= 3 OR pressure >= 0.3 (and not meeting above).
        (3, 0.0, 1.0, ReasoningMode.NORMAL),
        (0, 0.3, 1.0, ReasoningMode.NORMAL),
        (7, 0.5, 0.3, ReasoningMode.NORMAL),  # SWARM would need budget >= 0.6
        # Default easy routing → FURIOUS.
        (0, 0.0, 1.0, ReasoningMode.FURIOUS),
        (2, 0.29, 1.0, ReasoningMode.FURIOUS),
        (1, 0.1, 0.5, ReasoningMode.FURIOUS),
    ],
)
def test_choose_mode_threshold_table(
    task_size: int,
    evidence_pressure: float,
    budget_remaining: float,
    expected: ReasoningMode,
) -> None:
    """Sweep the routing threshold table: every documented branch is exercised."""

    actual = choose_mode(
        task_size=task_size,
        evidence_pressure=evidence_pressure,
        budget_remaining=budget_remaining,
    )
    assert actual is expected


# ---------------------------------------------------------------------------
# 5. Public-safety (boundary) pin
# ---------------------------------------------------------------------------


def test_modes_module_choose_mode_signature_returns_reasoning_mode() -> None:
    """Return-type annotation (string under PEP 563) names ReasoningMode."""

    import inspect

    sig = inspect.signature(choose_mode)
    # Under from __future__ import annotations, returns is a ForwardRef string.
    assert sig.return_annotation in {"ReasoningMode", inspect.Signature.empty} or (
        "ReasoningMode" in str(sig.return_annotation)
    ), f"choose_mode must return ReasoningMode, got {sig.return_annotation!r}"


def test_modes_module_pin_reasoning_mode_member_count() -> None:
    """The source pins exactly 4 ReasoningMode member declarations."""

    member_lines = re.findall(
        r"^\s+(FURIOUS|NORMAL|DEEP|SWARM)\s*=\s*['\"]\1['\"]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(member_lines) == 4, (
        f"modes module must declare exactly 4 ReasoningMode members, found {len(member_lines)}"
    )


def test_modes_module_no_typing_collections_at_runtime() -> None:
    """The modes module is library code; no collections usage at runtime."""

    assert "from collections" not in _MODULE_SOURCE
    assert "import collections" not in _MODULE_SOURCE


def test_modes_module_no_third_party_imports() -> None:
    """The modes module imports nothing from third-party libraries."""

    # Only stdlib (enum) + __future__ should appear at import-time.
    assert "import pydantic" not in _MODULE_SOURCE
    assert "import numpy" not in _MODULE_SOURCE
    assert "import requests" not in _MODULE_SOURCE
    assert "import httpx" not in _MODULE_SOURCE
