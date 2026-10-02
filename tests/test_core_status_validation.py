"""Structural / validation tests for ``oai2.core.status``.

Pins the top-level package-status pin module:

- :data:`PACKAGE_STATUS` -- the canonical lifecycle marker that every
  subsystem reports against. Today this is ``Status.PROPOSED``; future
  slices may bump it to ``EXPERIMENTAL`` / ``IMPLEMENTED`` as subsystems
  move out of design-only territory.

The :class:`Status` enum itself is re-exported from :mod:`oai2.core` and
its full wire-string contract is pinned in this file too -- it is the
enum every other ``STATUS`` constant and ``status=`` field in the
codebase uses, so a structural break here ripples everywhere.

Behavioural / integration coverage lives in
``tests/test_core_full_package_identity.py``; this file pins the *shape*
of the API and the invariants the live ORCHORDS runtime checks at
import time.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

import pytest

import oai2.core as core_package
from oai2.core import Status
from oai2.core import status as status_module

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "core" / "status.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_status_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "core/status.py is unexpectedly empty"


def test_status_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_status_module_docstring_mentions_status_marker() -> None:
    """The docstring names the role of the module (status marker)."""

    lowered = _MODULE_SOURCE.lower()
    assert "status" in lowered, "core/status.py docstring must mention 'status'"


def test_status_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_status_module_imports_status_from_c_relative_to_oai2_core() -> None:
    """``Status`` is imported relatively from the parent package."""

    match = re.search(
        r"^from\s+\.\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "core/status.py must import Status relatively from '.'"
    body = match.group(1)
    assert "Status" in body, f"Relative import from '.' must include Status (got: {body!r})"


def test_status_module_does_not_import_oai2_package() -> None:
    """No absolute ``import oai2`` / ``from oai2`` at line-start."""

    for pattern in (
        r"^import oai2\b",
        r"^from oai2\b",
    ):
        assert not re.search(pattern, _MODULE_SOURCE, re.MULTILINE), (
            f"core/status.py must not use {pattern!r}"
        )


def test_status_module_does_not_use_wildcard_imports() -> None:
    """No ``from ... import *`` statements."""

    assert not re.search(
        r"^from\s+\S+\s+import\s+\*",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "core/status.py must not use wildcard imports"


def test_status_module_does_not_import_submodules_or_helper_modules() -> None:
    """The module imports stdlib typing/IO helpers only if at all; no helpers."""

    # Only `from __future__ import annotations` and `from . import Status`
    # are expected. Catch any third import statement by matching the whole
    # clause (from/import … [as …]).
    import_lines = re.findall(
        r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+)(?:\s+as\s+\S+)?",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    # Filter out the `from __future__ import annotations` line.
    non_future = [s for s in import_lines if "__future__" not in s]
    assert len(non_future) == 1, f"Expected exactly 1 non-future import; got {non_future!r}"
    assert non_future[0].startswith("from . import"), (
        f"Only relative `from . import Status` is permitted (got: {non_future[0]!r})"
    )


# ---------------------------------------------------------------------------
# 2. ``__all__`` completeness
# ---------------------------------------------------------------------------


def test_status_module_declares_dunder_all() -> None:
    """``__all__`` is declared at module scope."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "core/status.py must declare __all__"


def test_status_module_dunder_all_has_exactly_one_name() -> None:
    """``__all__`` exposes exactly one public name."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    body = match.group(1)
    names = [n.strip().strip("\"'") for n in body.split(",") if n.strip()]
    assert names == ["PACKAGE_STATUS"], f"Expected __all__ == ['PACKAGE_STATUS']; got {names!r}"


def test_status_module_dunder_all_names_are_importable() -> None:
    """Every name in ``__all__`` resolves on the imported module object."""

    assert hasattr(status_module, "PACKAGE_STATUS")
    # Sanity, isinstance the import to bound this test to the documented symbol.
    assert isinstance(status_module.PACKAGE_STATUS, Status)


def test_status_module_package_status_is_not_re_exported_at_package_level() -> None:
    """``PACKAGE_STATUS`` lives in ``oai2.core.status``, NOT in ``oai2.core``.

    The module declares ``__all__ = ['PACKAGE_STATUS']`` but
    :mod:`oai2.core.__init__` does not import it -- so consumers must use
    ``from oai2.core.status import PACKAGE_STATUS`` explicitly. Pinning
    this structural choice prevents future contributors from accidentally
    adding a second ``packaging_status``-shaped name to the package surface.
    """

    assert not hasattr(core_package, "PACKAGE_STATUS"), (
        "PACKAGE_STATUS must not be re-exported at the oai2.core package level; "
        "consumers import it from oai2.core.status directly"
    )


def test_status_module_package_status_is_importable_via_oai2_core_status() -> None:
    """``PACKAGE_STATUS`` is reachable at ``oai2.core.status.PACKAGE_STATUS``."""

    assert core_package.status.PACKAGE_STATUS is status_module.PACKAGE_STATUS


# ---------------------------------------------------------------------------
# 3. Status enum (4 members + wire strings + StrEnum)
# ---------------------------------------------------------------------------


def test_status_is_a_str_enum() -> None:
    """``Status`` subclasses both :class:`str` and :class:`enum.StrEnum`."""

    assert issubclass(Status, StrEnum)
    assert issubclass(Status, str)


def test_status_has_exactly_four_members() -> None:
    """``Status`` exposes exactly the 4 documented lifecycle states."""

    members = list(Status)
    assert len(members) == 4, f"Expected 4 Status members; got {len(members)}"


def test_status_member_wire_strings_are_uppercase_pinned() -> None:
    """Every ``Status`` member has the documented uppercase wire string."""

    expected_wire = {
        "IMPLEMENTED": "IMPLEMENTED",
        "EXPERIMENTAL": "EXPERIMENTAL",
        "PROPOSED": "PROPOSED",
        "BLOCKED": "BLOCKED",
    }
    actual_wire = {member.name: member.value for member in Status}
    assert actual_wire == expected_wire, (
        f"Status wire-string drift: expected {expected_wire!r}; got {actual_wire!r}"
    )


def test_status_member_names_are_unique() -> None:
    """Every ``Status`` member name is unique (no shadow members)."""

    names = [member.name for member in Status]
    assert len(names) == len(set(names)), f"Duplicate Status member names: {names!r}"


def test_status_member_wire_strings_are_unique() -> None:
    """Every ``Status`` wire-string value is unique (no alias members)."""

    values = [member.value for member in Status]
    assert len(values) == len(set(values)), f"Duplicate Status wire values: {values!r}"


def test_status_resolves_known_wire_string_to_member() -> None:
    """``Status(wire)`` resolves each documented wire string to its member."""

    for wire in ("IMPLEMENTED", "EXPERIMENTAL", "PROPOSED", "BLOCKED"):
        assert Status(wire) is getattr(Status, wire), (
            f"Status({wire!r}) did not resolve to its member"
        )


def test_status_unknown_wire_string_raises_value_error() -> None:
    """Unknown wire strings raise ``ValueError`` (no silent fallback)."""

    with pytest.raises(ValueError):
        Status("not-a-real-status")


def test_status_str_returns_wire_string() -> None:
    """``str(member)`` returns the documented wire string."""

    for member in Status:
        assert str(member) == member.value


def test_status_member_equals_wire_string() -> None:
    """``Status.<X> == "<X>"`` (StrEnum equality with str)."""

    for member in Status:
        assert member == member.value
        assert member == member.name


# ---------------------------------------------------------------------------
# 4. PACKAGE_STATUS pin
# ---------------------------------------------------------------------------


def test_package_status_is_a_status_member() -> None:
    """``PACKAGE_STATUS`` is one of the 4 ``Status`` members."""

    assert isinstance(status_module.PACKAGE_STATUS, Status)


def test_package_status_value_is_currently_proposed() -> None:
    """The current package status is ``Status.PROPOSED``."""

    assert status_module.PACKAGE_STATUS is Status.PROPOSED


def test_package_status_wire_string_is_proposed() -> None:
    """The wire-string value of ``PACKAGE_STATUS`` is ``"PROPOSED"``."""

    assert status_module.PACKAGE_STATUS.value == "PROPOSED"


def test_package_status_is_module_level_constant() -> None:
    """``PACKAGE_STATUS`` is declared at module scope (not inside a function/class)."""

    # The source must contain a top-level `PACKAGE_STATUS = Status.PROPOSED`
    # assignment (column 0, not nested under a class or function).
    assert re.search(
        r"^PACKAGE_STATUS\s*=\s*Status\.PROPOSED",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "core/status.py must declare `PACKAGE_STATUS = Status.PROPOSED` at module scope"


def test_package_status_declared_exactly_once() -> None:
    """``PACKAGE_STATUS = ...`` appears exactly once in the source."""

    matches = re.findall(
        r"^PACKAGE_STATUS\s*=",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert len(matches) == 1, f"Expected exactly 1 PACKAGE_STATUS declaration; got {len(matches)}"


# ---------------------------------------------------------------------------
# 5. Public-safety boundary pin
# ---------------------------------------------------------------------------


def test_status_module_does_not_import_cloud_sdk() -> None:
    """The module must not import any cloud SDK (boto3 / azure / google.cloud)."""

    for needle in ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric"):
        assert needle not in _MODULE_SOURCE, f"core/status.py must not reference {needle!r}"


def test_status_module_has_no_hardcoded_credentials() -> None:
    """No ``api_key=``, no ``BEGIN PRIVATE KEY`` blocks."""

    assert "api_key=" not in _MODULE_SOURCE
    assert "BEGIN PRIVATE KEY" not in _MODULE_SOURCE


def test_status_module_does_not_print() -> None:
    """No ``print(`` or ``pprint(`` calls (status pin must be silent)."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_status_module_does_not_use_subprocess() -> None:
    """No ``subprocess`` / ``os.system`` references."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_status_module_does_not_import_requests_or_httpx() -> None:
    """No direct ``requests`` / ``urllib`` / ``httpx`` / ``aiohttp`` imports."""

    for needle in ("import requests", "urllib", "httpx", "aiohttp"):
        assert needle not in _MODULE_SOURCE, f"core/status.py must not reference {needle!r}"


def test_status_module_does_not_eval_or_exec() -> None:
    """No ``eval(`` or ``exec(`` calls (the pin must not execute dynamic code)."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_status_module_does_not_read_environment() -> None:
    """No ``os.environ`` / ``os.getenv`` access (the pin must be deterministic)."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_status_module_has_no_todo_or_fixme_markers() -> None:
    """No ``TODO`` / ``FIXME`` / ``XXX`` markers in the source."""

    for needle in ("TODO", "FIXME", "XXX"):
        assert needle not in _MODULE_SOURCE, f"core/status.py must not contain {needle!r}"


def test_status_module_does_not_reexport_status_enum() -> None:
    """The ``Status`` enum is re-exported from ``oai2.core``, not from this module."""

    match = re.search(
        r"^__all__\s*=\s*\[([^\]]+)\]",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None
    body = match.group(1)
    assert "Status" not in body, (
        "Status enum must not be in __all__ of core/status.py (it is in __all__ of core/__init__.py)"
    )


def test_status_module_declares_no_classes_or_functions() -> None:
    """The module is intentionally minimal: only a constant + __all__."""

    class_count = len(re.findall(r"^class\s+\w+", _MODULE_SOURCE, re.MULTILINE))
    func_count = len(re.findall(r"^(?:async\s+)?def\s+\w+\s*\(", _MODULE_SOURCE, re.MULTILINE))
    assert class_count == 0, f"Expected 0 classes; got {class_count}"
    assert func_count == 0, f"Expected 0 functions; got {func_count}"


def test_status_module_declares_no_dataclass_or_pydantic_model() -> None:
    """The module declares no ``@dataclass`` or Pydantic ``BaseModel`` subclasses."""

    assert "@dataclass" not in _MODULE_SOURCE
    assert "BaseModel" not in _MODULE_SOURCE
    assert "pydantic" not in _MODULE_SOURCE


def test_status_module_pins_one_module_level_constant() -> None:
    """Exactly one module-level constant is declared (``PACKAGE_STATUS``)."""

    # A module-level constant is `^[A-Z_][A-Z0-9_]*\s*=` at column 0.
    # PACKAGE_STATUS counts; ``__all__`` does not (no `=`).
    const_matches = re.findall(
        r"^([A-Z_][A-Z0-9_]*)\s*=",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert const_matches == ["PACKAGE_STATUS"], (
        f"Expected exactly one module-level constant named PACKAGE_STATUS; got {const_matches!r}"
    )


def test_status_module_is_importable_via_attribute_access() -> None:
    """``status_module.PACKAGE_STATUS`` is reachable via attribute access."""

    attr = status_module.PACKAGE_STATUS
    assert attr is Status.PROPOSED


def test_status_module_status_member_is_singleton_in_process_catalogue() -> None:
    """``Status.PROPOSED`` in this test is the same enum member across imports."""

    # Defensive: re-import the enum and confirm the singleton property holds
    # even when accessed via the relative-import path used in the module.
    from oai2.core import Status as StatusReloaded

    assert StatusReloaded is Status
    assert StatusReloaded.PROPOSED is Status.PROPOSED


def test_status_module_top_level_package_attribute_is_proposed() -> None:
    """``oai2.core.status.PACKAGE_STATUS`` is ``Status.PROPOSED`` (the canonical pin)."""

    assert core_package.status.PACKAGE_STATUS is Status.PROPOSED
