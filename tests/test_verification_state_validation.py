"""Structural / validation tests for ``oai2.verification.state``.

Pin the claim-state machine's public surface: the
:class:`ClaimStatus` and :class:`SupportNeed` :class:`enum.StrEnum`
wire strings, the :class:`ClaimState` Pydantic model constraints
(``extra="forbid"``, ``text`` length, ``attempts`` non-negative), and
the :class:`VerificationContext` mutation methods (``add``,
``update``, ``open``).

Behavioural end-to-end coverage lives in
``tests/test_verification.py``; this file pins the *shape* of the API
and the invariants the verification engine relies on.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import ValidationError

from oai2.core import ClaimId
from oai2.verification import (
    ClaimState,
    ClaimStatus,
    SupportNeed,
    VerificationContext,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "verification" / "state.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Module docstring + import surface
# ---------------------------------------------------------------------------


def test_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "verification/state.py is unexpectedly empty"


def test_module_has_docstring() -> None:
    """The module ships an overview mentioning claim state + verification."""

    assert _MODULE_SOURCE.startswith('"""')
    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert "ClaimState" in first_para
    assert "verification" in first_para.lower()


def test_module_uses_future_annotations() -> None:
    """``from __future__ import annotations`` is present (UP006-clean)."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_module_imports_use_relative_from_imports() -> None:
    """No wildcard imports; ``ClaimId`` is imported relatively from ``..core``."""

    assert "import *" not in _MODULE_SOURCE
    assert "from ..core import ClaimId" in _MODULE_SOURCE
    # No accidental absolute import of core
    assert not re.search(r"^import oai2\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.core\b", _MODULE_SOURCE, re.MULTILINE)


# ---------------------------------------------------------------------------
# __all__ completeness + package re-export
# ---------------------------------------------------------------------------


def test_dunder_all_lists_exactly_four_public_names() -> None:
    """The module's public surface is exactly 4 names, no more, no less."""

    import oai2.verification.state as mod

    assert isinstance(mod.__all__, list)
    assert set(mod.__all__) == {
        "ClaimState",
        "ClaimStatus",
        "SupportNeed",
        "VerificationContext",
    }
    assert len(mod.__all__) == 4


def test_each_all_name_is_importable_from_module() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    import oai2.verification.state as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ name missing: {name}"


def test_public_names_are_reexported_from_package() -> None:
    """Top-level ``oai2.verification`` re-exports the four public names."""

    import oai2.verification as pkg

    for name in (
        "ClaimState",
        "ClaimStatus",
        "SupportNeed",
        "VerificationContext",
    ):
        assert name in pkg.__all__, f"package re-export missing: {name}"


def test_package_re_export_preserves_identity() -> None:
    """Package re-exports point at the *same* class objects as the module."""

    import oai2.verification as pkg
    import oai2.verification.state as mod

    assert pkg.ClaimState is mod.ClaimState
    assert pkg.ClaimStatus is mod.ClaimStatus
    assert pkg.SupportNeed is mod.SupportNeed
    assert pkg.VerificationContext is mod.VerificationContext


# ---------------------------------------------------------------------------
# ClaimStatus StrEnum
# ---------------------------------------------------------------------------


def test_claim_status_is_a_strenum() -> None:
    """``ClaimStatus`` subclasses :class:`enum.StrEnum` so wire strings
    round-trip through JSON without a custom serializer."""

    assert issubclass(ClaimStatus, StrEnum)
    assert issubclass(ClaimStatus, str)


def test_claim_status_members_and_wire_strings() -> None:
    """``ClaimStatus`` has exactly 4 members with the documented wire values."""

    expected = {
        "SUPPORTED": "supported",
        "UNVERIFIED": "unverified",
        "CONFLICTING": "conflicting",
        "BLOCKED": "blocked",
    }
    actual = {member.name: str(member.value) for member in ClaimStatus}
    assert actual == expected
    assert len(ClaimStatus) == 4


def test_claim_status_member_names_are_unique() -> None:
    """No duplicate member names — ``ClaimStatus`` is a valid enum."""

    names = [member.name for member in ClaimStatus]
    assert len(names) == len(set(names))


def test_claim_status_wire_strings_are_unique() -> None:
    """No duplicate wire-string values — the enum is a true bijection."""

    values = [str(member.value) for member in ClaimStatus]
    assert len(values) == len(set(values))


def test_claim_status_lookup_by_wire_string() -> None:
    """``ClaimStatus("supported")`` resolves to ``SUPPORTED``."""

    assert ClaimStatus("supported") is ClaimStatus.SUPPORTED
    assert ClaimStatus("blocked") is ClaimStatus.BLOCKED


def test_claim_status_rejects_unknown_wire_string() -> None:
    """An unknown wire string raises ``ValueError`` — no silent fallback."""

    with pytest.raises(ValueError):
        ClaimStatus("not-a-real-status")


# ---------------------------------------------------------------------------
# SupportNeed StrEnum
# ---------------------------------------------------------------------------


def test_support_need_is_a_strenum() -> None:
    """``SupportNeed`` subclasses :class:`enum.StrEnum`."""

    assert issubclass(SupportNeed, StrEnum)
    assert issubclass(SupportNeed, str)


def test_support_need_members_and_wire_strings() -> None:
    """``SupportNeed`` has exactly 6 members with the documented wire values."""

    expected = {
        "SOURCE_FACT": "source_fact",
        "EXTERNAL_FACT": "external_fact",
        "RUNTIME": "runtime_behavior",
        "VISUAL": "visual_behavior",
        "CODE": "code_correctness",
        "NONE": "none",
    }
    actual = {member.name: str(member.value) for member in SupportNeed}
    assert actual == expected
    assert len(SupportNeed) == 6


def test_support_need_member_names_are_unique() -> None:
    """No duplicate member names — ``SupportNeed`` is a valid enum."""

    names = [member.name for member in SupportNeed]
    assert len(names) == len(set(names))


def test_support_need_wire_strings_are_unique() -> None:
    """No duplicate wire-string values — the enum is a true bijection."""

    values = [str(member.value) for member in SupportNeed]
    assert len(values) == len(set(values))


def test_support_need_none_is_the_default() -> None:
    """``SupportNeed.NONE`` carries the wire value ``"none"``."""

    assert SupportNeed.NONE.value == "none"
    assert str(SupportNeed.NONE) == "none"


# ---------------------------------------------------------------------------
# ClaimState Pydantic model
# ---------------------------------------------------------------------------


def test_claim_state_is_a_pydantic_basemodel() -> None:
    """``ClaimState`` subclasses :class:`pydantic.BaseModel`."""

    from pydantic import BaseModel

    assert issubclass(ClaimState, BaseModel)


def test_claim_state_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ClaimState(id=ClaimId("c1"), text="x", unknown_field="oops")  # type: ignore[call-arg]


def test_claim_state_minimal_construction() -> None:
    """Only ``id`` and ``text`` are required; everything else has one."""

    claim = ClaimState(id=ClaimId("c1"), text="hello")
    assert claim.id == "c1"
    assert claim.text == "hello"
    assert claim.support_need is SupportNeed.NONE
    assert claim.status is ClaimStatus.UNVERIFIED
    assert claim.attempts == 0
    assert claim.last_error is None


def test_claim_state_id_accepts_arbitrary_string() -> None:
    """``ClaimId`` is ``NewType("ClaimId", str)`` — runtime is plain str."""

    claim = ClaimState(id="any-string-at-all", text="x")  # type: ignore[arg-type]
    assert claim.id == "any-string-at-all"


def test_claim_state_text_rejects_empty_string() -> None:
    """``min_length=1`` — empty text raises ``ValidationError``."""

    with pytest.raises(ValidationError):
        ClaimState(id=ClaimId("c1"), text="")


def test_claim_state_text_rejects_overlong_string() -> None:
    """``max_length=4096`` — text longer than 4096 chars raises
    ``ValidationError``."""

    too_long = "x" * 4097
    with pytest.raises(ValidationError):
        ClaimState(id=ClaimId("c1"), text=too_long)


def test_claim_state_text_accepts_max_length_boundary() -> None:
    """A text of exactly 4096 chars is accepted (boundary inclusive)."""

    boundary = "x" * 4096
    claim = ClaimState(id=ClaimId("c1"), text=boundary)
    assert len(claim.text) == 4096


def test_claim_state_attempts_rejects_negative() -> None:
    """``attempts`` uses ``ge=0`` — negatives raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        ClaimState(id=ClaimId("c1"), text="x", attempts=-1)


def test_claim_state_attempts_accepts_zero() -> None:
    """``attempts=0`` is the documented default and boundary."""

    claim = ClaimState(id=ClaimId("c1"), text="x", attempts=0)
    assert claim.attempts == 0


def test_claim_state_last_error_accepts_none() -> None:
    """``last_error`` is ``str | None`` — ``None`` is accepted."""

    claim = ClaimState(id=ClaimId("c1"), text="x", last_error=None)
    assert claim.last_error is None


def test_claim_state_last_error_accepts_string() -> None:
    """``last_error`` accepts a non-empty string."""

    claim = ClaimState(id=ClaimId("c1"), text="x", last_error="timeout")
    assert claim.last_error == "timeout"


def test_claim_state_model_dump_round_trip() -> None:
    """``model_dump`` then ``model_validate`` round-trips losslessly."""

    original = ClaimState(
        id=ClaimId("c1"),
        text="hello",
        support_need=SupportNeed.SOURCE_FACT,
        status=ClaimStatus.SUPPORTED,
        attempts=3,
        last_error="boom",
    )
    dumped = original.model_dump()
    restored = ClaimState.model_validate(dumped)
    assert restored == original
    assert restored.id == original.id
    assert restored.text == original.text
    assert restored.support_need is original.support_need
    assert restored.status is original.status
    assert restored.attempts == original.attempts
    assert restored.last_error == original.last_error


# ---------------------------------------------------------------------------
# VerificationContext Pydantic model
# ---------------------------------------------------------------------------


def test_verification_context_is_a_pydantic_basemodel() -> None:
    """``VerificationContext`` subclasses :class:`pydantic.BaseModel`."""

    from pydantic import BaseModel

    assert issubclass(VerificationContext, BaseModel)


def test_verification_context_rejects_extra_fields() -> None:
    """``extra="forbid"`` — unknown fields raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        VerificationContext(unknown_field="oops")  # type: ignore[call-arg]


def test_verification_context_default_claims_is_empty_dict() -> None:
    """A fresh ``VerificationContext`` has no claims."""

    ctx = VerificationContext()
    assert ctx.claims == {}


def test_verification_context_default_factory_yields_independent_dicts() -> None:
    """Two fresh contexts must not share the same dict instance."""

    a = VerificationContext()
    b = VerificationContext()
    assert a.claims is not b.claims
    a.claims["k"] = ClaimState(id=ClaimId("k"), text="x")
    assert "k" not in b.claims


def test_verification_context_add_inserts_by_claim_id() -> None:
    """``add`` keys the inserted claim by ``claim.id``."""

    ctx = VerificationContext()
    claim = ClaimState(id=ClaimId("c1"), text="hello")
    ctx.add(claim)
    assert "c1" in ctx.claims
    assert ctx.claims["c1"] is claim


def test_verification_context_add_overwrites_existing_id() -> None:
    """Adding a second claim with the same id overwrites the prior entry."""

    ctx = VerificationContext()
    first = ClaimState(id=ClaimId("c1"), text="first")
    second = ClaimState(id=ClaimId("c1"), text="second")
    ctx.add(first)
    ctx.add(second)
    assert ctx.claims["c1"] is second
    assert len(ctx.claims) == 1


def test_verification_context_update_raises_keyerror_for_unknown_id() -> None:
    """``update`` raises ``KeyError`` when the claim id is unknown."""

    ctx = VerificationContext()
    with pytest.raises(KeyError):
        ctx.update("ghost", status=ClaimStatus.SUPPORTED)


def test_verification_context_update_replaces_with_model_copy() -> None:
    """``update`` replaces the entry with a copy that includes the changes."""

    ctx = VerificationContext()
    original = ClaimState(id=ClaimId("c1"), text="hello", attempts=0)
    ctx.add(original)
    ctx.update("c1", status=ClaimStatus.SUPPORTED, attempts=2)
    updated = ctx.claims["c1"]
    assert updated is not original
    assert updated.status is ClaimStatus.SUPPORTED
    assert updated.attempts == 2
    assert updated.text == "hello"  # unchanged
    assert updated.id == "c1"  # unchanged


def test_verification_context_update_does_not_revalidate_unknown_field() -> None:
    """``update`` does NOT re-validate ``extra="forbid"`` — ``model_copy``
    applies the update dict verbatim.

    Pydantic v2 ``model_copy(update=...)`` skips schema validation, so an
    unknown keyword in ``update`` lands on the model's ``__dict__`` and
    is silently attached. The schema-level ``extra="forbid"`` is only
    enforced at construction time. This test pins that semantic so any
    change to the source behaviour (e.g. switching to ``model_validate``
    or adding an explicit field filter) shows up here.
    """

    ctx = VerificationContext()
    ctx.add(ClaimState(id=ClaimId("c1"), text="x"))
    ctx.update("c1", bogus_field="oops")  # does NOT raise
    updated = ctx.claims["c1"]
    # The unknown field is attached to the instance dict but not exposed
    # in the schema-defined surface (``model_fields``).
    assert updated.__dict__.get("bogus_field") == "oops"
    assert "bogus_field" not in type(updated).model_fields


def test_verification_context_update_can_clear_last_error() -> None:
    """``update`` can reset ``last_error`` back to ``None``."""

    ctx = VerificationContext()
    ctx.add(ClaimState(id=ClaimId("c1"), text="x", last_error="boom"))
    ctx.update("c1", last_error=None)
    assert ctx.claims["c1"].last_error is None


def test_verification_context_open_returns_unverified_claims() -> None:
    """``open`` returns only claims with ``status is UNVERIFIED``."""

    ctx = VerificationContext()
    ctx.add(
        ClaimState(
            id=ClaimId("c_open"),
            text="x",
            support_need=SupportNeed.SOURCE_FACT,
            status=ClaimStatus.UNVERIFIED,
        )
    )
    ctx.add(
        ClaimState(
            id=ClaimId("c_done"),
            text="y",
            support_need=SupportNeed.SOURCE_FACT,
            status=ClaimStatus.SUPPORTED,
        )
    )
    open_ids = [c.id for c in ctx.open()]
    assert open_ids == ["c_open"]


def test_verification_context_open_excludes_non_unverified_statuses() -> None:
    """``open`` filters out SUPPORTED, CONFLICTING, and BLOCKED."""

    ctx = VerificationContext()
    for status, claim_id in [
        (ClaimStatus.SUPPORTED, "c_supported"),
        (ClaimStatus.CONFLICTING, "c_conflicting"),
        (ClaimStatus.BLOCKED, "c_blocked"),
    ]:
        ctx.add(ClaimState(id=ClaimId(claim_id), text="x", status=status))
    assert ctx.open() == []


def test_verification_context_open_returns_empty_when_no_claims() -> None:
    """An empty context returns ``[]``."""

    ctx = VerificationContext()
    assert ctx.open() == []


def test_verification_context_open_returns_list_not_iterator() -> None:
    """``open`` returns a ``list`` (concrete), so callers can iterate twice."""

    ctx = VerificationContext()
    ctx.add(ClaimState(id=ClaimId("c1"), text="x"))
    result = ctx.open()
    assert isinstance(result, list)
    # Iteration twice does not exhaust the result
    assert len(result) == len(result)


# ---------------------------------------------------------------------------
# Module source — public-safety boundary
# ---------------------------------------------------------------------------


def test_module_source_has_no_cloud_sdk_reference() -> None:
    """The state machine must not pull any cloud SDK or vendor helper."""

    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "gcp",
        "aws_access_key",
        "kubernetes",
        "docker",
    )
    for needle in forbidden:
        assert needle not in _MODULE_SOURCE, f"forbidden cloud reference: {needle}"


def test_module_source_has_no_hardcoded_api_key() -> None:
    """No long alphanumeric secret is hardcoded in the module source."""

    code_only = "\n".join(
        line for line in _MODULE_SOURCE.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"api_key\s*=\s*[\"']sk-[A-Za-z0-9]{16,}", code_only)
    assert not re.search(r"[\"']sk-[A-Za-z0-9]{16,}[\"']", code_only)
    assert "BEGIN PRIVATE KEY" not in code_only


def test_module_source_has_no_print_or_pprint() -> None:
    """The state machine is silent — no ``print`` / ``pprint``."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_module_source_has_no_subprocess_or_os_system() -> None:
    """No subprocess / os.system — the module is pure-Python state mgmt."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_module_source_has_no_direct_network_imports() -> None:
    """No direct ``requests`` / ``urllib`` / ``httpx`` imports."""

    assert "import requests" not in _MODULE_SOURCE
    assert "from urllib" not in _MODULE_SOURCE
    assert "import urllib" not in _MODULE_SOURCE
    assert "import httpx" not in _MODULE_SOURCE
    assert "from httpx" not in _MODULE_SOURCE


def test_module_source_has_no_eval_or_exec() -> None:
    """No ``eval`` / ``exec`` — runtime config is static."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_module_source_has_no_wildcard_imports() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


def test_module_source_does_not_read_environment_directly() -> None:
    """The state machine does not read ``os.environ`` — config is constructor-injected."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_module_source_has_no_outstanding_todo_markers() -> None:
    """No outstanding ``TODO`` / ``FIXME`` / ``XXX`` — the file is shipped
    as production code."""

    for marker in (r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b"):
        assert not re.search(marker, _MODULE_SOURCE), f"forbidden marker in source: {marker}"


def test_module_source_uses_strenum_for_wire_enums() -> None:
    """``ClaimStatus`` and ``SupportNeed`` are ``StrEnum`` so wire strings
    survive JSON round-trips."""

    assert "StrEnum" in _MODULE_SOURCE
    assert "class ClaimStatus(StrEnum)" in _MODULE_SOURCE
    assert "class SupportNeed(StrEnum)" in _MODULE_SOURCE


def test_module_source_uses_extra_forbid_on_models() -> None:
    """Both Pydantic models declare ``extra="forbid"`` — unknown fields
    are rejected, not silently ignored."""

    assert _MODULE_SOURCE.count('extra="forbid"') == 2


def test_module_source_uses_pydantic_basemodel() -> None:
    """``ClaimState`` and ``VerificationContext`` are :class:`BaseModel`
    subclasses — Pydantic v2 is the validation engine."""

    assert "BaseModel" in _MODULE_SOURCE
    assert _MODULE_SOURCE.count("(BaseModel)") == 2
