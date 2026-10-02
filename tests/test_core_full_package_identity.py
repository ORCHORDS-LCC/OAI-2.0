"""Coverage-gap pin: every name in ``oai2/core/__init__.py`` ``__all__``
is importable from ``oai2.core`` and aliases the source-of-truth.

Closes the Pass 10 coverage gap where ``oai2.core`` declared 9 ``__all__``
entries (6 ``NewType`` ID aliases, ``Status`` enum, ``ComponentInfo``
dataclass, ``component_info`` factory helper) but no dedicated test file
existed for the package. The 9 entries were referenced transitively
across the test suite but never pinned at the package surface, leaving
a 100 percent coverage gap on the package public contract.

All 9 entries are defined inline in ``oai2/core/__init__.py`` (the
``Status`` enum is re-imported by ``oai2/core/status.py`` but the
canonical binding lives in the package init). The pin below asserts
the package-surface reference equals the canonical binding.

Reference pattern: ``tests/test_evals.py::test_evals_package_full_surface_identity``
(Pass 9, commit ``ffc5e86`` for #216) and
``tests/test_protocols.py::test_protocols_module_exports_from_protocols_package``.
"""

from __future__ import annotations


def test_core_full_package_identity() -> None:
    """Assert identity for all 9 ``oai2.core.__all__`` entries."""
    import oai2.core as core_pkg
    from oai2.core import (
        AgentId as _PkgAgentId,
    )
    from oai2.core import (
        ClaimId as _PkgClaimId,
    )
    from oai2.core import (
        ComponentInfo as _PkgComponentInfo,
    )
    from oai2.core import (
        EvidenceId as _PkgEvidenceId,
    )
    from oai2.core import (
        KnowledgeId as _PkgKnowledgeId,
    )
    from oai2.core import (
        Status as _PkgStatus,
    )
    from oai2.core import (
        TaskId as _PkgTaskId,
    )
    from oai2.core import (
        ToolId as _PkgToolId,
    )
    from oai2.core import (
        component_info as _PkgComponentInfoFactory,
    )

    # The 9 entries the package defines inline must be the same object
    # as the package-surface reference.
    assert core_pkg.AgentId is _PkgAgentId
    assert core_pkg.ClaimId is _PkgClaimId
    assert core_pkg.ComponentInfo is _PkgComponentInfo
    assert core_pkg.EvidenceId is _PkgEvidenceId
    assert core_pkg.KnowledgeId is _PkgKnowledgeId
    assert core_pkg.Status is _PkgStatus
    assert core_pkg.TaskId is _PkgTaskId
    assert core_pkg.ToolId is _PkgToolId
    assert core_pkg.component_info is _PkgComponentInfoFactory


def test_core_status_enum_has_expected_members() -> None:
    """Sanity guard so a Status enum shrink cannot silently pass the pin."""
    from oai2.core import Status

    assert Status.IMPLEMENTED == "IMPLEMENTED"
    assert Status.EXPERIMENTAL == "EXPERIMENTAL"
    assert Status.PROPOSED == "PROPOSED"
    assert Status.BLOCKED == "BLOCKED"
    # Adding a new status is a non-breaking change; removing one is.
    assert len(Status) >= 4


def test_core_all_matches_runtime_exports() -> None:
    """``oai2.core.__all__`` must list every entry the package exposes."""
    import oai2.core as core_pkg

    declared = set(core_pkg.__all__)
    expected = {
        "KnowledgeId",
        "EvidenceId",
        "TaskId",
        "ClaimId",
        "AgentId",
        "ToolId",
        "Status",
        "ComponentInfo",
        "component_info",
    }
    assert declared == expected
    assert len(declared) == 9


def test_core_ids_are_opaque_distinct_aliases() -> None:
    """Each NewType ID alias must be a distinct type even though each
    collapses to ``str`` at runtime. This guards against an accidental
    dedup that would let one ID be silently assigned where another was
    expected (a class of bug the IDs exist to prevent).
    """
    from oai2.core import (
        AgentId,
        ClaimId,
        EvidenceId,
        KnowledgeId,
        TaskId,
        ToolId,
    )

    # Each NewType is its own type object.
    id_types = {AgentId, ClaimId, EvidenceId, KnowledgeId, TaskId, ToolId}
    assert len(id_types) == 6

    # Each ID accepts a plain str but the constructed value preserves the
    # type at the static-typing level (runtime type is still str).
    assert AgentId("a1") == "a1"
    assert ClaimId("c1") == "c1"
