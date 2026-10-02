"""Coverage-gap pin: every name in ``oai2/tools/__init__.py`` ``__all__``
is importable from ``oai2.tools`` and aliases the source-of-truth
defined in ``oai2.tools.dispatch``.

Closes the Pass 10 coverage gap where the existing
``tests/test_tools_dispatch.py`` exercised the dispatcher's six policy
stages and pinned identity for only 3 of the 5 ``__all__`` entries
(``DispatchPolicy``, ``DispatchStage``, ``ToolDispatcher``). The
remaining 2 entries ``DispatchDecision`` and ``default_dispatcher``
were not pinned at the package surface, leaving a 40 percent coverage
gap on the package public contract.

Reference pattern: ``tests/test_evals.py::test_evals_package_full_surface_identity``
(Pass 9, commit ``ffc5e86`` for #216) and
``tests/test_protocols.py::test_protocols_module_exports_from_protocols_package``.
"""

from __future__ import annotations


def test_tools_package_full_surface_identity() -> None:
    """Assert identity for all 5 ``oai2.tools.__all__`` entries."""
    import oai2.tools as tools_pkg
    from oai2.tools import (
        DispatchDecision as _PkgDispatchDecision,
    )
    from oai2.tools import (
        DispatchPolicy as _PkgDispatchPolicy,
    )
    from oai2.tools import (
        DispatchStage as _PkgDispatchStage,
    )
    from oai2.tools import (
        ToolDispatcher as _PkgToolDispatcher,
    )
    from oai2.tools import (
        default_dispatcher as _PkgDefaultDispatcher,
    )
    from oai2.tools.dispatch import (
        DispatchDecision as _SrcDispatchDecision,
    )
    from oai2.tools.dispatch import (
        DispatchPolicy as _SrcDispatchPolicy,
    )
    from oai2.tools.dispatch import (
        DispatchStage as _SrcDispatchStage,
    )
    from oai2.tools.dispatch import (
        ToolDispatcher as _SrcToolDispatcher,
    )
    from oai2.tools.dispatch import (
        default_dispatcher as _SrcDefaultDispatcher,
    )

    # The 5 entries the package re-exports from ``oai2.tools.dispatch``
    # must be the same object as the source-of-truth.
    assert tools_pkg.DispatchDecision is _SrcDispatchDecision
    assert tools_pkg.DispatchDecision is _PkgDispatchDecision
    assert tools_pkg.DispatchPolicy is _SrcDispatchPolicy
    assert tools_pkg.DispatchPolicy is _PkgDispatchPolicy
    assert tools_pkg.DispatchStage is _SrcDispatchStage
    assert tools_pkg.DispatchStage is _PkgDispatchStage
    assert tools_pkg.ToolDispatcher is _SrcToolDispatcher
    assert tools_pkg.ToolDispatcher is _PkgToolDispatcher
    assert tools_pkg.default_dispatcher is _SrcDefaultDispatcher
    assert tools_pkg.default_dispatcher is _PkgDefaultDispatcher


def test_tools_all_matches_runtime_exports() -> None:
    """``oai2.tools.__all__`` must list every entry the package exposes."""
    import oai2.tools as tools_pkg

    declared = set(tools_pkg.__all__)
    expected = {
        "DispatchDecision",
        "DispatchPolicy",
        "DispatchStage",
        "ToolDispatcher",
        "default_dispatcher",
    }
    assert declared == expected
    assert len(declared) == 5
