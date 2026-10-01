"""Dynamic package-surface import smoke test for oai2.knowledge.

The :mod:`oai2.knowledge` package exposes 100+ symbols via its
``__all__`` and is actively extended by remote commits (retrieval,
gc-lease, sweep, transport, qpipe-import, etc.). A static per-symbol
identity test would be stale the moment a new symbol lands.

This test reads ``__all__`` at runtime and asserts each named symbol
is importable as an attribute on the ``oai2.knowledge`` module. It
catches:

- A symbol listed in ``__all__`` that is not re-exported by the
  package (typo / dead reference).
- A re-export that resolves to ``None`` (silent import-cycle break).

It does not assert ``is``-identity against a sub-module because the
public contract for ``oai2.knowledge`` is its own ``__all__`` list;
sub-modules are internal source-of-truth and may move.

Refs: source-side audit of every ``oai2.*`` package's ``__all__``
identity coverage (#232 reconciliation, 2026-10-02).
"""

from __future__ import annotations

import pytest


def _all_symbols() -> list[str]:
    import oai2.knowledge as knowledge_pkg

    listed = getattr(knowledge_pkg, "__all__", None)
    assert isinstance(listed, list) and listed, (
        "oai2.knowledge.__all__ must be a non-empty list"
    )
    return list(listed)


def test_knowledge_package_all_list_is_nonempty() -> None:
    listed = _all_symbols()
    assert all(isinstance(s, str) and s for s in listed)
    # The package has 100+ public symbols; assert a sane lower bound.
    assert len(listed) >= 100, (
        f"oai2.knowledge.__all__ shrank unexpectedly: {len(listed)} entries"
    )


def test_knowledge_package_exports_every_all_symbol() -> None:
    """Every name in ``oai2.knowledge.__all__`` must resolve on the package."""
    import oai2.knowledge as knowledge_pkg

    missing: list[str] = []
    none_aliases: list[str] = []
    for symbol in _all_symbols():
        value = getattr(knowledge_pkg, symbol, None)
        if value is None:
            # Distinguish "missing attribute" from "value happens to be None".
            if not hasattr(knowledge_pkg, symbol):
                missing.append(symbol)
            else:
                none_aliases.append(symbol)
    assert not missing, (
        f"oai2.knowledge.__all__ lists names not present on the package: "
        f"{sorted(missing)}"
    )
    assert not none_aliases, (
        f"oai2.knowledge re-exports resolve to None (likely silent import-cycle "
        f"break): {sorted(none_aliases)}"
    )


@pytest.mark.parametrize("symbol", _all_symbols())
def test_knowledge_package_each_all_symbol_resolves(symbol: str) -> None:
    """Per-symbol parametrised smoke check for every ``__all__`` entry."""
    import oai2.knowledge as knowledge_pkg

    value = getattr(knowledge_pkg, symbol, None)
    assert value is not None, (
        f"oai2.knowledge.{symbol} resolves to None or is missing"
    )
