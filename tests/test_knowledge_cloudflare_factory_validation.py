"""Outer guard rails for oai2/knowledge/cloudflare_factory.py.

Pin the public-safety boundary of the assembly module that wires bound
Cloudflare Worker resources into one public-safe knowledge component
set. The factory itself contains no deployment identifiers or secrets
(declared in the module docstring); it just constructs the lower-level
runtime / transport / reader / writer / lease-store pieces around the
supplied D1/R2/Vectorize/KV bindings.

Companion to ``tests/test_cloudflare_factory.py`` (which exercises the
happy-path wiring flows against fakes). This file pins the wire-contract
surface so silent breakage cannot slip through:

* Module docstring pin (deployment-neutral, no cloud-provider secrets).
* UP006-clean imports (no ``typing.Mapping`` / ``typing.Sequence``
  runtime-use imports; bare ``dataclass`` from ``dataclasses``).
* ``__all__`` (2 names: ``CloudflareKnowledgeComponents``,
  ``build_cloudflare_knowledge_components``) + package-level re-export
  identity check.
* ``CloudflareKnowledgeComponents`` dataclass shape — 8 fields
  (reader / writer / lease_store / r2 / vectorize / kv / runtime /
  transport), ``slots=True``, ``frozen=True``, ``frozen=True`` rejects
  attribute mutation, slot-set membership preserved.
* ``build_cloudflare_knowledge_components`` signature — all 8 required
  kwargs are keyword-only (the ``*`` enforces it).
* ``build_cloudflare_knowledge_components.ensure_schema`` — non-bool
  rejected (rejects int / str / None / 0 / 1 / list / dict via the
  ``ensure_schema must be a boolean`` message; ``True`` / ``False`` are
  accepted because ``isinstance(True, bool) is True`` and the
  ``isinstance(x, bool)`` check is the load-bearing guard).
* Schema-batch sequencing — when ``ensure_schema=True`` the factory
  calls ``writer.ensure_schema()`` FIRST (knowledge tables before GC
  lease SQL because the GC lease references the authoritative
  ``knowledge_index`` liveness), THEN ``lease_store.ensure_schema()``;
  when ``ensure_schema=False`` both schema calls are skipped (the
  pre-migrated-worker path).
* One-D1-binding invariant — the factory threads the SAME ``d1``
  binding into reader + writer + lease_store (three independent
  ``D1DatabaseBinding``-typed objects, one source binding).
* Returned ``CloudflareKnowledgeComponents`` wires every public field —
  reader / writer / lease_store / r2 / vectorize / kv / runtime /
  transport — all 8 fields are non-None.
* ``CloudflareKnowledgeComponents`` is the documented public-safe
  return type — the factory does NOT return a tuple, dict, or raw
  dataclass.
* The factory is idempotent on the binding surface — calling it
  twice with the same bindings produces two independent component
  sets (no shared mutable state).
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from oai2.knowledge import CloudflareKnowledgeComponents
from oai2.knowledge.cloudflare_bindings_runtime import (
    CloudflareKvCache,
    CloudflareR2Store,
    CloudflareVectorizeStore,
)
from oai2.knowledge.cloudflare_factory import (
    CloudflareKnowledgeComponents as SourceComponents,
)
from oai2.knowledge.cloudflare_factory import (
    build_cloudflare_knowledge_components as SourceBuilder,
)
from oai2.knowledge.cloudflare_runtime import AsyncCloudflareKnowledgeRuntime
from oai2.knowledge.gc_lease_d1 import gc_lease_schema_statements
from oai2.knowledge.gc_lease_d1_runtime import D1GcLeaseStore
from oai2.knowledge.knowledge_d1 import knowledge_schema_statements
from oai2.knowledge.knowledge_d1_runtime import D1KnowledgeReader, D1KnowledgeWriter
from oai2.knowledge.worker_transport import KnowledgeWorkerTransport

# ---------------------------------------------------------------------------
# Test fakes (mirror tests/test_cloudflare_factory.py patterns so the
# validation surface can run independently).
# ---------------------------------------------------------------------------


@dataclass
class _FakeStatement:
    query: str
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> _FakeStatement:
        self.bound = values
        return self

    async def run(self) -> object:
        return {"success": True, "meta": {"changes": 0}, "results": []}

    async def first(self, column_name: str | None = None) -> object | None:
        if column_name == "revision":
            return 0
        return None


class _FakeD1:
    def __init__(self) -> None:
        self.batch_calls: list[list[_FakeStatement]] = []

    def prepare(self, query: str) -> _FakeStatement:
        return _FakeStatement(query=query)

    async def batch(self, statements: Sequence[object]) -> list[object]:
        copied = [s for s in statements if isinstance(s, _FakeStatement)]
        self.batch_calls.append(copied)
        return [{"success": True, "meta": {"changes": 0}} for _ in copied]


class _FakeR2:
    async def get(self, key: str) -> object | None:
        return None

    async def head(self, key: str) -> object | None:
        return None

    async def put(self, key: str, value: str) -> object | None:
        return object()

    async def delete(self, key: str) -> None:
        return None


class _FakeVectorize:
    async def upsert(self, vectors: object) -> object:
        return {"mutationId": "m1"}

    async def query(self, vector: object, options: object = None) -> object:
        return {"matches": []}


class _FakeKv:
    async def get(self, key: str) -> object | None:
        return None

    async def put(
        self,
        key: str,
        value: str,
        *,
        expirationTtl: int | None = None,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_module_docstring_declares_no_secrets_and_no_deployment_ids() -> None:
    """The module docstring must declare no deployment identifiers or secrets."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    doc = factory_mod.__doc__ or ""
    assert "no deployment identifiers or secrets" in doc.lower()


def test_module_source_does_not_contain_hardcoded_account_or_bucket_ids() -> None:
    """The module source must not contain hardcoded account / bucket / namespace IDs."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    source = inspect.getsource(factory_mod)
    forbidden_patterns = [
        "account_id=",
        "r2_bucket:",
        "vectorize_index=",
        "kv_namespace=",
        "wrangler",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in source, f"forbidden pattern {pattern!r} found in module source"


def test_module_does_not_transitively_import_workers_package() -> None:
    """The factory must not import the workers package directly."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    source = inspect.getsource(factory_mod)
    assert "from oai2.workers" not in source
    assert "import oai2.workers" not in source


def test_module_uses_no_runtime_install_path_for_typing_collections() -> None:
    """The module must NOT use ``typing.Mapping`` / ``typing.Sequence`` for runtime use."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    source = inspect.getsource(factory_mod)
    assert "from typing import Mapping" not in source
    assert "from typing import Sequence" not in source
    assert "typing.Mapping" not in source
    assert "typing.Sequence" not in source


# ---------------------------------------------------------------------------
# 2. __all__ completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_all_has_exactly_two_names() -> None:
    """``__all__`` must list exactly 2 public names."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    assert sorted(factory_mod.__all__) == [
        "CloudflareKnowledgeComponents",
        "build_cloudflare_knowledge_components",
    ]


def test_all_names_are_importable_from_module() -> None:
    """Every name in ``__all__`` must be importable from the module."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    for name in factory_mod.__all__:
        obj = getattr(factory_mod, name)
        assert obj is not None


def test_package_level_reexport_identity() -> None:
    """``oai2.knowledge`` must re-export the factory names by identity."""
    import oai2.knowledge as pkg

    assert pkg.CloudflareKnowledgeComponents is SourceComponents
    assert pkg.build_cloudflare_knowledge_components is SourceBuilder


# ---------------------------------------------------------------------------
# 3. CloudflareKnowledgeComponents dataclass shape
# ---------------------------------------------------------------------------


def test_components_dataclass_field_set_is_complete() -> None:
    """``CloudflareKnowledgeComponents`` must declare all 8 documented fields."""
    expected_fields = {
        "reader",
        "writer",
        "lease_store",
        "r2",
        "vectorize",
        "kv",
        "runtime",
        "transport",
    }
    actual_fields = set(CloudflareKnowledgeComponents.__dataclass_fields__.keys())
    assert actual_fields == expected_fields


def test_components_dataclass_is_slots_and_frozen() -> None:
    """``CloudflareKnowledgeComponents`` must be ``slots=True`` + ``frozen=True``."""
    params = CloudflareKnowledgeComponents.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_components_dataclass_rejects_attribute_mutation() -> None:
    """``CloudflareKnowledgeComponents`` must reject attribute mutation."""
    components = SourceComponents(
        reader=object(),  # type: ignore[arg-type]
        writer=object(),  # type: ignore[arg-type]
        lease_store=object(),  # type: ignore[arg-type]
        r2=object(),  # type: ignore[arg-type]
        vectorize=object(),  # type: ignore[arg-type]
        kv=object(),  # type: ignore[arg-type]
        runtime=object(),  # type: ignore[arg-type]
        transport=object(),  # type: ignore[arg-type]
    )
    with pytest.raises((AttributeError, Exception)):
        components.reader = object()  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# 4. build_cloudflare_knowledge_components signature + argument validation
# ---------------------------------------------------------------------------


def test_builder_signature_is_keyword_only_after_d1() -> None:
    """``build_cloudflare_knowledge_components`` must be keyword-only after ``d1``."""
    sig = inspect.signature(SourceBuilder)
    params = sig.parameters

    expected = {
        "d1",
        "r2",
        "vectorize",
        "kv",
        "embedding_version",
        "embedding_digest",
        "embedding_provider",
        "require_auth",
        "ensure_schema",
    }
    assert set(params.keys()) == expected
    for name, p in params.items():
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name} is {p.kind}, expected KEYWORD_ONLY"
        )


@pytest.mark.parametrize("bad_value", [1, 0, "yes", None, [], {"k": 1}, 1.0])
@pytest.mark.asyncio
async def test_builder_rejects_non_boolean_ensure_schema(bad_value: object) -> None:
    """``build_cloudflare_knowledge_components`` must reject non-bool ``ensure_schema``.

    Bool is load-bearing because ``isinstance(True, int) is True``, so the
    ``isinstance(ensure_schema, bool)`` check is the load-bearing guard.
    A refactor that switches to truthy checks would silently accept ``1``
    and skip the schema migration.
    """
    with pytest.raises(ValueError, match="ensure_schema must be a boolean"):
        await SourceBuilder(
            d1=_FakeD1(),
            r2=_FakeR2(),  # type: ignore[arg-type]
            vectorize=_FakeVectorize(),
            kv=_FakeKv(),
            embedding_version="embed-v1",
            embedding_digest="digest-v1",
            ensure_schema=bad_value,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_builder_accepts_boolean_true_and_false_for_ensure_schema() -> None:
    """``build_cloudflare_knowledge_components`` must accept ``True`` and ``False``."""
    d1 = _FakeD1()
    components = await SourceBuilder(
        d1=d1,
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=True,
    )
    assert isinstance(components, SourceComponents)
    d1 = _FakeD1()
    components = await SourceBuilder(
        d1=d1,
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )
    assert isinstance(components, SourceComponents)
    assert d1.batch_calls == []


# ---------------------------------------------------------------------------
# 5. Schema-batch sequencing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builder_runs_writer_schema_then_lease_schema_in_order() -> None:
    """``build_cloudflare_knowledge_components`` must run ``writer.ensure_schema`` BEFORE ``lease_store.ensure_schema``.

    The knowledge tables / corpus state are the authoritative source for
    ``knowledge_index`` liveness; the GC lease SQL references that
    liveness. Running the lease schema first would fail because the
    knowledge_index table does not exist yet.
    """
    d1 = _FakeD1()
    await SourceBuilder(
        d1=d1,
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
    )

    assert len(d1.batch_calls) == 2
    assert [stmt.query for stmt in d1.batch_calls[0]] == list(knowledge_schema_statements())
    assert [stmt.query for stmt in d1.batch_calls[1]] == list(gc_lease_schema_statements())


@pytest.mark.asyncio
async def test_builder_skips_schema_when_ensure_schema_false() -> None:
    """``build_cloudflare_knowledge_components`` must skip both schema batches when ``ensure_schema=False``."""
    d1 = _FakeD1()
    components = await SourceBuilder(
        d1=d1,
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert d1.batch_calls == []
    assert isinstance(components, SourceComponents)


# ---------------------------------------------------------------------------
# 6. One-D1-binding invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builder_threads_d1_binding_into_reader_writer_and_lease_store() -> None:
    """The factory must thread the SAME ``d1`` binding into reader + writer + lease_store."""
    d1 = _FakeD1()
    components = await SourceBuilder(
        d1=d1,
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
    )

    assert components.reader._db is d1  # noqa: SLF001
    assert components.writer._db is d1  # noqa: SLF001
    assert components.lease_store._db is d1  # noqa: SLF001


@pytest.mark.asyncio
async def test_builder_threads_r2_vectorize_kv_into_runtime() -> None:
    """The factory must thread ``r2`` / ``vectorize`` / ``kv`` into the runtime."""
    r2 = _FakeR2()
    vectorize = _FakeVectorize()
    kv = _FakeKv()
    components = await SourceBuilder(
        d1=_FakeD1(),
        r2=r2,  # type: ignore[arg-type]
        vectorize=vectorize,
        kv=kv,
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert isinstance(components.r2, CloudflareR2Store)
    assert isinstance(components.vectorize, CloudflareVectorizeStore)
    assert isinstance(components.kv, CloudflareKvCache)
    assert isinstance(components.runtime, AsyncCloudflareKnowledgeRuntime)


# ---------------------------------------------------------------------------
# 7. Returned CloudflareKnowledgeComponents fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builder_returns_components_with_all_eight_fields_populated() -> None:
    """The returned components must populate all 8 fields."""
    components = await SourceBuilder(
        d1=_FakeD1(),
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert components.reader is not None
    assert components.writer is not None
    assert components.lease_store is not None
    assert components.r2 is not None
    assert components.vectorize is not None
    assert components.kv is not None
    assert components.runtime is not None
    assert components.transport is not None


@pytest.mark.asyncio
async def test_builder_returns_reader_writer_lease_store_of_correct_types() -> None:
    """The factory must produce typed ``D1KnowledgeReader`` / ``D1KnowledgeWriter`` / ``D1GcLeaseStore`` instances."""
    components = await SourceBuilder(
        d1=_FakeD1(),
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert isinstance(components.reader, D1KnowledgeReader)
    assert isinstance(components.writer, D1KnowledgeWriter)
    assert isinstance(components.lease_store, D1GcLeaseStore)


@pytest.mark.asyncio
async def test_builder_returns_transport_knowledge_worker_transport_instance() -> None:
    """The factory must produce a ``KnowledgeWorkerTransport`` instance."""
    components = await SourceBuilder(
        d1=_FakeD1(),
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert isinstance(components.transport, KnowledgeWorkerTransport)


@pytest.mark.asyncio
async def test_builder_returns_components_dataclass_instance() -> None:
    """The factory must return a ``CloudflareKnowledgeComponents`` dataclass instance."""
    components = await SourceBuilder(
        d1=_FakeD1(),
        r2=_FakeR2(),  # type: ignore[arg-type]
        vectorize=_FakeVectorize(),
        kv=_FakeKv(),
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert isinstance(components, SourceComponents)
    assert isinstance(components, CloudflareKnowledgeComponents)


# ---------------------------------------------------------------------------
# 8. Idempotence on the binding surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builder_produces_independent_component_sets_per_call() -> None:
    """Calling the factory twice with the same bindings must produce two independent component sets."""
    d1 = _FakeD1()
    r2 = _FakeR2()
    vectorize = _FakeVectorize()
    kv = _FakeKv()

    first = await SourceBuilder(
        d1=d1,
        r2=r2,  # type: ignore[arg-type]
        vectorize=vectorize,
        kv=kv,
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )
    second = await SourceBuilder(
        d1=d1,
        r2=r2,  # type: ignore[arg-type]
        vectorize=vectorize,
        kv=kv,
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
        ensure_schema=False,
    )

    assert first.reader is not second.reader
    assert first.writer is not second.writer
    assert first.lease_store is not second.lease_store
    assert first.r2 is not second.r2
    assert first.vectorize is not second.vectorize
    assert first.kv is not second.kv
    assert first.runtime is not second.runtime
    assert first.transport is not second.transport


# ---------------------------------------------------------------------------
# 9. UP006 import contract
# ---------------------------------------------------------------------------


def test_module_source_uses_no_typing_collections_for_runtime() -> None:
    """The module source must NOT use ``typing.Mapping`` / ``typing.Sequence`` for runtime use."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    source = inspect.getsource(factory_mod)
    assert "from typing import Mapping" not in source
    assert "from typing import Sequence" not in source
    assert "typing.Mapping" not in source
    assert "typing.Sequence" not in source


def test_module_source_imports_from_sibling_modules() -> None:
    """The module source must import from the cloud-runtime / gc-lease / worker-transport siblings."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    source = inspect.getsource(factory_mod)
    assert "from .cloudflare_bindings_runtime import" in source
    assert "from .cloudflare_runtime import" in source
    assert "from .gc_lease_d1_runtime import" in source
    assert "from .knowledge_d1_runtime import" in source
    assert "from .worker_transport import" in source


def test_module_source_declares_all() -> None:
    """The module source must declare its public API via ``__all__``."""
    import oai2.knowledge.cloudflare_factory as factory_mod

    source = inspect.getsource(factory_mod)
    assert "__all__" in source
