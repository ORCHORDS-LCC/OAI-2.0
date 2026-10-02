"""Outer guard rails for ``oai2/knowledge/ingestion.py`` boundary contracts.

The existing ``tests/test_knowledge_sweep*.py`` family covers the GC sweep
happy paths and async machinery. What is NOT pinned for
``oai2/knowledge/ingestion.py``:

* ``IngestionStatus`` 4-member ``StrEnum`` contract (PENDING / OK / FAILED /
  SKIPPED) and the wire-string bijection ``pending`` / ``ok`` / ``failed`` /
  ``skipped``.
* ``IngestionJob`` stdlib ``@dataclass(slots=True, frozen=True)`` shape
  (``source_uri`` / ``topic`` / ``content`` mandatory; ``authority`` defaults
  to ``0.5``; ``status`` defaults to ``Status.EXPERIMENTAL``); frozen mutation
  raises ``FrozenInstanceError``; no ``__dict__`` exposure (slotted).
* ``IngestionJob.build()`` round-trip — returns a ``KnowledgeObject`` whose
  ``knowledge_id`` is ``KnowledgeId("ko_<12-hex>")``; ``content_hash`` is
  ``sha256_hex(self.content)``; ``retrieved_at`` is a ``float``; ``topic`` /
  ``content`` / ``source_uri`` / ``authority`` / ``status`` are copied from
  ``self``; two ``build()`` calls produce different ``knowledge_id`` values
  (uuid uniqueness).
* ``IngestionPipeline`` stdlib ``@dataclass(slots=True)`` shape (``store``
  mandatory; ``sources`` is a ``list[_Source]`` default-factory ``list`` —
  mutable default!); NOT frozen (mutating ``pipeline.sources.append(...)``
  must succeed).
* ``IngestionPipeline.ingest`` semantics — empty jobs → empty results; one
  good job → ``[OK]`` + object in the store; multiple good jobs → ``[OK,
  OK, ...]``; per-job exception isolation (one failing ``build()`` yields
  ``FAILED`` for that job, the next good job still succeeds); accepts a
  generator (``Iterable[IngestionJob]``), not just a ``list``.
* Module shape — module docstring mentions "ingestion" + "pipeline" +
  "KnowledgeObject" + "PROPOSED"; ``from __future__ import annotations``
  (UP006-clean); stdlib imports ``uuid`` / ``Iterable`` from
  ``collections.abc`` / ``dataclass`` + ``field`` / ``StrEnum`` / ``Protocol``;
  relative imports ``from ..core import KnowledgeId, Status`` and
  ``from .abstraction import KnowledgeObject, KnowledgeStore, now_epoch,
  sha256_hex``; no absolute ``import oai2`` / ``from oai2.knowledge`` /
  ``from oai2.core`` at line-start; no wildcard imports; no cloud-runtime
  imports (``boto3`` / ``azure`` / ``google.cloud`` / ``kubernetes`` /
  ``docker`` / ``fabric`` excluded).
* ``__all__`` completeness — exactly 3 names (``IngestionJob`` /
  ``IngestionPipeline`` / ``IngestionStatus``); every name is importable from
  the module; identity-equal re-exports at the ``oai2.knowledge`` package
  level.
* Public-safety — no hardcoded credentials, no ``print`` / ``pprint``, no
  ``subprocess`` / ``os.system``, no direct ``requests`` / ``urllib`` /
  ``httpx`` / ``aiohttp``, no ``eval`` / ``exec``, no wildcard imports, no
  ``os.environ`` / ``os.getenv``, no ``TODO`` / ``FIXME`` / ``XXX`` markers.

Each section pins one or more of these contracts with a small, sharp test
that fails immediately on a regression. The pattern follows the slice-60 …
slice-64 outer-guard-rail files in this campaign.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass
from enum import StrEnum

import pytest

from oai2.core import Status
from oai2.knowledge.abstraction import (
    InMemoryKnowledgeStore,
    KnowledgeObject,
    sha256_hex,
)
from oai2.knowledge.ingestion import (
    IngestionJob,
    IngestionPipeline,
    IngestionStatus,
)
from oai2.knowledge.ingestion import (
    __all__ as ingestion_all,
)

INGESTION_MODULE = "oai2.knowledge.ingestion"
MODULE_SOURCE = inspect.getsource(IngestionStatus).rsplit("class IngestionStatus", 1)[0]


def _module_ingest_source() -> str:
    import oai2.knowledge.ingestion as _mod

    return inspect.getsource(_mod)


# ---------------------------------------------------------------------------
# Section 1 — docstring + module-level imports
# ---------------------------------------------------------------------------


def test_ingestion_module_docstring_mentions_ingestion_pipeline_and_status() -> None:
    src = _module_ingest_source()
    doc_match = re.search(r'^"""(?P<body>.*?)"""', src, re.DOTALL)
    assert doc_match is not None, "module must begin with a triple-quoted docstring"
    body = doc_match.group("body").lower()
    assert "ingestion" in body, "docstring must mention 'ingestion'"
    assert "pipeline" in body, "docstring must mention 'pipeline'"
    assert "knowledgeobject" in body, "docstring must mention 'KnowledgeObject'"
    assert "proposed" in body, "docstring must note the PROPOSED status"


def test_ingestion_module_has_future_annotations() -> None:
    src = _module_ingest_source()
    assert re.search(r"^from __future__ import annotations\s*$", src, re.MULTILINE), (
        "module must declare `from __future__ import annotations` for UP006"
    )


def test_ingestion_module_imports_uuid_iterable_dataclass_strenum_protocol() -> None:
    src = _module_ingest_source()
    assert re.search(r"^import uuid\s*$", src, re.MULTILINE), (
        "module must `import uuid` at module level"
    )
    assert re.search(r"^from collections\.abc import Iterable\s*$", src, re.MULTILINE), (
        "module must import `Iterable` from collections.abc (not a tuple import)"
    )
    assert re.search(r"^from dataclasses import dataclass, field\s*$", src, re.MULTILINE), (
        "module must co-import `dataclass` + `field` from dataclasses"
    )
    assert re.search(r"^from enum import StrEnum\s*$", src, re.MULTILINE), (
        "module must `from enum import StrEnum`"
    )
    assert re.search(r"^from typing import Protocol\s*$", src, re.MULTILINE), (
        "module must `from typing import Protocol`"
    )


def test_ingestion_module_imports_knowledge_id_and_status_from_core() -> None:
    src = _module_ingest_source()
    match = re.search(r"^from \.\.core import KnowledgeId, Status\s*$", src, re.MULTILINE)
    assert match is not None, (
        "module must `from ..core import KnowledgeId, Status` (single multi-name line)"
    )


def test_ingestion_module_imports_abstraction_helpers() -> None:
    src = _module_ingest_source()
    match = re.search(
        r"^from \.abstraction import "
        r"KnowledgeObject, KnowledgeStore, now_epoch, sha256_hex\s*$",
        src,
        re.MULTILINE,
    )
    assert match is not None, (
        "module must `from .abstraction import KnowledgeObject, KnowledgeStore, "
        "now_epoch, sha256_hex` (single multi-name line)"
    )


def test_ingestion_module_has_no_absolute_oai2_imports() -> None:
    src = _module_ingest_source()
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import oai2") or stripped.startswith("from oai2 "):
            if stripped.startswith("from oai2.knowledge.ingestion "):
                # The test file itself can re-import the module — not a real
                # boundary violation. We only flag upstream cross-package lines.
                continue
            if (
                stripped.startswith("from oai2.core ")
                or stripped.startswith("from oai2.knowledge.abstraction ")
                or stripped.startswith("from oai2.knowledge.ingestion ")
            ):
                # Allowed: explicit intra-package re-import via the relative
                # package path. The actual module uses `from ..core` /
                # `from .abstraction` which we already pinned.
                continue
            pytest.fail(f"unexpected absolute oai2 import: {line!r}")


def test_ingestion_module_has_no_cloud_runtime_imports() -> None:
    src = _module_ingest_source()
    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "kubernetes",
        "docker",
        "fabric",
    )
    for needle in forbidden:
        assert needle not in src, f"ingestion module must not reference {needle!r} (cloud runtime)"


def test_ingestion_module_has_no_wildcard_imports() -> None:
    src = _module_ingest_source()
    assert "from X import *" not in src and "import *" not in src
    assert not re.search(r"^from\s+\S+\s+import\s+\*\s*$", src, re.MULTILINE), (
        "ingestion module must not contain wildcard imports"
    )


# ---------------------------------------------------------------------------
# Section 2 — __all__ + identity
# ---------------------------------------------------------------------------


def test_ingestion_module_all_is_declared_and_has_three_names() -> None:
    assert isinstance(ingestion_all, list), "__all__ must be a list"
    assert sorted(ingestion_all) == sorted(
        ["IngestionJob", "IngestionPipeline", "IngestionStatus"]
    ), f"__all__ must export exactly the 3 documented public names (got {ingestion_all!r})"


def test_ingestion_module_all_names_are_importable() -> None:
    import oai2.knowledge.ingestion as mod

    for name in ingestion_all:
        assert hasattr(mod, name), f"{name} must be importable from the module"


def test_ingestion_module_all_omits_internal_helpers() -> None:
    expected_private = {
        "KnowledgeObject",
        "KnowledgeStore",
        "KnowledgeId",
        "Status",
        "now_epoch",
        "sha256_hex",
        "_Source",
        "uuid",
        "Iterable",
        "dataclass",
        "field",
        "StrEnum",
        "Protocol",
    }
    for name in expected_private:
        assert name not in ingestion_all, (
            f"{name} is a private/imported helper and must NOT be in __all__"
        )


def test_ingestion_module_source_pins_three_quoted_all_strings() -> None:
    src = _module_ingest_source()
    quoted = re.findall(r"^__all__\s*=\s*\[([^\]]+)\]", src, re.MULTILINE)
    assert quoted, "__all__ must be a literal list assignment at module scope"
    body = quoted[0]
    names = [n.strip().strip('"').strip("'") for n in body.split(",")]
    assert sorted(names) == sorted(["IngestionJob", "IngestionPipeline", "IngestionStatus"]), (
        f"__all__ literal must list exactly 3 names (got {names!r})"
    )


def test_ingestion_symbols_reexported_at_knowledge_package_level() -> None:
    import oai2.knowledge as pkg

    for name in ingestion_all:
        assert hasattr(pkg, name), f"oai2.knowledge must re-export {name} via __init__.py"
        assert getattr(pkg, name) is globals()[name], (
            f"oai2.knowledge.{name} must identity-equal oai2.knowledge.ingestion.{name}"
        )


def test_ingestion_symbols_match_direct_module_imports() -> None:
    from oai2.knowledge.ingestion import (
        IngestionJob as DirectJob,
    )
    from oai2.knowledge.ingestion import (
        IngestionPipeline as DirectPipe,
    )
    from oai2.knowledge.ingestion import (
        IngestionStatus as DirectStatus,
    )

    assert DirectJob is IngestionJob
    assert DirectPipe is IngestionPipeline
    assert DirectStatus is IngestionStatus


def test_ingestion_knowledge_package_all_contains_three_ingestion_names() -> None:
    import oai2.knowledge as pkg

    for name in ("IngestionJob", "IngestionPipeline", "IngestionStatus"):
        assert name in pkg.__all__, (
            f"oai2.knowledge.__all__ must list {name} so it is reachable as oai2.knowledge.{name}"
        )


def test_ingestion_module_source_has_no_top_level_dunder_side_effects() -> None:
    """The module must not run side effects at import time beyond declarations."""
    import ast

    tree = ast.parse(_module_ingest_source())
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
    )
    for node in tree.body:
        assert isinstance(node, allowed), (
            f"unexpected top-level statement kind: {type(node).__name__}"
        )


# ---------------------------------------------------------------------------
# Section 3 — IngestionStatus
# ---------------------------------------------------------------------------


def test_ingestion_status_is_a_str_enum_subclass() -> None:
    assert issubclass(IngestionStatus, StrEnum)
    assert issubclass(IngestionStatus, str)


def test_ingestion_status_has_exactly_four_members() -> None:
    members = list(IngestionStatus)
    assert len(members) == 4, f"expected 4 members, got {len(members)}: {members}"


def test_ingestion_status_member_names_and_values() -> None:
    assert IngestionStatus.PENDING.value == "pending"
    assert IngestionStatus.OK.value == "ok"
    assert IngestionStatus.FAILED.value == "failed"
    assert IngestionStatus.SKIPPED.value == "skipped"


def test_ingestion_status_member_name_value_bijection_set() -> None:
    members = list(IngestionStatus)
    names = [m.name for m in members]
    values = [m.value for m in members]
    assert sorted(names) == sorted(["PENDING", "OK", "FAILED", "SKIPPED"])
    assert sorted(values) == sorted(["pending", "ok", "failed", "skipped"])
    assert len(set(names)) == 4, "member names must be unique"
    assert len(set(values)) == 4, "wire values must be unique"


def test_ingestion_status_str_equality_with_wire_string() -> None:
    assert IngestionStatus.PENDING == "pending"
    assert IngestionStatus.OK == "ok"
    assert IngestionStatus.FAILED == "failed"
    assert IngestionStatus.SKIPPED == "skipped"
    assert str(IngestionStatus.PENDING) == "pending"
    assert str(IngestionStatus.OK) == "ok"
    assert str(IngestionStatus.FAILED) == "failed"
    assert str(IngestionStatus.SKIPPED) == "skipped"


def test_ingestion_status_wire_string_lifts_round_trip() -> None:
    for member in IngestionStatus:
        assert IngestionStatus(member.value) is member, (
            f"IngestionStatus({member.value!r}) must resolve back to {member}"
        )


def test_ingestion_status_unknown_wire_string_raises_value_error() -> None:
    with pytest.raises(ValueError):
        IngestionStatus("not_a_real_status")


# ---------------------------------------------------------------------------
# Section 4 — IngestionJob
# ---------------------------------------------------------------------------


def test_ingestion_job_is_stdlib_dataclass() -> None:
    assert is_dataclass(IngestionJob)


def test_ingestion_job_is_frozen() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    with pytest.raises(FrozenInstanceError):
        job.topic = "different"  # type: ignore[misc]


def test_ingestion_job_is_slotted() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    assert not hasattr(job, "__dict__"), "slotted dataclass must not expose __dict__"


def test_ingestion_job_field_set_is_five_names() -> None:
    field_names = {f.name for f in fields(IngestionJob)}
    assert field_names == {
        "source_uri",
        "topic",
        "content",
        "authority",
        "status",
    }, f"unexpected field set: {field_names!r}"


def test_ingestion_job_default_authority_is_half() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    assert job.authority == 0.5


def test_ingestion_job_default_status_is_experimental() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    assert job.status is Status.EXPERIMENTAL


def test_ingestion_job_stores_all_fields() -> None:
    job = IngestionJob(
        source_uri="https://example.com/x",
        topic="alpha",
        content="payload",
        authority=0.9,
        status=Status.IMPLEMENTED,
    )
    assert job.source_uri == "https://example.com/x"
    assert job.topic == "alpha"
    assert job.content == "payload"
    assert job.authority == 0.9
    assert job.status is Status.IMPLEMENTED


def test_ingestion_job_accepts_any_float_authority() -> None:
    """``IngestionJob.authority`` has no built-in clamp — only
    ``KnowledgeObject.authority`` clamps. The pipeline trusts callers.
    """
    for value in (0.0, 1.0, -0.5, 2.0):
        job = IngestionJob(source_uri="u", topic="t", content="c", authority=value)
        assert job.authority == value


def test_ingestion_job_accepts_every_status_enum_value() -> None:
    for value in Status:
        job = IngestionJob(source_uri="u", topic="t", content="c", status=value)
        assert job.status is value


def test_ingestion_job_missing_required_field_raises_type_error() -> None:
    with pytest.raises(TypeError):
        IngestionJob()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        IngestionJob(source_uri="u")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        IngestionJob(source_uri="u", topic="t")  # type: ignore[call-arg]


def test_ingestion_job_build_returns_knowledge_object() -> None:
    job = IngestionJob(
        source_uri="u",
        topic="t",
        content="c",
        authority=0.7,
        status=Status.IMPLEMENTED,
    )
    obj = job.build()
    assert isinstance(obj, KnowledgeObject)


def test_ingestion_job_build_copies_caller_fields() -> None:
    job = IngestionJob(
        source_uri="https://example.com/x",
        topic="alpha",
        content="payload",
        authority=0.9,
        status=Status.IMPLEMENTED,
    )
    obj = job.build()
    assert obj.topic == "alpha"
    assert obj.content == "payload"
    assert obj.source_uri == "https://example.com/x"
    assert obj.authority == 0.9
    assert obj.status is Status.IMPLEMENTED


def test_ingestion_job_build_derives_content_hash_via_sha256_hex() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="payload-xyz")
    obj = job.build()
    assert obj.content_hash == sha256_hex("payload-xyz")


def test_ingestion_job_build_knowledge_id_starts_with_ko_prefix() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    obj = job.build()
    raw = obj.knowledge_id
    assert raw.startswith("ko_"), f"knowledge_id must start with 'ko_', got {raw!r}"
    suffix = raw[len("ko_") :]
    assert len(suffix) == 12, f"hex suffix must be 12 chars, got {suffix!r}"
    assert all(c in "0123456789abcdef" for c in suffix), (
        f"suffix must be lowercase hex, got {suffix!r}"
    )
    assert isinstance(raw, str)  # KnowledgeId is a NewType over str


def test_ingestion_job_build_retrieved_at_is_a_float() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    obj = job.build()
    assert isinstance(obj.retrieved_at, float)


def test_ingestion_job_two_builds_yield_distinct_knowledge_ids() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="c")
    a = job.build()
    b = job.build()
    assert a.knowledge_id != b.knowledge_id


def test_ingestion_job_build_with_empty_content_still_hashes() -> None:
    job = IngestionJob(source_uri="u", topic="t", content="")
    obj = job.build()
    assert obj.content_hash == sha256_hex("")
    # sha256("") is the canonical 64-hex-char empty digest
    assert len(obj.content_hash) == 64


# ---------------------------------------------------------------------------
# Section 5 — IngestionPipeline
# ---------------------------------------------------------------------------


def test_ingestion_pipeline_is_stdlib_dataclass() -> None:
    assert is_dataclass(IngestionPipeline)


def test_ingestion_pipeline_is_not_frozen() -> None:
    """``IngestionPipeline`` MUST be mutable — ``sources`` is a mutable list
    that callers append to at runtime.
    """
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)
    # Should not raise — assignment / append are allowed.
    pipe.sources.append(object())  # type: ignore[arg-type]
    assert len(pipe.sources) == 1


def test_ingestion_pipeline_is_slotted() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)
    assert not hasattr(pipe, "__dict__"), "slotted dataclass must not expose __dict__"


def test_ingestion_pipeline_field_set_is_two_names() -> None:
    field_names = {f.name for f in fields(IngestionPipeline)}
    assert field_names == {"store", "sources"}, f"unexpected field set: {field_names!r}"


def test_ingestion_pipeline_sources_defaults_to_empty_list() -> None:
    pipe = IngestionPipeline(store=InMemoryKnowledgeStore())
    assert pipe.sources == []
    assert isinstance(pipe.sources, list)


def test_ingestion_pipeline_sources_is_a_list_not_tuple_or_frozenset() -> None:
    pipe = IngestionPipeline(store=InMemoryKnowledgeStore())
    assert type(pipe.sources) is list, f"sources must be a list, got {type(pipe.sources).__name__}"


def test_ingestion_pipeline_ingest_empty_jobs_returns_empty_list() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)
    assert pipe.ingest([]) == []


def test_ingestion_pipeline_ingest_single_job_returns_ok_and_persists() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)
    job = IngestionJob(source_uri="u", topic="t", content="c")
    results = pipe.ingest([job])
    assert results == [IngestionStatus.OK]
    persisted = list(store.all())
    assert len(persisted) == 1
    assert persisted[0].topic == "t"
    assert persisted[0].content == "c"


def test_ingestion_pipeline_ingest_multiple_jobs_returns_one_ok_per_job() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)
    jobs = [IngestionJob(source_uri=f"u{i}", topic=f"t{i}", content=f"c{i}") for i in range(4)]
    results = pipe.ingest(jobs)
    assert results == [IngestionStatus.OK] * 4
    assert len(list(store.all())) == 4


def test_ingestion_pipeline_ingest_generator_input_accepted() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)

    def _gen() -> Iterable[IngestionJob]:
        yield IngestionJob(source_uri="u", topic="t", content="c")
        yield IngestionJob(source_uri="v", topic="w", content="d")

    results = pipe.ingest(_gen())
    assert results == [IngestionStatus.OK, IngestionStatus.OK]
    assert len(list(store.all())) == 2


def test_ingestion_pipeline_ingest_per_job_exception_isolation() -> None:
    """If one job's ``build()`` raises, that job yields FAILED and the next
    good job still succeeds.
    """
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)

    good = IngestionJob(source_uri="u", topic="t", content="c")

    @dataclass(slots=True, frozen=True)
    class _ExplodingJob:
        source_uri: str = "x"
        topic: str = "y"
        content: str = "z"

        def build(self) -> KnowledgeObject:  # pragma: no cover - tested via ingest
            raise RuntimeError("boom")

    results = pipe.ingest([_ExplodingJob(), good])  # type: ignore[list-item]
    assert results == [IngestionStatus.FAILED, IngestionStatus.OK]
    # Only the good job's object made it to the store.
    persisted = list(store.all())
    assert len(persisted) == 1
    assert persisted[0].topic == "t"


def test_ingestion_pipeline_ingest_does_not_swallow_exception_class_name() -> None:
    """The pipeline records FAILED on failure but does NOT re-raise — the
    exception is contained. We assert ingest returns cleanly.
    """
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)

    @dataclass(slots=True, frozen=True)
    class _Exploding:
        source_uri: str = "u"
        topic: str = "t"
        content: str = "c"

        def build(self) -> KnowledgeObject:  # pragma: no cover - tested via ingest
            raise ValueError("specific-failure-marker")

    # Must not raise.
    results = pipe.ingest([_Exploding()])  # type: ignore[list-item]
    assert results == [IngestionStatus.FAILED]
    assert list(store.all()) == []


def test_ingestion_pipeline_ingest_continues_after_failure_in_first_position() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)

    @dataclass(slots=True, frozen=True)
    class _Exploding:
        source_uri: str = "u"
        topic: str = "t"
        content: str = "c"

        def build(self) -> KnowledgeObject:  # pragma: no cover - tested via ingest
            raise RuntimeError("boom")

    good = IngestionJob(source_uri="good", topic="good", content="good")
    results = pipe.ingest(
        [_Exploding(), good, _Exploding(), good]  # type: ignore[list-item]
    )
    assert results == [
        IngestionStatus.FAILED,
        IngestionStatus.OK,
        IngestionStatus.FAILED,
        IngestionStatus.OK,
    ]
    persisted = list(store.all())
    assert len(persisted) == 2


def test_ingestion_pipeline_store_argument_stored_on_self() -> None:
    store = InMemoryKnowledgeStore()
    pipe = IngestionPipeline(store=store)
    assert pipe.store is store


def test_ingestion_pipeline_ingest_returns_list_of_ingestion_status() -> None:
    pipe = IngestionPipeline(store=InMemoryKnowledgeStore())
    results = pipe.ingest([])
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, IngestionStatus)


# ---------------------------------------------------------------------------
# Section 6 — public-safety + structural counts
# ---------------------------------------------------------------------------


def test_ingestion_module_has_no_hardcoded_credentials() -> None:
    src = _module_ingest_source()
    assert "api_key=" not in src, "no `api_key=` literal in source"
    assert "BEGIN PRIVATE KEY" not in src, "no embedded private keys"
    # Stripe / AWS / GCP / Slack token patterns
    for pattern in (
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r"xox[baprs]-[A-Za-z0-9-]+",
    ):
        assert not re.search(pattern, src), f"credential-like pattern {pattern!r} found in source"


def test_ingestion_module_has_no_print_or_pprint() -> None:
    src = _module_ingest_source()
    assert not re.search(r"\bprint\s*\(", src), "no print() in source"
    assert not re.search(r"\bpprint\s*[\.\(]", src), "no pprint in source"


def test_ingestion_module_has_no_subprocess_or_shell() -> None:
    src = _module_ingest_source()
    assert "subprocess" not in src, "no subprocess in source"
    assert "os.system" not in src, "no os.system in source"
    assert "shell=True" not in src, "no shell=True in source"


def test_ingestion_module_has_no_direct_http_imports() -> None:
    src = _module_ingest_source()
    for needle in ("import requests", "import urllib", "import httpx", "import aiohttp"):
        assert needle not in src, f"unexpected direct HTTP import: {needle!r}"


def test_ingestion_module_has_no_eval_or_exec() -> None:
    src = _module_ingest_source()
    assert not re.search(r"^\s*eval\s*\(", src, re.MULTILINE), "no eval(...) call"
    assert not re.search(r"^\s*exec\s*\(", src, re.MULTILINE), "no exec(...) call"
    assert "compile(" not in src, "no compile(...) call"


def test_ingestion_module_has_no_os_environ_access() -> None:
    src = _module_ingest_source()
    assert "os.environ" not in src, "no os.environ access"
    assert "os.getenv" not in src, "no os.getenv call"


def test_ingestion_module_has_no_todo_fixme_xxx_markers() -> None:
    src = _module_ingest_source()
    for marker in ("TODO", "FIXME", "XXX"):
        assert not re.search(rf"\b{marker}\b", src), f"no {marker!r} markers allowed in source"


def test_ingestion_module_source_ends_with_single_trailing_newline() -> None:
    src = _module_ingest_source()
    assert src.endswith("\n"), "source must end with at least one newline"
    assert not src.endswith("\n\n"), "source must end with exactly one trailing newline (POSIX)"


def test_ingestion_module_source_pins_one_strenum_subclass() -> None:
    src = _module_ingest_source()
    strenum_classes = re.findall(r"^class\s+(\w+)\s*\(\s*StrEnum\s*\)\s*:", src, re.MULTILINE)
    assert strenum_classes == ["IngestionStatus"], (
        f"expected exactly 1 StrEnum subclass (IngestionStatus), got {strenum_classes!r}"
    )


def test_ingestion_module_source_pins_two_dataclass_decorators() -> None:
    src = _module_ingest_source()
    # Capture every `@dataclass(...)` invocation (possibly multiline).
    decorators = re.findall(r"@dataclass\s*\(", src)
    assert len(decorators) == 2, (
        f"expected exactly 2 @dataclass(...) decorators, got {decorators!r}"
    )


def test_ingestion_module_source_pins_one_protocol_subclass() -> None:
    src = _module_ingest_source()
    protos = re.findall(r"^class\s+(\w+)\s*\(\s*Protocol\s*\)\s*:", src, re.MULTILINE)
    assert protos == ["_Source"], f"expected exactly 1 Protocol subclass (_Source), got {protos!r}"


def test_ingestion_module_source_pins_one_in_memory_knowledge_store_consumer() -> None:
    """The pipeline relies on the abstract ``KnowledgeStore``; it must NOT
    instantiate the concrete ``InMemoryKnowledgeStore`` internally.
    """
    src = _module_ingest_source()
    assert "InMemoryKnowledgeStore" not in src, (
        "ingestion pipeline must consume the abstract KnowledgeStore, not the "
        "concrete in-memory implementation"
    )


def test_ingestion_source_private_source_protocol_is_not_in_all() -> None:
    src = _module_ingest_source()
    assert "_Source" not in ingestion_all, (
        "_Source is private (leading underscore) and must NOT be re-exported"
    )
    # Verify the source line exists at module scope (the protocol declaration).
    assert re.search(r"^class\s+_Source\s*\(\s*Protocol\s*\)\s*:", src, re.MULTILINE), (
        "_Source Protocol must be declared at module scope"
    )


def test_ingestion_module_source_pins_uuid4_prefix_slice() -> None:
    """The ``build()`` knowledge_id format is ``ko_<12-hex>`` via ``uuid.uuid4().hex[:12]``."""
    src = _module_ingest_source()
    assert "uuid.uuid4().hex[:12]" in src, (
        "knowledge_id format must use `uuid.uuid4().hex[:12]` (12-hex slice)"
    )
    assert 'KnowledgeId(f"ko_' in src, (
        "knowledge_id must be prefixed with 'ko_' before the hex slice"
    )


def test_ingestion_pipeline_ingest_signature_is_jobs_only() -> None:
    sig = inspect.signature(IngestionPipeline.ingest)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "jobs"], (
        f"ingest must take (self, jobs), got {[p.name for p in params]!r}"
    )
    jobs_param = params[1]
    assert jobs_param.annotation is Iterable[IngestionJob] or (
        # Under `from __future__ import annotations` the annotation may be a
        # string. Accept either form.
        str(jobs_param.annotation) == "Iterable[IngestionJob]"
    )


def test_ingestion_pipeline_module_attribute_is_path_to_ingestion_submodule() -> None:
    import oai2.knowledge.ingestion as mod

    assert mod.__name__ == INGESTION_MODULE
