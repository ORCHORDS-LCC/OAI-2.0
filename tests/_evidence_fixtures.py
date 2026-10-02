"""Shared fixtures for the ``oai2.knowledge`` test surface.

The ``test_evidence_package.py``, ``test_evidence_package_validation.py``, and
``test_abstraction_validation.py`` files all need the same handful of
builders:

* ``tokens(text)`` -- a deterministic token counter used by every
  ``build_evidence_package(...)`` call.
* ``make_knowledge_object(...)`` -- a ``KnowledgeObject`` factory with a
  single uniform kwargs surface (``knowledge_id``, ``topic``, ``content``,
  ``content_hash``, ``authority``, ``status``, ``source_uri``,
  ``retrieved_at``, ``artifact_ref``). Each call site used to define its own
  variant with slightly different defaults; this module is the canonical
  form.
* ``make_retrieval_result(...)`` -- wrap one or more ``KnowledgeObject``
  instances in a matching ``RetrievalResult`` with scoring candidates.
* ``make_evidence_package(...)`` -- convenience that builds a small
  ``EvidencePackage`` from a list of objects at a generous budget.

This module's filename starts with an underscore so ``pytest``'s default
``test_*.py`` / ``*_test.py`` collection rule does not pick it up. Tests
import the builders explicitly via ``from tests._evidence_fixtures import
...``. Keep all helpers here pure (no ``pytest`` fixtures / no shared
state) so the module stays cheap to import and trivially testable in
isolation.
"""

from __future__ import annotations

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    EvidencePackage,
    KnowledgeObject,
    RetrievalCandidate,
    RetrievalResult,
    sha256_hex,
)
from oai2.knowledge.evidence_package import TokenCounter, build_evidence_package

__all__ = [
    "tokens",
    "make_knowledge_object",
    "make_retrieval_result",
    "make_evidence_package",
]


def tokens(text: str) -> int:
    """Deterministic fixture tokenizer.

    Production callers supply their real target tokenizer; tests use this
    lightweight splitter so output is independent of the model tokenizer.
    """
    return len(text.split())


def make_knowledge_object(
    knowledge_id: str = "ko_1",
    content: str = "claim text",
    *,
    topic: str = "retrieval",
    authority: float = 0.9,
    status: Status = Status.EXPERIMENTAL,
    content_hash: str | None = None,
    source_uri: str | None = "https://example.test/source",
    retrieved_at: float = 123.0,
    artifact_ref: str | None = "r2://fallback",
) -> KnowledgeObject:
    """Build a ``KnowledgeObject`` with all-args-explicit defaults.

    The defaults match the canonical evidence-package fixture: a stable
    ``knowledge_id``, ``retrieved_at=123.0``, ``status=EXPERIMENTAL``, and
    an ``artifact_ref`` so the object satisfies the "must have a provenance
    source" guard in ``build_evidence_package`` even when ``source_uri``
    is set to ``None`` in negative-path tests.

    ``knowledge_id`` and ``content`` are positional so the common call
    shape is ``make_knowledge_object("ko_1", "claim text")``; everything
    else is keyword-only.
    """
    return KnowledgeObject(
        knowledge_id=KnowledgeId(knowledge_id),
        topic=topic,
        content=content,
        content_hash=content_hash if content_hash is not None else sha256_hex(content),
        source_uri=source_uri,
        retrieved_at=retrieved_at,
        authority=authority,
        status=status,
        artifact_ref=artifact_ref,
    )


def make_retrieval_result(
    *objects: KnowledgeObject,
    topic: str = "retrieval",
) -> RetrievalResult:
    """Wrap ``KnowledgeObject`` instances in a matching ``RetrievalResult``.

    The candidate score decays linearly with index (``0.9, 0.8, 0.7, ...``)
    so tests that exercise ``evaluate_retrieval_package`` see a deterministic
    ranking signal that is independent of the order the objects happen to
    be passed in.
    """
    return RetrievalResult(
        topic=topic,
        objects=list(objects),
        candidates=[
            RetrievalCandidate(
                knowledge_id=obj.knowledge_id,
                content_hash=obj.content_hash,
                source_uri=obj.source_uri,
                score=0.9 - (i * 0.1),
            )
            for i, obj in enumerate(objects)
        ],
    )


def make_evidence_package(
    *objects: KnowledgeObject,
    token_budget: int = 1000,
    token_counter: TokenCounter = tokens,
    max_entries: int | None = None,
) -> EvidencePackage:
    """Convenience wrapper that builds an ``EvidencePackage`` from objects.

    The defaults (``token_budget=1000``, ``max_entries=None``) leave plenty
    of room so callers usually get the full object set back; tests that
    need a tight budget pass ``token_budget`` explicitly.
    """
    return build_evidence_package(
        make_retrieval_result(*objects),
        token_budget=token_budget,
        token_counter=token_counter,
        max_entries=max_entries,
    )
