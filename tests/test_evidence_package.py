from __future__ import annotations

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import KnowledgeObject, RetrievalCandidate, RetrievalResult, sha256_hex
from oai2.knowledge.evidence_package import (
    EVIDENCE_PACKAGE_VERSION,
    build_evidence_package,
    evaluate_retrieval_package,
)


def _tokens(text: str) -> int:
    # Deterministic fixture tokenizer. Production callers supply their real
    # target tokenizer counter.
    return len(text.split())


def _obj(
    knowledge_id: str,
    content: str,
    *,
    authority: float = 0.9,
    source_uri: str | None = "https://example.test/source",
) -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId(knowledge_id),
        topic="retrieval",
        content=content,
        content_hash=sha256_hex(content),
        source_uri=source_uri,
        retrieved_at=123.0,
        authority=authority,
        status=Status.EXPERIMENTAL,
        artifact_ref="r2://fallback",
    )


def _result(*objects: KnowledgeObject) -> RetrievalResult:
    return RetrievalResult(
        topic="retrieval",
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


def test_package_preserves_provenance_hash_authority_timestamp_and_score() -> None:
    obj = _obj("ko_1", "important verified claim")
    package = build_evidence_package(
        _result(obj),
        token_budget=100,
        token_counter=_tokens,
    )

    assert package.version == EVIDENCE_PACKAGE_VERSION == "1"
    assert package.insufficient_evidence is False
    assert len(package.entries) == 1
    entry = package.entries[0]
    assert entry.knowledge_id == obj.knowledge_id
    assert entry.content_hash == obj.content_hash
    assert entry.source_ref == obj.source_uri
    assert entry.authority == obj.authority
    assert entry.retrieved_at == obj.retrieved_at
    assert entry.score == pytest.approx(0.9)
    assert entry.snippet == "important verified claim"
    assert package.token_count == _tokens(package.render())


def test_package_truncates_snippet_but_never_drops_provenance_to_fit_budget() -> None:
    content = " ".join(f"token{i}" for i in range(40))
    obj = _obj("ko_long", content)
    full = build_evidence_package(
        _result(obj),
        token_budget=200,
        token_counter=_tokens,
    )
    assert full.entries[0].snippet == content

    metadata_only = build_evidence_package(
        _result(obj),
        token_budget=20,
        token_counter=_tokens,
    )
    assert metadata_only.token_count <= 20
    assert metadata_only.entries
    entry = metadata_only.entries[0]
    assert entry.content_hash == obj.content_hash
    assert entry.source_ref == obj.source_uri
    assert entry.snippet != content
    assert "…" in entry.snippet or entry.snippet == ""


def test_package_stops_at_budget_and_reports_insufficient_when_nothing_fits() -> None:
    first = _obj("ko_1", "first compact claim")
    second = _obj("ko_2", "second compact claim")
    package = build_evidence_package(
        _result(first, second),
        token_budget=12,
        token_counter=_tokens,
    )
    assert package.token_count <= 12
    assert len(package.entries) <= 1

    empty = build_evidence_package(
        _result(first),
        token_budget=1,
        token_counter=_tokens,
    )
    assert empty.entries == ()
    assert empty.insufficient_evidence is True
    assert empty.token_count == 0


def test_packaging_rejects_missing_candidate_evidence_or_hash_mismatch() -> None:
    obj = _obj("ko_1", "claim")
    missing = RetrievalResult(topic="retrieval", objects=[obj], candidates=[])
    with pytest.raises(ValueError, match="no candidate evidence"):
        build_evidence_package(missing, token_budget=100, token_counter=_tokens)

    mismatched = RetrievalResult(
        topic="retrieval",
        objects=[obj],
        candidates=[
            RetrievalCandidate(
                knowledge_id=obj.knowledge_id,
                content_hash=sha256_hex("different"),
                source_uri=obj.source_uri,
                score=0.5,
            )
        ],
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_evidence_package(mismatched, token_budget=100, token_counter=_tokens)


def test_packaging_requires_a_provenance_source() -> None:
    obj = _obj("ko_1", "claim", source_uri=None)
    obj.artifact_ref = None
    result = RetrievalResult(
        topic="retrieval",
        objects=[obj],
        candidates=[
            RetrievalCandidate(
                knowledge_id=obj.knowledge_id,
                content_hash=obj.content_hash,
                source_uri=None,
                score=0.5,
            )
        ],
    )
    with pytest.raises(ValueError, match="no provenance source"):
        build_evidence_package(result, token_budget=100, token_counter=_tokens)


def test_retrieval_metrics_cover_precision_recall_irrelevant_tokens_and_task_delta() -> None:
    relevant = _obj("ko_relevant", "relevant claim")
    irrelevant = _obj("ko_irrelevant", "irrelevant claim")
    package = build_evidence_package(
        _result(relevant, irrelevant),
        token_budget=100,
        token_counter=_tokens,
    )

    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_relevant", "ko_missing_relevant"},
        raw_source_tokens=200,
        task_success_with_retrieval=0.8,
        task_success_without_retrieval=0.5,
    )

    assert metrics.k == 2
    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.irrelevant_context_rate == pytest.approx(0.5)
    assert metrics.package_tokens == package.token_count
    assert metrics.compression_ratio == pytest.approx(package.token_count / 200)
    assert metrics.task_success_delta == pytest.approx(0.3)


def test_empty_package_metrics_are_explicit_not_padded() -> None:
    package = build_evidence_package(
        _result(_obj("ko_1", "claim")),
        token_budget=1,
        token_counter=_tokens,
    )
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_1"},
        raw_source_tokens=50,
        task_success_with_retrieval=0.0,
        task_success_without_retrieval=0.0,
    )
    assert metrics.k == 0
    assert metrics.precision_at_k == 0.0
    assert metrics.recall == 0.0
    assert metrics.irrelevant_context_rate == 0.0
    assert metrics.package_tokens == 0


def test_invalid_token_counter_fails_closed() -> None:
    obj = _obj("ko_1", "claim")

    def bad_counter(text: str) -> int:
        return -1

    with pytest.raises(ValueError, match="non-negative integer"):
        build_evidence_package(
            _result(obj),
            token_budget=100,
            token_counter=bad_counter,
        )



def test_evidence_package_exports_from_knowledge_package() -> None:
    from oai2.knowledge import EVIDENCE_PACKAGE_VERSION as ExportedVersion
    from oai2.knowledge import EvidencePackage as ExportedPackage
    from oai2.knowledge import EvidencePackageEntry as ExportedEntry
    from oai2.knowledge import RetrievalMetrics as ExportedMetrics
    from oai2.knowledge import build_evidence_package as ExportedBuild
    from oai2.knowledge import evaluate_retrieval_package as ExportedEvaluate
    from oai2.knowledge.evidence_package import (
        EvidencePackage,
        EvidencePackageEntry,
        RetrievalMetrics,
    )

    assert ExportedVersion == EVIDENCE_PACKAGE_VERSION == "1"
    assert ExportedPackage is EvidencePackage
    assert ExportedEntry is EvidencePackageEntry
    assert ExportedMetrics is RetrievalMetrics
    assert ExportedBuild is build_evidence_package
    assert ExportedEvaluate is evaluate_retrieval_package
