"""Enforce REQ-RET-022 against the actual joined evidence context.

The counters here are deterministic cost fixtures, not model-token benchmarks.
"""

from collections.abc import Callable

import pytest

from oai2.core import KnowledgeId
from oai2.knowledge.abstraction import KnowledgeObject, RetrievalCandidate, RetrievalResult
from oai2.knowledge.evidence_package import build_evidence_package


def _result() -> RetrievalResult:
    objects = [
        KnowledgeObject(
            knowledge_id=KnowledgeId(f"item-{i}"),
            topic="budget",
            content="one two three four five six seven eight nine ten",
            content_hash=str(i) * 64,
            source_uri=f"https://example.test/{i}",
            retrieved_at=123.0,
            authority=0.9,
        )
        for i in (1, 2)
    ]
    return RetrievalResult(
        topic="budget",
        objects=objects,
        candidates=[
            RetrievalCandidate(obj.knowledge_id, obj.content_hash, obj.source_uri)
            for obj in objects
        ],
    )


def test_join_separators_are_included_in_the_hard_budget() -> None:
    result = _result()
    full = build_evidence_package(result, token_budget=10000, token_counter=len)
    budget = sum(entry.tokens for entry in full.entries)
    package = build_evidence_package(result, token_budget=budget, token_counter=len)
    assert len(package.render()) <= budget
    assert package.token_count == len(package.render())
    for entry in package.entries:
        original = next(obj for obj in result.objects if obj.knowledge_id == entry.knowledge_id)
        assert entry.content_hash == original.content_hash
        assert entry.source_ref == original.source_uri
        assert entry.tokens == len(entry.render())


def _with_overhead(text: str) -> int:
    return len(text) + 23


def _with_boundary_cost(text: str) -> int:
    return len(text) + (80 if "\n\n" in text else 0)


@pytest.mark.parametrize("counter", [_with_overhead, _with_boundary_cost])
def test_whole_context_counter_not_entry_sum_drives_fit(counter: Callable[[str], int]) -> None:
    result = _result()
    full = build_evidence_package(result, token_budget=10000, token_counter=counter)
    budget = counter(full.render())
    exact = build_evidence_package(result, token_budget=budget, token_counter=counter)
    assert [entry.snippet for entry in exact.entries] == [obj.content for obj in result.objects]
    assert exact.token_count == budget
    assert exact.token_count == counter(exact.render())


@pytest.mark.parametrize("budget", [180, 240, 320, 400, 480])
def test_nonadditive_counter_never_overflows_or_misreports(budget: int) -> None:
    package = build_evidence_package(
        _result(), token_budget=budget, token_counter=_with_boundary_cost
    )
    assert package.token_count == _with_boundary_cost(package.render())
    assert package.token_count <= budget


def test_empty_package_accounts_for_counter_overhead() -> None:
    package = build_evidence_package(
        RetrievalResult(topic="empty"), token_budget=23, token_counter=_with_overhead
    )
    assert package.entries == ()
    assert package.insufficient_evidence is True
    assert package.token_count == _with_overhead(package.render()) == 23


def test_budget_below_empty_encoding_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty package"):
        build_evidence_package(
            RetrievalResult(topic="empty"), token_budget=22, token_counter=_with_overhead
        )
