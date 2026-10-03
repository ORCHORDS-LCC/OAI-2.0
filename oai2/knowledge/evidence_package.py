"""Compact provenance-rich evidence packages and retrieval metrics.

The formatter is tokenizer-agnostic: callers supply the exact token counter used
by the target runtime. This avoids pretending character/word counts are model
tokens while still enforcing a hard evidence-context budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..core import KnowledgeId
from .abstraction import KnowledgeObject, RetrievalCandidate, RetrievalResult

EVIDENCE_PACKAGE_VERSION = "1"
TokenCounter = Callable[[str], int]


@dataclass(slots=True, frozen=True)
class EvidencePackageEntry:
    knowledge_id: KnowledgeId
    content_hash: str
    source_ref: str
    authority: float
    retrieved_at: float
    snippet: str
    tokens: int
    score: float | None = None

    def render(self) -> str:
        score = "" if self.score is None else f" score={self.score:.6f}"
        return (
            f"[{self.knowledge_id}] hash={self.content_hash} "
            f"source={self.source_ref} authority={self.authority:.6f} "
            f"retrieved_at={self.retrieved_at:.6f}{score}\n"
            f"{self.snippet}"
        )


@dataclass(slots=True, frozen=True)
class EvidencePackage:
    version: str
    topic: str
    entries: tuple[EvidencePackageEntry, ...]
    token_count: int
    token_budget: int
    insufficient_evidence: bool

    def render(self) -> str:
        return "\n\n".join(entry.render() for entry in self.entries)


@dataclass(slots=True, frozen=True)
class RetrievalMetrics:
    k: int
    precision_at_k: float
    recall: float
    irrelevant_context_rate: float
    package_tokens: int
    raw_source_tokens: int
    compression_ratio: float
    task_success_delta: float
    # Whether the package carried NO evidence at all, so every rate above is
    # the result of a measurement that never happened.
    #
    # `EvidencePackage.insufficient_evidence` already knew this -- it is
    # `not selected` -- and `evaluate_retrieval_package` dropped it. Two
    # ordinary failures, retrieval returning nothing and a candidate that did
    # not fit the budget, both produced k=0 with precision 0.0, recall 0.0
    # and irrelevant_context_rate 0.0, and nothing on this record said so.
    #
    # `irrelevant_context_rate = 0.0` is the dangerous one: it reads as "this
    # context introduced no irrelevant material", which is the most flattering
    # number a retrieval metric can carry, and here it was produced by a
    # retrieval that retrieved nothing. It is the same failure as
    # `no_comparable_samples` in the numerical chain and `capability_measured`
    # in the promotion chain: an absence, rendered as a clean result.
    insufficient_evidence: bool

    def __post_init__(self) -> None:
        if self.insufficient_evidence and self.k > 0:
            raise ValueError(
                "insufficient_evidence is true but k > 0: a package that "
                "retrieved something is not an evidence-free package, and "
                "allowing the combination would let a measured rate and an "
                "unmeasured one be reported for the same package"
            )

    @property
    def measured(self) -> bool:
        """Whether the rates above describe a real retrieval.

        A property rather than a second stored flag, so it cannot disagree
        with ``k``: there is no way to construct a ``k=0`` record that claims
        to be measured.
        """
        return self.k > 0


def build_evidence_package(
    result: RetrievalResult,
    *,
    token_budget: int,
    token_counter: TokenCounter,
    max_entries: int | None = None,
) -> EvidencePackage:
    """Build a provenance-complete package within the exact rendered-text budget.

    The supplied token counter must be deterministic for a fixed text.
    """
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
        raise ValueError("token_budget must be a positive integer")
    # `max_entries=None` is the legitimate sentinel for "no truncation; keep the
    # full result set". Do not collapse this branch into a generic "must be a
    # positive int" guard — that would reject legitimate callers and break the
    # contract. The explicit `is not None` check below is load-bearing: the
    # ternary at line ~95 (`result.objects if max_entries is None else
    # result.objects[:max_entries]`) and the slice-18 parametrized negative-path
    # tests in tests/test_evidence_package_validation.py both depend on `None`
    # remaining accepted.
    if max_entries is not None and (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries <= 0
    ):
        raise ValueError("max_entries must be a positive integer when provided")
    if not callable(token_counter):
        raise ValueError("token_counter must be callable")

    candidates = {candidate.knowledge_id: candidate for candidate in result.candidates}
    selected: list[EvidencePackageEntry] = []
    used_tokens = _token_count(token_counter, "")
    if used_tokens > token_budget:
        raise ValueError("token_budget cannot fit the empty package encoding")

    objects = result.objects if max_entries is None else result.objects[:max_entries]
    for obj in objects:
        candidate = candidates.get(obj.knowledge_id)
        if candidate is None:
            # Packaging must not silently discard the retrieval evidence that
            # binds score/hash/provenance to the object.
            raise ValueError(
                f"retrieval object {obj.knowledge_id!s} has no candidate evidence"
            )
        if candidate.content_hash != obj.content_hash:
            raise ValueError(
                f"candidate hash mismatch for {obj.knowledge_id!s}"
            )
        source_ref = obj.source_uri or obj.artifact_ref
        if source_ref is None or not source_ref.strip():
            raise ValueError(
                f"retrieval object {obj.knowledge_id!s} has no provenance source"
            )

        prefix = "\n\n".join(entry.render() for entry in selected)
        if selected:
            prefix += "\n\n"
        entry = _fit_entry(
            obj,
            candidate,
            source_ref=source_ref,
            prefix=prefix,
            token_budget=token_budget,
            token_counter=token_counter,
        )
        if entry is None:
            break
        selected.append(entry)
        used_tokens = _token_count(token_counter, prefix + entry.render())

    return EvidencePackage(
        version=EVIDENCE_PACKAGE_VERSION,
        topic=result.topic,
        entries=tuple(selected),
        token_count=used_tokens,
        token_budget=token_budget,
        insufficient_evidence=not selected,
    )


def evaluate_retrieval_package(
    package: EvidencePackage,
    *,
    relevant_knowledge_ids: Iterable[KnowledgeId | str],
    raw_source_tokens: int,
    task_success_with_retrieval: float,
    task_success_without_retrieval: float,
) -> RetrievalMetrics:
    """Compute deterministic retrieval/package usefulness metrics."""
    if (
        isinstance(raw_source_tokens, bool)
        or not isinstance(raw_source_tokens, int)
        or raw_source_tokens <= 0
    ):
        raise ValueError("raw_source_tokens must be a positive integer")
    with_retrieval = _rate(task_success_with_retrieval, "task_success_with_retrieval")
    without_retrieval = _rate(
        task_success_without_retrieval,
        "task_success_without_retrieval",
    )

    relevant = {str(value) for value in relevant_knowledge_ids}
    if not relevant:
        raise ValueError("relevant_knowledge_ids must not be empty")

    retrieved = [str(entry.knowledge_id) for entry in package.entries]
    relevant_retrieved = sum(1 for knowledge_id in retrieved if knowledge_id in relevant)
    k = len(retrieved)
    precision = relevant_retrieved / k if k else 0.0
    recall = relevant_retrieved / len(relevant)
    irrelevant_rate = 1.0 - precision if k else 0.0

    return RetrievalMetrics(
        k=k,
        precision_at_k=precision,
        recall=recall,
        irrelevant_context_rate=irrelevant_rate,
        package_tokens=package.token_count,
        raw_source_tokens=raw_source_tokens,
        compression_ratio=package.token_count / raw_source_tokens,
        task_success_delta=with_retrieval - without_retrieval,
        # Derived from k so the record cannot disagree with itself, and
        # OR-ed with the package's own flag so a hand-built package that
        # claims evidence it does not have is still reported as such.
        insufficient_evidence=bool(package.insufficient_evidence) or k == 0,
    )


def _fit_entry(
    obj: KnowledgeObject,
    candidate: RetrievalCandidate,
    *,
    source_ref: str,
    prefix: str,
    token_budget: int,
    token_counter: TokenCounter,
) -> EvidencePackageEntry | None:
    words = " ".join(obj.content.split()).split(" ")
    # Empty bodies still need a provenance-bearing entry.
    if words == [""]:
        words = []

    def make(snippet: str) -> EvidencePackageEntry:
        provisional = EvidencePackageEntry(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            source_ref=source_ref,
            authority=obj.authority,
            retrieved_at=obj.retrieved_at,
            snippet=snippet,
            score=candidate.score,
            tokens=0,
        )
        tokens = _token_count(token_counter, provisional.render())
        return EvidencePackageEntry(
            knowledge_id=provisional.knowledge_id,
            content_hash=provisional.content_hash,
            source_ref=provisional.source_ref,
            authority=provisional.authority,
            retrieved_at=provisional.retrieved_at,
            snippet=provisional.snippet,
            score=provisional.score,
            tokens=tokens,
        )

    def fits(entry: EvidencePackageEntry) -> bool:
        return _token_count(token_counter, prefix + entry.render()) <= token_budget

    full = make(" ".join(words))
    if fits(full):
        return full

    # Search for a provenance-complete content prefix. Check the whole joined
    # package, since separators and tokenizer boundaries make costs non-additive.
    low = 0
    high = len(words)
    best: EvidencePackageEntry | None = None
    while low <= high:
        mid = (low + high) // 2
        snippet = " ".join(words[:mid])
        if mid < len(words) and snippet:
            snippet += " …"
        entry = make(snippet)
        if fits(entry):
            best = entry
            low = mid + 1
        else:
            high = mid - 1
    return best


def _token_count(token_counter: TokenCounter, text: str) -> int:
    value = token_counter(text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token_counter must return a non-negative integer")
    return value


def _rate(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


__all__ = [
    "EVIDENCE_PACKAGE_VERSION",
    "TokenCounter",
    "EvidencePackageEntry",
    "EvidencePackage",
    "RetrievalMetrics",
    "build_evidence_package",
    "evaluate_retrieval_package",
]
