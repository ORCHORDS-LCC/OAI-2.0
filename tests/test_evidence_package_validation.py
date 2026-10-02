"""Pin the input-validation surface of ``oai2.knowledge.evidence_package``.

The behavioural tests in ``tests/test_evidence_package.py`` and the budget tests
in ``tests/test_evidence_package_budget.py`` cover the positive path plus the
inner-tuple integrity checks (missing candidate / hash mismatch / missing
provenance / negative counter result). The module's outermost guard rails --
``token_budget`` shape, ``max_entries`` shape, ``token_counter`` callability,
the "empty package cannot fit the budget" feasibility check, ``raw_source_tokens``
shape, and the rate-in-range check -- are also public contracts: a future
refactor that drops one of them would silently widen the input domain and admit
it valid as a malformed call.

Each ``pytest.raises(ValueError, match=...)`` pins a single contract line in
``oai2/knowledge/evidence_package.py`` so regressions on those lines are
caught at unit-test granularity instead of as a downstream type error.
"""

from __future__ import annotations

import pytest

from oai2.knowledge import evaluate_retrieval_package
from oai2.knowledge.evidence_package import build_evidence_package
from tests._evidence_fixtures import (
    make_evidence_package,
)
from tests._evidence_fixtures import (
    make_knowledge_object as _obj,
)
from tests._evidence_fixtures import (
    make_retrieval_result as _result,
)
from tests._evidence_fixtures import (
    tokens as _tokens,
)


def _package(*objects):
    """A minimal valid package used by the ``evaluate_*`` tests.

    Thin wrapper over ``make_evidence_package`` so the body of each test
    reads as ``_package(obj)`` (clear "make me a package from this obj")
    rather than ``make_evidence_package(obj, token_budget=1000)``.
    """
    return make_evidence_package(*objects)


# ---------------------------------------------------------------------------
# build_evidence_package: input-shape guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_budget", [0, -1, -1000])
def test_build_evidence_package_rejects_non_positive_token_budget(
    bad_budget: int,
) -> None:
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="token_budget must be a positive integer"):
        build_evidence_package(
            _result(obj),
            token_budget=bad_budget,
            token_counter=_tokens,
        )


@pytest.mark.parametrize("bad_budget", ["100", 100.0, None, [100]])
def test_build_evidence_package_rejects_non_integer_token_budget(
    bad_budget: object,
) -> None:
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="token_budget must be a positive integer"):
        build_evidence_package(
            _result(obj),
            token_budget=bad_budget,  # type: ignore[arg-type]
            token_counter=_tokens,
        )


def test_build_evidence_package_rejects_bool_token_budget() -> None:
    """``bool`` is a subclass of ``int`` in Python; without the explicit guard,
    ``True`` would be accepted as ``token_budget=1`` and ``False`` would fail
    the ``<= 0`` branch with a confusing message. The guard short-circuits
    ``bool`` so the message stays consistent.
    """
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="token_budget must be a positive integer"):
        build_evidence_package(
            _result(obj),
            token_budget=True,  # bool is rejected at runtime by the explicit guard
            token_counter=_tokens,
        )


@pytest.mark.parametrize("bad_max_entries", [0, -1, -1000])
def test_build_evidence_package_rejects_non_positive_max_entries(
    bad_max_entries: int,
) -> None:
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="max_entries must be a positive integer when provided"):
        build_evidence_package(
            _result(obj),
            token_budget=100,
            token_counter=_tokens,
            max_entries=bad_max_entries,
        )


@pytest.mark.parametrize("bad_max_entries", ["3", 3.0, [3]])
def test_build_evidence_package_rejects_non_integer_max_entries(
    bad_max_entries: object,
) -> None:
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="max_entries must be a positive integer when provided"):
        build_evidence_package(
            _result(obj),
            token_budget=100,
            token_counter=_tokens,
            max_entries=bad_max_entries,  # type: ignore[arg-type]
        )


def test_build_evidence_package_rejects_bool_max_entries() -> None:
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="max_entries must be a positive integer when provided"):
        build_evidence_package(
            _result(obj),
            token_budget=100,
            token_counter=_tokens,
            max_entries=True,  # bool is rejected at runtime by the explicit guard
        )


# ---------------------------------------------------------------------------
# Positive-path pinning: `max_entries=None` is the legitimate sentinel for
# "keep the whole result set" and `max_entries=N` (positive int) truncates.
# The slice-18 negative-path coverage above locks the rejection branches, so
# these positive-path tests lock the accept branches. Together they pin the
# full contract so a refactor that "simplifies" the guard to require a
# positive int (and accidentally rejects `None`) cannot silently regress.
# ---------------------------------------------------------------------------


def test_build_evidence_package_accepts_none_max_entries_as_no_truncation_sentinel() -> None:
    objs = [_obj("ko_1", "first claim"), _obj("ko_2", "second claim")]
    package = build_evidence_package(
        _result(*objs),
        token_budget=1000,
        token_counter=_tokens,
        max_entries=None,  # sentinel: keep the full result set
    )
    assert len(package.entries) == 2


def test_build_evidence_package_accepts_positive_int_max_entries_as_truncation() -> None:
    objs = [
        _obj("ko_1", "first claim"),
        _obj("ko_2", "second claim"),
        _obj("ko_3", "third claim"),
    ]
    package = build_evidence_package(
        _result(*objs),
        token_budget=1000,
        token_counter=_tokens,
        max_entries=2,
    )
    assert len(package.entries) == 2


@pytest.mark.parametrize("bad_counter", [None, 42, "len", 3.14])
def test_build_evidence_package_rejects_non_callable_token_counter(
    bad_counter: object,
) -> None:
    obj = _obj("ko_1", "claim")
    with pytest.raises(ValueError, match="token_counter must be callable"):
        build_evidence_package(
            _result(obj),
            token_budget=100,
            token_counter=bad_counter,  # type: ignore[arg-type]
        )


def test_build_evidence_package_rejects_budget_too_small_for_empty_encoding() -> None:
    """Pin the line 92-93 feasibility check: a counter whose empty-string cost
    is positive can never fit "the empty package encoding", so the call must
    fail closed instead of returning an empty package that violates the
    budget invariant.
    """

    def overhead_counter(text: str) -> int:
        return len(text) + 50  # empty string already costs 50

    with pytest.raises(ValueError, match="token_budget cannot fit the empty package encoding"):
        build_evidence_package(
            _result(),
            token_budget=10,
            token_counter=overhead_counter,
        )


# ---------------------------------------------------------------------------
# evaluate_retrieval_package: input-shape guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_raw", [0, -1, -1000])
def test_evaluate_retrieval_package_rejects_non_positive_raw_source_tokens(
    bad_raw: int,
) -> None:
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="raw_source_tokens must be a positive integer"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["ko_1"],
            raw_source_tokens=bad_raw,
            task_success_with_retrieval=0.5,
            task_success_without_retrieval=0.4,
        )


@pytest.mark.parametrize("bad_raw", ["100", 100.0, None, [100]])
def test_evaluate_retrieval_package_rejects_non_integer_raw_source_tokens(
    bad_raw: object,
) -> None:
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="raw_source_tokens must be a positive integer"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["ko_1"],
            raw_source_tokens=bad_raw,  # type: ignore[arg-type]
            task_success_with_retrieval=0.5,
            task_success_without_retrieval=0.4,
        )


def test_evaluate_retrieval_package_rejects_bool_raw_source_tokens() -> None:
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="raw_source_tokens must be a positive integer"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["ko_1"],
            raw_source_tokens=True,  # bool is rejected at runtime by the explicit guard
            task_success_with_retrieval=0.5,
            task_success_without_retrieval=0.4,
        )


def test_evaluate_retrieval_package_rejects_empty_relevant_knowledge_ids() -> None:
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="relevant_knowledge_ids must not be empty"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=[],
            raw_source_tokens=100,
            task_success_with_retrieval=0.5,
            task_success_without_retrieval=0.4,
        )


# ---------------------------------------------------------------------------
# evaluate_retrieval_package: rate-in-range guards (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_rate",
    [-0.0001, -0.5, -1.0, 1.0001, 1.5, 2.0, float("inf"), float("nan")],
)
def test_evaluate_retrieval_package_rejects_out_of_range_task_success_with_retrieval(
    bad_rate: float,
) -> None:
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="task_success_with_retrieval must be between 0 and 1"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["ko_1"],
            raw_source_tokens=100,
            task_success_with_retrieval=bad_rate,
            task_success_without_retrieval=0.4,
        )


@pytest.mark.parametrize(
    "bad_rate",
    [-0.0001, -0.5, -1.0, 1.0001, 1.5, 2.0, float("inf"), float("nan")],
)
def test_evaluate_retrieval_package_rejects_out_of_range_task_success_without_retrieval(
    bad_rate: float,
) -> None:
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="task_success_without_retrieval must be between 0 and 1"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["ko_1"],
            raw_source_tokens=100,
            task_success_with_retrieval=0.5,
            task_success_without_retrieval=bad_rate,
        )


@pytest.mark.parametrize("bad_rate", ["0.5", True, None, [0.5]])
def test_evaluate_retrieval_package_rejects_non_numeric_task_success_rate(
    bad_rate: object,
) -> None:
    """``_rate`` accepts only ``int``/``float`` (excluding ``bool`` because of
    the explicit ``isinstance(value, bool)`` short-circuit). Non-numeric types
    must fail closed at validation, not later in the division.
    """
    package = _package(_obj("ko_1", "claim"))
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["ko_1"],
            raw_source_tokens=100,
            task_success_with_retrieval=bad_rate,  # type: ignore[arg-type]
            task_success_without_retrieval=0.4,
        )
