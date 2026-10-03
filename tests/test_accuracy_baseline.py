"""Tests for the llama.cpp runtime and the deterministic-hard suite.

Both halves are exercised offline under ``httpx.MockTransport``: a
correctness oracle that required a live model to test would itself be
unverifiable, which is the problem this work exists to fix.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from oai2.evals import BUILTIN_SUITES_NAMES, builtin_suite, run_suite
from oai2.evals.deterministic_hard import (
    CASE_COUNT,
    DERIVED_ANSWERS,
    deterministic_hard_suite,
)
from oai2.runtime.inference import InferenceRequest, PlaceholderRuntime
from oai2.runtime.llamacpp_runtime import (
    DEFAULT_SEED,
    LlamaServerError,
    LlamaServerRuntime,
)

BASE = "http://127.0.0.1:8851"


# --------------------------------------------------------------------------
# Oracle correctness — the most important test in this file.
# --------------------------------------------------------------------------


def test_deterministic_hard_oracles_are_correct() -> None:
    """Every expected answer must be re-derivable, not merely asserted.

    A wrong oracle reads as a model defect that does not exist. Each
    answer is recomputed here from the same facts the prompt states.
    """
    expected = {
        "days_in_a_week": "7",
        "arithmetic_multistep": str(17 * 23 - 19),
        "modular_arithmetic": str(98765 % 31),
        "count_letter_in_word": str("banana".count("a")),
        "geometric_sequence": str(2 * 16),
        "boolean_logic": str((True and False) or True),
        "unit_conversion": str(3 * 3600),
        "rate_word_problem": str(int(60 / 1.5)),
        "capital_city": "Canberra",
        "distinct_letter_count": str(len(set("orchestra"))),
        "string_reversal": "orchords"[::-1],
        "leap_year_length": "366",
    }
    assert DERIVED_ANSWERS == expected
    # Each recorded answer must actually satisfy its own case patterns.
    suite = deterministic_hard_suite()
    for case in suite.cases:
        answer = DERIVED_ANSWERS[case.case_id]
        assert any(re.search(p, answer) for p in case.expected_patterns), (
            f"{case.case_id}: derived answer {answer!r} matches none of {case.expected_patterns}"
        )


def test_oracle_answers_do_not_satisfy_the_wrong_case() -> None:
    """A case's own answer must not pass a different case's oracle.

    Without this, two loosely-worded patterns could both accept one
    answer and the score would silently measure the pattern set rather
    than the model's reasoning.
    """
    suite = deterministic_hard_suite()
    for case in suite.cases:
        for other in suite.cases:
            if other.case_id == case.case_id:
                continue
            answer = DERIVED_ANSWERS[other.case_id]
            # Short shared answers may legitimately overlap; the tight
            # (multi-character, distinctive) answers must not.
            if len(answer) < 4:
                continue
            assert not any(re.search(p, answer) for p in case.expected_patterns), (
                f"{other.case_id}'s answer {answer!r} also satisfies {case.case_id}'s oracle"
            )


def test_suite_has_exactly_twelve_cases_with_distinct_ids() -> None:
    suite = deterministic_hard_suite()
    assert len(suite.cases) == CASE_COUNT == 12
    ids = [c.case_id for c in suite.cases]
    assert len(set(ids)) == len(ids)
    for case in suite.cases:
        assert case.prompt.strip()
        assert case.expected_patterns
        assert case.notes


def test_suite_is_registered_with_the_canonical_harness() -> None:
    assert "deterministic_hard" in BUILTIN_SUITES_NAMES
    assert builtin_suite("deterministic_hard").suite_id == "deterministic_hard_12"


def test_every_case_scores_its_own_answer_and_rejects_wrong_ones() -> None:
    """The oracle discriminates: own answer passes, a wrong one fails."""
    suite = deterministic_hard_suite()
    for case in suite.cases:
        correct = run_suite_for_text(suite, case, DERIVED_ANSWERS[case.case_id])
        assert correct.n_passed == 1, f"{case.case_id} rejected its own answer"
        wrong = run_suite_for_text(suite, case, "the answer is definitely not that")
        assert wrong.n_passed == 0, f"{case.case_id} accepted a non-answer"


def run_suite_for_text(suite, case, text):  # noqa: ANN001, ANN202
    """Run a one-case suite against a runtime that always returns ``text``."""
    from oai2.runtime.inference import InferenceResponse

    only_one = suite.model_copy(update={"cases": [case]})

    class Fixed(PlaceholderRuntime):
        def generate(self, request: InferenceRequest) -> InferenceResponse:  # noqa: D102
            return InferenceResponse(text=text, tokens=1, elapsed_ms=0.0, device="t")

    return run_suite(only_one, Fixed())


# --------------------------------------------------------------------------
# LlamaServerRuntime
# --------------------------------------------------------------------------


def _props(**overrides: object) -> dict[str, object]:
    props = {
        "model_path": "/Users/orchords/models/normal/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "total_slots": 4,
        "default_generation_settings": {"n_ctx": 8192, "n_parallel": 4},
    }
    props.update(overrides)
    return props


def _completion(content: str) -> httpx.Response:
    body = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"completion_tokens": 5},
    }
    return httpx.Response(200, json=body)


def test_identity_is_read_from_props_not_assumed() -> None:
    """A correctness claim is a claim about specific weights.

    Deriving identity from a CLI flag would record intent, not what is
    resident, so the two could disagree silently.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/props"
        return httpx.Response(200, json=_props())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runtime = LlamaServerRuntime(base_url=BASE, model="alias", client=client)
    identity = runtime.serving_identity()
    assert identity["model_path"].endswith("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    assert identity["model_basename"] == "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
    assert identity["total_slots"] == 4
    assert identity["n_ctx"] == 8192


def test_unreachable_props_raises_rather_than_reporting_an_empty_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runtime = LlamaServerRuntime(base_url=BASE, client=client)
    with pytest.raises(LlamaServerError):
        runtime.serving_identity()


def test_generation_returns_content_and_never_the_reasoning_trace() -> None:
    """A reasoning trace must not be scored as the model's conclusion."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Canberra",
                        "reasoning_content": "Let me think about Australia... Canberra",
                    },
                }
            ],
            "usage": {"completion_tokens": 3},
        }
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runtime = LlamaServerRuntime(base_url=BASE, client=client)
    response = runtime.generate(InferenceRequest(prompt="Capital of Australia?"))
    assert response.text == "Canberra"
    assert "Let me think" not in response.text


def test_seed_is_only_sent_when_the_request_sets_one() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _completion("ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runtime = LlamaServerRuntime(base_url=BASE, client=client)
    runtime.generate(InferenceRequest(prompt="a"))
    runtime.generate(InferenceRequest(prompt="b", seed=DEFAULT_SEED))
    assert "seed" not in seen[0]
    assert seen[1]["seed"] == DEFAULT_SEED


def test_temperature_and_seed_reach_the_wire_so_runs_are_reproducible() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _completion("ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runtime = LlamaServerRuntime(base_url=BASE, client=client)
    runtime.generate(InferenceRequest(prompt="p", temperature=0.0, top_p=1.0, seed=7))
    assert seen[0]["temperature"] == 0.0
    assert seen[0]["top_p"] == 1.0
    assert seen[0]["seed"] == 7


def test_non_2xx_completion_raises_with_the_upstream_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runtime = LlamaServerRuntime(base_url=BASE, client=client)
    with pytest.raises(LlamaServerError, match="503"):
        runtime.generate(InferenceRequest(prompt="p"))


def test_the_placeholder_runtime_cannot_pass_the_hard_suite() -> None:
    """Guards against the suite being trivially satisfiable.

    The placeholder is the harness's offline stand-in. If it scored well
    on a 'deterministic hard' suite, the oracle would be measuring
    nothing.
    """
    report = run_suite(deterministic_hard_suite(), PlaceholderRuntime())
    assert report.n_passed < CASE_COUNT


# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------


def test_failure_taxonomy_separates_defects_that_a_bare_score_hides() -> None:
    """A 0/12 and a 12/12-all-wrong-the-same-way are different systems."""
    from scripts.accuracy_baseline import classify_failure

    # Right answer that the regex missed is an oracle gap, not a model error.
    assert (
        classify_failure("string_reversal", "sdrohcro", "sdrohcro extra prose")
        == "correct_but_unmatched"
    )
    # A numeric answer that is simply the wrong number.
    assert classify_failure("arithmetic_multistep", "372", "401") == "arithmetic_error"
    # Wrong operator handling.
    assert classify_failure("boolean_logic", "True", "False") == "logic_error"
    # A confidently wrong fact.
    assert classify_failure("capital_city", "Canberra", "Sydney") == "fact_error"
    # A right-permutation-wrong-order reversal.
    assert classify_failure("string_reversal", "sdrohcro", "orchords") == "wrong_permutation"
    # Emitted a character that does not exist in the input at all.
    assert classify_failure("string_reversal", "sdrohcro", "'drocorb'") == "symbolic_corruption"
    assert classify_failure("days_in_a_week", "7", "   ") == "empty_response"


def test_failure_taxonomy_flags_a_non_numeric_answer_separately() -> None:
    """Refusing to produce a number is a different defect from a wrong number."""
    from scripts.accuracy_baseline import classify_failure

    assert classify_failure("arithmetic_multistep", "372", "I cannot compute that") == (
        "non_numeric_answer"
    )
