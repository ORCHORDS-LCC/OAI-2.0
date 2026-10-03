"""``deterministic_hard`` — 12 cases with objectively checkable answers.

Why this suite exists
---------------------

Issue #240 carried an accuracy figure of ``0/12`` that could not be
reproduced or refuted: it was cited against
``evidence/latency-20261003/summary.json``, a path that has never existed
in this repository. A correctness number that no one can re-derive is not
evidence, and the audit was right to reject it.

This module supplies the missing half — a fixed task set whose answers
are decidable without a model, a human grader, or an LLM judge.

Oracle properties
-----------------

- **Deterministic.** Every expected answer is a fixed fact or a pure
  function of the prompt. Two runs at the same SHA must score identically.
- **Decidable by regex.** Scoring uses the harness's existing
  ``regex_or`` scorer. No model grades the model, so the oracle cannot
  drift or be talked into agreeing.
- **Independently checked.** Each answer below was computed, not
  recalled, and ``test_deterministic_hard_oracles_are_correct`` re-derives
  every one of them at test time. A wrong expected answer would otherwise
  read as a model defect that does not exist.
- **Deliberately hard.** These are exact-answer tasks that require
  multi-step arithmetic, enumeration or recall, chosen to be difficult
  for a small local model rather than easy. An easy suite would report a
  flattering number and misinform the gate.

Cases accept the numeral or, where noted in ``notes``, the English word
form. Refusing to count "seven" as a correct answer to a question about
how many days are in a week would measure format compliance instead of
correctness, which is its own kind of bad evidence.

Import note
-----------

``CapabilityCase`` / ``CapabilitySuite`` live in this package's
``__init__``, which imports this module to register the suite, so the case
data below is plain tuples and the model objects are built inside
:func:`deterministic_hard_suite`. The sibling submodules
(``regression``, ``truth``, ``truth_runner``) have no such cycle because
they define their own types instead of importing the package.
"""

from __future__ import annotations

#: ``(case_id, prompt, expected_patterns, notes)``. Prompt text is fixed,
#: and each prompt states the unit or format it wants so a failure
#: reflects reasoning rather than guessing which representation the grader
#: expected.
_CASE_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "days_in_a_week",
        "How many days are there in one week? Answer with a number.",
        (r"\b7\b", r"\bseven\b"),
        "accepts numeral or word; answer 7",
    ),
    (
        "arithmetic_multistep",
        "Compute 17 * 23 - 19. Answer with a number only.",
        (r"\b372\b",),
        "answer 372; requires multiply-then-subtract without reordering",
    ),
    (
        "modular_arithmetic",
        "What is the remainder when 98765 is divided by 31? Answer with a number only.",
        (r"\b30\b",),
        "answer 30; 5-digit division, not a memorised fact",
    ),
    (
        "count_letter_in_word",
        "How many times does the letter 'a' appear in the word 'banana'? "
        "Answer with a number only.",
        (r"\b3\b", r"\bthree\b"),
        "accepts numeral or word; answer 3",
    ),
    (
        "geometric_sequence",
        "What number comes next in this sequence: 2, 4, 8, 16, ...? Answer with a number only.",
        (r"\b32\b",),
        "answer 32; the term after the last one shown, not the last one",
    ),
    (
        "boolean_logic",
        "Evaluate: (True AND False) OR True. Answer with True or False only.",
        # Case-insensitive: the prompt shows "True", but a model may emit
        # "true", and scoring capitalisation would measure formatting.
        (r"(?i)\btrue\b",),
        "answer True; requires honouring the OR after the AND collapses to False",
    ),
    (
        "unit_conversion",
        "How many seconds are there in 3 hours? Answer with a number only.",
        (r"\b10,?800\b",),
        "answer 10800; comma tolerated, magnitude must be right",
    ),
    (
        "rate_word_problem",
        "A train travels 60 kilometres in 1.5 hours at constant speed. "
        "What is its average speed in kilometres per hour? Answer with a number only.",
        (r"\b40\b",),
        "answer 40; requires dividing, not multiplying",
    ),
    (
        "capital_city",
        "What is the capital city of Australia? Answer with the name only.",
        # Case-insensitive: a model may answer "Canberra" or "canberra",
        # and both are correct.
        (r"(?i)\bcanberra\b",),
        "answer Canberra; long-tail fact with a single unambiguous answer",
    ),
    (
        "distinct_letter_count",
        "How many distinct letters appear in the word 'orchestra'? Answer with a number only.",
        (r"\b8\b", r"\beight\b"),
        "accepts numeral or word; answer 8 (o,r,c,h,e,s,t,a); confusable with the 9 letters typed",
    ),
    (
        "string_reversal",
        "Reverse the string 'orchords'. Output only the reversed string.",
        (r"\bsdrohcro\b",),
        "answer sdrohcro; character-exact, so no normalisation is possible",
    ),
    (
        "leap_year_length",
        "How many days are there in a leap year? Answer with a number only.",
        (r"\b366\b",),
        "answer 366; confusable with 365, the common-year answer",
    ),
)

#: Independently computed answers. The test suite re-derives these rather
#: than trusting the regexes, so an edited oracle cannot silently pass.
DERIVED_ANSWERS: dict[str, str] = {
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

#: The size the ``0/12`` claim referred to, made explicit so it is never
#: conflated with a differently-sized suite.
CASE_COUNT = 12


def deterministic_hard_suite():  # noqa: ANN201 - resolves the package types lazily
    """The canonical 12-case deterministic-hard accuracy suite."""
    from . import CapabilityCase, CapabilitySuite

    return CapabilitySuite(
        suite_id="deterministic_hard_12",
        capability="deterministic_hard",
        description=(
            "Twelve fixed tasks with machine-checkable answers, used to "
            "re-baseline accuracy at a pinned SHA. The oracle is a regex "
            "over the response text; no model or human grades the output."
        ),
        cases=[
            CapabilityCase(
                case_id=case_id,
                capability="deterministic_hard",
                prompt=prompt,
                expected_patterns=patterns,
                notes=notes,
            )
            for case_id, prompt, patterns, notes in _CASE_SPECS
        ],
    )


__all__ = ["CASE_COUNT", "DERIVED_ANSWERS", "deterministic_hard_suite"]
