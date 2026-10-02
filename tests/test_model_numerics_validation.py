"""Outer guard rails for oai2/model/numerics.py.

Pin the public-safety boundary of the backend-neutral numerical-safety
policy module: ``NUMERICAL_POLICY_VERSION`` wire contract, the
``NumericalOperation`` six-member StrEnum, the ``PrecisionRule`` strict
non-empty-string contract (rejects bool via the ``isinstance(x, str)``
check — load-bearing because ``PrecisionRule.__post_init__`` is the
gate that prevents a bad reference rule from poisoning every downstream
numerical-check call), the ``NumericalFailure`` keyword-only constructor (the
structured failure metadata is what the audit trail consumes), and the
``check_numerics`` validation matrix (non-finite / out-of-range /
non-numeric / bool rejection paths).

Companion to ``tests/test_numerics.py`` (which exercises the happy-path
non-finite / range / coverage / package-reexport flows). This file
pins the wire-contract surface so silent breakage cannot slip through:

* Module docstring pin (deployment-neutral, no platform refs).
* UP006-clean imports.
* ``__all__`` (9 names) + package-level re-export identity.
* ``NUMERICAL_POLICY_VERSION`` literal pinning.
* ``NumericalOperation`` StrEnum value-set + membership + repr + .value pinning.
* ``REFERENCE_PRECISION_RULES`` covers every ``NumericalOperation`` + no
  extraneous operations + ``PrecisionRule`` field triple non-empty-string.
* ``PrecisionRule`` dataclass shape — 3 fields, ``slots=True``,
  ``frozen=True``, attribute-mutation rejection, keyword-only
  constructor (the load-bearing ``__post_init__`` accepts the three
  dtype strings positionally OR by keyword and validates all three).
* ``PrecisionRule.__post_init__`` rejection matrix — non-string dtype
  for any of the three fields (parametrized × bool / int / None /
  list / dict — the bool rejection is load-bearing via the
  ``isinstance(value, str)`` check; ``isinstance(True, str)`` is False,
  so a refactor that used ``isinstance(value, (str, int))`` would
  silently accept ``True`` as a valid dtype), empty string (after
  ``.strip()`` — the .strip() means a whitespace-only string is
  rejected), whitespace-only string.
* ``NumericalCheckResult`` dataclass shape — 6 fields, ``slots=True``,
  ``frozen=True``, attribute-mutation rejection.
* ``NumericalFailure`` structured metadata — keyword-only constructor
  signature (the 4 kwargs are ``operation`` / ``stage`` / ``reason`` /
  ``index=None`` — positional callers would silently mis-bind the
  metadata); attribute access for all 4 fields; ``__str__`` with-index
  suffix vs without-index suffix; subclass-of-RuntimeError guarantee.
* ``precision_rule(operation)`` lookup + KeyError on unknown operation.
* ``check_numerics`` signature — keyword-only kwargs after ``values``.
* ``check_numerics`` validation matrix — ``stage`` non-empty-string
  rejection (parametrized × None / int / list / dict / whitespace);
  ``max_abs`` finite + positive rejection (parametrized × bool /
  None / NaN / +inf / -inf / 0 / negative int / negative float — the
  ``isinstance(max_abs, bool)`` check is load-bearing because
  ``isinstance(True, int) is True`` so a refactor that used
  ``isinstance(max_abs, (int, float))`` would silently accept
  ``max_abs=True`` as ``max_abs=1.0``); ``values`` non-numeric rejection
  (parametrized × str / None / list / dict / object() — bool rejection
  via the ``isinstance(raw, bool)`` check is load-bearing for the same
  reason).
* ``check_numerics`` happy path — finite-only output peaks; correct
  ``max_abs_observed`` for finite-only input; correct
  ``checked_count`` for the input length; ``enabled=False`` returns a
  sentinel without reading values.
* ``extreme_value_fixtures()`` returns ``dict[NumericalOperation, tuple[float, ...]]``
  covering all 6 operations; each tuple is finite (no NaN / ±inf leak
  into the reference fixtures — deliberate injection is rejected on the
  non-finite check).
* Package-level identity check — every name in ``__all__`` re-exports
  by identity at the package level.
"""

from __future__ import annotations

import inspect
import math

import pytest

from oai2.model import (
    NUMERICAL_POLICY_VERSION,
    REFERENCE_PRECISION_RULES,
    NumericalCheckResult,
    NumericalFailure,
    NumericalOperation,
    PrecisionRule,
    check_numerics,
    extreme_value_fixtures,
    precision_rule,
)
from oai2.model import numerics as source_mod
from oai2.model.numerics import (
    NumericalFailure as SourceNumericalFailure,
)
from oai2.model.numerics import (
    PrecisionRule as SourcePrecisionRule,
)

# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_module_docstring_declares_versioned_policy_and_backend_neutral() -> None:
    """The module docstring must declare versioned-policy + backend-neutral."""
    doc = source_mod.__doc__ or ""
    assert "versioned" in doc.lower() or "version" in doc.lower()
    assert "backend-neutral" in doc.lower() or "backend neutral" in doc.lower()


def test_module_source_does_not_import_cloud_runtime() -> None:
    """The module must not import cloud runtime / Cloudflare / API modules."""
    source = inspect.getsource(source_mod)
    forbidden = [
        "from oai2.knowledge.cloudflare",
        "import oai2.knowledge.cloudflare",
        "from oai2.gateway",
        "import oai2.gateway",
        "requests.",
        "httpx.",
        "urllib.",
    ]
    for pattern in forbidden:
        assert pattern not in source, f"forbidden pattern {pattern!r} found in module source"


def test_module_source_does_not_contain_hardcoded_secrets() -> None:
    """The module source must not contain hardcoded credentials."""
    source = inspect.getsource(source_mod)
    forbidden = ["api_key=", "token=", "password=", "secret="]
    for pattern in forbidden:
        assert pattern not in source, f"forbidden pattern {pattern!r} found in module source"


def test_module_uses_no_runtime_install_path_for_typing_collections() -> None:
    """The module must NOT use ``typing.Mapping`` / ``typing.Sequence`` for runtime use."""
    source = inspect.getsource(source_mod)
    assert "from typing import Mapping" not in source
    assert "from typing import Sequence" not in source
    assert "typing.Mapping" not in source
    assert "typing.Sequence" not in source


# ---------------------------------------------------------------------------
# 2. __all__ completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_all_has_exactly_nine_names() -> None:
    """``__all__`` must list exactly 9 public names."""
    assert sorted(source_mod.__all__) == sorted(
        [
            "NUMERICAL_POLICY_VERSION",
            "NumericalOperation",
            "PrecisionRule",
            "REFERENCE_PRECISION_RULES",
            "NumericalCheckResult",
            "NumericalFailure",
            "precision_rule",
            "check_numerics",
            "extreme_value_fixtures",
        ]
    )


def test_all_names_are_importable_from_module() -> None:
    """Every name in ``__all__`` must be importable from the module."""
    for name in source_mod.__all__:
        obj = getattr(source_mod, name)
        assert obj is not None


def test_package_level_reexport_identity() -> None:
    """``oai2.model`` must re-export the numerics names by identity."""
    import oai2.model as pkg

    assert pkg.NUMERICAL_POLICY_VERSION is source_mod.NUMERICAL_POLICY_VERSION
    assert pkg.NumericalOperation is source_mod.NumericalOperation
    assert pkg.PrecisionRule is source_mod.PrecisionRule
    assert pkg.NumericalCheckResult is source_mod.NumericalCheckResult
    assert pkg.NumericalFailure is source_mod.NumericalFailure
    assert pkg.check_numerics is source_mod.check_numerics
    assert pkg.extreme_value_fixtures is source_mod.extreme_value_fixtures
    assert pkg.precision_rule is source_mod.precision_rule
    assert pkg.REFERENCE_PRECISION_RULES is source_mod.REFERENCE_PRECISION_RULES


# ---------------------------------------------------------------------------
# 3. NUMERICAL_POLICY_VERSION literal pinning
# ---------------------------------------------------------------------------


def test_numerical_policy_version_is_string_one() -> None:
    """``NUMERICAL_POLICY_VERSION`` must be the literal string ``"1"``."""
    assert NUMERICAL_POLICY_VERSION == "1"
    assert isinstance(NUMERICAL_POLICY_VERSION, str)


# ---------------------------------------------------------------------------
# 4. NumericalOperation StrEnum
# ---------------------------------------------------------------------------


def test_numerical_operation_strenum_value_set_is_pinned() -> None:
    """``NumericalOperation`` must declare exactly the 6 documented wire strings."""
    expected = {
        "attention_softmax",
        "normalization",
        "router_probabilities",
        "loss",
        "quantize_dequantize",
        "long_context_reduction",
    }
    actual = {op.value for op in NumericalOperation}
    assert actual == expected


def test_numerical_operation_member_count_is_six() -> None:
    """``NumericalOperation`` must have exactly 6 members."""
    assert len(list(NumericalOperation)) == 6


def test_numerical_operation_is_strenum_subclass() -> None:
    """``NumericalOperation`` must be a ``StrEnum`` subclass so values compare as strings."""
    from enum import StrEnum

    assert issubclass(NumericalOperation, StrEnum)
    # Wire contract: ``operation.value`` is the literal string
    assert NumericalOperation.ATTENTION_SOFTMAX.value == "attention_softmax"


def test_numerical_operation_lookup_by_value_works() -> None:
    """``NumericalOperation(value)`` must round-trip every wire string."""
    for op in NumericalOperation:
        roundtripped = NumericalOperation(op.value)
        assert roundtripped is op


# ---------------------------------------------------------------------------
# 5. PrecisionRule dataclass shape + validation
# ---------------------------------------------------------------------------


def test_precision_rule_field_set_is_complete() -> None:
    """``PrecisionRule`` must declare exactly 3 fields."""
    expected = {"compute_dtype", "storage_dtype", "accumulation_dtype"}
    actual = set(PrecisionRule.__dataclass_fields__.keys())
    assert actual == expected


def test_precision_rule_is_slots_and_frozen() -> None:
    """``PrecisionRule`` must be ``slots=True`` + ``frozen=True``."""
    params = PrecisionRule.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_precision_rule_rejects_attribute_mutation() -> None:
    """``PrecisionRule`` must reject attribute mutation."""
    rule = PrecisionRule("fp32", "bf16", "fp32")
    with pytest.raises((AttributeError, Exception)):
        rule.compute_dtype = "fp64"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_value",
    [True, False, 0, 1, 1.0, None, [], {"k": 1}, b"bytes"],
)
def test_precision_rule_rejects_non_string_compute_dtype(bad_value: object) -> None:
    """``PrecisionRule`` must reject non-string ``compute_dtype`` (the bool check is load-bearing)."""
    with pytest.raises(ValueError, match="compute_dtype must be a non-empty string"):
        PrecisionRule(bad_value, "bf16", "fp32")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value",
    [True, False, 0, 1, 1.0, None, [], {"k": 1}, b"bytes"],
)
def test_precision_rule_rejects_non_string_storage_dtype(bad_value: object) -> None:
    """``PrecisionRule`` must reject non-string ``storage_dtype``."""
    with pytest.raises(ValueError, match="storage_dtype must be a non-empty string"):
        PrecisionRule("fp32", bad_value, "fp32")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value",
    [True, False, 0, 1, 1.0, None, [], {"k": 1}, b"bytes"],
)
def test_precision_rule_rejects_non_string_accumulation_dtype(bad_value: object) -> None:
    """``PrecisionRule`` must reject non-string ``accumulation_dtype``."""
    with pytest.raises(ValueError, match="accumulation_dtype must be a non-empty string"):
        PrecisionRule("fp32", "bf16", bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_dtype", ["", "   ", "\t", "\n", " \t\n "])
def test_precision_rule_rejects_empty_or_whitespace_compute_dtype(
    bad_dtype: str,
) -> None:
    """``PrecisionRule`` must reject empty / whitespace-only ``compute_dtype``."""
    with pytest.raises(ValueError, match="compute_dtype must be a non-empty string"):
        PrecisionRule(bad_dtype, "bf16", "fp32")


@pytest.mark.parametrize("bad_dtype", ["", "   ", "\t", "\n", " \t\n "])
def test_precision_rule_rejects_empty_or_whitespace_storage_dtype(
    bad_dtype: str,
) -> None:
    """``PrecisionRule`` must reject empty / whitespace-only ``storage_dtype``."""
    with pytest.raises(ValueError, match="storage_dtype must be a non-empty string"):
        PrecisionRule("fp32", bad_dtype, "fp32")


@pytest.mark.parametrize("bad_dtype", ["", "   ", "\t", "\n", " \t\n "])
def test_precision_rule_rejects_empty_or_whitespace_accumulation_dtype(
    bad_dtype: str,
) -> None:
    """``PrecisionRule`` must reject empty / whitespace-only ``accumulation_dtype``."""
    with pytest.raises(ValueError, match="accumulation_dtype must be a non-empty string"):
        PrecisionRule("fp32", "bf16", bad_dtype)


def test_precision_rule_accepts_keyword_only_construction() -> None:
    """``PrecisionRule`` must accept the three dtypes as keyword arguments."""
    rule = PrecisionRule(
        compute_dtype="fp32",
        storage_dtype="bf16",
        accumulation_dtype="fp32",
    )
    assert rule.compute_dtype == "fp32"
    assert rule.storage_dtype == "bf16"
    assert rule.accumulation_dtype == "fp32"


def test_precision_rule_validates_all_three_fields_even_when_first_valid() -> None:
    """``PrecisionRule.__post_init__`` must validate ALL three fields, not short-circuit on the first."""
    with pytest.raises(ValueError, match="storage_dtype"):
        PrecisionRule("fp32", "", "fp32")
    with pytest.raises(ValueError, match="accumulation_dtype"):
        PrecisionRule("fp32", "bf16", "")


# ---------------------------------------------------------------------------
# 6. REFERENCE_PRECISION_RULES coverage + identity
# ---------------------------------------------------------------------------


def test_reference_precision_rules_cover_every_numerical_operation() -> None:
    """``REFERENCE_PRECISION_RULES`` must declare a rule for every ``NumericalOperation``."""
    assert set(REFERENCE_PRECISION_RULES.keys()) == set(NumericalOperation)


def test_reference_precision_rules_have_no_extraneous_operations() -> None:
    """``REFERENCE_PRECISION_RULES`` must not declare rules for any non-enum operation."""
    # If the rule table grows beyond the StrEnum, downstream code that
    # iterates the StrEnum will silently skip the new rule.
    assert len(REFERENCE_PRECISION_RULES) == len(NumericalOperation)


def test_reference_precision_rules_values_are_precision_rule_instances() -> None:
    """Every value in ``REFERENCE_PRECISION_RULES`` must be a ``PrecisionRule``."""
    for operation, rule in REFERENCE_PRECISION_RULES.items():
        assert isinstance(rule, PrecisionRule), (
            f"rule for {operation} is not a PrecisionRule instance"
        )


def test_reference_precision_rules_all_have_non_empty_dtypes() -> None:
    """Every rule in ``REFERENCE_PRECISION_RULES`` must declare non-empty dtypes."""
    for operation, rule in REFERENCE_PRECISION_RULES.items():
        assert rule.compute_dtype.strip(), f"{operation}.compute_dtype is empty"
        assert rule.storage_dtype.strip(), f"{operation}.storage_dtype is empty"
        assert rule.accumulation_dtype.strip(), f"{operation}.accumulation_dtype is empty"


# ---------------------------------------------------------------------------
# 7. NumericalCheckResult dataclass shape
# ---------------------------------------------------------------------------


def test_numerical_check_result_field_set_is_complete() -> None:
    """``NumericalCheckResult`` must declare exactly 6 fields."""
    expected = {
        "policy_version",
        "operation",
        "stage",
        "checked_count",
        "max_abs_observed",
        "enabled",
    }
    actual = set(NumericalCheckResult.__dataclass_fields__.keys())
    assert actual == expected


def test_numerical_check_result_is_slots_and_frozen() -> None:
    """``NumericalCheckResult`` must be ``slots=True`` + ``frozen=True``."""
    params = NumericalCheckResult.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_numerical_check_result_rejects_attribute_mutation() -> None:
    """``NumericalCheckResult`` must reject attribute mutation."""
    result = NumericalCheckResult(
        policy_version="1",
        operation=NumericalOperation.ATTENTION_SOFTMAX,
        stage="test",
        checked_count=0,
        max_abs_observed=0.0,
        enabled=True,
    )
    with pytest.raises((AttributeError, Exception)):
        result.stage = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. NumericalFailure structured metadata
# ---------------------------------------------------------------------------


def test_numerical_failure_is_runtime_error_subclass() -> None:
    """``NumericalFailure`` must be a ``RuntimeError`` subclass."""
    assert issubclass(NumericalFailure, RuntimeError)


def test_numerical_failure_signature_is_keyword_only_after_self() -> None:
    """``NumericalFailure.__init__`` must be keyword-only after ``self``."""
    sig = inspect.signature(SourceNumericalFailure.__init__)
    params = sig.parameters
    expected = {"self", "operation", "stage", "reason", "index"}
    assert set(params.keys()) == expected
    for name, p in params.items():
        if name == "self":
            continue
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name} is {p.kind}, expected KEYWORD_ONLY"
        )


def test_numerical_failure_without_index_omits_suffix() -> None:
    """``NumericalFailure`` without ``index`` must omit the ``at index`` suffix."""
    failure = NumericalFailure(
        operation=NumericalOperation.LOSS,
        stage="train.loss",
        reason="non_finite_value",
    )
    assert failure.operation is NumericalOperation.LOSS
    assert failure.stage == "train.loss"
    assert failure.reason == "non_finite_value"
    assert failure.index is None
    assert str(failure) == "loss/train.loss: non_finite_value"
    assert "at index" not in str(failure)


def test_numerical_failure_with_index_appends_suffix() -> None:
    """``NumericalFailure`` with ``index`` must append ``at index {N}``."""
    failure = NumericalFailure(
        operation=NumericalOperation.ATTENTION_SOFTMAX,
        stage="decode.attention",
        reason="range_exceeded",
        index=42,
    )
    assert failure.index == 42
    assert str(failure) == "attention_softmax/decode.attention: range_exceeded at index 42"


def test_numerical_failure_attribute_access() -> None:
    """``NumericalFailure`` attributes must be readable post-construction."""
    failure = NumericalFailure(
        operation=NumericalOperation.NORMALIZATION,
        stage="decode.norm",
        reason="non_numeric_value",
        index=7,
    )
    assert failure.operation is NumericalOperation.NORMALIZATION
    assert failure.stage == "decode.norm"
    assert failure.reason == "non_numeric_value"
    assert failure.index == 7


def test_numerical_failure_message_does_not_leak_raw_tensor_values() -> None:
    """The ``__str__`` of ``NumericalFailure`` must not contain any ``[`` / ``]`` (raw tensor leak)."""
    failure = NumericalFailure(
        operation=NumericalOperation.LOSS,
        stage="train.loss",
        reason="non_finite_value",
        index=3,
    )
    msg = str(failure)
    assert "[" not in msg
    assert "]" not in msg


# ---------------------------------------------------------------------------
# 9. precision_rule lookup
# ---------------------------------------------------------------------------


def test_precision_rule_returns_table_entry() -> None:
    """``precision_rule(operation)`` must return the matching table rule."""
    for operation in NumericalOperation:
        rule = precision_rule(operation)
        assert rule is REFERENCE_PRECISION_RULES[operation]


def test_precision_rule_raises_key_error_for_unknown_operation() -> None:
    """``precision_rule`` must raise ``KeyError`` for an unknown operation."""
    with pytest.raises(KeyError):
        precision_rule("not_a_real_operation")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. check_numerics signature
# ---------------------------------------------------------------------------


def test_check_numerics_signature_is_keyword_only_after_values() -> None:
    """``check_numerics`` must be keyword-only after ``values``."""
    sig = inspect.signature(check_numerics)
    params = sig.parameters
    expected = {"values", "operation", "stage", "max_abs", "enabled"}
    assert set(params.keys()) == expected
    for name, p in params.items():
        if name == "values":
            continue
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name} is {p.kind}, expected KEYWORD_ONLY"
        )


def test_check_numerics_max_abs_default_is_none() -> None:
    """``check_numerics`` must default ``max_abs`` to ``None`` (sentinel for unbounded)."""
    sig = inspect.signature(check_numerics)
    assert sig.parameters["max_abs"].default is None


def test_check_numerics_enabled_default_is_true() -> None:
    """``check_numerics`` must default ``enabled`` to ``True``."""
    sig = inspect.signature(check_numerics)
    assert sig.parameters["enabled"].default is True


# ---------------------------------------------------------------------------
# 11. check_numerics validation matrix — stage (non-empty-string)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_stage", [None, 0, 1, 1.0, True, False, [], {"k": 1}])
def test_check_numerics_rejects_non_string_stage(bad_stage: object) -> None:
    """``check_numerics`` must reject non-string ``stage``."""
    with pytest.raises(ValueError, match="stage must be a non-empty string"):
        check_numerics([0.0], operation=NumericalOperation.LOSS, stage=bad_stage)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_stage", ["", "   ", "\t", "\n", " \t\n "])
def test_check_numerics_rejects_empty_or_whitespace_stage(bad_stage: str) -> None:
    """``check_numerics`` must reject empty / whitespace-only ``stage``."""
    with pytest.raises(ValueError, match="stage must be a non-empty string"):
        check_numerics([0.0], operation=NumericalOperation.LOSS, stage=bad_stage)


# ---------------------------------------------------------------------------
# 12. check_numerics validation matrix — max_abs (finite + positive)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_max_abs",
    [True, False, float("nan"), float("inf"), float("-inf"), 0, 0.0, -1, -1.0, -1e9],
)
def test_check_numerics_rejects_invalid_max_abs(bad_max_abs: object) -> None:
    """``check_numerics`` must reject invalid ``max_abs`` (the bool check is load-bearing)."""
    with pytest.raises(ValueError, match="max_abs must be finite and > 0"):
        check_numerics(
            [0.0],
            operation=NumericalOperation.LOSS,
            stage="train.loss",
            max_abs=bad_max_abs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("good_max_abs", [1, 1.0, 1e9, 1.5e9])
def test_check_numerics_accepts_positive_finite_max_abs(good_max_abs: float) -> None:
    """``check_numerics`` must accept any positive finite ``max_abs``."""
    result = check_numerics(
        [0.0, 0.5, 0.7],
        operation=NumericalOperation.LOSS,
        stage="train.loss",
        max_abs=good_max_abs,
    )
    assert result.enabled is True
    assert result.checked_count == 3


# ---------------------------------------------------------------------------
# 13. check_numerics validation matrix — values (non-numeric + bool rejection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    ["0.0", None, True, False, [], {"k": 1}, b"bytes", object()],
)
def test_check_numerics_rejects_non_numeric_values(bad_value: object) -> None:
    """``check_numerics`` must reject non-numeric values (the bool rejection is load-bearing)."""
    with pytest.raises(NumericalFailure) as exc_info:
        check_numerics(
            [0.0, bad_value, 1.0],  # type: ignore[list-item]
            operation=NumericalOperation.LOSS,
            stage="train.loss",
        )
    failure = exc_info.value
    assert failure.reason == "non_numeric_value"
    assert failure.index == 1


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_check_numerics_rejects_non_finite_values(bad_value: float) -> None:
    """``check_numerics`` must reject NaN / ±inf values."""
    with pytest.raises(NumericalFailure) as exc_info:
        check_numerics(
            [0.0, bad_value, 1.0],
            operation=NumericalOperation.LOSS,
            stage="train.loss",
        )
    failure = exc_info.value
    assert failure.reason == "non_finite_value"
    assert failure.index == 1


@pytest.mark.parametrize(
    "values,peak_index",
    [
        ([0.0, 1.5, 1.0], 1),
        ([0.0, 1.0, 1.5], 2),
        ([-3.0, -1.0, -2.0], 0),
        ([1.0, 2.0, 3.0], 1),
        ([0.0, 0.0, 1.5, 0.5], 2),
    ],
)
def test_check_numerics_rejects_range_exceeded_with_correct_index(
    values: list[float],
    peak_index: int,
) -> None:
    """``check_numerics`` must reject range-exceeded and pin the FIRST violating index."""
    with pytest.raises(NumericalFailure) as exc_info:
        check_numerics(
            values,
            operation=NumericalOperation.LOSS,
            stage="train.loss",
            max_abs=1.0,
        )
    failure = exc_info.value
    assert failure.reason == "range_exceeded"
    assert failure.index == peak_index


# ---------------------------------------------------------------------------
# 14. check_numerics happy path
# ---------------------------------------------------------------------------


def test_check_numerics_returns_result_for_finite_correct_values() -> None:
    """``check_numerics`` must return a successful ``NumericalCheckResult`` for finite correct values."""
    result = check_numerics(
        [0.0, 0.5, 0.7],
        operation=NumericalOperation.ATTENTION_SOFTMAX,
        stage="decode.attention",
    )
    assert isinstance(result, NumericalCheckResult)
    assert result.policy_version == NUMERICAL_POLICY_VERSION
    assert result.operation is NumericalOperation.ATTENTION_SOFTMAX
    assert result.stage == "decode.attention"
    assert result.checked_count == 3
    assert result.max_abs_observed == pytest.approx(0.7)
    assert result.enabled is True


def test_check_numerics_returns_zero_count_when_input_empty() -> None:
    """``check_numerics`` must return ``checked_count=0`` and ``max_abs_observed=0.0`` for empty input."""
    result = check_numerics(
        [],
        operation=NumericalOperation.LOSS,
        stage="train.loss",
    )
    assert result.checked_count == 0
    assert result.max_abs_observed == 0.0
    assert result.enabled is True


def test_check_numerics_disabled_returns_sentinel_without_reading_values() -> None:
    """``check_numerics(enabled=False)`` must return the disabled sentinel without reading values."""
    result = check_numerics(
        [float("nan"), float("inf")],
        operation=NumericalOperation.NORMALIZATION,
        stage="decode.norm",
        enabled=False,
    )
    assert result.enabled is False
    assert result.checked_count == 0
    assert result.max_abs_observed == 0.0


def test_check_numerics_uses_max_abs_peak_for_max_abs_observed() -> None:
    """``check_numerics`` must set ``max_abs_observed`` to the absolute peak magnitude."""
    result = check_numerics(
        [0.1, -0.7, 0.3, 0.2],
        operation=NumericalOperation.LOSS,
        stage="train.loss",
    )
    assert result.max_abs_observed == pytest.approx(0.7)


def test_check_numerics_accepts_tuple_input() -> None:
    """``check_numerics`` must accept a tuple of floats."""
    result = check_numerics(
        (0.0, 0.5, 0.7),
        operation=NumericalOperation.ATTENTION_SOFTMAX,
        stage="decode.attention",
    )
    assert result.checked_count == 3


def test_check_numerics_accepts_list_input() -> None:
    """``check_numerics`` must accept a list of floats."""
    result = check_numerics(
        [0.0, 0.5, 0.7],
        operation=NumericalOperation.ATTENTION_SOFTMAX,
        stage="decode.attention",
    )
    assert result.checked_count == 3


# ---------------------------------------------------------------------------
# 15. extreme_value_fixtures coverage
# ---------------------------------------------------------------------------


def test_extreme_value_fixtures_returns_dict_for_every_operation() -> None:
    """``extreme_value_fixtures`` must return a rule for every ``NumericalOperation``."""
    fixtures = extreme_value_fixtures()
    assert set(fixtures.keys()) == set(NumericalOperation)


def test_extreme_value_fixtures_values_are_finite() -> None:
    """Every value in ``extreme_value_fixtures`` must be finite (no leak of NaN / ±inf)."""
    fixtures = extreme_value_fixtures()
    for operation, values in fixtures.items():
        for value in values:
            assert math.isfinite(value), f"{operation} fixture contains non-finite value: {value}"


def test_extreme_value_fixtures_values_are_tuples_of_floats() -> None:
    """Every value in ``extreme_value_fixtures`` must be a tuple of floats."""
    fixtures = extreme_value_fixtures()
    for operation, values in fixtures.items():
        assert isinstance(values, tuple), (
            f"{operation} fixture is {type(values).__name__}, expected tuple"
        )
        for value in values:
            assert isinstance(value, float), (
                f"{operation} fixture contains {type(value).__name__}, expected float"
            )


def test_extreme_value_fixtures_each_has_at_least_three_values() -> None:
    """Every operation in ``extreme_value_fixtures`` must have at least 3 finite values."""
    fixtures = extreme_value_fixtures()
    for operation, values in fixtures.items():
        assert len(values) >= 3, f"{operation} fixture has {len(values)} values, expected >= 3"


def test_extreme_value_fixtures_are_accepted_by_check_numerics() -> None:
    """Every ``extreme_value_fixtures`` entry must be accepted by ``check_numerics``."""
    fixtures = extreme_value_fixtures()
    for operation, values in fixtures.items():
        result = check_numerics(
            values,
            operation=operation,
            stage=f"fixture.{operation.value}",
        )
        assert result.enabled is True
        assert result.checked_count == len(values)
        assert math.isfinite(result.max_abs_observed)


# ---------------------------------------------------------------------------
# 16. Package-level integration sanity
# ---------------------------------------------------------------------------


def test_source_precision_rule_is_re_exported_as_oai2_precision_rule() -> None:
    """``oai2.model.PrecisionRule`` must be the same class as ``oai2.model.numerics.PrecisionRule``."""
    assert SourcePrecisionRule is PrecisionRule
    assert PrecisionRule.__module__ == "oai2.model.numerics"


def test_source_numerical_failure_is_re_exported_as_oai2_numerical_failure() -> None:
    """``oai2.model.NumericalFailure`` must be the same class as ``oai2.model.numerics.NumericalFailure``."""
    assert SourceNumericalFailure is NumericalFailure
    assert NumericalFailure.__module__ == "oai2.model.numerics"
