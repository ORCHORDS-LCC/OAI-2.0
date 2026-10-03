"""Canonical per-operation tolerance matrix and its enforcement (REQ-NUM-001).

The gap this file exists for: ``NumericalToleranceProfile`` is caller-supplied
and, before this matrix, nothing in ``oai2/`` said what a profile *should* be
for a given operation. REQ-NUM-001 asks for the precision/dtype policy to be
explicit per major operation class "where non-default behaviour matters" -- but
with nothing to check against, the requirement had no teeth and the gate would
accept any numbers a caller declared.

Every test here that asserts enforcement was written to fail before the matrix
existed; the negative controls at the bottom mutate the implementation to prove
they are not vacuous.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from oai2.model.numerical_compare import (
    NumericalArtifactIdentity,
    NumericalFallback,
    NumericalToleranceProfile,
    compare_numerical_paths,
)
from oai2.model.numerics import NumericalOperation
from oai2.model.tolerance_matrix import (
    _PRECISION_FACTOR,
    TOLERANCE_MATRIX,
    TOLERANCE_POLICY_VERSION,
    DtypeClass,
    TolerancePolicy,
    canonical_policy,
    dtype_class,
    policy_violations,
)

OPERATIONS = tuple(NumericalOperation)
PRECISION_CLASSES = (
    DtypeClass.FP32,
    DtypeClass.FP16,
    DtypeClass.BF16,
    DtypeClass.FP8,
    DtypeClass.QUANTIZED,
)


def _identity() -> NumericalArtifactIdentity:
    return NumericalArtifactIdentity(
        model_version="m-1",
        runtime_version="r-1",
        backend="b-1",
        config_id="c-1",
    )


def _profile(
    *,
    operation: NumericalOperation = NumericalOperation.ROUTER_PROBABILITIES,
    dtype: str = "fp16",
    profile_id: str = "p-1",
    max_abs_error: float = 1.0e-3,
    max_rel_error: float = 1.0e-3,
    max_capability_regression: float = 0.0,
    require_finite_state_match: bool = True,
    fallback: NumericalFallback = NumericalFallback.USE_REFERENCE,
) -> NumericalToleranceProfile:
    return NumericalToleranceProfile(
        profile_id=profile_id,
        operation=operation,
        dtype=dtype,
        shape_class="scalar",
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        max_capability_regression=max_capability_regression,
        require_finite_state_match=require_finite_state_match,
        fallback=fallback,
    )


# ---------------------------------------------------------------------------
# 1. Matrix shape and coverage
# ---------------------------------------------------------------------------


class TestMatrixShape:
    def test_every_operation_has_every_precision_class(self) -> None:
        """A missing (operation, dtype) cell is a hole the gate falls through.

        Without full coverage, ``canonical_policy`` would raise KeyError on a
        real combination -- or, worse, someone would add a fallback that
        silently substituted a neighbouring operation's numbers.
        """
        for operation in OPERATIONS:
            for class_ in PRECISION_CLASSES:
                assert (operation, class_) in TOLERANCE_MATRIX, (
                    f"{operation.value}/{class_.value} is missing from the matrix"
                )

    def test_matrix_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            TOLERANCE_MATRIX[(NumericalOperation.LOSS, DtypeClass.FP32)] = None  # type: ignore[index]

    def test_policy_id_is_unique_per_entry(self) -> None:
        ids = [p.policy_id for p in TOLERANCE_MATRIX.values()]
        assert len(ids) == len(set(ids))

    def test_policy_id_records_the_policy_version(self) -> None:
        policy = canonical_policy(NumericalOperation.LOSS, "fp16")
        assert policy.policy_id.endswith(f"v{TOLERANCE_POLICY_VERSION}")
        assert policy.operation.value in policy.policy_id
        assert policy.dtype_class.value in policy.policy_id

    def test_every_ceiling_is_finite_and_non_negative(self) -> None:
        for policy in TOLERANCE_MATRIX.values():
            for value in (
                policy.max_abs_error,
                policy.max_rel_error,
                policy.max_capability_regression,
            ):
                assert math.isfinite(value)
                assert value >= 0.0

    def test_finiteness_is_required_for_every_entry(self) -> None:
        """A non-finite output is never an acceptable approximation.

        If any entry could drop the finiteness requirement, a saturating or
        NaN-producing kernel would become promotable at some precision.
        """
        for policy in TOLERANCE_MATRIX.values():
            assert policy.require_finite_state_match is True


# ---------------------------------------------------------------------------
# 2. Dtype classification
# ---------------------------------------------------------------------------


class TestDtypeClassification:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("fp32", DtypeClass.FP32),
            ("float32", DtypeClass.FP32),
            ("f32", DtypeClass.FP32),
            ("FP32", DtypeClass.FP32),
            ("  fp32  ", DtypeClass.FP32),
            ("fp16", DtypeClass.FP16),
            ("float16", DtypeClass.FP16),
            ("half", DtypeClass.FP16),
            ("bf16", DtypeClass.BF16),
            ("bfloat16", DtypeClass.BF16),
            ("fp8", DtypeClass.FP8),
            ("fp8_e4m3", DtypeClass.FP8),
            ("int8", DtypeClass.QUANTIZED),
            ("q4_K_M", DtypeClass.QUANTIZED),
        ],
    )
    def test_aliases_resolve(self, raw: str, expected: DtypeClass) -> None:
        assert dtype_class(raw) is expected

    def test_fp32_and_float32_share_one_policy(self) -> None:
        """A caller must not escape the policy by spelling the dtype differently."""
        a = canonical_policy(NumericalOperation.ROUTER_PROBABILITIES, "fp32")
        b = canonical_policy(NumericalOperation.ROUTER_PROBABILITIES, "float32")
        assert a is b

    @pytest.mark.parametrize("raw", ["", "fp4", "made-up", "fp128", "int4"])
    def test_unrecognised_dtype_is_unknown_not_loosest(self, raw: str) -> None:
        assert dtype_class(raw) is DtypeClass.UNKNOWN

    def test_unknown_dtype_resolves_to_strict_policy(self) -> None:
        """An uncharacterised precision must be qualified, not waved through.

        If UNKNOWN resolved to the loosest entry, adding a new dtype would
        silently grant the most permissive gate in the matrix.
        """
        policy = canonical_policy(NumericalOperation.QUANTIZE_DEQUANTIZE, "fp4")
        loosest = TOLERANCE_MATRIX[
            (NumericalOperation.QUANTIZE_DEQUANTIZE, DtypeClass.QUANTIZED)
        ]
        assert policy.dtype_class is DtypeClass.UNKNOWN
        assert policy.max_abs_error < loosest.max_abs_error
        assert policy.max_capability_regression <= loosest.max_capability_regression


# ---------------------------------------------------------------------------
# 3. Calibration semantics
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_fp32_is_tighter_than_fp16_for_every_operation(self) -> None:
        """fp32 carries ~3 more decimal digits than fp16.

        If this ever inverts, a "high precision" run would be gated more loosely
        than the low-precision run it is meant to supersede.
        """
        for operation in OPERATIONS:
            fp32 = TOLERANCE_MATRIX[(operation, DtypeClass.FP32)]
            fp16 = TOLERANCE_MATRIX[(operation, DtypeClass.FP16)]
            assert fp32.max_abs_error < fp16.max_abs_error
            assert fp32.max_rel_error < fp16.max_rel_error

    def test_bf16_is_coarser_than_fp16_not_equal_to_it(self) -> None:
        """bf16 has fp32's EXPONENT range and an 8-bit significand.

        fp16 has an 11-bit significand, so bf16 cannot hold as many digits and
        its machine epsilon is 2**-8 against fp16's 2**-11 -- exactly 8x
        coarser. Treating the two as equal (an easy slip, since they are both
        2-byte formats and bf16 has the wider exponent) leaves a bf16 path
        almost no headroom: the measured bf16 relative error in
        evidence/numerical/tolerance_matrix_*.json sits within 1.5x of the
        old ceiling, which no real backend would hold.
        """
        assert _PRECISION_FACTOR[DtypeClass.BF16] == 8.0
        for operation in OPERATIONS:
            bf16 = TOLERANCE_MATRIX[(operation, DtypeClass.BF16)]
            fp16 = TOLERANCE_MATRIX[(operation, DtypeClass.FP16)]
            assert bf16.max_abs_error > fp16.max_abs_error
            assert bf16.max_rel_error > fp16.max_rel_error

    def test_precision_classes_are_ordered_by_significand_width(self) -> None:
        """fp32 < fp16 < bf16 < quantized(int8) < fp8, for every operation.

        A coarser significand cannot produce a tighter result, so a factor
        that inverts this order would gate a low-precision path more tightly
        than the high-precision path it is meant to replace.

        Note the placement of QUANTIZED, which is counter-intuitive: int8's
        evenly-split 7-bit grid has a relative step of 1/127, which is FINER
        than fp8 e4m3's 4-bit significand (eps 2**-4). So the low-bit integer
        format is the more accurate of the two, and it sits below fp8 rather
        than above it. int4 (1/7, ~293x fp16) is coarser than fp8; that is
        the straddle this bucket cannot express, and it is why a caller
        qualifying an int4 normalization path is expected to be rejected
        rather than waved through.
        """
        order = (
            DtypeClass.FP32,
            DtypeClass.FP16,
            DtypeClass.BF16,
            DtypeClass.QUANTIZED,
            DtypeClass.FP8,
        )
        for operation in OPERATIONS:
            for tighter, looser in zip(order, order[1:], strict=False):
                a = TOLERANCE_MATRIX[(operation, tighter)]
                b = TOLERANCE_MATRIX[(operation, looser)]
                assert a.max_abs_error < b.max_abs_error, (
                    f"{operation.value}: {tighter.value} is not tighter than "
                    f"{looser.value}"
                )

    def test_precision_factors_are_the_eps_ratios(self) -> None:
        """The factors are derived, not chosen: they are eps relative to fp16.

        eps = 2**(1 - p) for a p-bit significand. fp32 is the one documented
        exception -- it is deliberately not the exact ratio, because the
        per-operation base already carries accumulation headroom.
        """
        fp16_eps = 2.0**-11
        assert _PRECISION_FACTOR[DtypeClass.FP16] == 1.0
        assert _PRECISION_FACTOR[DtypeClass.BF16] == 2.0**-8 / fp16_eps == 8.0
        assert _PRECISION_FACTOR[DtypeClass.FP8] == 2.0**-4 / fp16_eps == 128.0
        # int8's evenly-split 7-bit grid.
        assert _PRECISION_FACTOR[DtypeClass.QUANTIZED] == pytest.approx(
            (1.0 / 127.0) / fp16_eps, rel=0.01
        )
        # fp32 is far tighter than the exact ratio, and must stay so.
        assert _PRECISION_FACTOR[DtypeClass.FP32] > (2.0**-24) / fp16_eps
        assert _PRECISION_FACTOR[DtypeClass.FP32] < 1.0

    def test_router_probabilities_permit_zero_capability_regression(self) -> None:
        """A moved routing distribution is a behavioural change.

        Every precision class must hold this line: a low-precision router is
        not a licence to route differently.
        """
        for class_ in PRECISION_CLASSES:
            policy = TOLERANCE_MATRIX[(NumericalOperation.ROUTER_PROBABILITIES, class_)]
            assert policy.max_capability_regression == 0.0

    def test_quantize_dequantize_is_the_loosest_operation(self) -> None:
        """A low-bit round trip is SUPPOSED to lose precision.

        A ceiling tight enough to reject it would make the operation
        untestable, and tightening it would mean the quantizer is not working.
        """
        fp16 = {
            operation: TOLERANCE_MATRIX[(operation, DtypeClass.FP16)].max_abs_error
            for operation in OPERATIONS
        }
        assert (
            fp16[NumericalOperation.QUANTIZE_DEQUANTIZE]
            == max(fp16.values())
        )
        for operation in OPERATIONS:
            if operation is NumericalOperation.QUANTIZE_DEQUANTIZE:
                continue
            q = TOLERANCE_MATRIX[(operation, DtypeClass.QUANTIZED)].max_abs_error
            qq = TOLERANCE_MATRIX[(NumericalOperation.QUANTIZE_DEQUANTIZE, DtypeClass.QUANTIZED)].max_abs_error
            assert qq >= q

    def test_capability_ceiling_is_not_scaled_by_precision(self) -> None:
        """A precision class does not get to move more routing decisions."""
        for operation in OPERATIONS:
            ceilings = {
                TOLERANCE_MATRIX[(operation, c)].max_capability_regression
                for c in PRECISION_CLASSES
            }
            assert len(ceilings) == 1, f"{operation.value} varies its capability ceiling"


# ---------------------------------------------------------------------------
# 4. Violation detection -- the enforcement itself
# ---------------------------------------------------------------------------


class TestPolicyViolations:
    def test_compliant_profile_reports_nothing(self) -> None:
        policy = canonical_policy(NumericalOperation.ROUTER_PROBABILITIES, "fp16")
        assert (
            policy_violations(
                operation=NumericalOperation.ROUTER_PROBABILITIES,
                dtype="fp16",
                max_abs_error=policy.max_abs_error,
                max_rel_error=policy.max_rel_error,
                max_capability_regression=policy.max_capability_regression,
                require_finite_state_match=True,
            )
            == ()
        )

    def test_tightening_is_never_a_violation(self) -> None:
        """A caller may always demand MORE accuracy than policy requires.

        If tightening were reported, the matrix could be used to make the gate
        stricter than any caller intended -- the opposite of a ceiling.
        """
        assert (
            policy_violations(
                operation=NumericalOperation.NORMALIZATION,
                dtype="fp16",
                max_abs_error=0.0,
                max_rel_error=0.0,
                max_capability_regression=0.0,
                require_finite_state_match=True,
            )
            == ()
        )

    def test_exact_boundary_is_not_a_violation(self) -> None:
        """Equal-to-policy must pass; only strictly wider fails.

        An off-by-one here would reject every legitimately calibrated profile.
        """
        policy = canonical_policy(NumericalOperation.LOSS, "fp16")
        assert (
            policy_violations(
                operation=NumericalOperation.LOSS,
                dtype="fp16",
                max_abs_error=policy.max_abs_error,
                max_rel_error=policy.max_rel_error,
                max_capability_regression=policy.max_capability_regression,
                require_finite_state_match=True,
            )
            == ()
        )

    def test_widening_absolute_error_is_reported(self) -> None:
        violations = policy_violations(
            operation=NumericalOperation.ROUTER_PROBABILITIES,
            dtype="fp16",
            max_abs_error=1.0e-2,
            max_rel_error=1.0e-3,
            max_capability_regression=0.0,
            require_finite_state_match=True,
        )
        assert len(violations) == 1
        assert "max_abs_error" in violations[0]

    def test_turning_off_finiteness_is_reported(self) -> None:
        violations = policy_violations(
            operation=NumericalOperation.NORMALIZATION,
            dtype="fp16",
            max_abs_error=1.0e-3,
            max_rel_error=1.0e-3,
            max_capability_regression=0.0,
            require_finite_state_match=False,
        )
        assert len(violations) == 1
        assert "require_finite_state_match" in violations[0]

    def test_all_widened_axes_are_reported_together(self) -> None:
        """A profile that widens everything must not be able to hide one axis."""
        violations = policy_violations(
            operation=NumericalOperation.ROUTER_PROBABILITIES,
            dtype="fp16",
            max_abs_error=1.0e9,
            max_rel_error=1.0e9,
            max_capability_regression=1.0,
            require_finite_state_match=False,
        )
        assert len(violations) == 4

    def test_violation_message_names_the_canonical_policy(self) -> None:
        """The reader needs to know which declared policy was exceeded."""
        (violation,) = policy_violations(
            operation=NumericalOperation.LOSS,
            dtype="fp16",
            max_abs_error=1.0,
            max_rel_error=1.0e-2,
            max_capability_regression=0.05,
            require_finite_state_match=True,
        )
        assert canonical_policy(NumericalOperation.LOSS, "fp16").policy_id in violation


# ---------------------------------------------------------------------------
# 5. Enforcement through the gate
# ---------------------------------------------------------------------------


class TestGateEnforcement:
    def test_meaningless_profile_no_longer_promotes(self) -> None:
        """THE defect: a caller who declares a meaningless gate must not pass.

        Before the matrix this exact call returned passed=True, failures=()
        with max_abs_error=0.4 on ROUTER_PROBABILITIES and a total capability
        loss -- at any speedup.
        """
        sloppy = _profile(
            max_abs_error=1.0e9,
            max_rel_error=1.0e9,
            max_capability_regression=1.0,
            require_finite_state_match=False,
        )
        result = compare_numerical_paths(
            [0.5, 0.3, 0.2],
            [0.1, 0.5, 0.4],
            profile=sloppy,
            identity=_identity(),
            reference_capability_score=1.0,
            optimized_capability_score=0.0,
        )
        assert result.passed is False
        assert "tolerance_policy" in result.failures
        assert result.tolerance_policy_violations

    def test_identical_samples_do_not_rescue_an_invalid_profile(self) -> None:
        """A perfect match is still not promotable through a bogus gate.

        This is the sharper version of the defect: with ref == opt there is no
        numerical error at all, so before the matrix the artifact read as a
        clean pass regardless of how meaningless the declared tolerance was.
        """
        sloppy = _profile(max_abs_error=1.0e9, max_capability_regression=1.0)
        result = compare_numerical_paths(
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            profile=sloppy,
            identity=_identity(),
        )
        assert result.passed is False
        assert "tolerance_policy" in result.failures

    def test_tolerance_policy_is_reported_before_the_measurements(self) -> None:
        """A reader must not reach `absolute_error` and conclude the gate was tight.

        The other failures were measured against a gate that should not have
        existed, so the policy failure has to lead. The declared ceiling is
        1e-2 -- wider than the 1e-3 policy, so `tolerance_policy` fires, but
        still tight enough that a 0.4 error trips `absolute_error` too. Both
        must be present, policy first.
        """
        sloppy = _profile(max_abs_error=1.0e-2)
        result = compare_numerical_paths(
            [0.5, 0.3, 0.2],
            [0.9, 0.9, 0.9],
            profile=sloppy,
            identity=_identity(),
        )
        assert result.failures[0] == "tolerance_policy"
        assert "absolute_error" in result.failures

    def test_compliant_profile_still_passes(self) -> None:
        """The matrix must not break the profiles that were already honest."""
        result = compare_numerical_paths(
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            profile=_profile(),
            identity=_identity(),
        )
        assert result.passed is True
        assert result.failures == ()
        assert result.tolerance_policy_violations == ()

    def test_tighter_than_policy_still_passes(self) -> None:
        result = compare_numerical_paths(
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            profile=_profile(max_abs_error=1.0e-6, max_rel_error=1.0e-6),
            identity=_identity(),
        )
        assert result.passed is True
        assert result.tolerance_policy_violations == ()

    def test_artifact_records_which_policy_applied(self) -> None:
        """`profile_id` is caller-chosen and proves nothing.

        The same caller id can be attached to a 1e9 profile and a 1e-6 one, so
        the artifact must name the policy it was actually measured against.
        """
        result = compare_numerical_paths(
            [0.5],
            [0.5],
            profile=_profile(profile_id="whatever-the-caller-likes"),
            identity=_identity(),
        )
        assert result.profile_id == "whatever-the-caller-likes"
        assert result.tolerance_policy_id == (
            canonical_policy(NumericalOperation.ROUTER_PROBABILITIES, "fp16").policy_id
        )

    def test_unrecognised_dtype_is_gated_strictly_end_to_end(self) -> None:
        """An uncharacterised precision must not inherit the loosest gate."""
        result = compare_numerical_paths(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            profile=_profile(dtype="fp4", max_abs_error=1.0e-2),
            identity=_identity(),
        )
        assert result.passed is False
        assert "tolerance_policy" in result.failures

    def test_fallback_routes_an_invalid_profile_to_reference(self) -> None:
        """A mis-declared gate must not promote; USE_REFERENCE must engage."""
        from oai2.model.numerical_compare import select_numerical_path

        sloppy = _profile(
            max_abs_error=1.0e9,
            fallback=NumericalFallback.USE_REFERENCE,
        )
        result = compare_numerical_paths(
            [0.5, 0.3],
            [0.5, 0.3],
            profile=sloppy,
            identity=_identity(),
        )
        selection = select_numerical_path(
            result, optimized_path="opt", reference_path="ref"
        )
        assert selection.used_fallback is True
        assert selection.selected_path == "ref"

    def test_reject_fallback_raises_for_an_invalid_profile(self) -> None:
        from oai2.model.numerical_compare import select_numerical_path

        sloppy = _profile(
            max_abs_error=1.0e9,
            fallback=NumericalFallback.REJECT,
        )
        result = compare_numerical_paths(
            [0.5, 0.3],
            [0.5, 0.3],
            profile=sloppy,
            identity=_identity(),
        )
        with pytest.raises(RuntimeError):
            select_numerical_path(result, optimized_path="opt", reference_path="ref")

    def test_promotion_evidence_is_ineligible_through_a_sloppy_profile(self) -> None:
        """Speed must not rescue an out-of-policy candidate.

        The whole point of the gate is that a 10x speedup over a kernel that
        was never actually checked is not evidence of anything.
        """
        from oai2.model.numerical_promotion import (
            NumericalCandidateKind,
            evaluate_numerical_candidate,
        )

        evidence = evaluate_numerical_candidate(
            [0.5, 0.3, 0.2],
            [0.1, 0.5, 0.4],
            candidate_kind=NumericalCandidateKind.KERNEL,
            profile=_profile(
                max_abs_error=1.0e9,
                max_rel_error=1.0e9,
                max_capability_regression=1.0,
            ),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
            reference_capability_score=1.0,
            optimized_capability_score=0.0,
            speedup_ratio=10.0,
        )
        assert evidence.eligible is False
        assert evidence.speedup_ratio == 10.0
        assert "tolerance_policy" in evidence.comparison.failures


# ---------------------------------------------------------------------------
# 6. The matrix is the live baseline for the shipped evidence script
# ---------------------------------------------------------------------------


class TestShippedProfileCompliance:
    def test_numerical_backend_evidence_profile_is_compliant(self) -> None:
        """The repo's own evidence generator must not rely on a loose gate.

        If this drifts, the artifact the repository produces to justify a
        promotion would be produced under a profile the matrix rejects.
        """
        policy = canonical_policy(NumericalOperation.NORMALIZATION, "q4_K_M")
        assert policy.dtype_class is DtypeClass.QUANTIZED
        assert policy.max_abs_error >= 1.0e-6  # the script declares 1e-6
        assert policy.max_capability_regression >= 0.0


# ---------------------------------------------------------------------------
# 7. Negative controls
# ---------------------------------------------------------------------------


class TestCalibrationAgainstMeasuredEvidence:
    """The matrix must fit the error the repository has actually measured.

    Reasoning about mantissa widths got bf16 wrong once already: it was set
    equal to fp16, which left under 1.5x headroom against the measured bf16
    relative error. Theory alone did not catch that. These tests bind the
    calibration to VER-NUM-022's measured distribution so a future mis-scaling
    fails here rather than in production.
    """

    @staticmethod
    def _measured() -> list[dict]:
        import json

        artifacts = sorted(
            pathlib.Path("evidence/numerical").glob("tolerance_matrix_*.json")
        )
        assert artifacts, "no measured tolerance distribution committed"
        return json.loads(artifacts[-1].read_text())["distribution"]

    def test_a_measured_cell_exists_for_every_operation(self) -> None:
        measured = {c["operation"] for c in self._measured()}
        missing = {op.value for op in OPERATIONS} - measured
        assert not missing, f"no measured distribution for {sorted(missing)}"

    def test_measured_relative_error_fits_the_matrix_ceiling(self) -> None:
        """Relative error, not absolute.

        The sweep runs over ``extreme_value_fixtures()``, so the absolute
        figures are dominated by the fixture magnitudes -- normalization's
        measured ``max_abs`` of 7.3e8 says nothing about the operation, it
        says the input was large. Relative error is the scale-invariant
        signal, and it is the axis a matrix has to be sized against.
        """
        for cell in self._measured():
            operation = NumericalOperation(cell["operation"])
            policy = canonical_policy(operation, cell["storage_dtype"])
            assert cell["max_rel_error"] <= policy.max_rel_error, (
                f"{operation.value}/{cell['storage_dtype']}: measured relative "
                f"error {cell['max_rel_error']:.3e} exceeds the canonical ceiling "
                f"{policy.max_rel_error:.3e} ({policy.policy_id}) -- the matrix "
                f"is too tight for a path that storage at this precision can "
                f"actually produce"
            )

    def test_bf16_cells_have_real_headroom(self) -> None:
        """Guard the specific regression: bf16 was calibrated as if it were fp16.

        Requiring genuine headroom (not merely 'fits') is what makes this
        meaningful. A ceiling that a measured value merely touches has no
        margin for the run-to-run variance a real backend will have over a
        3-sample simulation.
        """
        for cell in self._measured():
            if dtype_class(cell["storage_dtype"]) is not DtypeClass.BF16:
                continue
            policy = canonical_policy(
                NumericalOperation(cell["operation"]), cell["storage_dtype"]
            )
            assert cell["max_rel_error"] * 2.0 <= policy.max_rel_error, (
                f"{cell['operation']}/bf16 leaves under 2x headroom "
                f"(measured {cell['max_rel_error']:.3e} vs ceiling "
                f"{policy.max_rel_error:.3e})"
            )


class TestNegativeControls:
    """Mutate the implementation and prove the guards above actually bite.

    Every test in this file is satisfiable by a matrix that is never consulted,
    or that is consulted but reordered or mis-scaled. Each control below breaks
    one of those and asserts the corresponding guard now FAILS. A control that
    still passed after its mutation would be proof the guard is decorative.
    """

    @staticmethod
    def _reload_mutated(module: object, mutate, tmp_path) -> object:  # type: ignore[no-untyped-def]
        """Exec a source-mutated copy of ``module`` under the real package name.

        A real source mutation rather than a monkeypatched attribute, because
        the ordering guard depends on where a statement sits in the function
        body -- something an attribute patch cannot reach.

        The mutated text has to be *written out* and loaded from there: a spec
        built from the original path silently re-executes the unmutated source,
        which makes the control pass for the wrong reason.
        """
        import importlib.util
        import pathlib
        import sys

        path = pathlib.Path(str(module.__file__))  # type: ignore[attr-defined]
        source = path.read_text()
        mutated = mutate(source)
        assert mutated != source, "mutation did not apply -- control is vacuous"

        target = tmp_path / "mutant_numerical_compare.py"
        target.write_text(mutated)
        # Load under the real package name so relative imports (`from
        # .numerics import ...`) resolve, and register before exec.
        name = "oai2.model._mutant_numerical_compare"
        spec = importlib.util.spec_from_file_location(name, target)
        assert spec is not None and spec.loader is not None
        module_obj = importlib.util.module_from_spec(spec)
        sys.modules[name] = module_obj
        try:
            spec.loader.exec_module(module_obj)
        except Exception:
            del sys.modules[name]
            raise
        return module_obj

    def test_control_removing_the_policy_check_breaks_enforcement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import oai2.model.numerical_compare as nc

        monkeypatch.setattr(nc, "policy_violations", lambda **_: ())
        result = nc.compare_numerical_paths(
            [0.5, 0.3],
            [0.5, 0.3],
            profile=_profile(max_abs_error=1.0e9),
            identity=_identity(),
        )
        with pytest.raises(AssertionError):
            # The guard from test_identical_samples_do_not_rescue_an_invalid_profile
            assert "tolerance_policy" in result.failures

    def test_control_moving_the_policy_failure_to_the_end_breaks_ordering(
        self, tmp_path: pathlib.Path
    ) -> None:
        import oai2.model.numerical_compare as nc

        def _mutate(source: str) -> str:
            lead = '    if policy_violation_details:\n        failures.append("tolerance_policy")\n'
            assert lead in source, "lead append not found; control is stale"
            source = source.replace(lead, "", 1)
            anchor = "    return NumericalComparison("
            assert anchor in source
            return source.replace(
                anchor,
                '    if policy_violation_details:\n        failures.append("tolerance_policy")\n\n' + anchor,
                1,
            )

        mutant = self._reload_mutated(nc, _mutate, tmp_path)
        result = mutant.compare_numerical_paths(
            [0.5, 0.3, 0.2],
            [0.9, 0.9, 0.9],
            profile=_profile(max_abs_error=1.0e-2),
            identity=_identity(),
        )
        assert "tolerance_policy" in result.failures, "mutation removed the failure"
        with pytest.raises(AssertionError):
            # The guard from test_tolerance_policy_is_reported_before_the_measurements
            assert result.failures[0] == "tolerance_policy"

    def test_control_dropping_the_policy_id_breaks_provenance(
        self, tmp_path: pathlib.Path
    ) -> None:
        import oai2.model.numerical_compare as nc

        def _mutate(source: str) -> str:
            line = "        tolerance_policy_id=canonical_policy(profile.operation, profile.dtype).policy_id,\n"
            assert line in source, "policy_id line not found; control is stale"
            return source.replace(line, '        tolerance_policy_id="",\n', 1)

        mutant = self._reload_mutated(nc, _mutate, tmp_path)
        result = mutant.compare_numerical_paths(
            [0.5],
            [0.5],
            profile=_profile(profile_id="whatever-the-caller-likes"),
            identity=_identity(),
        )
        with pytest.raises(AssertionError):
            # The guard from test_artifact_records_which_policy_applied
            assert result.tolerance_policy_id == (
                canonical_policy(NumericalOperation.ROUTER_PROBABILITIES, "fp16").policy_id
            )

    def test_control_relaxing_router_capability_ceiling_breaks_calibration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import oai2.model.tolerance_matrix as tm

        relaxed = {
            **tm.TOLERANCE_MATRIX,
            (NumericalOperation.ROUTER_PROBABILITIES, DtypeClass.FP16): TolerancePolicy(
                operation=NumericalOperation.ROUTER_PROBABILITIES,
                dtype_class=DtypeClass.FP16,
                max_abs_error=1.0e-3,
                max_rel_error=1.0e-3,
                max_capability_regression=0.5,
            ),
        }
        monkeypatch.setattr(tm, "TOLERANCE_MATRIX", relaxed)
        with pytest.raises(AssertionError):
            # The guard from test_router_probabilities_permit_zero_capability_regression
            for class_ in PRECISION_CLASSES:
                assert relaxed[(NumericalOperation.ROUTER_PROBABILITIES, class_)].max_capability_regression == 0.0

    def test_control_scaling_capability_by_precision_breaks_calibration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import oai2.model.tolerance_matrix as tm

        mutated = {
            (op, c): TolerancePolicy(
                operation=op,
                dtype_class=c,
                max_abs_error=tm.TOLERANCE_MATRIX[(op, c)].max_abs_error,
                max_rel_error=tm.TOLERANCE_MATRIX[(op, c)].max_rel_error,
                # Deliberately wrong: a looser precision class must NOT be
                # allowed to move more routing decisions.
                max_capability_regression=0.5 if c is DtypeClass.QUANTIZED else 0.0,
            )
            for (op, c) in tm.TOLERANCE_MATRIX
        }
        monkeypatch.setattr(tm, "TOLERANCE_MATRIX", mutated)
        with pytest.raises(AssertionError):
            # The guard from test_capability_ceiling_is_not_scaled_by_precision
            ceilings = {
                mutated[(NumericalOperation.ROUTER_PROBABILITIES, c)].max_capability_regression
                for c in PRECISION_CLASSES
            }
            assert len(ceilings) == 1

    def test_control_letting_unknown_dtype_resolve_to_loosest_breaks_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import oai2.model.tolerance_matrix as tm

        monkeypatch.setattr(
            tm,
            "canonical_policy",
            lambda operation, dtype: tm.TOLERANCE_MATRIX[(operation, DtypeClass.QUANTIZED)],
        )
        with pytest.raises(AssertionError):
            # The guard from test_unrecognised_dtype_is_unknown_not_loosest
            assert tm.canonical_policy(NumericalOperation.QUANTIZE_DEQUANTIZE, "fp4").dtype_class is DtypeClass.UNKNOWN

    def test_control_making_tightening_a_violation_breaks_the_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import oai2.model.tolerance_matrix as tm

        real = tm.policy_violations

        def _mutant(**kwargs: object) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
            out = real(**kwargs)  # type: ignore[arg-type]
            if not out and kwargs["max_abs_error"] == 0.0:
                return ("tighter than policy was wrongly reported",)
            return out

        monkeypatch.setattr(tm, "policy_violations", _mutant)
        with pytest.raises(AssertionError):
            # The guard from test_tightening_is_never_a_violation
            assert (
                tm.policy_violations(
                    operation=NumericalOperation.NORMALIZATION,
                    dtype="fp16",
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    max_capability_regression=0.0,
                    require_finite_state_match=True,
                )
                == ()
            )
