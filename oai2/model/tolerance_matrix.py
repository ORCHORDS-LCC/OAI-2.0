"""Canonical per-operation numerical tolerance policy.

WHY THIS EXISTS
---------------
``NumericalToleranceProfile`` is caller-supplied, and until now nothing in
``oai2/`` said what a profile *should* be for a given operation. REQ-NUM-001
asks for the precision/dtype policy to be explicit per major operation class
where non-default behaviour matters -- but with no registry, that requirement
had nothing to check against and the gate would accept any numbers a caller
declared.

Measured on the prior ``main``, a caller could declare this for
``ROUTER_PROBABILITIES`` -- the operation whose outputs are routing
probabilities:

    max_abs_error             = 1e9
    max_rel_error             = 1e9
    max_capability_regression = 1.0     # 100% capability loss permitted
    require_finite_state_match = False

and the gate passed a kernel whose router probabilities were wrong by
**0.4 absolute**. Every one of those declarations is individually legal and
collectively meaningless: there was no baseline to be measured against.

WHAT THIS MODULE IS
-------------------
A declared, versioned policy per ``(NumericalOperation, dtype class)``. It is
data, not enforcement -- :func:`policy_violations` is what
``compare_numerical_paths`` consults, so a profile that is looser than the
declared policy fails the comparison with ``tolerance_policy`` rather than
quietly widening the gate.

Calibration
-----------
Tolerances are set from the semantics of each operation, not chosen to make
tests pass:

- Within one operation, the ceiling is scaled by how coarsely each format's
  significand can hold a value: ``fp32`` < ``fp16`` < ``bf16`` < quantized
  (int8) < ``fp8``. ``bf16`` is the case that is easy to get backwards -- it
  carries ``fp32``'s *exponent range* with an 8-bit significand against
  ``fp16``'s 11, so it is **coarser** than ``fp16``, not a midpoint between
  the two. int8 landing below fp8 is the other counter-intuitive one.
- ``QUANTIZE_DEQUANTIZE`` is deliberately the loosest. A low-bit round trip
  is *supposed* to lose precision; a tolerance that rejected it would make
  the operation untestable.
- ``ROUTER_PROBABILITIES`` permits **zero** capability regression. A router
  whose output distribution moved has changed which expert runs, which is a
  behavioural change, not a rounding difference.
- Finiteness is required for every operation class. A non-finite output is
  never an acceptable approximation of a finite one.
- Capability ceilings are behavioural, so they are **not** scaled by
  precision class. A coarser format does not get to move more routing
  decisions.

The factors are cross-checked against the measured error distribution in
``evidence/numerical/tolerance_matrix_*.json`` (VER-NUM-022), which is what
exposed the bf16 factor being set equal to ``fp16``: the measured bf16
relative error left the old ceiling under 1.5x of headroom, which no real
backend would hold.

Extending the matrix is a policy decision, not a refactor: add the
``(operation, dtype class)`` entry and the profile id changes with it, so a
loosened policy is visible in any artifact that records ``profile_id``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .numerics import NumericalOperation

#: Bump when any entry below changes. Artifacts record this so a comparison
#: made under a different policy cannot be read as if it were made under this
#: one.
TOLERANCE_POLICY_VERSION = "1"


class DtypeClass(StrEnum):
    """Precision classes the matrix is keyed on.

    The class is what matters, not the exact dtype string: a caller writing
    ``"float32"`` and one writing ``"fp32"`` are subject to the same policy.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    QUANTIZED = "quantized"
    UNKNOWN = "unknown"


_DTYPE_ALIASES: Mapping[str, DtypeClass] = MappingProxyType(
    {
        "fp32": DtypeClass.FP32,
        "float32": DtypeClass.FP32,
        "f32": DtypeClass.FP32,
        "fp16": DtypeClass.FP16,
        "float16": DtypeClass.FP16,
        "f16": DtypeClass.FP16,
        "half": DtypeClass.FP16,
        "bf16": DtypeClass.BF16,
        "bfloat16": DtypeClass.BF16,
        "fp8": DtypeClass.FP8,
        "float8": DtypeClass.FP8,
        "fp8_e4m3": DtypeClass.FP8,
        "fp8_e5m2": DtypeClass.FP8,
        "int8": DtypeClass.QUANTIZED,
        "uint8": DtypeClass.QUANTIZED,
        "q4": DtypeClass.QUANTIZED,
        "q4_k_m": DtypeClass.QUANTIZED,
        "q5": DtypeClass.QUANTIZED,
        "q6": DtypeClass.QUANTIZED,
        "q8": DtypeClass.QUANTIZED,
    }
)


def dtype_class(dtype: str) -> DtypeClass:
    """Classify a dtype string. Unrecognised dtypes get :attr:`DtypeClass.UNKNOWN`.

    Unknown is a real class, not a silent pass-through to the loosest
    policy: it selects :data:`_UNKNOWN_POLICY`, which is deliberately strict
    enough that an unrecognised precision cannot quietly widen the gate.
    """
    return _DTYPE_ALIASES.get(dtype.strip().lower(), DtypeClass.UNKNOWN)


@dataclass(slots=True, frozen=True)
class TolerancePolicy:
    """The declared ceiling for one operation at one precision class."""

    operation: NumericalOperation
    dtype_class: DtypeClass
    max_abs_error: float
    max_rel_error: float
    max_capability_regression: float
    require_finite_state_match: bool = True

    @property
    def policy_id(self) -> str:
        return f"canonical-{self.operation.value}-{self.dtype_class.value}-v{TOLERANCE_POLICY_VERSION}"


def _p(
    operation: NumericalOperation,
    dtype_class: DtypeClass,
    max_abs_error: float,
    max_rel_error: float,
    max_capability_regression: float,
) -> TolerancePolicy:
    return TolerancePolicy(
        operation=operation,
        dtype_class=dtype_class,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        max_capability_regression=max_capability_regression,
    )


# Per-operation ceilings at fp16, the default serving precision. The other
# precision classes are derived by rounding factor rather than invented, so
# the relationship between them is explicit and checkable.
_FP16_BASE: Mapping[NumericalOperation, tuple[float, float, float]] = MappingProxyType(
    {
        # Outputs are O(1). A normalization kernel that drifts by more than
        # ~1e-3 changes downstream activations materially.
        NumericalOperation.NORMALIZATION: (1.0e-3, 1.0e-3, 0.01),
        # Attention weights are probabilities; drift compounds across heads
        # and layers, so the per-layer ceiling is set for the accumulation.
        NumericalOperation.ATTENTION_SOFTMAX: (1.0e-3, 1.0e-3, 0.0),
        # Router outputs are discrete routing decisions. Moving the
        # distribution moves which expert runs: a behavioural change, never
        # a rounding artefact, hence zero capability regression.
        NumericalOperation.ROUTER_PROBABILITIES: (1.0e-3, 1.0e-3, 0.0),
        # Loss is O(1-10) and feeds gradients, so the relative ceiling
        # carries the tolerance.
        NumericalOperation.LOSS: (1.0e-2, 1.0e-2, 0.05),
        # Reduction accumulates across the context window; the error grows
        # with length, so the ceiling is looser than a single step.
        NumericalOperation.LONG_CONTEXT_REDUCTION: (1.0e-2, 1.0e-2, 0.05),
        # A low-bit round trip is SUPPOSED to lose precision. A ceiling
        # tight enough for fp16 would make the operation untestable, and
        # tightening it would mean the quantizer is not doing its job.
        NumericalOperation.QUANTIZE_DEQUANTIZE: (1.0e-1, 5.0e-2, 0.10),
    }
)

#: Relative factor applied to the fp16 ceiling for each precision class.
#:
#: These are the ratios of each format's machine epsilon to fp16's, because
#: storage rounding is the one error an optimized path cannot avoid -- the
#: significand simply does not hold the digits. eps is ``2**(1 - p)`` for a
#: ``p``-bit significand, so with fp16 at ``p=11`` (``eps = 2**-11``):
#:
#: ==========  =====  ==============  ==========  ==========================
#: format      ``p``  eps             ratio      note
#: ==========  =====  ==============  ==========  ==========================
#: fp32        24     ``2**-24``      ~1e-2       see the note below
#: fp16        11     ``2**-11``      1.0        the reference
#: bf16        8      ``2**-8``       8.0        fp32's exponent, fp16-ish
#:                                                    significand -- but
#:                                                    COARSER than fp16
#: fp8 e4m3    4      ``2**-4``       128.0      the more precise fp8
#: int8 grid   --     ``1/127``       16.1       7 bits + sign, evenly split
#: ==========  =====  ==============  ==========  ==========================
#:
#: ``fp32`` is deliberately NOT the exact eps ratio (~1.2e-4). The per-operation
#: base below already carries headroom for accumulated error across a whole
#: operation, and fp32 only needs to be unambiguously tighter than fp16 -- not
#: so tight that a real fp32 kernel is rejected for a rounding artefact.
#:
#: ``quantized`` is sized at the int8 grid, the tighter end of a family that
#: runs to int4 (``1/7``, ~293x). Note this puts QUANTIZED *below* FP8 in
#: precision: int8's evenly-split 7-bit grid has a relative step of 1/127,
#: which is finer than fp8 e4m3's 4-bit significand. The bucket therefore
#: straddles fp8 rather than sitting wholly above or below it, and an int4
#: path is expected to be rejected by these ceilings rather than waved
#: through -- which is the right answer, and the reason the genuinely lossy
#: low-bit round trip is admitted by ``QUANTIZE_DEQUANTIZE``'s own base
#: instead.
_PRECISION_FACTOR: Mapping[DtypeClass, float] = MappingProxyType(
    {
        DtypeClass.FP32: 1.0e-2,
        DtypeClass.FP16: 1.0,
        DtypeClass.BF16: 8.0,
        DtypeClass.FP8: 128.0,
        DtypeClass.QUANTIZED: 16.0,
    }
)

#: Strict ceiling for an unrecognised dtype. Deliberately tighter than the
#: quantized class: an operation nobody has characterised should be
#: qualified before it is trusted, not waved through.
_UNKNOWN_POLICY: TolerancePolicy = TolerancePolicy(
    operation=NumericalOperation.NORMALIZATION,
    dtype_class=DtypeClass.UNKNOWN,
    max_abs_error=1.0e-3,
    max_rel_error=1.0e-3,
    max_capability_regression=0.0,
)


def _build_matrix() -> Mapping[tuple[NumericalOperation, DtypeClass], TolerancePolicy]:
    entries: dict[tuple[NumericalOperation, DtypeClass], TolerancePolicy] = {}
    for operation, (abs_err, rel_err, cap_reg) in _FP16_BASE.items():
        for class_ in (DtypeClass.FP32, DtypeClass.FP16, DtypeClass.BF16):
            factor = _PRECISION_FACTOR[class_]
            # A capability ceiling is behavioural, not numeric: a precision
            # class does not get to move more routing decisions.
            entries[(operation, class_)] = TolerancePolicy(
                operation=operation,
                dtype_class=class_,
                max_abs_error=round(abs_err * factor, 12),
                max_rel_error=round(rel_err * factor, 12),
                max_capability_regression=cap_reg,
            )
        # fp8 and low-bit quantized share the deliberately loose ceiling.
        factor = _PRECISION_FACTOR[DtypeClass.FP8]
        entries[(operation, DtypeClass.FP8)] = TolerancePolicy(
            operation=operation,
            dtype_class=DtypeClass.FP8,
            max_abs_error=round(abs_err * factor, 12),
            max_rel_error=round(rel_err * factor, 12),
            max_capability_regression=cap_reg,
        )
    for operation, (abs_err, rel_err, cap_reg) in _FP16_BASE.items():
        factor = _PRECISION_FACTOR[DtypeClass.QUANTIZED]
        entries[(operation, DtypeClass.QUANTIZED)] = TolerancePolicy(
            operation=operation,
            dtype_class=DtypeClass.QUANTIZED,
            max_abs_error=round(abs_err * factor, 12),
            max_rel_error=round(rel_err * factor, 12),
            max_capability_regression=cap_reg,
        )
    return MappingProxyType(entries)


#: The canonical policy matrix, keyed by ``(operation, dtype class)``.
TOLERANCE_MATRIX: Mapping[tuple[NumericalOperation, DtypeClass], TolerancePolicy] = (
    _build_matrix()
)


def canonical_policy(operation: NumericalOperation, dtype: str) -> TolerancePolicy:
    """Return the declared policy for ``operation`` at ``dtype``'s class.

    An unrecognised dtype resolves to the strict :data:`_UNKNOWN_POLICY`
    rather than to the loosest entry, so qualifying a new precision is
    something you have to do on purpose.
    """
    class_ = dtype_class(dtype)
    if class_ is DtypeClass.UNKNOWN:
        return _UNKNOWN_POLICY
    return TOLERANCE_MATRIX[(operation, class_)]


def policy_violations(
    *,
    operation: NumericalOperation,
    dtype: str,
    max_abs_error: float,
    max_rel_error: float,
    max_capability_regression: float,
    require_finite_state_match: bool,
) -> tuple[str, ...]:
    """Report every axis on which a profile is LOOSER than the declared policy.

    Tighter than policy is never a violation: a caller may always demand more
    accuracy than the matrix requires. Only widening is reported, so this
    cannot be used to make the gate stricter than a caller intended.
    """
    policy = canonical_policy(operation, dtype)
    violations: list[str] = []
    if max_abs_error > policy.max_abs_error:
        violations.append(
            f"max_abs_error {max_abs_error:g} exceeds canonical "
            f"{policy.max_abs_error:g} for {policy.policy_id}"
        )
    if max_rel_error > policy.max_rel_error:
        violations.append(
            f"max_rel_error {max_rel_error:g} exceeds canonical "
            f"{policy.max_rel_error:g} for {policy.policy_id}"
        )
    if max_capability_regression > policy.max_capability_regression:
        violations.append(
            f"max_capability_regression {max_capability_regression:g} exceeds "
            f"canonical {policy.max_capability_regression:g} for {policy.policy_id}"
        )
    if policy.require_finite_state_match and not require_finite_state_match:
        violations.append(
            f"require_finite_state_match is off but {policy.policy_id} requires it"
        )
    return tuple(violations)


__all__ = [
    "TOLERANCE_MATRIX",
    "TOLERANCE_POLICY_VERSION",
    "DtypeClass",
    "TolerancePolicy",
    "canonical_policy",
    "dtype_class",
    "policy_violations",
]
