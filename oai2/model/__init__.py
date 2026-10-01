"""OAI-2.0 model architecture primitives."""

from .numerics import (
    NUMERICAL_POLICY_VERSION,
    NumericalCheckResult,
    NumericalFailure,
    NumericalOperation,
    PrecisionRule,
    REFERENCE_PRECISION_RULES,
    check_numerics,
    extreme_value_fixtures,
    precision_rule,
)

__all__ = [
    "NUMERICAL_POLICY_VERSION",
    "NumericalCheckResult",
    "NumericalFailure",
    "NumericalOperation",
    "PrecisionRule",
    "REFERENCE_PRECISION_RULES",
    "check_numerics",
    "extreme_value_fixtures",
    "precision_rule",
]
