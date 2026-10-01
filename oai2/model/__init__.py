"""OAI-2.0 model architecture primitives."""

from .numerical_compare import (
    NumericalArtifactIdentity,
    NumericalComparison,
    NumericalFallback,
    NumericalSelection,
    NumericalToleranceProfile,
    compare_numerical_paths,
    select_numerical_path,
)
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
    "NumericalArtifactIdentity",
    "NumericalComparison",
    "NumericalFallback",
    "NumericalSelection",
    "NumericalToleranceProfile",
    "compare_numerical_paths",
    "select_numerical_path",
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
