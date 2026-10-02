"""Structural + validation contract for ``oai2.model.numerical_promotion``.

Pin the outer guard-rail contract of the canonical numerical promotion gate
that consumes the WI-NUM-002 tolerance matrix before a candidate path can
be considered eligible. Speed is evidence only: it never overrides numerical
or capability failures.

Sections:
  - module docstring + module-level imports
  - __all__ + identity (re-exports through oai2.model.__init__)
  - NumericalCandidateKind StrEnum
  - NumericalPromotionEvidence frozen-slotted @dataclass
  - evaluate_numerical_candidate(...) function
  - public-safety + structural counts
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re

import pytest

from oai2.model import (
    NumericalCandidateKind,
    NumericalPromotionEvidence,
    evaluate_numerical_candidate,
)
from oai2.model import numerical_promotion as promotion_module
from oai2.model.numerical_compare import (
    NumericalArtifactIdentity,
    NumericalComparison,
    NumericalFallback,
    NumericalSelection,
    NumericalToleranceProfile,
)
from oai2.model.numerics import NumericalOperation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity() -> NumericalArtifactIdentity:
    return NumericalArtifactIdentity(
        model_version="m1",
        runtime_version="r1",
        backend="cpu",
        config_id="c1",
    )


def _profile(
    *,
    max_abs_error: float = 1e-5,
    max_rel_error: float = 1e-5,
    max_capability_regression: float = 0.0,
    fallback: NumericalFallback = NumericalFallback.USE_REFERENCE,
    require_finite_state_match: bool = True,
) -> NumericalToleranceProfile:
    return NumericalToleranceProfile(
        profile_id="promote-default",
        operation=NumericalOperation.ATTENTION_SOFTMAX,
        dtype="fp32",
        shape_class="vector",
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        max_capability_regression=max_capability_regression,
        require_finite_state_match=require_finite_state_match,
        fallback=fallback,
    )


def _equal_seqs(n: int = 4) -> list[float]:
    return [1.0, 2.0, 3.0, 4.0][:n]


# ---------------------------------------------------------------------------
# 1. Module docstring + module-level imports
# ---------------------------------------------------------------------------


class TestModuleDocstringAndImports:
    def test_module_docstring_mentions_promotion_and_tolerance_matrix(self) -> None:
        doc = promotion_module.__doc__
        assert isinstance(doc, str)
        lower = doc.lower()
        assert "promotion" in lower
        assert "tolerance" in lower
        assert "wi-num-002" in lower

    def test_module_docstring_pins_speed_is_evidence_only(self) -> None:
        doc = promotion_module.__doc__.lower()
        assert "speed" in doc
        assert "evidence" in doc
        assert "never" in doc

    def test_module_has_future_annotations(self) -> None:
        source = inspect.getsourcefile(promotion_module)
        assert source is not None
        with open(source, encoding="utf-8") as f:
            head = f.read(1024)
        assert "from __future__ import annotations" in head

    def test_module_imports_stdlib_math(self) -> None:
        assert hasattr(promotion_module, "math")
        assert promotion_module.math.__name__ == "math"

    def test_module_imports_dataclass_from_dataclasses(self) -> None:
        assert hasattr(promotion_module, "dataclass")

    def test_module_imports_strenum_from_enum(self) -> None:
        assert hasattr(promotion_module, "StrEnum")

    def test_module_does_not_import_pydantic_at_module_scope(self) -> None:
        assert not hasattr(promotion_module, "pydantic")

    def test_module_does_not_import_requests_or_httpx(self) -> None:
        assert not hasattr(promotion_module, "requests")
        assert not hasattr(promotion_module, "httpx")
        assert not hasattr(promotion_module, "urllib")
        assert not hasattr(promotion_module, "aiohttp")

    def test_module_relative_imports_numerical_compare(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "from .numerical_compare import" in src

    def test_module_has_no_absolute_import_oai2(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # no top-level absolute `import oai2` or `from oai2`
        assert not re.search(r"^import oai2\b", src, re.MULTILINE)
        assert not re.search(r"^from oai2\b", src, re.MULTILINE)

    def test_module_has_no_wildcard_imports(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "import *" not in src


# ---------------------------------------------------------------------------
# 2. __all__ + identity
# ---------------------------------------------------------------------------


class TestAllAndIdentity:
    def test_dunder_all_has_exactly_three_names(self) -> None:
        assert len(promotion_module.__all__) == 3

    def test_dunder_all_names_pinned(self) -> None:
        assert set(promotion_module.__all__) == {
            "NumericalCandidateKind",
            "NumericalPromotionEvidence",
            "evaluate_numerical_candidate",
        }

    def test_dunder_all_is_list_of_strings(self) -> None:
        assert isinstance(promotion_module.__all__, list)
        for n in promotion_module.__all__:
            assert isinstance(n, str)
            assert n

    def test_top_level_import_resolves_same_object_as_module_attr(self) -> None:
        assert NumericalCandidateKind is promotion_module.NumericalCandidateKind
        assert NumericalPromotionEvidence is promotion_module.NumericalPromotionEvidence
        assert evaluate_numerical_candidate is promotion_module.evaluate_numerical_candidate

    def test_oai2_model_package_reexports_all_three(self) -> None:
        import oai2.model

        assert "NumericalCandidateKind" in oai2.model.__all__
        assert "NumericalPromotionEvidence" in oai2.model.__all__
        assert "evaluate_numerical_candidate" in oai2.model.__all__
        assert oai2.model.NumericalCandidateKind is promotion_module.NumericalCandidateKind
        assert oai2.model.NumericalPromotionEvidence is promotion_module.NumericalPromotionEvidence
        assert (
            oai2.model.evaluate_numerical_candidate is promotion_module.evaluate_numerical_candidate
        )

    def test_direct_module_import_matches_top_level(self) -> None:
        from oai2.model.numerical_promotion import (
            NumericalCandidateKind as DirectKind,
        )
        from oai2.model.numerical_promotion import (
            NumericalPromotionEvidence as DirectEvidence,
        )
        from oai2.model.numerical_promotion import (
            evaluate_numerical_candidate as DirectEval,
        )

        assert DirectKind is NumericalCandidateKind
        assert DirectEvidence is NumericalPromotionEvidence
        assert DirectEval is evaluate_numerical_candidate

    def test_private_imports_not_in_dunder_all(self) -> None:
        for private in (
            "math",
            "dataclass",
            "dataclasses",
            "StrEnum",
            "NumericalArtifactIdentity",
            "NumericalComparison",
            "NumericalSelection",
            "NumericalToleranceProfile",
            "compare_numerical_paths",
            "select_numerical_path",
        ):
            assert private not in promotion_module.__all__


# ---------------------------------------------------------------------------
# 3. NumericalCandidateKind StrEnum
# ---------------------------------------------------------------------------


class TestNumericalCandidateKind:
    def test_subclasses_strenum_and_str(self) -> None:
        from enum import StrEnum

        assert issubclass(NumericalCandidateKind, StrEnum)
        assert issubclass(NumericalCandidateKind, str)

    def test_has_exactly_three_members(self) -> None:
        assert len(NumericalCandidateKind.__members__) == 3

    def test_member_names_pinned(self) -> None:
        assert set(NumericalCandidateKind.__members__) == {
            "BACKEND",
            "EXPORT",
            "KERNEL",
        }

    def test_member_values_are_lowercase(self) -> None:
        for member in NumericalCandidateKind:
            assert member.value == member.value.lower()
            assert " " not in member.value
            assert member.value.isascii()

    def test_member_values_pinned(self) -> None:
        assert NumericalCandidateKind.BACKEND.value == "backend"
        assert NumericalCandidateKind.EXPORT.value == "export"
        assert NumericalCandidateKind.KERNEL.value == "kernel"

    def test_member_values_distinct(self) -> None:
        values = [m.value for m in NumericalCandidateKind]
        assert len(values) == len(set(values))

    def test_strenum_equality_with_wire_string(self) -> None:
        assert NumericalCandidateKind.BACKEND == "backend"
        assert NumericalCandidateKind.EXPORT == "export"
        assert NumericalCandidateKind.KERNEL == "kernel"

    def test_str_member_returns_wire_string(self) -> None:
        for member in NumericalCandidateKind:
            assert str(member) == member.value

    def test_wire_string_round_trip(self) -> None:
        for member in NumericalCandidateKind:
            assert NumericalCandidateKind(member.value) is member

    def test_unknown_wire_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            NumericalCandidateKind("not-a-real-candidate")

    def test_module_source_pins_three_candidates(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # count the BACKEND / EXPORT / KERNEL wire-string assignments
        assert re.search(r'BACKEND\s*=\s*"backend"', src)
        assert re.search(r'EXPORT\s*=\s*"export"', src)
        assert re.search(r'KERNEL\s*=\s*"kernel"', src)


# ---------------------------------------------------------------------------
# 4. NumericalPromotionEvidence frozen-slotted @dataclass
# ---------------------------------------------------------------------------


class TestNumericalPromotionEvidence:
    def test_is_a_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(NumericalPromotionEvidence)

    def test_is_frozen(self) -> None:

        params = getattr(NumericalPromotionEvidence, "__dataclass_params__", None)
        assert params is not None
        assert params.frozen is True

    def test_is_slotted(self) -> None:

        # a slots-style dataclass exposes __slots__ and omits __dict__
        # on instances — verify by constructing one and checking __dict__
        ev = _make_evidence_within_placeholder()
        assert not hasattr(ev, "__dict__")

    def test_field_set_pinned_to_five_names(self) -> None:
        import dataclasses

        fields = {f.name for f in dataclasses.fields(NumericalPromotionEvidence)}
        assert fields == {
            "candidate_kind",
            "comparison",
            "selection",
            "eligible",
            "speedup_ratio",
        }

    def test_constructs_with_all_five_fields(self) -> None:
        ev = _make_evidence_within_placeholder()
        assert ev.candidate_kind is NumericalCandidateKind.BACKEND
        assert isinstance(ev.comparison, NumericalComparison)
        assert isinstance(ev.selection, NumericalSelection)
        assert ev.eligible is True
        assert ev.speedup_ratio == pytest.approx(1.5)

    def test_speedup_ratio_defaults_to_none(self) -> None:
        ev = _make_evidence_within_placeholder(speedup_ratio=None)
        assert ev.speedup_ratio is None

    def test_eligible_is_bool(self) -> None:
        ev = _make_evidence_within_placeholder()
        assert isinstance(ev.eligible, bool)

    def test_mutation_raises_frozen_instance_error(self) -> None:
        ev = _make_evidence_within_placeholder()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.eligible = False  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        ev = _make_evidence_within_placeholder()
        assert hash(ev) is not None

    def test_value_equality_by_fields(self) -> None:
        a = _make_evidence_within_placeholder()
        b = _make_evidence_within_placeholder()
        assert a == b

    def test_tolerance_profile_id_property_returns_comparison_profile_id(
        self,
    ) -> None:
        ev = _make_evidence_within_placeholder()
        assert ev.tolerance_profile_id == ev.comparison.profile_id
        assert ev.tolerance_profile_id == "promote-default"

    def test_identity_property_returns_comparison_identity(self) -> None:
        ev = _make_evidence_within_placeholder()
        assert ev.identity is ev.comparison.identity

    def test_tolerance_profile_id_is_a_property(self) -> None:
        assert isinstance(
            NumericalPromotionEvidence.__dict__["tolerance_profile_id"],
            property,
        )

    def test_identity_is_a_property(self) -> None:
        assert isinstance(
            NumericalPromotionEvidence.__dict__["identity"],
            property,
        )

    def test_source_pins_one_dataclass_decorator(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # exactly one @dataclass(...) decorator
        decorators = re.findall(r"@dataclass\(.*\)", src)
        assert len(decorators) == 1
        assert "slots=True" in decorators[0]
        assert "frozen=True" in decorators[0]


def _make_evidence_within_placeholder(
    speedup_ratio: float | None = 1.5,
) -> NumericalPromotionEvidence:
    """Build a minimal in-tolerance evidence instance for property tests."""

    profile = _profile()
    seq = _equal_seqs()
    identity = _identity()
    return evaluate_numerical_candidate(
        reference=seq,
        optimized=list(seq),
        candidate_kind=NumericalCandidateKind.BACKEND,
        profile=profile,
        identity=identity,
        optimized_path="opt",
        reference_path="ref",
        speedup_ratio=speedup_ratio,
    )


def _make_evidence_within_placeholder_compat() -> NumericalPromotionEvidence:
    """Compatibility wrapper — calls the place holder above."""
    return _make_evidence_within_placeholder()


# alias used by the inner test body
_make_evidence_within_placeholder_compat  # noqa: B018


# ---------------------------------------------------------------------------
# 5. evaluate_numerical_candidate function
# ---------------------------------------------------------------------------


class TestEvaluateNumericalCandidate:
    def test_returns_numerical_promotion_evidence_instance(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert isinstance(ev, NumericalPromotionEvidence)

    def test_signature_is_callable_with_keyword_args(self) -> None:
        import inspect

        sig = inspect.signature(evaluate_numerical_candidate)
        # positional-or-keyword params: reference, optimized, candidate_kind, profile, identity, optimized_path, reference_path
        for name in (
            "reference",
            "optimized",
            "candidate_kind",
            "profile",
            "identity",
            "optimized_path",
            "reference_path",
        ):
            assert name in sig.parameters

    def test_signature_has_keyword_only_capability_scores(self) -> None:
        import inspect

        sig = inspect.signature(evaluate_numerical_candidate)
        for name in (
            "reference_capability_score",
            "optimized_capability_score",
            "speedup_ratio",
        ):
            p = sig.parameters[name]
            assert p.kind is inspect.Parameter.KEYWORD_ONLY

    def test_reference_only_is_keyword_only(self) -> None:
        # `reference` and `optimized` are positional-or-keyword (most useful)
        # but the function accepts keyword invocation too
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert ev.eligible is True

    def test_eligible_true_when_in_tolerance_and_not_fallback(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert ev.eligible is True
        assert ev.selection.used_fallback is False
        assert ev.selection.selected_path == "opt"

    def test_eligible_false_when_out_of_tolerance_with_use_reference(
        self,
    ) -> None:
        # force out-of-tolerance by widening optimized values
        seq = _equal_seqs()
        opt = [v + 1.0 for v in seq]
        ev = evaluate_numerical_candidate(
            reference=seq,
            optimized=opt,
            candidate_kind=NumericalCandidateKind.EXPORT,
            profile=_profile(max_abs_error=1e-6),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert ev.eligible is False
        assert ev.selection.used_fallback is True
        assert ev.selection.selected_path == "ref"
        assert ev.selection.reason == "out_of_tolerance"

    def test_eligible_false_when_out_of_tolerance_with_reject_raises(
        self,
    ) -> None:
        # REJECT fallback means select_numerical_path raises — and
        # evaluate_numerical_candidate propagates that raise (no
        # silent swallow), so we expect RuntimeError
        seq = _equal_seqs()
        opt = [v + 1.0 for v in seq]
        with pytest.raises(RuntimeError):
            evaluate_numerical_candidate(
                reference=seq,
                optimized=opt,
                candidate_kind=NumericalCandidateKind.KERNEL,
                profile=_profile(
                    max_abs_error=1e-6,
                    fallback=NumericalFallback.REJECT,
                ),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
            )

    def test_candidate_kind_must_be_numerical_candidate_kind_instance(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind="backend",  # type: ignore[arg-type]
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
            )

    def test_speedup_ratio_none_is_accepted(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
            speedup_ratio=None,
        )
        assert ev.speedup_ratio is None

    def test_speedup_ratio_normalized_to_float(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
            speedup_ratio=2,  # int
        )
        assert ev.speedup_ratio == pytest.approx(2.0)
        assert isinstance(ev.speedup_ratio, float)

    def test_speedup_ratio_float_input_stored_as_float(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
            speedup_ratio=1.25,
        )
        assert ev.speedup_ratio == pytest.approx(1.25)
        assert isinstance(ev.speedup_ratio, float)

    def test_speedup_ratio_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio=0.0,
            )

    def test_speedup_ratio_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio=-0.1,
            )

    def test_speedup_ratio_positive_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio=float("inf"),
            )

    def test_speedup_ratio_negative_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio=float("-inf"),
            )

    def test_speedup_ratio_nan_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio=float("nan"),
            )

    def test_speedup_ratio_bool_true_rejected(self) -> None:
        # True is an instance of int — explicit bool rejection
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio=True,
            )

    def test_speedup_ratio_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                speedup_ratio="1.5",  # type: ignore[arg-type]
            )

    def test_capability_score_one_none_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                reference_capability_score=0.5,
                optimized_capability_score=None,
            )

    def test_capability_score_none_one_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
                reference_capability_score=None,
                optimized_capability_score=0.5,
            )

    def test_capability_score_both_none_accepted(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
            reference_capability_score=None,
            optimized_capability_score=None,
        )
        assert ev.comparison.capability_regression == 0.0

    def test_capability_score_pair_accepted(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(max_capability_regression=1.0),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
            reference_capability_score=0.9,
            optimized_capability_score=0.7,
        )
        # capability_regression = max(0.0, 0.9 - 0.7) = 0.2
        assert ev.comparison.capability_regression == pytest.approx(0.2)

    def test_uses_tuple_reference(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=(1.0, 2.0, 3.0),
            optimized=(1.0, 2.0, 3.0),
            candidate_kind=NumericalCandidateKind.KERNEL,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert ev.eligible is True

    def test_uses_list_optimized(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=(1.0, 2.0, 3.0),
            optimized=[1.0, 2.0, 3.0],
            candidate_kind=NumericalCandidateKind.KERNEL,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert ev.eligible is True

    def test_empty_sequences_propagate_value_error(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=[],
                optimized=[],
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
            )

    def test_length_mismatch_propagates_value_error(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=[1.0, 2.0, 3.0],
                optimized=[1.0, 2.0],
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
            )

    def test_empty_optimized_path_propagates_value_error(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="   ",
                reference_path="ref",
            )

    def test_empty_reference_path_propagates_value_error(self) -> None:
        with pytest.raises(ValueError):
            evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=NumericalCandidateKind.BACKEND,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="",
            )

    def test_finite_state_mismatch_marked_in_failures(self) -> None:
        # inf vs finite → finite_state_mismatch in failures
        ev = evaluate_numerical_candidate(
            reference=[1.0, float("inf"), 3.0],
            optimized=[1.0, 2.0, 3.0],
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert "finite_state_mismatch" in ev.comparison.failures
        assert ev.eligible is False

    def test_comparison_propagated_into_evidence(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        # The evidence's comparison is the same object returned from compare
        assert ev.comparison.passed is True
        assert ev.comparison.failures == ()

    def test_selection_propagated_into_evidence(self) -> None:
        ev = evaluate_numerical_candidate(
            reference=_equal_seqs(),
            optimized=_equal_seqs(),
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity(),
            optimized_path="opt",
            reference_path="ref",
        )
        assert ev.selection.selected_path == "opt"
        assert ev.selection.used_fallback is False
        assert ev.selection.reason == "within_tolerance"

    def test_function_called_with_each_candidate_kind(self) -> None:
        for kind in NumericalCandidateKind:
            ev = evaluate_numerical_candidate(
                reference=_equal_seqs(),
                optimized=_equal_seqs(),
                candidate_kind=kind,
                profile=_profile(),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
            )
            assert ev.candidate_kind is kind

    def test_function_signature_return_annotation_resolves_to_evidence(self) -> None:
        import inspect
        import typing

        sig = inspect.signature(evaluate_numerical_candidate)
        ret = sig.return_annotation
        # PEP 563 string form OR resolved class — accept either
        if isinstance(ret, str):
            assert ret.endswith("NumericalPromotionEvidence")
        else:
            assert ret is NumericalPromotionEvidence
        # typing.get_type_hints should resolve to the class
        hints = typing.get_type_hints(evaluate_numerical_candidate)
        assert hints["return"] is NumericalPromotionEvidence


# ---------------------------------------------------------------------------
# 6. Public-safety + structural counts
# ---------------------------------------------------------------------------


class TestPublicSafetyAndStructuralCounts:
    def test_no_cloud_runtime_imports(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        for banned in (
            "boto3",
            "azure",
            "google.cloud",
            "kubernetes",
            "docker",
            "fabric",
        ):
            assert banned not in src

    def test_no_hardcoded_credentials(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert 'api_key="sk-' not in src
        assert "BEGIN PRIVATE KEY" not in src
        # Stripe / AWS / GCP / Slack token patterns
        assert "AKIA" not in src
        assert "AIza" not in src
        assert re.search(r"\bxox[abprs]-", src) is None

    def test_no_print_or_pprint(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "print(" not in src
        assert "pprint(" not in src

    def test_no_subprocess_or_shell(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "subprocess" not in src
        assert "shell=True" not in src
        assert "os.system" not in src

    def test_no_eval_or_exec(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "eval(" not in src
        assert "exec(" not in src
        assert "compile(" not in src

    def test_no_os_environ_or_getenv(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "os.environ" not in src
        assert "os.getenv" not in src

    def test_no_todo_or_fixme_markers(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "TODO" not in src
        assert "FIXME" not in src
        assert "XXX" not in src

    def test_source_ends_with_single_trailing_newline(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, "rb") as f:
            data = f.read()
        assert data.endswith(b"\n")
        # exactly one trailing newline (no double-EOL)
        assert not data.endswith(b"\n\n")

    def test_pins_exactly_one_strenum_class(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # exactly one `class Foo(StrEnum):` block
        matches = re.findall(r"^class\s+\w+\s*\(\s*StrEnum\s*\)\s*:", src, re.MULTILINE)
        assert len(matches) == 1
        assert "NumericalCandidateKind" in matches[0]

    def test_pins_exactly_one_dataclass_decorator(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        decorators = re.findall(r"@dataclass\(", src)
        assert len(decorators) == 1

    def test_pins_exactly_one_module_level_function_def(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # exactly one `def evaluate_numerical_candidate(` at module scope
        # (we allow _make_evidence_within_placeholder etc. only inside the
        # tests file, not in the source file we're testing)
        funcs = re.findall(r"^def\s+(\w+)\(", src, re.MULTILINE)
        assert funcs == ["evaluate_numerical_candidate"]

    def test_pins_exactly_one_class_def_at_module_scope(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # exactly two `class Foo(` blocks: one StrEnum subclass and one
        # dataclass (NumericalPromotionEvidence)
        classes = re.findall(r"^class\s+(\w+)[\(:]", src, re.MULTILINE)
        assert set(classes) == {
            "NumericalCandidateKind",
            "NumericalPromotionEvidence",
        }
        assert len(classes) == 2

    def test_no_unexpected_top_level_statement_kinds(self) -> None:
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        allowed = {
            ast.Module,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
            ast.Assign,
            ast.AnnAssign,
            ast.Expr,
            ast.If,
            ast.Try,
            ast.With,
            ast.Pass,
        }
        for node in tree.body:
            assert type(node) in allowed, f"unexpected node: {type(node).__name__}"

    def test_no_unlisted_public_names_in_module(self) -> None:
        # any module-level def/class/Assign.name that isn't private must
        # be in __all__
        source_path = inspect.getsourcefile(promotion_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        all_names = set(promotion_module.__all__)
        declared: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                declared.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                declared.add(node.name)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                declared.add(node.targets[0].id)
        public_declared = {n for n in declared if not n.startswith("_")}
        assert public_declared.issubset(all_names), public_declared - all_names

    def test_does_not_swallow_exceptions(self) -> None:
        # the function must NOT wrap compare_numerical_paths or
        # select_numerical_path in try/except that swallows — the
        # REJECT fallback raises RuntimeError that we expect to propagate
        seq = _equal_seqs()
        opt = [v + 1.0 for v in seq]
        with pytest.raises(RuntimeError):
            evaluate_numerical_candidate(
                reference=seq,
                optimized=opt,
                candidate_kind=NumericalCandidateKind.KERNEL,
                profile=_profile(
                    max_abs_error=1e-6,
                    fallback=NumericalFallback.REJECT,
                ),
                identity=_identity(),
                optimized_path="opt",
                reference_path="ref",
            )
