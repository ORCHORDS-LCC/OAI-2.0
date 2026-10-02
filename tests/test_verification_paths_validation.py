"""Structural / validation tests for ``oai2.verification.paths``.

Pin the WI-TRUTH-001 evidence-path adapters: the three
``assess_*_evidence`` functions (``assess_repository_evidence``,
``assess_tool_runtime_evidence``, ``assess_web_evidence``) that wrap
:class:`ClaimEvidencePolicy.assess` for each runtime evidence path,
plus the shared ``_validate_binding_classes`` private helper.

Behavioural end-to-end coverage lives in ``tests/test_verification_paths.py``;
this file pins the *shape* of the API and the invariants the call
sequencer relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from oai2.core import EvidenceId
from oai2.verification import (
    ClaimClass,
    ClaimEvidencePolicy,
    Evidence,
    EvidenceAssessment,
    EvidenceBinding,
    EvidenceClass,
)
from oai2.verification.paths import (
    assess_repository_evidence,
    assess_tool_runtime_evidence,
    assess_web_evidence,
)
from oai2.verification.paths import (
    assess_repository_evidence as assess_repository_evidence_from_module,
)
from oai2.verification.paths import (
    assess_tool_runtime_evidence as assess_tool_runtime_evidence_from_module,
)
from oai2.verification.paths import (
    assess_web_evidence as assess_web_evidence_from_module,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "verification" / "paths.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Module docstring + import surface
# ---------------------------------------------------------------------------


def test_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "verification/paths.py is unexpectedly empty"


def test_module_has_docstring() -> None:
    """The module ships an overview of the runtime-facing path adapters."""

    assert _MODULE_SOURCE.startswith('"""')
    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert first_para, "module docstring is empty"


def test_module_docstring_mentions_evidence_paths() -> None:
    """The overview mentions ``evidence`` (path adapters deal in evidence)."""

    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert "evidence" in first_para.lower()


def test_module_uses_future_annotations() -> None:
    """``from __future__ import annotations`` is present (UP006-clean)."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_module_imports_use_relative_evidence_and_policy() -> None:
    """``EvidenceClass`` is from ``.evidence`` and the policy types are from
    ``.policy``; no absolute ``from oai2`` imports."""

    # EvidenceClass from .evidence (single-line form)
    evidence_import = re.search(
        r"^from \.evidence\s+import\s+([^\n]+)", _MODULE_SOURCE, re.MULTILINE
    )
    assert evidence_import, "missing relative evidence import line"
    assert "EvidenceClass" in evidence_import.group(1)

    # ClaimClass, ClaimEvidencePolicy, EvidenceAssessment, EvidenceBinding from
    # .policy. The source uses a parenthesised multi-line import — capture the
    # entire block including newlines with a DOTALL search.
    assert "from .policy import (" in _MODULE_SOURCE, "missing relative policy import block"
    for name in ("ClaimClass", "ClaimEvidencePolicy", "EvidenceAssessment", "EvidenceBinding"):
        assert re.search(rf"^\s+{name}\s*,?\s*$", _MODULE_SOURCE, re.MULTILINE), (
            f"missing relative import: {name}"
        )

    # No accidental absolute imports.
    assert not re.search(r"^import oai2\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.evidence\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.policy\b", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^from oai2\.verification\b", _MODULE_SOURCE, re.MULTILINE)


def test_module_imports_collections_abc_sequence() -> None:
    """The module uses ``collections.abc.Sequence`` for its public bindings argument."""

    assert "from collections.abc import Sequence" in _MODULE_SOURCE


def test_module_has_no_wildcard_imports() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


# ---------------------------------------------------------------------------
# __all__ completeness + package re-export
# ---------------------------------------------------------------------------


def test_dunder_all_lists_exactly_three_public_names() -> None:
    """The module's public surface is exactly 3 names — no more, no less."""

    import oai2.verification.paths as mod

    assert isinstance(mod.__all__, list)
    assert set(mod.__all__) == {
        "assess_repository_evidence",
        "assess_tool_runtime_evidence",
        "assess_web_evidence",
    }
    assert len(mod.__all__) == 3


def test_each_all_name_is_importable_from_module() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    import oai2.verification.paths as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ name missing: {name}"


def test_public_names_are_reexported_from_verification_package() -> None:
    """Top-level ``oai2.verification`` re-exports the 3 path-adapter names."""

    import oai2.verification as pkg

    for name in (
        "assess_repository_evidence",
        "assess_tool_runtime_evidence",
        "assess_web_evidence",
    ):
        assert name in pkg.__all__, f"package re-export missing: {name}"


def test_package_reexport_preserves_identity() -> None:
    """Package re-exports point at the *same* functions as the module."""

    import oai2.verification as pkg
    import oai2.verification.paths as mod

    assert pkg.assess_repository_evidence is mod.assess_repository_evidence
    assert pkg.assess_tool_runtime_evidence is mod.assess_tool_runtime_evidence
    assert pkg.assess_web_evidence is mod.assess_web_evidence


def test_module_imports_match_top_level() -> None:
    """``from oai2.verification.paths import X`` matches the top-level re-export."""

    assert assess_repository_evidence is assess_repository_evidence_from_module
    assert assess_tool_runtime_evidence is assess_tool_runtime_evidence_from_module
    assert assess_web_evidence is assess_web_evidence_from_module


# ---------------------------------------------------------------------------
# assess_repository_evidence
# ---------------------------------------------------------------------------


def _policy() -> ClaimEvidencePolicy:
    return ClaimEvidencePolicy(current_external_max_age_seconds=60.0)


def _evidence(
    evidence_id: str,
    cls: EvidenceClass,
    *,
    observed_at: float = 0.0,
) -> Evidence:
    return Evidence(
        id=EvidenceId(evidence_id),
        cls=cls,
        summary="a piece of evidence",
        observed_at=observed_at,
    )


def test_assess_repository_evidence_is_a_function() -> None:
    """``assess_repository_evidence`` is a regular function."""

    import inspect

    assert callable(assess_repository_evidence)
    assert inspect.isfunction(assess_repository_evidence)


def test_assess_repository_evidence_signature() -> None:
    """``(policy, bindings, *, current_state_version)`` — ``current_state_version``
    is the only keyword-only parameter."""

    import inspect

    sig = inspect.signature(assess_repository_evidence)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["policy", "bindings", "current_state_version"]
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY


def test_assess_repository_evidence_empty_bindings_returns_evidence_assessment() -> None:
    """Empty bindings still produce an :class:`EvidenceAssessment` for the
    correct claim class (``REPOSITORY_STATE``)."""

    out = assess_repository_evidence(
        _policy(),
        (),
        current_state_version="v1",
    )
    assert isinstance(out, EvidenceAssessment)
    assert out.claim_class is ClaimClass.REPOSITORY_STATE


def test_assess_repository_evidence_accepts_repo_source_binding() -> None:
    """``REPO_SOURCE`` is one of the three accepted repository evidence classes."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.REPO_SOURCE),
        state_version="v1",
    )
    out = assess_repository_evidence(
        _policy(),
        (binding,),
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.REPOSITORY_STATE


def test_assess_repository_evidence_accepts_change_history_binding() -> None:
    """``CHANGE_HISTORY`` is one of the three accepted repository evidence classes."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.CHANGE_HISTORY),
        state_version="v1",
    )
    out = assess_repository_evidence(
        _policy(),
        (binding,),
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.REPOSITORY_STATE


def test_assess_repository_evidence_accepts_deterministic_binding() -> None:
    """``DETERMINISTIC`` is one of the three accepted repository evidence classes."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.DETERMINISTIC),
        state_version="v1",
    )
    out = assess_repository_evidence(
        _policy(),
        (binding,),
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.REPOSITORY_STATE


def test_assess_repository_evidence_accepts_mixed_class_bindings() -> None:
    """A mixed-tuple of the three accepted classes is allowed in one call."""

    bindings = (
        EvidenceBinding(
            evidence=_evidence("e1", EvidenceClass.REPO_SOURCE),
            state_version="v1",
        ),
        EvidenceBinding(
            evidence=_evidence("e2", EvidenceClass.CHANGE_HISTORY),
            state_version="v1",
        ),
        EvidenceBinding(
            evidence=_evidence("e3", EvidenceClass.DETERMINISTIC),
            state_version="v1",
        ),
    )
    out = assess_repository_evidence(
        _policy(),
        bindings,
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.REPOSITORY_STATE


def test_assess_repository_evidence_rejects_runtime_obs_binding() -> None:
    """``RUNTIME_OBS`` is not a repository evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.RUNTIME_OBS),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="repository"):
        assess_repository_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_repository_evidence_rejects_external_binding() -> None:
    """``EXTERNAL`` is not a repository evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.EXTERNAL),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="repository"):
        assess_repository_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_repository_evidence_rejects_visual_obs_binding() -> None:
    """``VISUAL_OBS`` is not a repository evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.VISUAL_OBS),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="repository"):
        assess_repository_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_repository_evidence_rejects_hypothesis_binding() -> None:
    """``HYPOTHESIS`` is not a repository evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.HYPOTHESIS),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="repository"):
        assess_repository_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_repository_evidence_requires_current_state_version() -> None:
    """``current_state_version`` is required — empty raises ``ValueError``."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.REPO_SOURCE),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="current_state_version"):
        assess_repository_evidence(
            _policy(),
            (binding,),
            current_state_version="",
        )


# ---------------------------------------------------------------------------
# assess_tool_runtime_evidence
# ---------------------------------------------------------------------------


def test_assess_tool_runtime_evidence_is_a_function() -> None:
    """``assess_tool_runtime_evidence`` is a regular function."""

    import inspect

    assert callable(assess_tool_runtime_evidence)
    assert inspect.isfunction(assess_tool_runtime_evidence)


def test_assess_tool_runtime_evidence_signature() -> None:
    """``(policy, bindings, *, current_state_version)``."""

    import inspect

    sig = inspect.signature(assess_tool_runtime_evidence)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["policy", "bindings", "current_state_version"]
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY


def test_assess_tool_runtime_evidence_empty_bindings_returns_evidence_assessment() -> None:
    """Empty bindings produce an :class:`EvidenceAssessment` for the
    correct claim class (``TOOL_RUNTIME_OBSERVATION``)."""

    out = assess_tool_runtime_evidence(
        _policy(),
        (),
        current_state_version="v1",
    )
    assert isinstance(out, EvidenceAssessment)
    assert out.claim_class is ClaimClass.TOOL_RUNTIME_OBSERVATION


def test_assess_tool_runtime_evidence_accepts_runtime_obs_binding() -> None:
    """``RUNTIME_OBS`` is one of the two accepted tool/runtime evidence classes."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.RUNTIME_OBS),
        state_version="v1",
    )
    out = assess_tool_runtime_evidence(
        _policy(),
        (binding,),
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.TOOL_RUNTIME_OBSERVATION


def test_assess_tool_runtime_evidence_accepts_deterministic_binding() -> None:
    """``DETERMINISTIC`` is one of the two accepted tool/runtime evidence classes."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.DETERMINISTIC),
        state_version="v1",
    )
    out = assess_tool_runtime_evidence(
        _policy(),
        (binding,),
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.TOOL_RUNTIME_OBSERVATION


def test_assess_tool_runtime_evidence_accepts_mixed_class_bindings() -> None:
    """A mixed-tuple of the two accepted classes is allowed in one call."""

    bindings = (
        EvidenceBinding(
            evidence=_evidence("e1", EvidenceClass.RUNTIME_OBS),
            state_version="v1",
        ),
        EvidenceBinding(
            evidence=_evidence("e2", EvidenceClass.DETERMINISTIC),
            state_version="v1",
        ),
    )
    out = assess_tool_runtime_evidence(
        _policy(),
        bindings,
        current_state_version="v1",
    )
    assert out.claim_class is ClaimClass.TOOL_RUNTIME_OBSERVATION


def test_assess_tool_runtime_evidence_rejects_repo_source_binding() -> None:
    """``REPO_SOURCE`` is not a tool/runtime evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.REPO_SOURCE),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="tool/runtime"):
        assess_tool_runtime_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_tool_runtime_evidence_rejects_change_history_binding() -> None:
    """``CHANGE_HISTORY`` is not a tool/runtime evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.CHANGE_HISTORY),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="tool/runtime"):
        assess_tool_runtime_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_tool_runtime_evidence_rejects_external_binding() -> None:
    """``EXTERNAL`` is not a tool/runtime evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.EXTERNAL),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="tool/runtime"):
        assess_tool_runtime_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


def test_assess_tool_runtime_evidence_rejects_visual_obs_binding() -> None:
    """``VISUAL_OBS`` is not a tool/runtime evidence class."""

    binding = EvidenceBinding(
        evidence=_evidence("e1", EvidenceClass.VISUAL_OBS),
        state_version="v1",
    )
    with pytest.raises(ValueError, match="tool/runtime"):
        assess_tool_runtime_evidence(
            _policy(),
            (binding,),
            current_state_version="v1",
        )


# ---------------------------------------------------------------------------
# assess_web_evidence
# ---------------------------------------------------------------------------


def test_assess_web_evidence_is_a_function() -> None:
    """``assess_web_evidence`` is a regular function."""

    import inspect

    assert callable(assess_web_evidence)
    assert inspect.isfunction(assess_web_evidence)


def test_assess_web_evidence_signature() -> None:
    """``(policy, bindings, *, current, now=None)`` — both ``current`` and ``now``
    are keyword-only."""

    import inspect

    sig = inspect.signature(assess_web_evidence)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["policy", "bindings", "current", "now"]
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[3].kind is inspect.Parameter.KEYWORD_ONLY
    # ``now`` defaults to None
    assert params[3].default is None


def test_assess_web_evidence_current_true_uses_external_current_fact_claim_class() -> None:
    """``current=True`` selects the ``EXTERNAL_CURRENT_FACT`` claim class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL))
    out = assess_web_evidence(
        _policy(),
        (binding,),
        current=True,
        now=1000.0,
    )
    assert out.claim_class is ClaimClass.EXTERNAL_CURRENT_FACT


def test_assess_web_evidence_current_false_uses_external_stable_fact_claim_class() -> None:
    """``current=False`` selects the ``EXTERNAL_STABLE_FACT`` claim class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL))
    out = assess_web_evidence(
        _policy(),
        (binding,),
        current=False,
    )
    assert out.claim_class is ClaimClass.EXTERNAL_STABLE_FACT


def test_assess_web_evidence_accepts_now_none_when_current() -> None:
    """``now`` is keyword-only and defaults to ``None``; passing ``None``
    explicitly is also accepted for the call-shape contract (the policy
    layer will raise its own ``ValueError`` if needed)."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL))
    with pytest.raises(ValueError):
        # policy.assess raises when now is None and freshness is required.
        assess_web_evidence(
            _policy(),
            (binding,),
            current=True,
            now=None,
        )


def test_assess_web_evidence_accepts_now_value_when_fixed() -> None:
    """``now`` accepts an explicit float."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL))
    out = assess_web_evidence(
        _policy(),
        (binding,),
        current=True,
        now=1700000000.0,
    )
    assert out.claim_class is ClaimClass.EXTERNAL_CURRENT_FACT


def test_assess_web_evidence_accepts_external_binding() -> None:
    """``EXTERNAL`` is the only accepted web evidence class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL))
    out = assess_web_evidence(
        _policy(),
        (binding,),
        current=False,
    )
    assert out.claim_class is ClaimClass.EXTERNAL_STABLE_FACT


def test_assess_web_evidence_rejects_repo_source_binding() -> None:
    """``REPO_SOURCE`` is not a web evidence class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.REPO_SOURCE))
    with pytest.raises(ValueError, match="web"):
        assess_web_evidence(
            _policy(),
            (binding,),
            current=False,
        )


def test_assess_web_evidence_rejects_deterministic_binding() -> None:
    """``DETERMINISTIC`` is not a web evidence class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.DETERMINISTIC))
    with pytest.raises(ValueError, match="web"):
        assess_web_evidence(
            _policy(),
            (binding,),
            current=False,
        )


def test_assess_web_evidence_rejects_runtime_obs_binding() -> None:
    """``RUNTIME_OBS`` is not a web evidence class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.RUNTIME_OBS))
    with pytest.raises(ValueError, match="web"):
        assess_web_evidence(
            _policy(),
            (binding,),
            current=False,
        )


def test_assess_web_evidence_rejects_change_history_binding() -> None:
    """``CHANGE_HISTORY`` is not a web evidence class."""

    binding = EvidenceBinding(evidence=_evidence("e1", EvidenceClass.CHANGE_HISTORY))
    with pytest.raises(ValueError, match="web"):
        assess_web_evidence(
            _policy(),
            (binding,),
            current=False,
        )


# ---------------------------------------------------------------------------
# _validate_binding_classes private helper
# ---------------------------------------------------------------------------


def test_validate_binding_classes_helper_is_callable() -> None:
    """``_validate_binding_classes`` exists on the module and is callable."""

    import oai2.verification.paths as mod

    assert hasattr(mod, "_validate_binding_classes")
    assert callable(mod._validate_binding_classes)


def test_validate_binding_classes_not_in_dunder_all() -> None:
    """The helper is private — it must NOT appear in ``__all__``."""

    import oai2.verification.paths as mod

    assert "_validate_binding_classes" not in mod.__all__


def test_validate_binding_classes_passes_when_all_in_accepted_set() -> None:
    """When every binding's class is in the accepted set, no exception is raised."""

    import oai2.verification.paths as mod

    bindings = (
        EvidenceBinding(evidence=_evidence("e1", EvidenceClass.REPO_SOURCE)),
        EvidenceBinding(evidence=_evidence("e2", EvidenceClass.CHANGE_HISTORY)),
    )
    mod._validate_binding_classes(
        bindings,
        {EvidenceClass.REPO_SOURCE, EvidenceClass.CHANGE_HISTORY, EvidenceClass.DETERMINISTIC},
        "repository",
    )


def test_validate_binding_classes_raises_value_error_on_wrong_class() -> None:
    """A binding whose class is NOT in the accepted set raises ``ValueError``."""

    import oai2.verification.paths as mod

    bindings = (EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL)),)
    with pytest.raises(ValueError):
        mod._validate_binding_classes(
            bindings,
            {EvidenceClass.REPO_SOURCE},
            "repository",
        )


def test_validate_binding_classes_error_message_includes_path_name() -> None:
    """The error message identifies the offending evidence path."""

    import oai2.verification.paths as mod

    bindings = (EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL)),)
    with pytest.raises(ValueError) as exc:
        mod._validate_binding_classes(
            bindings,
            {EvidenceClass.REPO_SOURCE},
            "my-custom-path-name",
        )
    assert "my-custom-path-name" in str(exc.value)


def test_validate_binding_classes_error_message_lists_accepted_classes() -> None:
    """The error message lists the accepted class wire strings (sorted)."""

    import oai2.verification.paths as mod

    bindings = (EvidenceBinding(evidence=_evidence("e1", EvidenceClass.EXTERNAL)),)
    with pytest.raises(ValueError) as exc:
        mod._validate_binding_classes(
            bindings,
            {EvidenceClass.REPO_SOURCE, EvidenceClass.CHANGE_HISTORY, EvidenceClass.DETERMINISTIC},
            "repository",
        )
    message = str(exc.value)
    for cls in (
        EvidenceClass.REPO_SOURCE,
        EvidenceClass.CHANGE_HISTORY,
        EvidenceClass.DETERMINISTIC,
    ):
        assert str(cls.value) in message, f"missing accepted class: {cls.value}"


def test_validate_binding_classes_empty_bindings_passes() -> None:
    """An empty bindings tuple passes validation (no class to validate)."""

    import oai2.verification.paths as mod

    mod._validate_binding_classes(
        (),
        {EvidenceClass.REPO_SOURCE, EvidenceClass.CHANGE_HISTORY},
        "repository",
    )


# ---------------------------------------------------------------------------
# Public-safety boundary
# ---------------------------------------------------------------------------


def test_module_source_has_no_cloud_sdk_imports() -> None:
    """No cloud SDK / orchestration / network dependency surface in the source."""

    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "kubernetes",
        "docker",
        "fabric",
    )
    for marker in forbidden:
        assert marker not in _MODULE_SOURCE, f"forbidden import marker: {marker}"


def test_module_source_has_no_hardcoded_secrets() -> None:
    """No literal API-key / private-key / token strings."""

    forbidden_patterns = (
        r"sk-[A-Za-z0-9_-]{8,}",
        r"api_key\s*=\s*['\"]sk-",
        r"BEGIN PRIVATE KEY",
        r"BEGIN RSA PRIVATE KEY",
        r"AWS_SECRET_ACCESS_KEY",
    )
    for pat in forbidden_patterns:
        assert not re.search(pat, _MODULE_SOURCE), f"secret pattern: {pat}"


def test_module_source_has_no_print_or_pprint() -> None:
    """No stdout printing — path adapters are silent."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_module_source_has_no_subprocess_or_os_system() -> None:
    """No subprocess / ``os.system`` invocations — path adapters are pure."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_module_source_has_no_network_requests() -> None:
    """No HTTP client surface — the path adapters do not talk to the network."""

    forbidden = ("requests.", "urllib.request", "httpx.", "aiohttp.")
    for marker in forbidden:
        assert marker not in _MODULE_SOURCE, f"network marker: {marker}"


def test_module_source_has_no_eval_or_exec() -> None:
    """No dynamic code execution."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_module_source_has_no_env_access() -> None:
    """No ``os.environ`` / ``os.getenv`` — the path adapters are deterministic."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_module_source_has_no_todo_or_fixme() -> None:
    """No ``# TODO`` / ``# FIXME`` / ``# XXX`` markers."""

    forbidden = ("# TODO", "# FIXME", "# XXX")
    for marker in forbidden:
        assert marker not in _MODULE_SOURCE, f"todo marker: {marker}"


def test_module_source_pins_three_public_functions() -> None:
    """The source declares exactly three top-level public assess_*_evidence functions."""

    # Each is ``def <name>(...):`` at column 0.
    matches = re.findall(r"^def\s+(assess_[a-z_]+_evidence)\(", _MODULE_SOURCE, re.MULTILINE)
    assert set(matches) == {
        "assess_repository_evidence",
        "assess_tool_runtime_evidence",
        "assess_web_evidence",
    }
    assert len(matches) == 3


def test_module_source_pins_one_private_helper() -> None:
    """The source declares exactly one private ``_validate_binding_classes`` helper."""

    matches = re.findall(r"^def\s+(_validate_binding_classes)\(", _MODULE_SOURCE, re.MULTILINE)
    assert matches == ["_validate_binding_classes"]


def test_module_source_pins_three_dunder_all_entries() -> None:
    """The source's ``__all__`` lists exactly 3 public names."""

    all_match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert all_match, "missing __all__ block"
    all_block = all_match.group(1)
    expected = {
        "assess_repository_evidence",
        "assess_tool_runtime_evidence",
        "assess_web_evidence",
    }
    actual = set(re.findall(r'"([a-z_]+)"', all_block))
    assert actual == expected
    assert len(actual) == 3
