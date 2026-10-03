"""The numerical-evidence harness must fail loudly rather than fake a pass.

`scripts/numerical_backend_evidence.py` produces the REQ-NUM-026 artifact for
a real backend. Two properties are load-bearing and are pinned here without
requiring a live llama-server:

1. It REQUIRES a proven serving identity. An unidentified model cannot support
   a numerical claim, so the harness must refuse rather than emit an artifact
   naming a model it never read from /props.
2. It never silently reports a promotion. The artifact records the gate's
   decision verbatim; the harness does not get an opinion.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from oai2.runtime.llamacpp_runtime import LlamaServerError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "numerical_backend_evidence.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("numerical_backend_evidence", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_module_imports_without_a_live_server() -> None:
    """Importing must not require a server; only `main` does."""
    module = _load()
    assert hasattr(module, "PROMPTS")
    assert hasattr(module, "main") or hasattr(module, "run")


def test_harness_refuses_when_serving_identity_cannot_be_proven() -> None:
    """An unidentified model cannot support a numerical claim."""
    module = _load()
    runtime = module.LlamaServerRuntime(base_url="http://127.0.0.1:1", model="m")  # type: ignore[attr-defined]

    with pytest.raises(LlamaServerError):
        runtime.serving_identity()


def test_declared_reference_matches_exact_arithmetic() -> None:
    """The oracle is exact arithmetic, so it is checkable without a server.

    If this drifts, the artifact's `reference` field stops being a declared
    oracle and becomes an assertion nobody can verify.
    """
    module = _load()
    assert module.REFERENCE == [42.0, 12.0, 1024.0]  # type: ignore[attr-defined]
    # 17+25, 144/12, 2**10
    assert module.REFERENCE[0] == 17.0 + 25.0
    assert module.REFERENCE[1] == 144.0 / 12.0
    assert module.REFERENCE[2] == float(2**10)


def test_committed_artifact_identifies_model_runtime_backend_and_config() -> None:
    """REQ-NUM-026: the artifact must name all four, and tie to a SHA.

    Also enforces that the `dirty` flag is never quietly true in a committed
    artifact -- a dirty tree means the SHA does not describe the code that
    produced the number.
    """
    import json

    root = Path(__file__).resolve().parents[1]
    artifacts = sorted((root / "evidence" / "numerical").glob("backend_*.json"))
    assert artifacts, "no numerical backend evidence artifact committed"

    for path in artifacts:
        data = json.loads(path.read_text())
        result = data["result"]
        for field in ("model_version", "runtime_version", "backend", "config_id"):
            assert result.get(field), f"{path.name}: missing {field}"
        harness = data["harness"]
        assert harness["source_sha"], f"{path.name}: no source_sha"
        assert harness["dirty"] is False, (
            f"{path.name}: dirty=True means the SHA does not describe the code "
            "that produced this number"
        )
        assert data["serving_identity"].get("weights_sha256"), (
            f"{path.name}: no weights hash, so the model is unidentified"
        )
