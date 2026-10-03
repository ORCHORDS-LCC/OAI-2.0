"""Run the REAL llama.cpp NORMAL lane through the canonical promotion gate.

This is the first real BACKEND candidate available in this repository
(`llamacpp_runtime.py` arrived on main in a133744), and #217's remaining
closure work is exactly this: REQ-NUM-026 (artifacts identifying
model/runtime/backend/config versions) and AC-NUM-021 together.

Serving identity is read from /props, never asserted. Weights hash is
computed from the file the server reports. Both are recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from oai2.model import (
    NumericalArtifactIdentity,
    NumericalFallback,
    NumericalOperation,
    NumericalToleranceProfile,
)
from oai2.model.numerical_promotion import (
    NumericalCandidateKind,
    evaluate_numerical_candidate,
)
from oai2.runtime.inference import InferenceRequest
from oai2.runtime.llamacpp_runtime import (
    DEFAULT_SEED,
    LlamaServerRuntime,
)

# A deterministic numeric probe, pinned at temperature 0 so the measurement
# is reproducible across runs and comparable across SHAs.
PROMPTS = [
    "Compute 17 + 25. Reply with only the integer.",
    "Compute 144 / 12. Reply with only the integer.",
    "Compute 2^10. Reply with only the integer.",
]

#: The declared oracle for REQ-NUM-021: exact integer arithmetic. Written
#: literally (not derived) so a test can pin that it IS the exact answer --
#: an oracle that drifts into an assertion is not an oracle.
REFERENCE = [42.0, 12.0, 1024.0]


def main() -> int:
    BASE = "http://127.0.0.1:8851"

    runtime = LlamaServerRuntime(base_url=BASE, model="smollm2-1.7b-q4km")
    identity = runtime.serving_identity()
    print("=== SERVING IDENTITY (read from /props) ===")
    for k, v in identity.items():
        print(f"  {k:20s} = {v}")

    path = identity.get("model_path")
    weights_sha = None
    if path and os.path.exists(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        weights_sha = h.hexdigest()
    print(f"  {'weights_sha256':20s} = {weights_sha}")
    print(f"  {'weights_bytes':20s} = {os.path.getsize(path) if path else None}")
    print()




    def ask(prompt: str) -> float:
        resp = runtime.generate(
            InferenceRequest(
                prompt=prompt,
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                seed=DEFAULT_SEED,
            )
        )
        text = (resp.text or "").strip()
        digits = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
        if not digits:
            raise SystemExit(f"no numeric output from: {text!r}")
        return float(digits)


    print("=== LIVE GENERATION (llama-server NORMAL lane, temperature=0.0) ===")
    optimized: list[float] = []
    for p in PROMPTS:
        v = ask(p)
        optimized.append(v)
        print(f"  {p[:44]:46s} -> {v}")
    print()

    reference = list(REFERENCE)
    print(f"  declared reference (exact arithmetic) = {reference}")
    print()

    profile = NumericalToleranceProfile(
        profile_id="arithmetic-infer-fp32-scalar-v1",
        operation=NumericalOperation.NORMALIZATION,
        dtype="q4_K_M",
        shape_class="scalar",
        max_abs_error=1.0e-6,
        max_rel_error=1.0e-6,
        require_finite_state_match=True,
        fallback=NumericalFallback.USE_REFERENCE,
    )

    artifact = NumericalArtifactIdentity(
        model_version=identity.get("model_alias") or "unknown",
        runtime_version=f"llama.cpp/{weights_sha[:12] if weights_sha else 'unknown'}",
        backend=f"llamacpp-server:{BASE}",
        config_id=f"q4_K_M@seed{DEFAULT_SEED}@temp0.0",
    )

    evidence = evaluate_numerical_candidate(
        reference,
        optimized,
        candidate_kind=NumericalCandidateKind.BACKEND,
        profile=profile,
        identity=artifact,
        optimized_path=f"llamacpp:{BASE}",
        reference_path="exact-arithmetic",
        speedup_ratio=None,
    )

    c = evidence.comparison
    print("=== NUMERICAL COMPARISON ARTIFACT (REQ-NUM-026) ===")
    print(f"  profile_id        = {c.profile_id}")
    print(f"  model_version     = {c.identity.model_version}")
    print(f"  runtime_version   = {c.identity.runtime_version}")
    print(f"  backend           = {c.identity.backend}")
    print(f"  config_id         = {c.identity.config_id}")
    print(f"  sample_count      = {c.sample_count}")
    print(f"  max_abs_error     = {c.max_abs_error}")
    print(f"  max_rel_error     = {c.max_rel_error}")
    print(f"  finite_state_match= {c.finite_state_match}")
    print(f"  passed            = {c.passed}")
    print(f"  failures          = {c.failures}")
    print()
    print("=== PROMOTION DECISION ===")
    print(f"  eligible      = {evidence.eligible}")
    print(f"  used_fallback = {evidence.selection.used_fallback}")
    print(f"  selected      = {getattr(evidence.selection, 'selected_path', None)}")
    print()
    print("NOTE: correctness of the SUITE is a separate question from whether")
    print("this backend is numerically equivalent on the probed operation.")
    print("This artifact claims only the latter.")
    print()

    # ---------------------------------------------------------------- artifact
    # REQ-NUM-026: the artifact must identify model/runtime/backend/config
    # versions, and it must be tied to the exact source SHA that produced it --
    # the same rule evidence/accuracy/README.md states, for the same reason: a
    # number nobody can re-derive is not evidence.

    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # Dirtiness is measured over TRACKED files only. Including untracked
    # paths would be circular: the artifact being written is itself
    # untracked at the moment this runs, so a full `git status --porcelain`
    # reports dirty=True for every run and the field can never mean
    # anything. What matters is whether the CODE that produced the number
    # was modified, which is exactly what `--untracked-files=no` asks.
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "harness": {
            "script": "scripts/numerical_backend_evidence.py",
            "source_sha": rev,
            "dirty": dirty,
            "gate": "oai2.model.numerical_promotion.evaluate_numerical_candidate",
        },
        "serving_identity": {
            **{k: v for k, v in identity.items()},
            "weights_sha256": weights_sha,
            "weights_bytes": os.path.getsize(path) if path else None,
        },
        "operation": {
            "profile_id": c.profile_id,
            "operation": profile.operation.value,
            "dtype": profile.dtype,
            "shape_class": profile.shape_class,
            "max_abs_error_tolerance": profile.max_abs_error,
            "max_rel_error_tolerance": profile.max_rel_error,
            "require_finite_state_match": profile.require_finite_state_match,
            "fallback": profile.fallback.value,
        },
        "samples": {
            "prompts": PROMPTS,
            "reference": reference,
            "optimized": optimized,
            "reference_basis": "exact integer arithmetic (declared oracle)",
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": DEFAULT_SEED,
                "max_tokens": 16,
            },
        },
        "result": {
            "candidate_kind": "BACKEND",
            "model_version": c.identity.model_version,
            "runtime_version": c.identity.runtime_version,
            "backend": c.identity.backend,
            "config_id": c.identity.config_id,
            "sample_count": c.sample_count,
            "measured_max_abs_error": c.max_abs_error,
            "measured_max_rel_error": c.max_rel_error,
            "finite_state_match": c.finite_state_match,
            "passed": c.passed,
            "failures": list(c.failures),
            "eligible": evidence.eligible,
            "used_fallback": evidence.selection.used_fallback,
        },
        "scope": (
            "This artifact claims only that the llama.cpp NORMAL lane was run "
            "through the canonical numerical promotion gate on the probed "
            "operation, and what the gate decided. It is NOT a claim that the "
            "backend is or is not fit for production, and NOT an accuracy "
            "figure for any suite."
        ),
    }

    out_dir = Path("evidence/numerical")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"backend_noran_{rev[:12]}.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n")
    print(f"=== ARTIFACT WRITTEN ===\n  {out}")


if __name__ == "__main__":
    raise SystemExit(main())
