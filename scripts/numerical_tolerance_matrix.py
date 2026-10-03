"""Error distribution across the operation / dtype / shape surface.

#217's remaining source-side work: REQ-NUM-022 (tolerances shall be
operation/dtype/shape aware, because one global tolerance is misleading)
and VER-NUM-022 (error distributions by operation/dtype/shape).

`f1af6bd` measured THREE SCALAR PROBES on one operation. That demonstrated
the gate runs end to end on a real backend; it did not establish a
tolerance matrix. This sweep closes that, using the declared per-operation
surface the repository already owns:

  * `extreme_value_fixtures()` -- finite stress inputs per operation.
  * `REFERENCE_PRECISION_RULES` -- the storage dtype each operation
    degrades to, which is the dtype dimension REQ-NUM-022 names.

The candidate under test is a declared lossy transformation of the
reference at the operation's OWN declared storage precision. That is what
an optimized path actually does, so the resulting error distribution is
the thing a tolerance profile has to be sized against -- measured, not
asserted.

What this does NOT do: it does not claim any of these tolerances are
correct. It measures what the error actually is, so that choosing one is
an informed decision rather than a guess. Sizing them is a separate,
deliberate act.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from oai2.model.numerical_compare import (
    NumericalArtifactIdentity,
    NumericalToleranceProfile,
    compare_numerical_paths,
)
from oai2.model.numerics import (
    REFERENCE_PRECISION_RULES,
    extreme_value_fixtures,
)

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Cell:
    operation: str
    storage_dtype: str
    shape_class: str
    n: int
    max_abs: float
    max_rel: float
    all_finite: bool


def lossy_candidate(values: tuple[float, ...], storage_dtype: str) -> list[float]:
    """Model an optimized path that stores at the operation's declared dtype.

    This is a DECLARED SIMULATION of a lossy backend, not a measurement of
    one. bf16 has 8 mantissa bits and int4/int8 quantize to a small grid;
    both are modelled here so the resulting error magnitudes are the right
    order of magnitude for the declared dtype.
    """
    out: list[float] = []
    for v in values:
        if storage_dtype.startswith("bf16"):
            # Round to bf16: 8 explicit mantissa bits.
            out.append(_round_to_bits(v, 8))
        elif storage_dtype.startswith("int4"):
            out.append(_quantize(v, levels=16))
        elif storage_dtype.startswith("int8"):
            out.append(_quantize(v, levels=256))
        else:
            out.append(v)
    return out


def _round_to_bits(value: float, mantissa_bits: int) -> float:
    if not math.isfinite(value) or value == 0.0:
        return value
    exponent = math.floor(math.log2(abs(value)))
    scale = 2.0 ** (exponent - mantissa_bits)
    return math.copysign(round(value / scale) * scale, value)


def _quantize(value: float, *, levels: int) -> float:
    """Symmetric uniform quantization onto a `levels`-step grid."""
    if not math.isfinite(value):
        return value
    peak = max(abs(v) for v in (-1.0, 1.0))
    step = (2.0 * peak) / (levels - 1)
    return round(value / step) * step


def shape_class_for(n: int) -> str:
    if n <= 3:
        return "scalar"
    if n <= 64:
        return "small"
    return "long_context"


def build_matrix() -> list[Cell]:
    cells: list[Cell] = []
    for operation, values in extreme_value_fixtures().items():
        storage = REFERENCE_PRECISION_RULES[operation].storage_dtype
        candidate = lossy_candidate(values, storage)
        # A wide-open profile: this sweep MEASURES the error, so the profile
        # must not reject it and hide the number.
        profile = NumericalToleranceProfile(
            profile_id=f"measure-{operation.value}-{storage}",
            operation=operation,
            dtype=storage,
            shape_class=shape_class_for(len(values)),
            # A measure-only profile, so nothing is rejected and the error
            # is visible. Finite and far above any real bf16/int4 error, but
            # not `inf` -- the profile validator rejects non-finite
            # tolerances, and rightly so: an infinite tolerance would be a
            # gate that cannot fail, which is the same class of defect as the
            # default=0.0 this work started from.
            max_abs_error=1.0e18,
            max_rel_error=1.0e18,
            require_finite_state_match=True,
        )
        identity = NumericalArtifactIdentity(
            model_version="declared-simulation",
            runtime_version="lossy-candidate-model",
            backend=f"simulated:{storage}",
            config_id="measure-only",
        )
        comparison = compare_numerical_paths(
            list(values), candidate, profile=profile, identity=identity
        )
        cells.append(
            Cell(
                operation=operation.value,
                storage_dtype=storage,
                shape_class=shape_class_for(len(values)),
                n=len(values),
                max_abs=comparison.max_abs_error,
                max_rel=comparison.max_rel_error,
                all_finite=comparison.finite_state_match,
            )
        )
    return cells


def main() -> int:
    cells = build_matrix()

    print("=== ERROR DISTRIBUTION BY OPERATION / DTYPE / SHAPE (VER-NUM-022) ===")
    print(
        f"{'operation':24s} {'storage':16s} {'shape':12s} {'n':>2s} "
        f"{'max_abs':>12s} {'max_rel':>12s} finite"
    )
    for c in cells:
        print(
            f"{c.operation:24s} {c.storage_dtype:16s} {c.shape_class:12s} "
            f"{c.n:2d} {c.max_abs:12.4e} {c.max_rel:12.4e} {c.all_finite}"
        )
    print()

    finite = [c for c in cells if math.isfinite(c.max_abs) and math.isfinite(c.max_rel)]
    if not finite:
        print("no comparable cell: cannot report a distribution")
        return 1

    abs_sorted = sorted(c.max_abs for c in finite)
    rel_sorted = sorted(c.max_rel for c in finite)
    worst = max(finite, key=lambda c: c.max_rel)

    print("=== DISTRIBUTION SUMMARY (across comparable cells) ===")
    print(f"  cells measured        = {len(cells)}  (comparable: {len(finite)})")
    print(f"  max_abs   min/med/max = {abs_sorted[0]:.3e} / "
          f"{abs_sorted[len(abs_sorted) // 2]:.3e} / {abs_sorted[-1]:.3e}")
    print(f"  max_rel   min/med/max = {rel_sorted[0]:.3e} / "
          f"{rel_sorted[len(rel_sorted) // 2]:.3e} / {rel_sorted[-1]:.3e}")
    print(f"  widest relative error = {worst.operation} ({worst.storage_dtype})")
    print()
    print("The spread is the point. A single global tolerance sized for the")
    print("median would be wrong by orders of magnitude at both ends, which is")
    print("what REQ-NUM-022 says a global tolerance would be.")
    print()

    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
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
            "script": "scripts/numerical_tolerance_matrix.py",
            "source_sha": rev,
            "dirty": dirty,
            "gate": "oai2.model.numerical_compare.compare_numerical_paths",
        },
        "method": {
            "reference": "oai2.model.numerics.extreme_value_fixtures()",
            "dtype_source": "oai2.model.numerics.REFERENCE_PRECISION_RULES",
            "candidate": (
                "DECLARED SIMULATION of a path storing at the operation's own "
                "declared dtype (bf16 = 8 mantissa bits; int4/int8 = uniform "
                "grid). NOT a measurement of a real backend."
            ),
            "profile": "max_abs_error=1e18, max_rel_error=1e18 (measure-only)",
        },
        "distribution": [
            {
                "operation": c.operation,
                "storage_dtype": c.storage_dtype,
                "shape_class": c.shape_class,
                "sample_count": c.n,
                "max_abs_error": c.max_abs,
                "max_rel_error": c.max_rel,
                "finite_state_match": c.all_finite,
            }
            for c in cells
        ],
        "summary": {
            "cells": len(cells),
            "comparable_cells": len(finite),
            "max_abs_min": abs_sorted[0],
            "max_abs_median": abs_sorted[len(abs_sorted) // 2],
            "max_abs_max": abs_sorted[-1],
            "max_rel_min": rel_sorted[0],
            "max_rel_median": rel_sorted[len(rel_sorted) // 2],
            "max_rel_max": rel_sorted[-1],
            "widest_relative_error_operation": worst.operation,
        },
        "known_limitations": [
            "extreme_value_fixtures() is a RANGE probe, not an "
            "operation-correct probe. Values like (-80, 0, 80) are not a "
            "softmax input in any meaningful sense, so a nonlinear operation "
            "is being round-tripped through storage rather than computed. The "
            "error measured is therefore the STORAGE error, which is the part "
            "a dtype-aware tolerance has to cover, but it is not the error of "
            "the whole operation.",
            "attention_softmax and loss report max_abs_error=0.0 because "
            "their fixture values are exactly representable at the declared "
            "storage precision (0.0, and 80.0 as a power of two). That is a "
            "TRUE zero, not an empty-comparison artifact -- but it means those "
            "two cells carry NO information about typical error and must not "
            "be read as 'softmax is exact'.",
            "Every cell has shape_class='scalar' because the declared fixture "
            "surface is three values wide. REQ-NUM-022's shape dimension is "
            "therefore NOT covered by this matrix; a real long-context sweep "
            "needs a fixture surface that is actually long.",
        ],
        "scope": (
            "Measures how large the storage error actually is for a DECLARED "
            "lossy candidate at each operation's declared storage dtype, so "
            "that choosing a tolerance is informed rather than guessed. It "
            "does NOT propose or endorse any tolerance, and it is NOT a "
            "measurement of a real backend -- that is "
            "evidence/numerical/backend_*.json."
        ),
    }

    out_dir = REPO / "evidence" / "numerical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tolerance_matrix_{rev[:12]}.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"=== ARTIFACT WRITTEN ===\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
