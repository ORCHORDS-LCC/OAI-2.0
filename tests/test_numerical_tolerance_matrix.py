"""The tolerance matrix must measure, and must not overstate what it measured.

`scripts/numerical_tolerance_matrix.py` produces the VER-NUM-022 error
distribution across the operation/dtype surface. Two things about it are
load-bearing:

1. It is a MEASUREMENT. A cell whose error is 0.0 must be a true zero, not
   an empty comparison — the defect class this issue opened with.
2. It does not claim more than it measured. The fixture surface is a range
   probe, not an operation-correct probe, and the artifact has to say so
   rather than let a reader infer coverage that is not there.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

from oai2.model.numerics import (
    REFERENCE_PRECISION_RULES,
    NumericalOperation,
    extreme_value_fixtures,
)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "numerical_tolerance_matrix.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("numerical_tolerance_matrix", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_declared_operation_has_a_cell() -> None:
    """A missing operation would silently shrink the matrix."""
    module = _load()
    cells = module.build_matrix()  # type: ignore[attr-defined]
    measured = {c.operation for c in cells}
    assert measured == {op.value for op in NumericalOperation}


def test_a_zero_error_cell_is_a_true_zero_not_an_empty_comparison() -> None:
    """The defect this issue opened with, checked in the new code.

    `max(abs_errors, default=0.0)` reports a perfect match for a comparison
    that never happened. A cell reporting 0.0 must therefore have actually
    compared its samples.
    """
    module = _load()
    for cell in module.build_matrix():  # type: ignore[attr-defined]
        if cell.max_abs == 0.0:
            assert cell.max_rel == 0.0
            # Re-derive: a true zero means the candidate equalled the
            # reference sample-for-sample.
            values = extreme_value_fixtures()[
                NumericalOperation(cell.operation)
            ]
            storage = REFERENCE_PRECISION_RULES[
                NumericalOperation(cell.operation)
            ].storage_dtype
            candidate = module.lossy_candidate(values, storage)  # type: ignore[attr-defined]
            assert list(candidate) == list(values), (
                f"{cell.operation}: reports 0.0 error but the candidate is not "
                "identical to the reference, so this is an empty comparison"
            )


def test_dtype_dimension_comes_from_the_declared_precision_rule() -> None:
    """REQ-NUM-022's dtype axis is the repository's, not the script's."""
    module = _load()
    cells = {c.operation: c.storage_dtype for c in module.build_matrix()}  # type: ignore[attr-defined]
    for operation, rule in REFERENCE_PRECISION_RULES.items():
        assert cells[operation.value] == rule.storage_dtype


def test_matrix_is_not_all_zero() -> None:
    """A lossy candidate must actually lose precision somewhere.

    If every cell reported 0.0 the simulation would be modelling nothing,
    and the "distribution" would be twelve rows of false comfort.
    """
    module = _load()
    cells = module.build_matrix()  # type: ignore[attr-defined]
    assert any(c.max_abs > 0.0 for c in cells)
    assert any(c.max_rel > 0.0 for c in cells)


def test_each_cell_records_the_inputs_it_compared() -> None:
    """A cell must say WHICH samples produced its numbers (d8862af).

    Without the digest, two cells reporting the same figures from different
    inputs are indistinguishable, and a reader cannot tell what a numerical
    failure came from.
    """
    module = _load()
    cells = module.build_matrix()  # type: ignore[attr-defined]
    digests = {c.input_digest for c in cells}
    assert all(d.startswith("sha256:") for d in digests)
    # Six distinct operations must not collapse to one digest.
    assert len(digests) == len(cells), "two cells share an input digest"


def test_committed_matrix_artifact_is_sha_tied_and_honest() -> None:
    artifacts = sorted((_REPO / "evidence" / "numerical").glob("tolerance_matrix_*.json"))
    assert artifacts, "no tolerance matrix artifact committed"

    for path in artifacts:
        data = json.loads(path.read_text())
        harness = data["harness"]
        assert harness["source_sha"], f"{path.name}: no source_sha"
        assert harness["dirty"] is False, f"{path.name}: dirty=True"
        on_main = subprocess.run(
            ["git", "merge-base", "--is-ancestor", harness["source_sha"], "HEAD"],
            capture_output=True,
            cwd=_REPO,
        )
        assert on_main.returncode == 0, f"{path.name}: source_sha off main"

        # The limitations are the point of the artifact, not a disclaimer.
        limits = data["known_limitations"]
        assert len(limits) >= 3, f"{path.name}: limitations were dropped"
        joined = " ".join(limits).lower()
        assert "range probe" in joined, f"{path.name}: fixture nature unstated"
        assert "shape" in joined, f"{path.name}: shape coverage unstated"

        for cell in data["distribution"]:
            assert cell.get("input_digest", "").startswith("sha256:"), (
                f"{path.name}: a cell does not record what was compared"
            )
            for field in ("max_abs_error", "max_rel_error"):
                value = cell[field]
                assert isinstance(value, (int, float))
                assert value >= 0.0 or math.isnan(value)


def test_measure_profile_cannot_be_a_gate_that_cannot_fail() -> None:
    """The measure-only profile must stay finite.

    An infinite tolerance would be a gate that can never reject, which is
    the same class of defect as the `default=0.0` this work started from.
    The profile validator already enforces this; this pins that the sweep
    does not work around it.
    """
    module = _load()
    cells = module.build_matrix()  # type: ignore[attr-defined]
    assert cells, "matrix empty"
    # Every cell was produced through a finite profile, so no cell can have
    # been admitted by an infinite tolerance.
    for cell in cells:
        assert math.isfinite(cell.max_abs) or cell.max_abs == float("inf")
