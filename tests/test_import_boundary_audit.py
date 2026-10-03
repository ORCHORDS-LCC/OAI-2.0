"""The import-boundary audit must measure the real import graph.

`scripts/import_boundary_audit.py` produces the dependency inventory
RISK-FLEET-011 (#242) asks for. Two properties are load-bearing, and both
were violated by the first version of the script:

1. **Reachability.** Scanning every file under a package over-reports.
   `oai2/runtime/mlx_hot_runtime.py` still has a module-level
   `from mlx_lm import load`, but nothing on the `oai2.runtime` import path
   loads it. Scanning by directory reported `mlx_lm` as import-time for
   `oai2.runtime` -- directly contradicting the behavioural proof that
   `import oai2.runtime` succeeds with MLX blocked. A measurement that
   disagrees with observed behaviour is worse than no measurement.
2. **Module vs symbol.** `from .admission import AdmissionAction` names the
   *module* `oai2.runtime.admission`. Appending the imported name yielded
   `oai2.runtime.admission.AdmissionAction`, which resolves to nothing, so
   reachability collapsed to a single module per package.

So these tests pin the script against the behaviour it is supposed to
describe, rather than against its own output.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "import_boundary_audit.py"
_MLX_HOT_RUNTIME = "oai2.runtime.mlx_hot_runtime"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("import_boundary_audit", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report() -> dict[str, dict[str, object]]:
    module = _load()
    result = module.audit(_REPO)  # type: ignore[attr-defined]
    assert isinstance(result, dict)
    return result  # type: ignore[return-value]


@pytest.mark.parametrize(
    "package",
    ["oai2", "oai2.runtime", "oai2.server", "oai2.agents", "oai2.knowledge",
     "oai2.tools"],
)
def test_no_portable_package_imports_a_backend(report, package: str) -> None:
    """REQ-FLEET-013: no reachable module may import MLX at module scope."""
    entry = report[package]
    assert entry["backend_at_import_time"] == [], (
        f"{package} imports a backend at import time: "
        f"{entry['backend_at_import_time']}"
    )


def test_mlx_hot_runtime_is_not_reachable_from_the_runtime_package(
    report,
) -> None:
    """The specific regression `140b99b` fixed.

    `oai2/runtime/__init__.py` used `from .mlx_hot_runtime import
    MLXHotRuntime`, which pulled `mlx_lm` into the import graph. If anyone
    restores an eager import -- directly, or by adding another module that
    imports it at module scope -- this bites.
    """
    module = _load()
    reachable = module.reachable(_REPO, "oai2.runtime")  # type: ignore[attr-defined]
    assert _MLX_HOT_RUNTIME not in reachable, (
        "oai2.runtime.mlx_hot_runtime is on the import path again; MLX would "
        "be required merely to import the package"
    )


def test_reachability_finds_more_than_the_package_init(report) -> None:
    """Guards the relative-import resolution.

    The broken version resolved `from .admission import X` to
    `oai2.runtime.admission.X`, found no file, and reported every package as
    a single reachable module. A graph that never grows past the entry point
    is a graph that is not being walked.
    """
    for package in ("oai2.runtime", "oai2.agents", "oai2.knowledge"):
        assert report[package]["modules_reachable"] > 1, (
            f"{package}: only the package __init__ was reachable, so the "
            f"import graph is not being walked"
        )


def test_model_module_defers_its_mlx_imports() -> None:
    """`model.py` states the contract this whole audit depends on.

    Its docstring calls it "the only module in the runtime that imports MLX"
    and every MLX import is annotated "keep this module dependency-free at
    parse time". That claim is load-bearing: it is the precedent the package
    boundary now follows, so it must stay true.

    Note what is *not* being claimed. `model.py` does import `pydantic` at
    module scope, and that is correct -- `pydantic` is pure Python and
    portable. The contract is specifically about *backends*, so this asserts
    no backend at import time, not no third-party imports at all.
    """
    module = _load()
    scanned = module.scan_module(_REPO, "oai2.runtime.model")  # type: ignore[attr-defined]
    assert scanned is not None
    backends = [
        name
        for name in scanned.import_time
        if name.split(".")[0].startswith(("mlx", "torch", "onnx", "tensorflow"))
    ]
    assert backends == [], (
        f"oai2.runtime.model gained a module-level backend import: {backends}"
    )
    assert any(name.startswith("mlx") for name in scanned.deferred), (
        "expected oai2.runtime.model to still defer its MLX imports; the "
        "module was probably refactored and this assertion needs revisiting"
    )


def test_audit_agrees_with_observed_import_behaviour() -> None:
    """The static audit and the real interpreter must tell the same story.

    This is the check that catches a resolver bug. An earlier version of the
    audit reported no backend edge for `oai2.server` and `oai2.agents` while
    `import oai2.server` demonstrably raised `ImportError` with MLX blocked --
    two bugs (a double-prefixed absolute import, and a level-2 relative
    import resolved against the module instead of its package) that both
    failed *silently*, by dropping an edge rather than raising.

    A measurement that contradicts observed behaviour is worse than no
    measurement, so the two are pinned against each other here.
    """
    module = _load()
    packages = list(module.PORTABLE_PACKAGES)  # type: ignore[attr-defined]
    code = (
        "import sys, importlib, importlib.abc, json\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, f, p=None, t=None):\n"
        "        if f.split('.')[0] in ('mlx', 'mlx_lm'):\n"
        "            raise ImportError(f)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        f"out = {{}}\n"
        f"for name in {packages!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        out[name] = True\n"
        "    except ImportError:\n"
        "        out[name] = False\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    observed = json.loads(proc.stdout.strip().splitlines()[-1])

    predicted = {
        name: not (module.audit(_REPO)[name]["backend_at_import_time"])  # type: ignore[attr-defined]
        for name in packages
    }
    assert predicted == observed, (
        "the static audit and the real interpreter disagree about which "
        f"packages need a backend.\n  audit: {predicted}\n  actual: {observed}"
    )


def test_reachability_crosses_a_two_level_relative_import() -> None:
    """Exercises the relative-import resolver against a real edge.

    `oai2.agents.agent_loop` reaches `oai2.runtime` via `from ..runtime import
    ...` -- a *level-2* relative import. The first version of the resolver
    treated the importing module's own dotted name as its package, so that
    resolved to `oai2.agents.runtime`, which is not a file; the edge was
    dropped silently.

    The agreement test above cannot catch that on its own: once `140b99b`
    removed the MLX edge there is no backend edge left to drop, so a broken
    resolver and a correct one agree about the thing that test compares. This
    pins the edge directly.
    """
    module = _load()
    reachable = module.reachable(_REPO, "oai2.agents")  # type: ignore[attr-defined]
    assert "oai2.runtime" in reachable, (
        "oai2.agents no longer reaches oai2.runtime; a level-2 relative import "
        "is being resolved against the module instead of its package"
    )


def test_audit_runs_without_the_backends_installed() -> None:
    """The audit must describe a tree whose dependencies are absent.

    It is `ast`-only by design, so it can report on a machine that could not
    import the package at all. If it ever starts importing, it silently stops
    being usable as pre-merge evidence.
    """
    code = (
        "import sys, importlib.abc, runpy\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, f, p=None, t=None):\n"
        "        if f.split('.')[0] in ('mlx','mlx_lm','pydantic','fastapi'):\n"
        "            raise ImportError('blocked: '+f)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        f"runpy.run_path({str(_SCRIPT)!r}, run_name='__main__')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, (
        f"the audit imported the tree it is meant to analyse statically:\n"
        f"{proc.stderr[-2000:]}"
    )
    assert "THIRD-PARTY IMPORTS ON THE PORTABLE PATH" in proc.stdout
