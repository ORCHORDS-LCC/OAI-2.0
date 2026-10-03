"""Which third-party imports does the portable path actually require?

RISK-FLEET-011 (#242) asks for dependency inventories before and after the
portable/native split, and REQ-FLEET-013 requires that importing the common
package not need an unavailable backend. ``140b99b`` fixed the MLX case; the
open question is whether any *other* third-party import has the same
eager-import defect, and whether the packages that were already clean stayed
clean.

This walks the intra-package import graph with ``ast`` -- it never imports
anything, so it can report on a tree whose dependencies are not installed --
and separates two things a naive grep cannot:

  * **import-time**: bindings evaluated when the module body runs. These are
    what make an import fail on a machine lacking the dependency.
  * **deferred**: bindings inside a function or method body. Cheap at import
    time; only required when that code path actually runs.

The distinction is the whole point. ``oai2.runtime.model`` and
``oai2.runtime.mlx_hot_runtime`` both reference ``mlx_lm``; only one of them
did so at module scope, and that is exactly the defect ``140b99b`` removed.

Reachability matters just as much. Scanning every file under a package
over-reports: ``mlx_hot_runtime.py`` still has a module-level ``from mlx_lm
import load``, but nothing on the ``oai2.runtime`` import path loads it any
more. Only modules actually reachable through import-time edges are counted.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Packages a portable worker must be able to import on a CPU-only box.
#: ``oai2.knowledge`` and ``oai2.tools`` are included deliberately: they were
#: already clean, and a clean package is a regression risk, not a safe zone.
PORTABLE_PACKAGES = [
    "oai2",
    "oai2.runtime",
    "oai2.server",
    "oai2.agents",
    "oai2.knowledge",
    "oai2.tools",
]

#: Module-level import of one of these is a hard failure without the
#: corresponding hardware. Reported by name so the reader does not have to
#: recognise it in a sorted list.
BACKEND_HINTS = ("mlx", "torch", "onnx", "tensorflow", "jax", "paddle")


@dataclass
class ModuleImports:
    """Third-party imports of one module, split by when they are required."""

    module: str
    import_time: set[str] = field(default_factory=set)
    deferred: set[str] = field(default_factory=set)


def _third_party(name: str) -> bool:
    """True for a distribution that is neither stdlib nor this project."""
    root = name.split(".")[0]
    if not root or root == "oai2":
        return False
    return root not in sys.stdlib_module_names


def _local_targets(dotted: str, stmt: ast.stmt, *, is_package: bool) -> list[str]:
    """Intra-package module names imported by one module-level statement.

    ``from .admission import AdmissionAction`` names the *module*
    ``pkg.admission``; ``AdmissionAction`` is a symbol inside it. Only
    ``from . import mlx_hot_runtime``, where the names really are modules,
    needs the name appended.

    ``is_package`` matters for relative imports: a level-1 import is relative
    to the module's own *containing package*, which for ``pkg.sub.mod`` is
    ``pkg.sub`` -- not ``pkg.sub.mod``. Treating the module name as a package
    makes ``from ..runtime import x`` resolve to ``pkg.sub.runtime`` instead
    of ``pkg.runtime``, silently dropping the edge.
    """
    if isinstance(stmt, ast.Import):
        return [a.name for a in stmt.names if a.name.startswith("oai2")]
    if isinstance(stmt, ast.ImportFrom):
        if stmt.level:
            pkg = dotted.split(".") if is_package else dotted.split(".")[:-1]
            up = stmt.level - 1
            if up:
                pkg = pkg[:-up] if up <= len(pkg) else []
            if stmt.module:
                return [".".join([*pkg, stmt.module])]
            return [".".join([*pkg, a.name]) for a in stmt.names]
        # Absolute: ``stmt.module`` is already fully qualified. Re-deriving a
        # package prefix here would produce a path that resolves to nothing,
        # silently dropping the edge.
        if stmt.module and stmt.module.startswith("oai2"):
            return [stmt.module]
        return []
    return []


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
        return None


def _module_path(root: Path, dotted: str) -> Path | None:
    if not dotted:
        return None
    base = root / Path(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def scan_module(root: Path, dotted: str) -> ModuleImports | None:
    """Split one module's third-party imports into import-time and deferred."""
    path = _module_path(root, dotted)
    if path is None:
        return None
    tree = _parse(path)
    if tree is None:
        return None
    out = ModuleImports(module=dotted)

    def record(stmt: ast.stmt, deferred: bool) -> None:
        if isinstance(stmt, ast.Import):
            names = [a.name for a in stmt.names]
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level or not stmt.module:
                return
            names = [stmt.module]
        else:
            return
        for name in names:
            if _third_party(name):
                (out.deferred if deferred else out.import_time).add(name)

    for stmt in tree.body:
        record(stmt, deferred=False)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    record(sub, deferred=True)
    return out


def reachable(root: Path, package: str) -> list[str]:
    """Modules loaded when ``package`` is imported, following import-time edges.

    Function-local imports are deliberately not followed: the module they name
    is not loaded at import time, and following it would over-report exactly
    the deferral this audit exists to measure.
    """
    seen: set[str] = set()
    order: list[str] = []
    queue = [package]
    while queue:
        dotted = queue.pop(0)
        if dotted in seen:
            continue
        path = _module_path(root, dotted)
        if path is None:
            continue
        tree = _parse(path)
        if tree is None:
            continue
        seen.add(dotted)
        order.append(dotted)
        is_package = path.name == "__init__.py"
        for stmt in tree.body:
            queue.extend(_local_targets(dotted, stmt, is_package=is_package))
    return order


def audit(root: Path) -> dict[str, object]:
    report: dict[str, object] = {}
    for package in PORTABLE_PACKAGES:
        modules = reachable(root, package)
        import_time: set[str] = set()
        deferred: set[str] = set()
        for dotted in modules:
            scanned = scan_module(root, dotted)
            if scanned is None:
                continue
            import_time |= scanned.import_time
            deferred |= scanned.deferred
        # Required at import time anywhere is not "deferred" anywhere.
        deferred -= import_time
        report[package] = {
            "modules_reachable": len(modules),
            "third_party_import_time": sorted(import_time),
            "third_party_deferred": sorted(deferred),
            "backend_at_import_time": sorted(
                n for n in import_time if n.split(".")[0].startswith(BACKEND_HINTS)
            ),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    report = audit(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 78)
    print("THIRD-PARTY IMPORTS ON THE PORTABLE PATH (REQ-FLEET-013)")
    print("=" * 78)
    for package, data in report.items():
        assert isinstance(data, dict)
        backend = data["backend_at_import_time"]
        flag = "   <-- BACKEND AT IMPORT TIME" if backend else ""
        print(f"\n{package}  ({data['modules_reachable']} modules reachable){flag}")
        required = data["third_party_import_time"]
        print(f"  import-time ({len(required)}): {', '.join(required) or 'none'}")
        lazy = data["third_party_deferred"]
        print(f"  deferred   ({len(lazy)}): {', '.join(lazy) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
