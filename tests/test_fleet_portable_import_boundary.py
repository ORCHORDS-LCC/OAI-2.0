"""REQ-FLEET-013: the common package must import without an Apple/GPU backend.

#242 requires that a fresh CPU-only Linux environment can install and import
the common package and exercise a real control handshake with no MLX/GPU
packages present. MLX is a *backend*: choosing a backend is the caller's
decision, so the package boundary must not make that choice at import time.

The defect this pins was an eager ``from .mlx_hot_runtime import
MLXHotRuntime`` in :mod:`oai2.runtime`, and ``mlx_hot_runtime`` imports
``mlx_lm`` at module scope. So ``import oai2.runtime`` -- and everything
downstream of it, including :mod:`oai2.server` and :mod:`oai2.agents` --
raised ``ImportError`` on any machine without Apple silicon. Measured on
the pre-fix SHA, with ``mlx``/``mlx_lm`` blocked at import:

    oai2           OK
    oai2.runtime   FAIL -> No module named 'mlx_lm'
    oai2.server    FAIL -> No module named 'mlx_lm'
    oai2.agents    FAIL -> No module named 'mlx_lm'
    oai2.knowledge OK
    oai2.tools     OK

The tests run in a subprocess with a real import blocker, because the
defect is about what a *fresh interpreter* can import. In-process patching
would import the package before the blocker existed and prove nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

#: Packages a portable worker must be able to import on a CPU-only box.
#: ``oai2.knowledge`` and ``oai2.tools`` were already clean and are listed
#: deliberately: they are the control-path neighbours that a future eager
#: import would silently break, so the guard must cover them too.
PORTABLE_PACKAGES = [
    "oai2",
    "oai2.runtime",
    "oai2.server",
    "oai2.agents",
    "oai2.knowledge",
    "oai2.tools",
]

#: Installed as a meta-path finder so the block applies to every import
#: path, including a lazy attribute access that happens later.
_BLOCKER = """
import sys, importlib.abc

class _BlockAppleBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("mlx", "mlx_lm"):
            raise ImportError("No module named %r (simulated CPU-only Linux)" % fullname)
        return None

sys.meta_path.insert(0, _BlockAppleBackend())
for _name in [n for n in sys.modules if n.split(".")[0] in ("mlx", "mlx_lm")]:
    del sys.modules[_name]
"""


def _run(body: str) -> dict[str, object]:
    """Run ``body`` in a fresh interpreter with Apple backends blocked."""
    script = _BLOCKER + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _import_attempts() -> str:
    """Import every portable package, recording each one's outcome."""
    return (
        "import importlib, json\n"
        f"packages = {PORTABLE_PACKAGES!r}\n"
        "out = {}\n"
        "for name in packages:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        out[name] = 'OK'\n"
        "    except ImportError as exc:\n"
        "        out[name] = f'ImportError: {exc}'\n"
        "print(json.dumps(out))\n"
    )


def test_common_packages_import_without_apple_backend() -> None:
    """REQ-FLEET-013 / AC-FLEET-011: no portable package needs MLX."""
    outcomes = _run(_import_attempts())
    assert isinstance(outcomes, dict)
    assert outcomes, "no packages attempted"
    failed = {name: why for name, why in outcomes.items() if why != "OK"}
    assert not failed, (
        "importing the common package required an Apple/GPU backend: "
        f"{failed}"
    )


def test_package_import_does_not_pull_in_the_mlx_runtime() -> None:
    """The sharp form of the invariant: importing must not *load* the backend.

    Asserting only that no ``ImportError`` is raised would still pass if the
    backend were somehow importable-but-present. The defect is an eager
    import, so the property to pin is that ``oai2.runtime.mlx_hot_runtime``
    is absent from ``sys.modules`` after ``import oai2.runtime``.
    """
    loaded = _run(
        "import json, sys\n"
        "import oai2.runtime\n"
        "import oai2.server\n"
        "import oai2.agents\n"
        "print(json.dumps(sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in ('mlx', 'mlx_lm') or m.endswith('mlx_hot_runtime')\n"
        ")))\n"
    )
    assert loaded == [], (
        "importing the common package eagerly loaded the MLX backend: "
        f"{loaded}"
    )


def test_mlx_export_fails_as_import_error_not_attribute_error() -> None:
    """Absent MLX must surface the real cause, not a phantom missing export.

    An ``AttributeError`` here would tell an operator that the package does
    not define ``MLXHotRuntime`` at all -- a false statement about the public
    contract -- and would hide the actual missing dependency.
    """
    outcome = _run(
        "import json\n"
        "import oai2.runtime\n"
        "try:\n"
        "    oai2.runtime.MLXHotRuntime\n"
        "except ImportError as exc:\n"
        "    print(json.dumps({'kind': 'ImportError', 'msg': str(exc)}))\n"
        "except AttributeError as exc:\n"
        "    print(json.dumps({'kind': 'AttributeError', 'msg': str(exc)}))\n"
        "else:\n"
        "    print(json.dumps({'kind': 'resolved', 'msg': ''}))\n"
    )
    assert isinstance(outcome, dict)
    assert outcome.get("kind") == "ImportError", (
        "reaching MLXHotRuntime without MLX must raise ImportError naming the "
        f"missing dependency, got {outcome!r}"
    )
    assert "mlx" in str(outcome.get("msg", "")).lower(), (
        f"ImportError does not name the missing dependency: {outcome!r}"
    )


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """Opposite direction: a lazy ``__getattr__`` must not swallow typos.

    A ``__getattr__`` that returned something for every name would turn
    ``oai2.runtime.Typoed`` into a silent success. Only the declared lazy
    exports may resolve.
    """
    outcome = _run(
        "import json\n"
        "import oai2.runtime\n"
        "import oai2.server\n"
        "out = {}\n"
        "for mod in (oai2.runtime, oai2.server):\n"
        "    try:\n"
        "        mod.DefinitelyNotARealExport\n"
        "        out[mod.__name__] = 'resolved'\n"
        "    except AttributeError:\n"
        "        out[mod.__name__] = 'AttributeError'\n"
        "print(json.dumps(out))\n"
    )
    assert isinstance(outcome, dict)
    assert outcome == {
        "oai2.runtime": "AttributeError",
        "oai2.server": "AttributeError",
    }, f"lazy __getattr__ is too permissive: {outcome!r}"


def test_native_mlx_export_identity_is_preserved() -> None:
    """Opposite direction: on Apple silicon nothing may change.

    The lazy boundary exists so a CPU-only box can import the package. It
    must not alter the native path: ``MLXHotRuntime`` still resolves to the
    real class, is still the same object through both re-exports, is still
    advertised in ``__all__`` and is still discoverable via ``dir()``.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json
                try:
                    import mlx_lm  # noqa: F401
                except ImportError:
                    print(json.dumps({"skipped": True}))
                    raise SystemExit(0)
                import oai2.runtime
                import oai2.server
                from oai2.runtime import MLXHotRuntime as a
                from oai2.server import MLXHotRuntime as b
                print(json.dumps({
                    "skipped": False,
                    "class": f"{a.__module__}.{a.__qualname__}",
                    "same_object": a is b,
                    "identity_pinned": getattr(oai2.runtime, "MLXHotRuntime") is a,
                    "in_runtime_all": "MLXHotRuntime" in oai2.runtime.__all__,
                    "in_server_all": "MLXHotRuntime" in oai2.server.__all__,
                    "in_dir": "MLXHotRuntime" in dir(oai2.runtime),
                }))
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stderr
    result = json.loads(probe.stdout.strip().splitlines()[-1])
    if result.get("skipped"):
        pytest.skip("MLX is not installed; native path cannot be checked here")
    assert result == {
        "skipped": False,
        "class": "oai2.runtime.mlx_hot_runtime.MLXHotRuntime",
        "same_object": True,
        "identity_pinned": True,
        "in_runtime_all": True,
        "in_server_all": True,
        "in_dir": True,
    }, f"native MLX export contract changed: {result!r}"
