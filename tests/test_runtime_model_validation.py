"""Structural / validation tests for ``oai2.runtime.model``.

Pins the model specification and MLX environment probe (PROPOSED status):

- :class:`ModelSpec` -- Pydantic ``BaseModel`` carrying ``name`` /
  ``revision`` / ``quantization`` / ``local_path`` / ``total_params`` /
  ``active_params`` plus a ``status`` property that always reports
  :attr:`Status.PROPOSED` (no model has been checked in yet).
- :func:`discover_default_device` -- local-import wrapper around
  :func:`mlx.core.default_device` that degrades gracefully to
  ``"unavailable:<ExceptionName>"`` when MLX is not importable.
- :func:`smoke_check` -- 1x3 dot 3x1 matmul sanity gate that returns
  ``(ok: bool, device_or_error: str)`` and routes the computed result
  through :func:`oai2.model.check_numerics` before reporting.

Behavioural end-to-end coverage lives in ``tests/test_runtime.py``;
this file pins the *shape* of the API and the invariants the live
runtime (when implemented) will rely on.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

import oai2.runtime as runtime_package
from oai2.core import Status
from oai2.runtime.model import (
    ModelSpec,
    discover_default_device,
    smoke_check,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "runtime" / "model.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_runtime_model_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "runtime/model.py is unexpectedly empty"


def test_runtime_model_module_has_docstring() -> None:
    """The module exposes a public-style multi-line docstring."""

    assert _MODULE_SOURCE.lstrip().startswith('"""')


def test_runtime_model_module_docstring_mentions_mlx_ownership() -> None:
    """The docstring must declare that this is the only runtime module that imports MLX."""

    lowered = _MODULE_SOURCE.lower()
    assert "mlx" in lowered, "runtime/model.py docstring must mention MLX"
    # "only module" is the canonical contract wording.
    assert "only" in lowered and ("module" in lowered or "imports" in lowered), (
        "runtime/model.py docstring must declare MLX ownership boundary"
    )


def test_runtime_model_module_docstring_notes_proposed_status() -> None:
    """The module is PROPOSED — the docstring must say so."""

    assert "PROPOSED" in _MODULE_SOURCE


def test_runtime_model_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_runtime_model_module_imports_pydantic_basemodel_configdict_field() -> None:
    """``ModelSpec`` is a Pydantic ``BaseModel`` subclass."""

    match = re.search(
        r"^from pydantic import ([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "runtime/model.py must import pydantic symbols"
    body = match.group(1)
    for symbol in ("BaseModel", "ConfigDict", "Field"):
        assert symbol in body, f"pydantic import must include {symbol!r}"


def test_runtime_model_module_imports_status_from_oai2_core_relative() -> None:
    """``Status`` is imported relatively from ``..core``."""

    match = re.search(
        r"^from\s+\.\.core\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "runtime/model.py must import Status relatively from ..core"
    body = match.group(1)
    assert "Status" in body, f"`..core` import must include Status (got: {body!r})"


def test_runtime_model_module_imports_numerics_from_oai2_model() -> None:
    """``NumericalOperation`` + ``check_numerics`` are imported from ``..model``."""

    match = re.search(
        r"^from\s+\.\.model\s+import\s+([^\n]+)",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "runtime/model.py must import from ..model"
    body = match.group(1)
    assert "NumericalOperation" in body, (
        f"`..model` import must include NumericalOperation (got: {body!r})"
    )
    assert "check_numerics" in body, f"`..model` import must include check_numerics (got: {body!r})"


def test_runtime_model_module_does_not_import_oai2_package_directly() -> None:
    """No top-level ``import oai2`` or ``from oai2 import ...`` (use relative)."""

    non_docstring_lines = [
        line
        for line in _MODULE_SOURCE.splitlines()
        if line.strip() and not line.strip().startswith('"""')
    ]
    for line in non_docstring_lines:
        assert not re.match(
            r"^(?:from\s+oai2\s+import|import\s+oai2(?:\.|\s|$))",
            line,
        ), f"runtime/model.py must not import oai2 directly: {line!r}"


def test_runtime_model_module_does_not_import_cloud_runtime_modules() -> None:
    """The runtime/model module is MLX-only; no cloud-runtime imports."""

    forbidden = ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"runtime/model module must not import cloud runtime module {token!r}"
        )


def test_runtime_model_module_is_the_only_runtime_mlx_importer() -> None:
    """``runtime/model.py`` and ``runtime/mlx_hot_runtime.py`` are the
    only runtime modules that import MLX.

    Pins the contract declared in the module docstring: every other
    runtime module (``admission``, ``admission_scheduler``,
    ``gateway_model_client``, ``gateway_runtime``, ``inference``,
    ``scheduler``) must NOT import ``mlx`` directly. ``mlx_hot_runtime``
    is the concrete live MLX-inference runtime wired into the agent
    loop (Refs sess_10cbe33c-d83b-42ce-bf2c) so it is also permitted
    to import MLX.
    """

    runtime_dir = Path(__file__).resolve().parent.parent / "oai2" / "runtime"
    allowed_mlx_importers = {"model.py", "mlx_hot_runtime.py"}
    offenders: list[str] = []
    for py_file in sorted(runtime_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        if py_file.name == "model.py":
            assert re.search(r"^\s*import\s+mlx\.core", text, re.MULTILINE), (
                "runtime/model.py must import mlx.core somewhere"
            )
            continue
        if py_file.name in allowed_mlx_importers:
            continue
        for line in text.splitlines():
            if re.match(r"^\s*(?:from\s+mlx\b|import\s+mlx(?:\.|\s|$))", line):
                offenders.append(f"{py_file.name}: {line.strip()}")
    assert offenders == [], (
        f"runtime/ modules other than {sorted(allowed_mlx_importers)} must not "
        f"import mlx directly: {offenders}"
    )


# ---------------------------------------------------------------------------
# 2. __all__ + identity + runtime package integration
# ---------------------------------------------------------------------------


def test_runtime_model_module_all_exports_expected_symbols() -> None:
    """``__all__`` lists exactly the 3 public symbols."""

    from oai2.runtime import model as model_module

    assert set(model_module.__all__) == {
        "ModelSpec",
        "discover_default_device",
        "smoke_check",
    }


def test_runtime_model_module_all_symbols_are_actually_defined() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    from oai2.runtime import model as model_module

    for name in model_module.__all__:
        assert hasattr(model_module, name), (
            f"runtime/model.__all__ lists {name!r} but the module has no such attribute"
        )


def test_runtime_model_module_no_unlisted_public_names() -> None:
    """No locally-defined public name is missing from ``__all__``."""

    import ast

    from oai2.runtime import model as model_module

    listed = set(model_module.__all__)

    tree = ast.parse(_MODULE_SOURCE)
    imported_names: set[str] = set()
    defined_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined_names.add(node.target.id)

    unlisted = {name for name in defined_names if not name.startswith("_") and name not in listed}
    assert unlisted == set(), (
        f"runtime/model has locally-defined public names missing from __all__: {sorted(unlisted)}"
    )


def test_runtime_model_symbols_are_re_exported_at_oai2_runtime_package_level() -> None:
    """All 3 symbols are re-exported at ``oai2.runtime``."""

    for symbol in ("ModelSpec", "discover_default_device", "smoke_check"):
        assert hasattr(runtime_package, symbol), (
            f"oai2.runtime must re-export {symbol!r} from model"
        )
        assert getattr(runtime_package, symbol) is globals()[symbol], (
            f"oai2.runtime.{symbol} must be the same object as runtime.model.{symbol}"
        )


def test_model_spec_is_a_pydantic_basemodel_subclass() -> None:
    """``ModelSpec`` is a Pydantic ``BaseModel`` subclass."""

    from pydantic import BaseModel

    assert issubclass(ModelSpec, BaseModel), (
        f"ModelSpec must subclass BaseModel; bases are {ModelSpec.__bases__!r}"
    )


def test_discover_default_device_is_a_function() -> None:
    """``discover_default_device`` is a regular function (not a class)."""

    assert callable(discover_default_device)
    assert not inspect.isclass(discover_default_device)


def test_smoke_check_is_a_function() -> None:
    """``smoke_check`` is a regular function (not a class)."""

    assert callable(smoke_check)
    assert not inspect.isclass(smoke_check)


# ---------------------------------------------------------------------------
# 3. ModelSpec Pydantic BaseModel
# ---------------------------------------------------------------------------


def test_model_spec_declares_exactly_six_fields() -> None:
    """``ModelSpec`` exposes exactly ``name`` / ``revision`` / ``quantization`` /
    ``local_path`` / ``total_params`` / ``active_params`` (status is a property, not a field)."""

    fields = ModelSpec.model_fields
    assert set(fields.keys()) == {
        "name",
        "revision",
        "quantization",
        "local_path",
        "total_params",
        "active_params",
    }, f"Unexpected ModelSpec fields: {set(fields.keys())}"


def test_model_spec_forbids_extra_fields() -> None:
    """``ModelSpec`` rejects unknown kwargs (``extra='forbid'``)."""

    with pytest.raises(ValidationError) as excinfo:
        ModelSpec(name="m", unknown_field=42)  # type: ignore[call-arg]
    assert "unknown_field" in str(excinfo.value)


def test_model_spec_default_construction_uses_name_main() -> None:
    """Defaults: name required, revision "main", quantization/local_path/params all None."""

    spec = ModelSpec(name="test-model")
    assert spec.name == "test-model"
    assert spec.revision == "main"
    assert spec.quantization is None
    assert spec.local_path is None
    assert spec.total_params is None
    assert spec.active_params is None


def test_model_spec_name_field_min_length_is_one() -> None:
    """``name`` has ``min_length=1`` — empty string is rejected."""

    with pytest.raises(ValidationError):
        ModelSpec(name="")


def test_model_spec_name_field_accepts_non_empty_string() -> None:
    """``name="x"`` is valid (single-character name is allowed)."""

    spec = ModelSpec(name="x")
    assert spec.name == "x"


def test_model_spec_revision_default_is_main() -> None:
    """``revision`` default is ``"main"`` (the design pin)."""

    spec = ModelSpec(name="m")
    assert spec.revision == "main"


def test_model_spec_revision_can_be_overridden() -> None:
    """``revision="v1.2.3"`` is accepted."""

    spec = ModelSpec(name="m", revision="v1.2.3")
    assert spec.revision == "v1.2.3"


def test_model_spec_quantization_accepts_known_levels() -> None:
    """``quantization`` accepts the documented wire-form levels (``"4bit"``, ``"8bit"``)."""

    for q in ("4bit", "8bit", "fp16", "bf16"):
        spec = ModelSpec(name="m", quantization=q)
        assert spec.quantization == q


def test_model_spec_local_path_accepts_a_filesystem_path() -> None:
    """``local_path`` accepts a filesystem path string."""

    spec = ModelSpec(name="m", local_path="/var/models/test")
    assert spec.local_path == "/var/models/test"


def test_model_spec_total_params_default_is_none() -> None:
    """``total_params`` defaults to ``None`` (unannotated)."""

    spec = ModelSpec(name="m")
    assert spec.total_params is None


def test_model_spec_total_params_accepts_zero_and_above() -> None:
    """``total_params`` is bounded ``ge=0`` (0, 1, 1_000_000 all valid)."""

    for n in (0, 1, 1_000_000, 7_000_000_000):
        spec = ModelSpec(name="m", total_params=n)
        assert spec.total_params == n


def test_model_spec_total_params_rejects_negative() -> None:
    """``total_params=-1`` is rejected (``ge=0``)."""

    with pytest.raises(ValidationError):
        ModelSpec(name="m", total_params=-1)


def test_model_spec_total_params_rejects_fractional_float() -> None:
    """``total_params=1.5`` is rejected via Pydantic v2 ``int_from_float``."""

    with pytest.raises(ValidationError) as excinfo:
        ModelSpec(name="m", total_params=1.5)  # type: ignore[arg-type]
    assert "int_from_float" in str(excinfo.value)


def test_model_spec_active_params_default_is_none() -> None:
    """``active_params`` defaults to ``None`` (unannotated)."""

    spec = ModelSpec(name="m")
    assert spec.active_params is None


def test_model_spec_active_params_accepts_zero_and_above() -> None:
    """``active_params`` is bounded ``ge=0`` (0, 1, 1_000_000 all valid)."""

    for n in (0, 1, 1_000_000):
        spec = ModelSpec(name="m", active_params=n)
        assert spec.active_params == n


def test_model_spec_active_params_rejects_negative() -> None:
    """``active_params=-1`` is rejected (``ge=0``)."""

    with pytest.raises(ValidationError):
        ModelSpec(name="m", active_params=-1)


def test_model_spec_status_is_a_property_returning_proposed() -> None:
    """``status`` is a property that always returns ``Status.PROPOSED``."""

    spec = ModelSpec(name="m")
    # The `status` attribute is a property on the class, not a model field.
    assert isinstance(ModelSpec.__dict__["status"], property), (
        "ModelSpec.status must be a @property (not a Pydantic field)"
    )
    assert spec.status is Status.PROPOSED


def test_model_spec_status_property_always_returns_proposed_regardless_of_state() -> None:
    """``status`` is a constant pin: any ``ModelSpec`` reports ``Status.PROPOSED``."""

    spec_a = ModelSpec(name="a")
    spec_b = ModelSpec(name="b", total_params=7_000_000_000, active_params=1_000_000_000)
    spec_c = ModelSpec(name="c", quantization="4bit")
    assert spec_a.status is Status.PROPOSED
    assert spec_b.status is Status.PROPOSED
    assert spec_c.status is Status.PROPOSED


def test_model_spec_status_not_in_model_fields() -> None:
    """``status`` is a property, not a model field — does not appear in ``model_fields``."""

    assert "status" not in ModelSpec.model_fields, (
        "ModelSpec.status must be a @property, not a Pydantic field"
    )


def test_model_spec_model_dump_round_trips_every_field() -> None:
    """``model_dump()`` round-trips a fully populated ModelSpec (without status)."""

    spec = ModelSpec(
        name="my-model",
        revision="v1",
        quantization="4bit",
        local_path="/tmp/m",
        total_params=1_000_000_000,
        active_params=100_000_000,
    )
    dumped = spec.model_dump()
    assert dumped == {
        "name": "my-model",
        "revision": "v1",
        "quantization": "4bit",
        "local_path": "/tmp/m",
        "total_params": 1_000_000_000,
        "active_params": 100_000_000,
    }


# ---------------------------------------------------------------------------
# 4. discover_default_device
# ---------------------------------------------------------------------------


def test_discover_default_device_returns_a_string() -> None:
    """``discover_default_device()`` returns a non-empty ``str``."""

    device = discover_default_device()
    assert isinstance(device, str)
    assert device, "discover_default_device() returned an empty string"


def test_discover_default_device_signature_is_parameterless() -> None:
    """``discover_default_device()`` takes no parameters."""

    sig = inspect.signature(discover_default_device)
    assert len(list(sig.parameters.values())) == 0, (
        f"discover_default_device must take no parameters; got {sig.parameters!r}"
    )


def test_discover_default_device_with_mlx_available_returns_real_device() -> None:
    """With MLX installed, the result is a real device string (not ``unavailable:...``)."""

    device = discover_default_device()
    assert not device.startswith("unavailable:"), (
        f"MLX is available in this environment; expected a real device string, got {device!r}"
    )


def test_discover_default_device_with_mlx_unavailable_returns_graceful_string() -> None:
    """When MLX import fails, the result is ``"unavailable:<ExceptionName>"``."""

    # Patch the local import site to raise ImportError, then re-import the
    # function so it re-runs the import-block under the mock.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlx.core" or name.startswith("mlx."):
            raise ImportError("simulated mlx unavailability")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    with mock.patch.object(builtins, "__import__", side_effect=fake_import):
        # Re-evaluate the import block of discover_default_device by reloading the module.
        import importlib

        import oai2.runtime.model as model_module

        importlib.reload(model_module)
        try:
            device = model_module.discover_default_device()
        finally:
            # Always restore the original module so other tests are not affected.
            importlib.reload(model_module)

    assert isinstance(device, str)
    assert device.startswith("unavailable:"), (
        f"Expected an 'unavailable:<ExceptionName>' string, got {device!r}"
    )
    assert "ImportError" in device, (
        f"Expected ImportError in the unavailable string, got {device!r}"
    )


def test_discover_default_device_is_pure_no_mutation() -> None:
    """Two consecutive calls return the same device string."""

    assert discover_default_device() == discover_default_device()


# ---------------------------------------------------------------------------
# 5. smoke_check
# ---------------------------------------------------------------------------


def test_smoke_check_returns_a_two_tuple() -> None:
    """``smoke_check()`` returns ``tuple[bool, str]`` of length 2."""

    result = smoke_check()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_smoke_check_first_element_is_a_bool() -> None:
    """``smoke_check()[0]`` is a ``bool``."""

    ok, _ = smoke_check()
    assert isinstance(ok, bool)


def test_smoke_check_second_element_is_a_string() -> None:
    """``smoke_check()[1]`` is a non-empty ``str``."""

    _, device_or_error = smoke_check()
    assert isinstance(device_or_error, str)
    assert device_or_error, "smoke_check() second element is an empty string"


def test_smoke_check_signature_is_parameterless() -> None:
    """``smoke_check()`` takes no parameters."""

    sig = inspect.signature(smoke_check)
    assert len(list(sig.parameters.values())) == 0, (
        f"smoke_check must take no parameters; got {sig.parameters!r}"
    )


def test_smoke_check_return_annotation_is_tuple_bool_str() -> None:
    """``smoke_check()`` is annotated to return ``tuple[bool, str]``."""

    sig = inspect.signature(smoke_check)
    ann = sig.return_annotation
    # Under PEP 563, the annotation may be either resolved or its string form.
    assert (
        ann is tuple[bool, str]
        or ann == "tuple[bool, str]"
        or (isinstance(ann, str) and "tuple[bool, str]" in ann)
    ), f"smoke_check return annotation must be tuple[bool, str]; got {ann!r}"


def test_smoke_check_with_mlx_available_returns_true() -> None:
    """With MLX installed, the smoke check returns ``(True, device)``."""

    ok, device = smoke_check()
    assert ok is True, f"smoke_check ok must be True with MLX available; got (ok={ok}, {device!r})"
    assert not device.startswith("unavailable:"), (
        f"smoke_check second element must be a real device string; got {device!r}"
    )


def test_smoke_check_device_matches_discover_default_device() -> None:
    """``smoke_check`` reports the same device that ``discover_default_device`` reports."""

    _, smoke_device = smoke_check()
    assert smoke_device == discover_default_device(), (
        f"smoke_check device ({smoke_device!r}) must equal discover_default_device "
        f"({discover_default_device()!r})"
    )


def test_smoke_check_with_mlx_unavailable_returns_false_and_error() -> None:
    """When MLX import fails, ``smoke_check`` returns ``(False, error_string)``."""

    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlx.core" or name.startswith("mlx."):
            raise ImportError("simulated mlx unavailability")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    with mock.patch.object(builtins, "__import__", side_effect=fake_import):
        import importlib

        import oai2.runtime.model as model_module

        importlib.reload(model_module)
        try:
            ok, message = model_module.smoke_check()
        finally:
            importlib.reload(model_module)

    assert ok is False, f"smoke_check ok must be False when MLX is unavailable; got {ok!r}"
    assert isinstance(message, str)
    assert "ImportError" in message, (
        f"smoke_check error message must mention ImportError; got {message!r}"
    )


def test_smoke_check_routes_result_through_check_numerics() -> None:
    """``smoke_check`` calls ``check_numerics`` with the LONG_CONTEXT_REDUCTION op."""

    # The smoke check is observably dependent on check_numerics — if the
    # numerics gate were skipped or misused, the contract would change.
    # We assert the import path is real and the call site uses the
    # LONG_CONTEXT_REDUCTION operation (a static source check).
    assert "check_numerics" in _MODULE_SOURCE
    assert "NumericalOperation.LONG_CONTEXT_REDUCTION" in _MODULE_SOURCE


def test_smoke_check_stage_label_is_runtime_smoke_matmul_reduction() -> None:
    """``smoke_check`` uses ``stage="runtime.smoke.matmul_reduction"`` (a static source check)."""

    assert 'stage="runtime.smoke.matmul_reduction"' in _MODULE_SOURCE


def test_smoke_check_matmul_is_1x3_dot_3x1() -> None:
    """The matmul is ``[1,2,3] @ [[1],[1],[1]]`` which equals ``[6]`` (static source check)."""

    assert "mx.array([1.0, 2.0, 3.0])" in _MODULE_SOURCE
    assert "mx.array([[1.0], [1.0], [1.0]])" in _MODULE_SOURCE


def test_smoke_check_pure_repeated_call_is_stable() -> None:
    """Two consecutive calls return the same tuple contents."""

    assert smoke_check() == smoke_check()


# ---------------------------------------------------------------------------
# 6. Public-safety boundary
# ---------------------------------------------------------------------------


def test_runtime_model_module_does_not_contain_cloud_credentials() -> None:
    """No hard-coded API keys / private keys / bearer tokens in the source."""

    forbidden_patterns = (
        r"api[_-]?key\s*=\s*['\"]sk-",
        r"BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY",
        r"AKIA[0-9A-Z]{16}",  # AWS access key id pattern.
        r"AIza[0-9A-Za-z\-_]{35}",  # GCP API key pattern.
        r"xox[baprs]-[0-9A-Za-z\-]+",  # Slack token pattern.
    )
    for pattern in forbidden_patterns:
        assert not re.search(pattern, _MODULE_SOURCE, re.IGNORECASE), (
            f"runtime/model module must not contain credential-like pattern {pattern!r}"
        )


def test_runtime_model_module_does_not_contain_print_or_pprint_calls() -> None:
    """No top-level ``print(...)`` or ``pprint(...)`` calls in module code."""

    non_docstring_lines = [
        line
        for line in _MODULE_SOURCE.splitlines()
        if line.strip() and not line.strip().startswith('"""')
    ]
    for line in non_docstring_lines:
        assert not re.search(r"\bprint\s*\(", line), (
            f"runtime/model module must not call print(): {line!r}"
        )
        assert not re.search(r"\bpprint\s*\(", line), (
            f"runtime/model module must not call pprint(): {line!r}"
        )


def test_runtime_model_module_does_not_import_subprocess() -> None:
    """No ``subprocess`` / ``os.system`` / shell escape hatches."""

    forbidden = ("subprocess", "os.system", "shell=True")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, f"runtime/model module must not import/exec {token!r}"


def test_runtime_model_module_does_not_import_network_clients() -> None:
    """No outbound HTTP / RPC clients (``requests``, ``urllib``, ``httpx``)."""

    forbidden = (
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "import httpx",
        "from httpx",
        "import aiohttp",
        "from aiohttp",
    )
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"runtime/model module must not import network client {token!r}"
        )


def test_runtime_model_module_does_not_use_eval_or_exec() -> None:
    """No dynamic code execution primitives."""

    forbidden = (r"\beval\s\(", r"\bexec\s\(", r"\bcompile\s\(")
    for token in forbidden:
        assert not re.search(token, _MODULE_SOURCE), f"runtime/model module must not use {token!r}"


def test_runtime_model_module_does_not_use_wildcard_imports() -> None:
    """No ``from X import *`` statements (preserves explicit surface)."""

    assert not re.search(
        r"^from\s+\S+\s+import\s+\*",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "runtime/model module must not use wildcard imports"


def test_runtime_model_module_does_not_read_environment_variables() -> None:
    """No ``os.environ`` / ``os.getenv`` access (no runtime config leak)."""

    forbidden = ("os.environ", "os.getenv")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"runtime/model module must not read environment via {token!r}"
        )


def test_runtime_model_module_does_not_contain_todo_or_fixme_markers() -> None:
    """No ``TODO`` / ``FIXME`` / ``XXX`` markers — the scaffold is finished."""

    forbidden = ("TODO", "FIXME", "XXX")
    for token in forbidden:
        assert not re.search(
            rf"\b{re.escape(token)}\b",
            _MODULE_SOURCE,
        ), f"runtime/model module must not contain {token!r} markers"


def test_runtime_model_module_source_is_well_terminated() -> None:
    """Source ends with a single trailing newline (POSIX)."""

    assert _MODULE_SOURCE.endswith("\n"), (
        "runtime/model module source must end with a trailing newline"
    )
    assert not _MODULE_SOURCE.endswith("\n\n\n"), (
        "runtime/model module source must not have multiple trailing newlines"
    )


def test_runtime_model_module_has_no_top_level_dunder_side_effects() -> None:
    """No top-level statements beyond imports, defs, class defs, and ``__all__``."""

    import ast

    tree = ast.parse(_MODULE_SOURCE)
    allowed = (
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
    )
    for node in tree.body:
        assert isinstance(node, allowed), (
            f"runtime/model module has unexpected top-level statement "
            f"{type(node).__name__} at line {node.lineno}"
        )
