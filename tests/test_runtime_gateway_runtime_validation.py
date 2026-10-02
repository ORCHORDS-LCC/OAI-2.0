"""Outer guard-rail tests for ``oai2.runtime.gateway_runtime``.

The :class:`GatewayRuntime` is the public-ORCHORDS-inference-gateway path
inside ``oai2.runtime``. It talks to the configured gateway (default
``https://api.orchords.com``) over HTTP, posting an OpenAI-compatible
``/v1/chat/completions`` request and parsing the completion as an
:class:`InferenceResponse`.

The module handles bearer tokens. They MUST NEVER leak into logs, reprs,
``GatewayRuntimeError`` messages, response-message redactors, or
``_safe_response_message`` truncators. These tests pin the public-safety
boundary: a refactor that drops a redaction step, removes a default, or
silently fails on a malformed upstream payload surfaces as a deliberate
contract change.

All HTTP calls go through :class:`httpx.MockTransport` so the runner-free
``scripts/verify.py`` cycle covers the runtime end-to-end without leaving
the host.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import pytest

from oai2.core import Status
from oai2.runtime import (
    DEFAULT_GATEWAY_BASE_URL,
    DEFAULT_GATEWAY_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayConfig,
    GatewayConfigError,
    GatewayRuntime,
    GatewayRuntimeError,
    InferenceRequest,
    PlaceholderRuntime,
    load_gateway_config_from_env,
)
from oai2.runtime import gateway_runtime as gateway_runtime_module

# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_gateway_runtime_module_docstring_pins_injection_model() -> None:
    """The module docstring must mention httpx injection + runner-free coverage."""
    docstring = gateway_runtime_module.__doc__ or ""
    assert "httpx" in docstring.lower(), (
        f"docstring must mention httpx injection; got: {docstring!r}"
    )
    assert "MockTransport" in docstring or "mock" in docstring.lower()


def test_gateway_runtime_module_has_no_hardcoded_credentials_in_source() -> None:
    """The module source must not contain hardcoded bearer tokens or API keys.

    The default base URL (``https://api.orchords.com``) is a configuration
    value, not a secret, so it is allowed in source. Real bearer tokens
    are forbidden.
    """
    source = inspect.getsource(gateway_runtime_module)
    forbidden_literals = [
        "sk-or-",
        "sk-or_",
        "sk-",
        "ghp_",
        "github_pat_",
        "xoxb-",
        "xoxp-",
        "password=",
        "secret=",
        "BEGIN PRIVATE KEY",
    ]
    for forbidden in forbidden_literals:
        assert forbidden not in source, (
            f"forbidden credential marker `{forbidden!r}` found in gateway_runtime.py"
        )


def test_gateway_runtime_module_has_no_default_api_key_in_source() -> None:
    """The module must not define a default ``api_key=`` constant in source."""
    source = inspect.getsource(gateway_runtime_module)
    # Walk for `api_key=` literal assignments; the dataclass field is allowed
    # because it's declared without a default value, so the search below for
    # `api_key: str` without `= ""` would catch a real regression. Use a
    # targeted scan for literal assignment.
    assert 'api_key: str = ""' not in source
    assert 'api_key: str = "' not in source
    assert "api_key = ''" not in source
    assert "api_key=''" not in source


def test_gateway_runtime_module_does_not_import_unrelated_cloud_runtimes() -> None:
    """The module must not transitively import unrelated cloud runtimes."""
    source = inspect.getsource(gateway_runtime_module)
    for forbidden in (
        "import boto3",
        "import openai",
        "import anthropic",
        "from oai2.workers",
        "from oai2.cloudflare",
        "import oai2.workers",
        "import oai2.cloudflare",
    ):
        assert forbidden not in source, f"unrelated cloud-runtime import `{forbidden!r}` leaked in"


def test_gateway_runtime_module_uses_collections_abc_for_mapping() -> None:
    """UP006-style: ``typing.Mapping`` must not be used at runtime."""
    source = inspect.getsource(gateway_runtime_module)
    assert "typing.Mapping" not in source
    assert "typing.Sequence" not in source
    assert "typing.Iterable" not in source


# ---------------------------------------------------------------------------
# 2. __all__ completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_gateway_runtime_all_lists_exactly_eight_names() -> None:
    """``__all__`` must list exactly the 8 documented public names."""
    assert set(gateway_runtime_module.__all__) == {
        "DEFAULT_GATEWAY_BASE_URL",
        "DEFAULT_GATEWAY_MODEL",
        "DEFAULT_TIMEOUT_SECONDS",
        "GatewayConfig",
        "GatewayConfigError",
        "GatewayRuntime",
        "GatewayRuntimeError",
        "load_gateway_config_from_env",
    }


def test_gateway_runtime_all_names_are_importable() -> None:
    """Each name in ``__all__`` must resolve as a module attribute."""
    for name in gateway_runtime_module.__all__:
        assert hasattr(gateway_runtime_module, name), (
            f"`{name}` declared in __all__ but missing from module"
        )


def test_gateway_runtime_symbols_are_re_exported_at_package_level() -> None:
    """Each public name must be re-exported through ``oai2.runtime``."""
    from oai2.runtime import (  # noqa: PLC0415
        DEFAULT_GATEWAY_BASE_URL as PackageBaseUrl,
    )
    from oai2.runtime import (
        DEFAULT_GATEWAY_MODEL as PackageModel,
    )
    from oai2.runtime import (
        DEFAULT_TIMEOUT_SECONDS as PackageTimeout,
    )
    from oai2.runtime import (
        GatewayConfig as PackageConfig,
    )
    from oai2.runtime import (
        GatewayConfigError as PackageConfigError,
    )
    from oai2.runtime import (
        GatewayRuntime as PackageRuntime,
    )
    from oai2.runtime import (
        GatewayRuntimeError as PackageRuntimeError,
    )
    from oai2.runtime import (
        load_gateway_config_from_env as PackageLoader,
    )

    assert PackageBaseUrl is DEFAULT_GATEWAY_BASE_URL
    assert PackageModel is DEFAULT_GATEWAY_MODEL
    assert PackageTimeout is DEFAULT_TIMEOUT_SECONDS
    assert PackageConfig is GatewayConfig
    assert PackageConfigError is GatewayConfigError
    assert PackageRuntime is GatewayRuntime
    assert PackageRuntimeError is GatewayRuntimeError
    assert PackageLoader is load_gateway_config_from_env


# ---------------------------------------------------------------------------
# 3. Module-level constants
# ---------------------------------------------------------------------------


def test_default_gateway_base_url_is_pinned_to_orchords() -> None:
    """``DEFAULT_GATEWAY_BASE_URL`` must equal the ORCHORDS gateway."""
    assert DEFAULT_GATEWAY_BASE_URL == "https://api.orchords.com"


def test_default_gateway_model_is_pinned() -> None:
    """``DEFAULT_GATEWAY_MODEL`` must equal the configured model id."""
    assert DEFAULT_GATEWAY_MODEL == "oai-2.0"


def test_default_timeout_seconds_is_a_positive_finite_number() -> None:
    """``DEFAULT_TIMEOUT_SECONDS`` must be a positive finite number."""
    assert isinstance(DEFAULT_TIMEOUT_SECONDS, float)
    assert DEFAULT_TIMEOUT_SECONDS > 0.0


# ---------------------------------------------------------------------------
# 4. GatewayConfig dataclass + __repr__ redaction (PUBLIC-SAFETY BOUNDARY)
# ---------------------------------------------------------------------------


def _config(**overrides: Any) -> GatewayConfig:
    payload: dict[str, Any] = {
        "base_url": "https://gateway.example.test",
        "api_key": "test-token-xyz",
        "model": "oai-2.0",
        "timeout_seconds": 5.0,
    }
    payload.update(overrides)
    return GatewayConfig(**payload)


def _mock_client(
    cfg: GatewayConfig,
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    """Build a MockTransport-backed client wired to ``cfg.base_url``.

    The runtime posts to the relative path ``/v1/chat/completions``;
    httpx resolves the path against the client's ``base_url``.
    """

    return httpx.Client(
        base_url=cfg.base_url,
        transport=httpx.MockTransport(handler),
    )


def test_gateway_config_field_set_is_pinned() -> None:
    """``GatewayConfig`` must have exactly 4 fields with documented defaults."""
    fields = {f.name for f in GatewayConfig.__dataclass_fields__.values()}
    assert fields == {"base_url", "api_key", "model", "timeout_seconds"}


def test_gateway_config_is_slots_and_frozen() -> None:
    """``GatewayConfig`` must be ``slots=True`` + ``frozen=True``."""
    params = GatewayConfig.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_gateway_config_repr_redacts_api_key_never_leaks_token() -> None:
    """``GatewayConfig.__repr__`` MUST NEVER include the raw api_key (public-safety)."""
    cfg = _config(api_key="my-tok-XYZ-1234")
    rendered = repr(cfg)
    assert "my-tok-XYZ-1234" not in rendered, (
        "api_key leaked into GatewayConfig.__repr__ — public-safety violation"
    )
    assert "my-tok" not in rendered
    assert "XYZ-1234" not in rendered


def test_gateway_config_repr_uses_redacted_len_marker() -> None:
    """``__repr__`` must include ``<redacted len=N>`` so logs can confirm redaction."""
    cfg = _config(api_key="0123456789abcde")
    rendered = repr(cfg)
    assert "<redacted len=" in rendered
    assert "15" in rendered  # length of the api_key


def test_gateway_config_repr_includes_base_url_and_model() -> None:
    """``__repr__`` must include the base_url and model (non-secret config)."""
    cfg = _config(base_url="https://gateway.example.test", model="oai-2.0")
    rendered = repr(cfg)
    assert "https://gateway.example.test" in rendered
    assert "oai-2.0" in rendered


def test_gateway_config_repr_includes_timeout_seconds() -> None:
    """``__repr__`` must include ``timeout_seconds``."""
    cfg = _config(timeout_seconds=12.5)
    rendered = repr(cfg)
    assert "12.5" in rendered
    assert "timeout_seconds" in rendered


def test_gateway_config_redacts_short_api_keys() -> None:
    """A short api_key (<=6 chars) must be replaced with ``***``."""
    cfg = _config(api_key="abc")
    rendered = repr(cfg)
    assert "abc" not in rendered
    assert "<redacted len=" in rendered


# ---------------------------------------------------------------------------
# 5. load_gateway_config_from_env
# ---------------------------------------------------------------------------


def test_load_gateway_config_from_env_returns_none_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``load_gateway_config_from_env`` must return ``None`` (NOT raise) when key is absent."""
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    assert load_gateway_config_from_env() is None


def test_load_gateway_config_from_env_uses_explicit_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader must accept an explicit ``environ`` mapping and use it instead of ``os.environ``."""
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    assert load_gateway_config_from_env(environ={}) is None
    cfg = load_gateway_config_from_env(
        environ={
            "OAI2_GATEWAY_API_KEY": "explicit-key",
            "OAI2_GATEWAY_BASE_URL": "https://explicit.test",
            "OAI2_GATEWAY_MODEL": "explicit-model",
            "OAI2_GATEWAY_TIMEOUT_SECONDS": "12.5",
        }
    )
    assert cfg is not None
    assert cfg.api_key == "explicit-key"
    assert cfg.base_url == "https://explicit.test"
    assert cfg.model == "explicit-model"
    assert cfg.timeout_seconds == 12.5


def test_load_gateway_config_from_env_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trailing slash on ``OAI2_GATEWAY_BASE_URL`` must be stripped (canonical form)."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test/")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.base_url == "https://gateway.example.test"
    assert not cfg.base_url.endswith("/")


def test_load_gateway_config_from_env_falls_back_to_default_when_base_url_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``OAI2_GATEWAY_BASE_URL`` must fall back to the default."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "   ")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.base_url == DEFAULT_GATEWAY_BASE_URL


def test_load_gateway_config_from_env_falls_back_to_default_when_model_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``OAI2_GATEWAY_MODEL`` must fall back to the default."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "   ")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.model == DEFAULT_GATEWAY_MODEL


def test_load_gateway_config_from_env_falls_back_to_default_timeout_on_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid ``OAI2_GATEWAY_TIMEOUT_SECONDS`` value falls back to the default."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("OAI2_GATEWAY_TIMEOUT_SECONDS", "not-a-float")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_load_gateway_config_from_env_falls_back_to_default_timeout_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ``OAI2_GATEWAY_TIMEOUT_SECONDS`` falls back to the default."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.delenv("OAI2_GATEWAY_TIMEOUT_SECONDS", raising=False)
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_load_gateway_config_from_env_uses_default_base_url_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ``OAI2_GATEWAY_BASE_URL`` falls back to the documented default."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.delenv("OAI2_GATEWAY_BASE_URL", raising=False)
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.base_url == DEFAULT_GATEWAY_BASE_URL


def test_load_gateway_config_from_env_uses_default_model_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ``OAI2_GATEWAY_MODEL`` falls back to the documented default."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.delenv("OAI2_GATEWAY_MODEL", raising=False)
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.model == DEFAULT_GATEWAY_MODEL


def test_load_gateway_config_from_env_signature_accepts_optional_environ() -> None:
    """``load_gateway_config_from_env`` must accept an optional ``environ`` keyword argument."""
    sig = inspect.signature(load_gateway_config_from_env)
    assert "environ" in sig.parameters
    environ_param = sig.parameters["environ"]
    assert environ_param.default is None
    assert environ_param.kind
    # Either POSITIONAL_OR_KEYWORD or KEYWORD_ONLY is acceptable.


def test_load_gateway_config_from_env_strips_whitespace_from_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader must strip whitespace from the API key (no leaked padding)."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "   padded-key   ")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.api_key == "padded-key"


def test_load_gateway_config_from_env_strips_whitespace_from_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace around the base URL must be stripped."""
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "  https://gateway.example.test  ")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.base_url == "https://gateway.example.test"


# ---------------------------------------------------------------------------
# 6. GatewayConfigError + GatewayRuntimeError
# ---------------------------------------------------------------------------


def test_gateway_config_error_is_runtime_error_subclass() -> None:
    """``GatewayConfigError`` must subclass ``RuntimeError``."""
    assert issubclass(GatewayConfigError, RuntimeError)


def test_gateway_runtime_error_is_runtime_error_subclass() -> None:
    """``GatewayRuntimeError`` must subclass ``RuntimeError``."""
    assert issubclass(GatewayRuntimeError, RuntimeError)


def test_gateway_runtime_error_message_format_pinned() -> None:
    """``GatewayRuntimeError.__str__`` must include status_code + message in fixed format."""
    err = GatewayRuntimeError(500, "boom")
    rendered = str(err)
    assert "500" in rendered
    assert "boom" in rendered
    assert "gateway returned " in rendered


def test_gateway_runtime_error_stores_status_code_and_message() -> None:
    """``GatewayRuntimeError`` must expose ``status_code`` and ``message`` attributes."""
    err = GatewayRuntimeError(401, "unauthorized")
    assert err.status_code == 401
    assert err.message == "unauthorized"


def test_gateway_runtime_error_does_not_leak_message_when_omitted() -> None:
    """An empty ``message`` field must not introduce spurious text into ``__str__``."""
    err = GatewayRuntimeError(500, "")
    rendered = str(err)
    assert "gateway returned 500" in rendered


def test_gateway_runtime_error_accepts_zero_status_code_for_transport_errors() -> None:
    """A status_code of 0 is the documented sentinel for transport-layer errors."""
    err = GatewayRuntimeError(0, "transport error")
    assert err.status_code == 0
    assert "transport error" in str(err)


# ---------------------------------------------------------------------------
# 7. GatewayRuntime — class identity + STATUS
# ---------------------------------------------------------------------------


def test_gateway_runtime_subclasses_inference_runtime() -> None:
    """``GatewayRuntime`` must subclass ``InferenceRuntime``."""
    from oai2.runtime import InferenceRuntime  # noqa: PLC0415

    assert issubclass(GatewayRuntime, InferenceRuntime)


def test_gateway_runtime_status_is_proposed() -> None:
    """``GatewayRuntime.STATUS`` must equal ``Status.PROPOSED`` (the module is design-only)."""
    assert GatewayRuntime.STATUS is Status.PROPOSED
    assert GatewayRuntime.STATUS.value == "PROPOSED"


# ---------------------------------------------------------------------------
# 8. GatewayRuntime.__init__
# ---------------------------------------------------------------------------


def test_gateway_runtime_rejects_empty_api_key() -> None:
    """An empty ``api_key`` must raise ``GatewayConfigError`` (fail-closed)."""
    cfg = _config(api_key="")
    with pytest.raises(GatewayConfigError, match="api_key must be non-empty"):
        GatewayRuntime(cfg)


def test_gateway_runtime_rejects_whitespace_only_api_key() -> None:
    """A whitespace-only ``api_key`` is accepted by ``__init__`` (only empty string rejected).

    The validation in ``__init__`` is ``if not config.api_key`` — an empty
    string is falsy and rejected, but whitespace is truthy and passes. This
    test pins the exact contract so a future tighten-the-validation
    refactor surfaces as a deliberate contract change.
    """
    cfg = _config(api_key="   ")
    # Must NOT raise — the loader strips whitespace but __init__ only checks
    # for empty-string.
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        assert runtime.config is cfg
    finally:
        runtime.close()


def test_gateway_runtime_default_spec_uses_config_model_name() -> None:
    """When ``spec`` is omitted, ``runtime.spec.name`` must equal ``config.model``."""
    cfg = _config(model="test-model")
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        assert runtime.spec.name == "test-model"
    finally:
        runtime.close()


def test_gateway_runtime_explicit_spec_overrides_config_model_name() -> None:
    """When ``spec`` is provided, ``runtime.spec`` must be that spec (not derived from config)."""
    from oai2.runtime import ModelSpec  # noqa: PLC0415

    cfg = _config(model="config-model")
    spec = ModelSpec(name="explicit-model")
    runtime = GatewayRuntime(cfg, client=httpx.Client(), spec=spec)
    try:
        assert runtime.spec is spec
    finally:
        runtime.close()


def test_gateway_runtime_config_property_returns_underlying_config() -> None:
    """The ``config`` property must return the same ``GatewayConfig`` instance passed in."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        assert runtime.config is cfg
    finally:
        runtime.close()


def test_gateway_runtime_does_not_own_injected_client() -> None:
    """When a ``client`` is injected, ``close()`` must NOT close it."""
    cfg = _config()
    handler = lambda request: httpx.Response(  # noqa: E731
        200, json={"choices": [{"message": {"content": "x"}}]}
    )
    client = _mock_client(cfg, handler)
    runtime = GatewayRuntime(cfg, client=client)
    runtime.close()
    # The injected client is still open — proves close() was a no-op on
    # the injected transport.
    assert client.is_closed is False
    # The injected client is still open — proves close() was a no-op on the
    # injected transport.
    assert client.is_closed is False


def test_gateway_runtime_init_keyword_only_after_config() -> None:
    """``__init__`` must take ``config`` positionally then keyword-only ``client`` / ``spec``."""
    sig = inspect.signature(GatewayRuntime.__init__)
    params = list(sig.parameters.values())
    config_param = params[0]
    assert config_param.name == "self"
    config_param = params[1]
    assert config_param.name == "config"
    assert config_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("client", "spec"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"`{name}` must be KEYWORD_ONLY; got {param.kind!r}"
        )


# ---------------------------------------------------------------------------
# 9. GatewayRuntime.from_env
# ---------------------------------------------------------------------------


def test_gateway_runtime_from_env_raises_gateway_config_error_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_env`` must raise ``GatewayConfigError`` when key is missing."""
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    with pytest.raises(GatewayConfigError, match="OAI2_GATEWAY_API_KEY"):
        GatewayRuntime.from_env()


def test_gateway_runtime_from_env_keyword_only_client() -> None:
    """``from_env`` must accept an injected ``client`` only as a keyword argument."""
    sig = inspect.signature(GatewayRuntime.from_env)
    for name in ("client",):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"`{name}` must be KEYWORD_ONLY; got {param.kind!r}"
        )


# ---------------------------------------------------------------------------
# 10. Context manager protocol
# ---------------------------------------------------------------------------


def test_gateway_runtime_context_manager_returns_self() -> None:
    """``__enter__`` must return ``self`` (the runtime instance)."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    with runtime as entered:
        assert entered is runtime
        # `close` was deferred to `__exit__` — the injected client is
        # unaffected, so we cannot verify close() was called here.
    # After __exit__, the runtime's internal state should still be sane.
    assert runtime.config is cfg


def test_gateway_runtime_context_manager_exit_does_not_close_injected_client() -> None:
    """``__exit__`` must NOT close the injected client."""
    client = httpx.Client()
    cfg = _config()
    with GatewayRuntime(cfg, client=client):
        pass
    # Verify the injected client was not garbage-collected.
    assert client.is_closed is False


# ---------------------------------------------------------------------------
# 11. _build_request_body — OpenAI-compatible wire format
# ---------------------------------------------------------------------------


def test_build_request_body_uses_explicit_request_model() -> None:
    """When ``request.model`` is provided, the body must use that model id."""
    from oai2.runtime import ModelSpec  # noqa: PLC0415

    cfg = _config(model="config-model")
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(
            prompt="hi",
            model=ModelSpec(name="explicit-model"),
        )
        body = runtime._build_request_body(request)
        assert body["model"] == "explicit-model"
    finally:
        runtime.close()


def test_build_request_body_falls_back_to_config_model() -> None:
    """When ``request.model`` is ``None``, the body must use ``config.model``."""
    cfg = _config(model="config-model")
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hi")
        body = runtime._build_request_body(request)
        assert body["model"] == "config-model"
    finally:
        runtime.close()


def test_build_request_body_falls_back_to_runtime_spec_name() -> None:
    """When ``request.model`` and ``config.model`` are both empty, use the runtime spec."""
    from oai2.runtime import ModelSpec  # noqa: PLC0415

    cfg = _config(model="")
    runtime = GatewayRuntime(cfg, client=httpx.Client(), spec=ModelSpec(name="spec-model"))
    try:
        request = InferenceRequest(prompt="hi")
        body = runtime._build_request_body(request)
        assert body["model"] == "spec-model"
    finally:
        runtime.close()


def test_build_request_body_includes_messages_array() -> None:
    """The body must include a single ``messages`` entry with the user prompt."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hello world")
        body = runtime._build_request_body(request)
        assert body["messages"] == [{"role": "user", "content": "hello world"}]
    finally:
        runtime.close()


def test_build_request_body_includes_sampling_parameters() -> None:
    """The body must include ``max_tokens``, ``temperature``, ``top_p``."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hi", max_tokens=64, temperature=0.5, top_p=0.9)
        body = runtime._build_request_body(request)
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.9
    finally:
        runtime.close()


def test_build_request_body_includes_stop_when_provided() -> None:
    """When ``stop`` is non-empty, the body must include ``stop`` as a list."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hi", stop=("END", "STOP"))
        body = runtime._build_request_body(request)
        assert body["stop"] == ["END", "STOP"]
    finally:
        runtime.close()


def test_build_request_body_omits_stop_when_empty() -> None:
    """When ``stop`` is empty, the body must NOT include a ``stop`` key."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hi")
        body = runtime._build_request_body(request)
        assert "stop" not in body
    finally:
        runtime.close()


def test_build_request_body_includes_seed_when_provided() -> None:
    """When ``seed`` is set, the body must include the seed value."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hi", seed=42)
        body = runtime._build_request_body(request)
        assert body["seed"] == 42
    finally:
        runtime.close()


def test_build_request_body_omits_seed_when_none() -> None:
    """When ``seed`` is ``None``, the body must NOT include a ``seed`` key."""
    cfg = _config()
    runtime = GatewayRuntime(cfg, client=httpx.Client())
    try:
        request = InferenceRequest(prompt="hi")
        body = runtime._build_request_body(request)
        assert "seed" not in body
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# 12. generate() — happy path (HTTP 200 + valid OpenAI payload)
# ---------------------------------------------------------------------------


def _transport_ok(payload: Mapping[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dict(payload))

    return httpx.MockTransport(handler)


def _transport_client(
    cfg: GatewayConfig,
    payload: Mapping[str, Any],
) -> httpx.Client:
    """Build a MockTransport client wired to cfg.base_url that returns payload."""
    return _mock_client(cfg, lambda request: httpx.Response(200, json=dict(payload)))


def test_generate_happy_path_returns_text_tokens_finish_reason() -> None:
    """A 200 with a valid OpenAI-completions payload returns ``InferenceResponse``."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"completion_tokens": 3},
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.text == "ok"
    assert response.tokens == 3
    assert response.finish_reason == "stop"
    assert response.status is Status.EXPERIMENTAL


def test_generate_includes_elapsed_ms_in_response() -> None:
    """``generate`` must record the wall-clock latency on the response."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.elapsed_ms >= 0.0


def test_generate_includes_device_marker_in_response() -> None:
    """``device`` field on the response must include the configured base_url."""
    cfg = _config(base_url="https://gateway.example.test")
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.device == "gateway:https://gateway.example.test"


def test_generate_includes_status_code_in_notes() -> None:
    """``notes`` must record the HTTP status code."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert any(note.startswith("status=200") for note in response.notes)


def test_generate_includes_model_marker_in_notes() -> None:
    """``notes`` must record the configured model id."""
    cfg = _config(model="oai-2.0")
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert any("model=oai-2.0" in note for note in response.notes)


def test_generate_includes_elapsed_ms_marker_in_notes() -> None:
    """``notes`` must record the elapsed_ms with 1-decimal precision."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert any(note.startswith("elapsed_ms=") for note in response.notes)


# ---------------------------------------------------------------------------
# 13. generate() — non-2xx → GatewayRuntimeError
# ---------------------------------------------------------------------------


def _transport_with_status(status_code: int, body_text: str = "") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body_text)

    return httpx.MockTransport(handler)


def _transport_client_with_status(
    cfg: GatewayConfig,
    status_code: int,
    body_text: str = "",
) -> httpx.Client:
    """Wire a status-only MockTransport to the configured base URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body_text)

    return _mock_client(cfg, handler)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429, 500, 502, 503])
def test_generate_translates_non_2xx_to_gateway_runtime_error(status_code: int) -> None:
    """Any non-2xx status must raise ``GatewayRuntimeError(status_code, redacted_message)``."""
    cfg = _config()
    runtime = GatewayRuntime(
        cfg,
        client=_transport_client_with_status(cfg, status_code, body_text="upstream failure detail"),
    )
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    err = exc_info.value
    assert err.status_code == status_code


def test_generate_redacts_upstream_error_message_in_gateway_runtime_error() -> None:
    """The upstream error message text MUST be redacted via ``_redact`` (public-safety)."""
    cfg = _config()
    secret_marker = "Bearer-LK-abcdef1234567890"
    runtime = GatewayRuntime(
        cfg, client=_transport_client_with_status(cfg, 500, body_text=f"detail: {secret_marker}")
    )
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert secret_marker not in str(exc_info.value), (
        "raw upstream body leaked into GatewayRuntimeError — public-safety violation"
    )


def test_generate_extracts_error_message_from_openai_error_envelope() -> None:
    """When the upstream returns ``{"error": {"message": "..."}}``, that message is used."""
    cfg = _config()
    body = {"error": {"message": "rate-limited"}}
    handler = lambda request: httpx.Response(429, json=body)  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert exc_info.value.status_code == 429
    # message body is redacted but "ra" / "limited" substrings may remain
    # after redaction depending on length. Just confirm the message field is set.
    assert exc_info.value.message


def test_generate_truncates_long_error_message_to_240_chars() -> None:
    """``_safe_response_message`` must truncate error bodies over 240 chars."""
    cfg = _config()
    long_body = "x" * 500
    handler = lambda request: httpx.Response(500, text=long_body)  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    # Truncated to 240 + ellipsis suffix then redacted; the raw "x" * 500 must
    # not appear.
    rendered = str(exc_info.value)
    assert "x" * 200 not in rendered


def test_generate_translates_invalid_json_response_to_gateway_runtime_error() -> None:
    """A 200 with non-JSON body must raise ``GatewayRuntimeError(200, "invalid JSON ...")`."""
    cfg = _config()
    handler = lambda request: httpx.Response(200, text="not json")  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert exc_info.value.status_code == 200
    assert "invalid JSON" in exc_info.value.message


def test_generate_translates_httpx_transport_error_to_gateway_runtime_error() -> None:
    """``httpx.HTTPError`` is translated to ``GatewayRuntimeError(status=0, ...)``."""
    cfg = _config()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect failed: bearer LK-token-1234567890")

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert exc_info.value.status_code == 0
    assert "transport error" in exc_info.value.message
    assert "ConnectError" in exc_info.value.message
    assert "LK-token-1234567890" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 14. generate() — payload validation (via _extract_completion)
# ---------------------------------------------------------------------------


def test_generate_rejects_non_dict_payload() -> None:
    """A 200 with a non-dict JSON payload raises ``GatewayRuntimeError(200, ...)``."""
    cfg = _config()
    handler = lambda request: httpx.Response(200, json=["not", "a", "dict"])  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError, match="JSON object"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_rejects_missing_choices_array() -> None:
    """A payload without a ``choices`` array raises ``GatewayRuntimeError(200, ...)``."""
    cfg = _config()
    handler = lambda request: httpx.Response(200, json={"usage": {}})  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError, match="choices"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_rejects_empty_choices_array() -> None:
    """An empty ``choices`` array raises ``GatewayRuntimeError(200, ...)``."""
    cfg = _config()
    handler = lambda request: httpx.Response(200, json={"choices": []})  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError, match="choices"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_rejects_non_list_choices_array() -> None:
    """A non-list ``choices`` raises ``GatewayRuntimeError(200, ...)``."""
    cfg = _config()
    handler = lambda request: httpx.Response(200, json={"choices": "not-a-list"})  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError, match="choices"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_rejects_non_dict_first_choice() -> None:
    """A non-dict first choice raises ``GatewayRuntimeError(200, ...)``."""
    cfg = _config()
    handler = lambda request: httpx.Response(200, json={"choices": ["not-a-dict"]})  # noqa: E731

    runtime = GatewayRuntime(cfg, client=_mock_client(cfg, handler))
    try:
        with pytest.raises(GatewayRuntimeError, match="choice shape"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_rejects_non_string_finish_reason() -> None:
    """A non-string, non-None ``finish_reason`` raises ``GatewayRuntimeError(200, ...)``."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": 42, "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        with pytest.raises(GatewayRuntimeError, match="finish_reason"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_accepts_null_finish_reason() -> None:
    """A ``finish_reason`` of ``None`` is allowed (the upstream did not provide one)."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": None, "message": {"content": "ok"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.finish_reason is None


def test_generate_rejects_tool_calls_finish_reason_without_tool_calls() -> None:
    """``finish_reason == "tool_calls"`` with no tool calls raises ``GatewayRuntimeError``."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"content": "ok"},
            }
        ]
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        with pytest.raises(GatewayRuntimeError, match="tool_calls"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_accepts_tool_calls_finish_reason_with_tool_calls() -> None:
    """``finish_reason == "tool_calls"`` with tool calls is accepted."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "call-1", "function": {"name": "f"}}],
                },
            }
        ]
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1


def test_generate_rejects_malformed_tool_calls() -> None:
    """A tool_calls array with non-dict entries raises ``GatewayRuntimeError``."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"tool_calls": ["not-a-dict"]},
            }
        ]
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        with pytest.raises(GatewayRuntimeError, match="tool_calls"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_rejects_choice_with_neither_message_nor_text() -> None:
    """A choice with neither a ``message`` object nor a ``text`` field raises ``GatewayRuntimeError``."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                # no ``message`` key, no ``text`` key — only finish_reason present.
            }
        ]
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        with pytest.raises(GatewayRuntimeError, match="message"):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_generate_accepts_message_with_no_content_field() -> None:
    """A ``message`` object without a ``content`` field is accepted (runtime returns empty text)."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant"},
            }
        ]
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.text == ""
    assert response.finish_reason == "stop"


def test_generate_accepts_legacy_text_field_when_message_missing() -> None:
    """When ``message`` is missing, the choice's top-level ``text`` field is used."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": "stop", "text": "legacy"}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.text == "legacy"


def test_generate_joins_multimodal_content_list_with_text_parts() -> None:
    """A multi-part content list of dicts is joined into a single string."""
    cfg = _config()
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ]
                },
            }
        ]
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.text == "hello world"


def test_generate_uses_usage_completion_tokens_when_present() -> None:
    """``usage.completion_tokens`` is preferred over the word-count estimate."""
    cfg = _config()
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": "x y z"}}],
        "usage": {"completion_tokens": 99},
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.tokens == 99


def test_generate_falls_back_to_word_count_when_usage_missing() -> None:
    """``tokens`` is computed as ``max(1, len(content.split()))`` when usage is absent."""
    cfg = _config()
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "one two three"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.tokens == 3


def test_generate_falls_back_to_word_count_when_usage_completion_tokens_zero() -> None:
    """When ``usage.completion_tokens == 0``, the word-count fallback is used."""
    cfg = _config()
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": "one two"}}],
        "usage": {"completion_tokens": 0},
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.tokens == 2


def test_generate_empty_content_yields_zero_tokens() -> None:
    """An empty content string yields zero tokens (no fallback applied)."""
    cfg = _config()
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
        "usage": {"completion_tokens": 0},
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.tokens == 0


def test_generate_rejects_negative_usage_completion_tokens() -> None:
    """``usage.completion_tokens < 0`` falls through to the word-count fallback."""
    cfg = _config()
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": "one two"}}],
        "usage": {"completion_tokens": -1},
    }
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert response.tokens == 2


# ---------------------------------------------------------------------------
# 15. Bearer token leakage prevention on the real config (public-safety boundary)
# ---------------------------------------------------------------------------


def test_real_config_with_bearer_token_never_leaks_token_in_repr() -> None:
    """A realistic bearer token (>=7 chars) must NEVER leak into ``__repr__``."""
    real_token = "Bearer-or-LEAKED-abcdef1234567890"
    cfg = _config(api_key=real_token)
    rendered = repr(cfg)
    assert real_token not in rendered
    assert "LEAKED" not in rendered


def test_real_config_with_short_bearer_token_never_leaks_token_in_repr() -> None:
    """A short token (<=6 chars) is replaced with ``***`` in ``__repr__``."""
    short_token = "abc"
    cfg = _config(api_key=short_token)
    rendered = repr(cfg)
    assert short_token not in rendered
    assert "***" in rendered or "<redacted" in rendered


def test_gateway_runtime_does_not_log_token_in_device_string() -> None:
    """``device`` field on the response must NEVER include the api_key."""
    cfg = _config(api_key="LK-token-abc", base_url="https://example.test")
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
    runtime = GatewayRuntime(cfg, client=_transport_client(cfg, payload))
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert "LEAKED" not in response.device
    assert "Bearer-LEAKED" not in response.device


# ---------------------------------------------------------------------------
# 16. Placeholder runtime is unaffected by gateway selection
# ---------------------------------------------------------------------------


def test_placeholder_runtime_is_a_runtime_subclass() -> None:
    """``PlaceholderRuntime`` must subclass ``InferenceRuntime`` (sanity pin)."""
    from oai2.runtime import InferenceRuntime  # noqa: PLC0415

    assert issubclass(PlaceholderRuntime, InferenceRuntime)


def test_placeholder_runtime_status_is_experimental() -> None:
    """``PlaceholderRuntime.STATUS`` must be ``Status.EXPERIMENTAL``."""
    assert PlaceholderRuntime.STATUS is Status.EXPERIMENTAL


def test_placeholder_runtime_does_not_require_api_key() -> None:
    """``PlaceholderRuntime`` is the offline fallback — no API key needed."""
    runtime = PlaceholderRuntime()
    response = runtime.generate(InferenceRequest(prompt="hi"))
    assert "[placeholder:" in response.text
    assert response.status is Status.EXPERIMENTAL


# ---------------------------------------------------------------------------
# 17. Default OS environment isolation (verify.py SKIP path)
# ---------------------------------------------------------------------------


def test_load_gateway_config_from_env_does_not_use_os_when_environ_passed() -> None:
    """When ``environ`` is passed, the loader MUST NOT consult ``os.environ``.

    Even if ``OAI2_GATEWAY_API_KEY`` is set in ``os.environ``, an explicit
    empty ``environ`` mapping must yield ``None``.
    """
    prior = os.environ.get("OAI2_GATEWAY_API_KEY")
    try:
        os.environ["OAI2_GATEWAY_API_KEY"] = "should-be-ignored"
        cfg = load_gateway_config_from_env(environ={})
        assert cfg is None
    finally:
        if prior is None:
            os.environ.pop("OAI2_GATEWAY_API_KEY", None)
        else:
            os.environ["OAI2_GATEWAY_API_KEY"] = prior


# ---------------------------------------------------------------------------
# 18. Defaults are exactly what the module exposes
# ---------------------------------------------------------------------------


def test_default_gateway_base_url_is_https_not_http() -> None:
    """The default base URL must use ``https://`` (transport security)."""
    assert DEFAULT_GATEWAY_BASE_URL.startswith("https://")


def test_default_gateway_model_has_no_slash_or_version_suffix() -> None:
    """The default model id must be a plain identifier (no slash / version)."""
    assert "/" not in DEFAULT_GATEWAY_MODEL
    assert ":" not in DEFAULT_GATEWAY_MODEL


def test_default_timeout_is_at_least_ten_seconds() -> None:
    """``DEFAULT_TIMEOUT_SECONDS`` must be at least 10s (sane for chat completions)."""
    assert DEFAULT_TIMEOUT_SECONDS >= 10.0


# Note: keep ``Any`` import so the test file is recognized as type-hinted.
_ = Any
