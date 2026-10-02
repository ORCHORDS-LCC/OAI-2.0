"""Gateway runtime — talks to the public ORCHORDS inference gateway.

:class:`GatewayRuntime` is an :class:`InferenceRuntime` that POSTs an
OpenAI-compatible ``/v1/chat/completions`` request to a remote gateway
(defaults to ``https://api.orchords.com``) and returns the parsed
completion as an :class:`InferenceResponse`.

Design points:

- The transport (``httpx.Client``) is injected so tests can use
  ``httpx.MockTransport`` for deterministic, runner-free coverage. No
  live call is ever required to exercise the runtime.
- The base URL, bearer token, and model id all come from configuration
  helpers in :mod:`oai2.runtime.gateway_config`. None of the values are
  hard-coded into source — credentials live in a gitignored ``.env``.
- Streaming is intentionally NOT supported in this revision. The
  reference runtime measures wall-clock latency for one completion
  at a time; speculative decoding, batching, and tool-calling live
  in higher layers.
- On non-2xx responses the runtime raises
  :class:`GatewayRuntimeError` with the upstream status and a
  redacted message body. Caller decides whether to retry or fall back
  to :class:`PlaceholderRuntime`.

Status: PROPOSED. The runtime imports successfully and the request /
response mapping is exercised by ``tests/test_gateway_runtime.py``
under ``httpx.MockTransport``. Live end-to-end verification requires
``OAI2_GATEWAY_API_KEY`` to be set in a local gitignored ``.env``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ..core import Status
from .inference import InferenceRequest, InferenceResponse, InferenceRuntime
from .model import ModelSpec

DEFAULT_GATEWAY_BASE_URL = "https://api.orchords.com"
DEFAULT_GATEWAY_MODEL = "oai-2.0"
DEFAULT_TIMEOUT_SECONDS = 60.0


class GatewayConfigError(RuntimeError):
    """Raised when the gateway client is constructed without required config."""


class GatewayRuntimeError(RuntimeError):
    """Raised when the upstream gateway returns a non-2xx response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"gateway returned {status_code}: {message}")
        self.status_code = status_code
        self.message = message


@dataclass(slots=True, frozen=True)
class GatewayConfig:
    """Static configuration for :class:`GatewayRuntime`.

    Built from environment variables via :func:`load_gateway_config_from_env`.
    Never includes the API key in its ``repr`` to keep accidental logs
    safe.
    """

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __repr__(self) -> str:
        return (
            f"GatewayConfig(base_url={self.base_url!r}, "
            f"model={self.model!r}, "
            f"api_key=<redacted len={len(self.api_key)}>, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )


def load_gateway_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> GatewayConfig | None:
    """Return a :class:`GatewayConfig` if ``OAI2_GATEWAY_API_KEY`` is set.

    A missing key returns ``None`` so callers can fall back to
    :class:`PlaceholderRuntime` without raising. Any other required
    value (base URL, model) falls back to its documented default.
    """

    env = os.environ if environ is None else environ
    api_key = env.get("OAI2_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = env.get("OAI2_GATEWAY_BASE_URL", DEFAULT_GATEWAY_BASE_URL).strip()
    if not base_url:
        base_url = DEFAULT_GATEWAY_BASE_URL
    base_url = base_url.rstrip("/")
    model = env.get("OAI2_GATEWAY_MODEL", DEFAULT_GATEWAY_MODEL).strip()
    if not model:
        model = DEFAULT_GATEWAY_MODEL
    raw_timeout = env.get("OAI2_GATEWAY_TIMEOUT_SECONDS", "").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
    else:
        timeout = DEFAULT_TIMEOUT_SECONDS
    return GatewayConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout,
    )


def _redact(value: str) -> str:
    """Return ``value`` with any token-like substring replaced."""

    if not value:
        return value
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}***{value[-3:]}"


class GatewayRuntime(InferenceRuntime):
    """Inference runtime backed by an OpenAI-compatible HTTP gateway.

    The HTTP client is injected via the ``client`` parameter so tests
    can pass an ``httpx.Client`` wrapping ``httpx.MockTransport``. In
    production, ``GatewayRuntime.from_env()`` builds a real client
    from the configured base URL.
    """

    STATUS = Status.PROPOSED

    def __init__(
        self,
        config: GatewayConfig,
        *,
        client: httpx.Client | None = None,
        spec: ModelSpec | None = None,
    ) -> None:
        if not config.api_key:
            raise GatewayConfigError(
                "GatewayConfig.api_key must be non-empty",
            )
        super().__init__(spec or ModelSpec(name=config.model))
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "oai2-gateway-runtime/0.1",
            },
        )

    @classmethod
    def from_env(
        cls,
        *,
        client: httpx.Client | None = None,
    ) -> GatewayRuntime:
        """Build a runtime from ``OAI2_GATEWAY_*`` env vars.

        Raises :class:`GatewayConfigError` when the API key is missing.
        """

        config = load_gateway_config_from_env()
        if config is None:
            raise GatewayConfigError(
                "OAI2_GATEWAY_API_KEY is not set; cannot build GatewayRuntime",
            )
        return cls(config, client=client)

    @property
    def config(self) -> GatewayConfig:
        return self._config

    def close(self) -> None:
        """Close the underlying HTTP client if this runtime owns it."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GatewayRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        body = self._build_request_body(request)
        start = time.perf_counter()
        try:
            response = self._client.post(
                "/v1/chat/completions",
                json=body,
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            raise GatewayRuntimeError(
                status_code=0,
                message=f"transport error after {elapsed_ms:.1f}ms: "
                f"{type(exc).__name__}: {_redact(str(exc))}",
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if response.status_code < 200 or response.status_code >= 300:
            raise GatewayRuntimeError(
                status_code=response.status_code,
                message=_safe_response_message(response),
            )
        try:
            payload = response.json()
        except ValueError:
            raise GatewayRuntimeError(
                status_code=response.status_code,
                message="invalid JSON completion response",
            ) from None
        text, tokens, finish_reason, tool_calls = _extract_completion(payload)
        return InferenceResponse(
            text=text,
            tokens=tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            elapsed_ms=elapsed_ms,
            device=f"gateway:{self._config.base_url}",
            status=Status.EXPERIMENTAL,
            notes=[
                f"model={self._config.model}",
                f"status={response.status_code}",
                f"elapsed_ms={elapsed_ms:.1f}",
            ],
        )

    def _build_request_body(self, request: InferenceRequest) -> dict[str, Any]:
        spec = request.model or self.spec
        model_id = spec.name or self._config.model
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": request.prompt},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            body["stop"] = list(request.stop)
        if request.seed is not None:
            body["seed"] = request.seed
        return body


def _safe_response_message(response: httpx.Response) -> str:
    """Return a small redacted representation of an error body."""

    text = response.text or ""
    if len(text) > 240:
        text = text[:240] + "…"
    try:
        parsed = response.json()
        if isinstance(parsed, dict) and "error" in parsed:
            error = parsed["error"]
            if isinstance(error, dict) and "message" in error:
                return _redact(str(error["message"]))
    except (json.JSONDecodeError, ValueError):
        pass
    return _redact(text)


def _extract_completion(
    payload: Any,
) -> tuple[str, int, str | None, tuple[dict[str, Any], ...]]:
    """Preserve upstream completion state; never turn missing metadata into success."""

    if not isinstance(payload, dict):
        raise GatewayRuntimeError(200, "completion response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GatewayRuntimeError(
            status_code=200,
            message="missing or empty 'choices' array in response",
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise GatewayRuntimeError(
            status_code=200,
            message=f"unexpected choice shape: {type(first).__name__}",
        )
    finish_reason = first.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise GatewayRuntimeError(200, "finish_reason must be a string or null")
    message = first.get("message")
    tool_calls: tuple[dict[str, Any], ...] = ()
    content = ""
    if isinstance(message, dict):
        raw_calls = message.get("tool_calls")
        if raw_calls is not None:
            if not isinstance(raw_calls, list) or any(
                not isinstance(call, dict) for call in raw_calls
            ):
                raise GatewayRuntimeError(200, "tool_calls must be an array of objects or null")
            tool_calls = tuple(raw_calls)
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            content = "".join(
                str(part.get("text", "")) for part in raw_content if isinstance(part, dict)
            )
    elif isinstance(first.get("text"), str):
        content = first["text"]
    else:
        raise GatewayRuntimeError(200, "choice must contain a message object or text")
    if finish_reason == "tool_calls" and not tool_calls:
        raise GatewayRuntimeError(200, "tool_calls finish_reason requires tool calls")
    usage = payload.get("usage") or {}
    completion_tokens = 0
    if isinstance(usage, dict):
        raw = usage.get("completion_tokens")
        if isinstance(raw, int) and raw >= 0:
            completion_tokens = raw
    if completion_tokens == 0:
        completion_tokens = max(1, len(content.split())) if content else 0
    return content, completion_tokens, finish_reason, tool_calls


__all__ = [
    "DEFAULT_GATEWAY_BASE_URL",
    "DEFAULT_GATEWAY_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "GatewayConfig",
    "GatewayConfigError",
    "GatewayRuntime",
    "GatewayRuntimeError",
    "load_gateway_config_from_env",
]
