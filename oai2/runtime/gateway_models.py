"""Cloud-model discovery + working-model fallback for the public gateway.

The OAI-2.0 :class:`GatewayRuntime` is configured with one model id
(``OAI2_GATEWAY_MODEL``, default ``"oai-1.2"``). When that exact model
is unavailable on the gateway (404), temporarily down (503), or
unreachable from the local host (502), the runtime needs a way to
discover what is exposed and pick the first model that answers.

This module owns that contract — independent of any specific runtime
or scheduler — so multiple call sites (the runtime's
``select_runtime`` factory, :mod:`scripts.gateway_smoke`, the
:mod:`scripts.gateway_probe_models` CLI, and the ``gateway-reach``
sub-check in :mod:`scripts.verify`) can share the same source of
truth for "what does the cloud offer right now".

Design points
-------------

- **Allowlist, not enum.** ``KNOWN_CLOUD_MODELS`` is the explicit set
  of model ids the ORCHORDS cloud has historically exposed. Callers
  may pass any string, but resolution against unknown ids fails fast
  with :class:`UnknownModelError`.
- **No live calls in tests.** Every public helper accepts an
  :class:`httpx.Client` so unit tests can drive
  :class:`httpx.MockTransport`. The :func:`discover_cloud_models`
  helper, :func:`probe_model`, and :func:`resolve_working_model` are
  all runner-free.
- **Public-safe errors.** Token-like substrings in upstream messages
  are redacted before being stored on :class:`ModelProbe` so the
  result is safe to log.
- **Deterministic.** :func:`resolve_working_model` walks candidates
  in input order; the first reachable model wins. The full
  per-model :class:`ModelProbe` list is preserved on the result so
  callers can diagnose failures.
- **Status semantics.** ``ModelProbe.reachable`` is True when the
  upstream responded with HTTP 2xx and at least one non-empty
  completion choice. 4xx, 5xx, transport errors, and empty
  responses are all non-reachable.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


#: The set of model ids the public ORCHORDS cloud has historically
#: exposed via ``GET /v1/models``. New ids must be added here before
#: any runtime treats them as expected (this is the contract #237
#: tracks). Order is not significant — :func:`resolve_working_model`
#: walks whatever the caller passes in input order.
KNOWN_CLOUD_MODELS: frozenset[str] = frozenset(
    {
        "oai-1.0",
        "oai-1.2",
        "orchordsai-gpt",
        "orchordsai-m3",
    },
)


class CloudModelError(RuntimeError):
    """Base class for cloud-model discovery / probe failures."""


class UnknownModelError(CloudModelError):
    """Raised when a caller passes a model id outside :data:`KNOWN_CLOUD_MODELS`."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"unknown cloud model id: {model_id!r}")
        self.model_id = model_id


class CloudDiscoveryError(CloudModelError):
    """Raised when ``GET /v1/models`` returns an unusable payload."""

    def __init__(self, base_url: str, message: str) -> None:
        super().__init__(f"cloud model discovery failed for {base_url}: {message}")
        self.base_url = base_url
        self.message = message


@dataclass(slots=True, frozen=True)
class ModelProbe:
    """One probe attempt of a single cloud model id.

    ``latency_ms`` is ``None`` when the request never produced a usable
    response (transport error, malformed payload, no choices). ``error``
    is a short redacted upstream message or ``None`` on success.
    """

    model_id: str
    status_code: int
    reachable: bool
    latency_ms: float | None
    error: str | None

    def __repr__(self) -> str:
        return (
            f"ModelProbe(model_id={self.model_id!r}, "
            f"status_code={self.status_code}, "
            f"reachable={self.reachable}, "
            f"latency_ms={self.latency_ms}, "
            f"error={self.error!r})"
        )


@dataclass(slots=True, frozen=True)
class WorkingModelResolution:
    """Outcome of :func:`resolve_working_model`.

    ``selected_model_id`` is ``None`` when no candidate answered
    successfully; ``probes`` carries the per-model result for every
    candidate so callers can print a status table.
    """

    selected_model_id: str | None
    probes: tuple[ModelProbe, ...]


def _redact(value: str) -> str:
    """Redact token-like substrings from arbitrary upstream text."""

    if not value:
        return value
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}***{value[-3:]}"


def _ensure_known(model_id: str) -> str:
    """Return ``model_id`` when it is in :data:`KNOWN_CLOUD_MODELS`."""

    if model_id not in KNOWN_CLOUD_MODELS:
        raise UnknownModelError(model_id)
    return model_id


def discover_cloud_models(
    client: httpx.Client,
    *,
    path: str = "/v1/models",
) -> list[str]:
    """Return the model ids advertised by the gateway.

    A 2xx response with a ``data`` array of ``{"id": str}`` objects
    is the only accepted shape. Any other status or payload raises
    :class:`CloudDiscoveryError` so callers don't silently treat a
    misconfigured cloud as empty.
    """

    try:
        response = client.get(path)
    except httpx.HTTPError as exc:
        raise CloudDiscoveryError(
            str(client.base_url),
            f"transport error: {type(exc).__name__}: {_redact(str(exc))}",
        ) from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise CloudDiscoveryError(
            str(client.base_url),
            f"HTTP {response.status_code}: {_redact(response.text or '')}",
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise CloudDiscoveryError(
            str(client.base_url),
            f"invalid JSON: {_redact(str(exc))}",
        ) from exc

    if not isinstance(payload, dict) or "data" not in payload:
        raise CloudDiscoveryError(
            str(client.base_url),
            "payload missing 'data' array",
        )
    data = payload["data"]
    if not isinstance(data, list):
        raise CloudDiscoveryError(
            str(client.base_url),
            "'data' is not an array",
        )
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            raw_id = item.get("id")
            if isinstance(raw_id, str) and raw_id:
                ids.append(raw_id)
    return ids


def probe_model(
    client: httpx.Client,
    model_id: str,
    *,
    prompt: str = "ping",
    max_tokens: int = 8,
    timeout: float | None = None,
) -> ModelProbe:
    """Send a minimal chat-completions probe and return the result.

    The probe is intentionally cheap (one user turn, max_tokens=8,
    temperature=0) so it can be used as a liveness check without
    spending meaningful tokens. Errors are caught and reflected on
    the returned :class:`ModelProbe` — this function never raises.
    """

    _ensure_known(model_id)
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    kwargs: dict[str, Any] = {"json": body}
    if timeout is not None:
        kwargs["timeout"] = timeout
    start = time.perf_counter()
    try:
        response = client.post("/v1/chat/completions", **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ModelProbe(
            model_id=model_id,
            status_code=0,
            reachable=False,
            latency_ms=elapsed_ms,
            error=_redact(f"{type(exc).__name__}: {exc}"),
        )

    if response.status_code < 200 or response.status_code >= 300:
        return ModelProbe(
            model_id=model_id,
            status_code=response.status_code,
            reachable=False,
            latency_ms=elapsed_ms,
            error=_extract_error_message(response),
        )

    try:
        payload = response.json()
    except ValueError:
        return ModelProbe(
            model_id=model_id,
            status_code=response.status_code,
            reachable=False,
            latency_ms=elapsed_ms,
            error="invalid JSON completion response",
        )

    if not _has_non_empty_choice(payload):
        return ModelProbe(
            model_id=model_id,
            status_code=response.status_code,
            reachable=False,
            latency_ms=elapsed_ms,
            error="completion has no non-empty choice",
        )
    return ModelProbe(
        model_id=model_id,
        status_code=response.status_code,
        reachable=True,
        latency_ms=elapsed_ms,
        error=None,
    )


def resolve_working_model(
    client: httpx.Client,
    candidates: Sequence[str] | Iterable[str],
    *,
    prompt: str = "ping",
    max_tokens: int = 8,
    timeout: float | None = None,
) -> WorkingModelResolution:
    """Probe ``candidates`` in order and return the first reachable id.

    Every candidate is probed once and its :class:`ModelProbe` is
    appended to ``probes`` regardless of outcome, so callers can
    print the full diagnostic trail. Iteration short-circuits on
    the first reachable hit; unknown ids are recorded as failed
    probes (never raised) so callers see exactly why they were
    rejected. When ``candidates`` is empty, the result is
    ``WorkingModelResolution(selected_model_id=None, probes=())``.
    """

    probes: list[ModelProbe] = []
    for raw in candidates:
        if not isinstance(raw, str) or not raw:
            continue
        if any(probe.model_id == raw for probe in probes):
            continue
        if raw not in KNOWN_CLOUD_MODELS:
            probes.append(
                ModelProbe(
                    model_id=raw,
                    status_code=0,
                    reachable=False,
                    latency_ms=None,
                    error=f"unknown cloud model id: {raw!r}",
                ),
            )
            return WorkingModelResolution(
                selected_model_id=None,
                probes=tuple(probes),
            )
        probe = probe_model(
            client,
            raw,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        probes.append(probe)
        if probe.reachable:
            return WorkingModelResolution(
                selected_model_id=raw,
                probes=tuple(probes),
            )

    return WorkingModelResolution(
        selected_model_id=None,
        probes=tuple(probes),
    )


def _extract_error_message(response: httpx.Response) -> str:
    """Pull a redacted ``error.message`` from a chat-completions error body."""

    text = response.text or ""
    if len(text) > 240:
        text = text[:240] + "…"
    try:
        parsed = response.json()
    except ValueError:
        return _redact(text)
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return _redact(message)
    return _redact(text)


def _has_non_empty_choice(payload: Any) -> bool:
    """Return True iff ``payload`` has at least one non-empty choice."""

    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return True
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            return True
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return True
    return False


__all__ = [
    "KNOWN_CLOUD_MODELS",
    "CloudModelError",
    "UnknownModelError",
    "CloudDiscoveryError",
    "ModelProbe",
    "WorkingModelResolution",
    "discover_cloud_models",
    "probe_model",
    "resolve_working_model",
]