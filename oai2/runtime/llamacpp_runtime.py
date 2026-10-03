"""llama.cpp server runtime — drives the *real* production serving path.

:class:`LlamaServerRuntime` is an :class:`InferenceRuntime` that POSTs an
OpenAI-compatible ``/v1/chat/completions`` request to a running
``llama-server`` (the NORMAL lane on ``127.0.0.1:8851``) and returns the
parsed completion as an :class:`InferenceResponse`.

Why this exists
---------------

:mod:`oai2.evals` could already score a suite, but the only runtimes it
could reach were :class:`PlaceholderRuntime` (fabricated output) and
:class:`GatewayRuntime` (a remote API). Neither is the path production
traffic takes. The consequence was concrete: an accuracy figure of
``0/12`` was asserted against ``evidence/latency-20261003/summary.json``,
a file that has never existed in this repository, so the number could not
be reproduced or refuted by anyone. This runtime is the missing link that
lets an existing suite be measured against the actual deployed server.

Design points
-------------

- The transport (``httpx.Client``) is injected so tests use
  ``httpx.MockTransport``; no live call is required for coverage.
- Serving identity is **read from ``/props``**, never inferred from a CLI
  flag. A runtime that guesses which weights it is talking to cannot
  support a correctness claim, because the claim would be about an
  unidentified model.
- ``temperature`` defaults to ``0.0`` and ``seed`` to a fixed value.
  An accuracy oracle requires a deterministic serving path; a runtime
  that sampled at ``temperature=0.7`` would produce a score that changed
  on every run and could not be compared across SHAs.
- This runtime implements ``generate`` only. Tool-calling, speculative
  decoding and batching belong to higher layers, as in
  :mod:`oai2.runtime.gateway_runtime`.

Status: IMPLEMENTED — exercised offline under ``httpx.MockTransport`` by
``tests/test_llamacpp_runtime.py`` and live against the production
NORMAL lane by ``scripts/accuracy_baseline.py``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from ..core import Status
from .inference import InferenceRequest, InferenceResponse, InferenceRuntime
from .model import ModelSpec

DEFAULT_BASE_URL = "http://127.0.0.1:8851"
DEFAULT_MODEL = "smollm2-1.7b-q4km"
DEFAULT_TIMEOUT_SECONDS = 120.0

#: Fixed seed. A reproducibility suite must not inherit per-run sampling.
DEFAULT_SEED = 20261004


class LlamaServerError(RuntimeError):
    """Raised when the llama-server returns a non-2xx response."""


class LlamaServerRuntime(InferenceRuntime):
    """An :class:`InferenceRuntime` backed by a running ``llama-server``."""

    STATUS: Status = Status.IMPLEMENTED

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        spec: ModelSpec | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("OAI2_LLAMACPP_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("OAI2_LLAMACPP_MODEL") or DEFAULT_MODEL
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds)
        # Identity is discovered, never asserted; see `serving_identity`.
        self._identity: dict[str, Any] | None = None
        super().__init__(spec or ModelSpec(name=self.model))

    # -- identity ---------------------------------------------------------

    def serving_identity(self, *, refresh: bool = False) -> dict[str, Any]:
        """Read what the server is actually serving, from ``/props``.

        A correctness number is a claim about a *specific* model artifact.
        Deriving that artifact from a CLI flag records what somebody
        intended to load, not what is resident, so the two can silently
        disagree. Failures propagate as :class:`LlamaServerError` rather
        than returning an empty dict, so an unproven identity can never be
        mistaken for a proven one.
        """
        if self._identity is not None and not refresh:
            return self._identity
        try:
            response = self.client.get(f"{self.base_url}/props")
        except httpx.HTTPError as exc:
            raise LlamaServerError(f"could not reach {self.base_url}/props: {exc}") from exc
        if response.status_code != 200:
            raise LlamaServerError(f"{self.base_url}/props returned {response.status_code}")
        props = response.json()
        gen = props.get("default_generation_settings") or {}
        identity = {
            "base_url": self.base_url,
            "model_alias": self.model,
            "model_path": props.get("model_path"),
            "model_basename": (
                os.path.basename(props["model_path"]) if props.get("model_path") else None
            ),
            "total_slots": props.get("total_slots"),
            "n_ctx": gen.get("n_ctx"),
        }
        self._identity = identity
        return identity

    # -- inference --------------------------------------------------------

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        messages: list[dict[str, Any]] = list(request.messages) or [
            {"role": "user", "content": request.prompt}
        ]
        payload: dict[str, Any] = {
            "model": (request.model.name if request.model else self.model),
            "messages": messages,
            "max_tokens": request.max_tokens,
            # Determinism is the point: an accuracy oracle that samples
            # differently on every run cannot be compared across SHAs.
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)

        started = time.perf_counter()
        try:
            response = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LlamaServerError(f"completion request failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise LlamaServerError(
                f"{self.base_url} returned {response.status_code}: {response.text[:200]}"
            )
        body = response.json()
        text = self._extract_text(body)
        return InferenceResponse(
            text=text,
            tokens=int((body.get("usage") or {}).get("completion_tokens") or 0),
            elapsed_ms=elapsed_ms,
            device=f"llamacpp:{self.base_url}",
        )

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        """Pull the assistant text out of an OpenAI-shaped completion.

        ``reasoning`` models interleave a ``reasoning_content`` channel;
        only ``content`` is the answer, so a reasoning trace is never
        scored as if it were the model's conclusion.
        """
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        return content if isinstance(content, str) else json.dumps(content)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_SEED",
    "LlamaServerError",
    "LlamaServerRuntime",
]
