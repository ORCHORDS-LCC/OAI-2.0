"""Adapter from :class:`GatewayRuntime` to q-pipe's ``ModelClient`` seam.

q-pipe's held-out eval runner (``HeldOutRunner``) talks to chat-completions
endpoints through a narrow protocol::

    client.chat(messages, *, max_tokens, temperature) -> ChatReply

where ``ChatReply`` only needs ``.content: str`` and
``.finish_reason: str | None``. This module ships a thin adapter that lets
the runner drive ``api.orchords.com`` through OAI-2.0's
:class:`GatewayRuntime` without either repo importing the other.

Design choices
--------------

- **Composition over inheritance.** ``GatewayModelClient`` owns a
  ``GatewayRuntime`` and delegates the HTTP call to it. The runtime stays
  general-purpose (``generate(request) -> InferenceResponse``); the
  adapter narrows the surface to ``chat(messages, ...) -> reply`` so it
  satisfies q-pipe's ``ModelClient`` Protocol structurally.

- **Duck-typed reply.** The runner only reads two attributes
  (``content`` and ``finish_reason``). Returning a local frozen dataclass
  with those fields means OAI-2.0 does not need to import q-pipe's
  ``ChatReply`` and stays a self-contained package.

- **Deterministic message flattening.** OpenAI-style ``messages`` are
  collapsed into a single ``InferenceRequest.prompt`` because
  ``GatewayRuntime`` posts a one-message chat-completions body. System
  / user / assistant roles are rendered as ``Role: <content>`` lines so
  the model still sees the structure.

This file is deliberately small and runner-free: every test uses
``httpx.MockTransport`` so nothing here requires a live endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gateway_runtime import GatewayRuntime
from .inference import InferenceRequest

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 256


@dataclass(frozen=True, slots=True)
class ChatReply:
    """Duck-typed reply shape compatible with q-pipe's ``ModelClient`` seam.

    q-pipe's runner reads ``reply.content`` and ``reply.finish_reason``.
    Anything else (``reasoning``, ``tool_calls``, ``raw``) is preserved on
    a side-car dict so future integrations have a place to put metadata
    without breaking the structural contract.
    """

    content: str
    finish_reason: str | None = "stop"
    reasoning: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] | None = None


class GatewayModelClient:
    """Adapter that exposes :class:`GatewayRuntime` as a ``ModelClient``.

    Example::

        runtime = GatewayRuntime.from_env()
        client = GatewayModelClient(runtime)
        reply = client.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=64,
            temperature=0.0,
        )
        assert reply.content.startswith("...")
        runtime.close()

    Use as a context manager to ensure the underlying ``httpx`` client
    is closed even when an exception escapes ``chat``::

        with GatewayRuntime.from_env() as runtime:
            client = GatewayModelClient(runtime)
            ...
    """

    def __init__(self, runtime: GatewayRuntime) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> GatewayRuntime:
        """Underlying runtime — exposed for diagnostics and tests."""

        return self._runtime

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatReply:
        """Return a chat reply for ``messages`` via the gateway runtime.

        Parameters mirror q-pipe's ``ModelClient.chat`` exactly so this
        object satisfies the Protocol structurally. ``max_tokens`` and
        ``temperature`` fall back to the runtime's InferenceRequest
        defaults when ``None`` is passed.
        """

        prompt = _flatten_messages(messages)
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=_coerce_max_tokens(max_tokens),
            temperature=_coerce_temperature(temperature),
        )
        response = self._runtime.generate(request)
        return ChatReply(
            content=response.text,
            finish_reason=response.finish_reason,
            reasoning="",
            tool_calls=response.tool_calls,
            raw={
                "elapsed_ms": response.elapsed_ms,
                "device": response.device,
                "status": response.status.name,
            },
        )

    def close(self) -> None:
        """Close the underlying httpx client. Safe to call multiple times."""

        self._runtime.close()

    def __enter__(self) -> GatewayModelClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Render OpenAI-style messages into a single prompt string.

    Each non-empty message becomes ``"<role>: <content>"`` on its own
    line. Empty messages are skipped silently. The result ends with a
    trailing newline so the gateway sees a complete user turn.
    """

    rendered: list[str] = []
    for message in messages:
        content = message.get("content")
        if not content:
            continue
        role = str(message.get("role", "user"))
        rendered.append(f"{role}: {content}")
    if not rendered:
        # Empty messages list — surface a deterministic marker rather
        # than letting InferenceRequest reject an empty prompt.
        return "user: \n"
    return "\n".join(rendered) + "\n"


def _coerce_max_tokens(value: int | None) -> int:
    return DEFAULT_MAX_TOKENS if value is None else int(value)


def _coerce_temperature(value: float | None) -> float:
    return DEFAULT_TEMPERATURE if value is None else float(value)


__all__ = [
    "ChatReply",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "GatewayModelClient",
]
