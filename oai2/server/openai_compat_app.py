"""OpenAI-compatible FastAPI surface for a locally-resident MLX hot runtime.

This app exposes the same wire shape that ZCode and other OpenAI
chat-completions clients already speak, but is backed by the
:class:`oai2.runtime.MLXHotRuntime` instead of the upstream
`api.orchords.com` gateway. The point is to give the host full
control of the system prompt and tool scaffold — the upstream
gateway hard-injects its own brand identity into every request, so
the model always answers "I'm OrchordsAI" regardless of what the
host says. This local endpoint passes the host's messages through
unmodified.

This module is the path ZCode actually uses: its ``local-oai2``
provider is configured against ``http://127.0.0.1:9100/v1`` and the
default ``--port`` below is 9100, so a change here is a change to the
serving path, not to a side door.

Status: EXPERIMENTAL (Refs #240, #186, sess_10cbe33c-d83b-42ce-bf2c).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from oai2.agents.composer import (
    PrefixSpec,
    TemplateError,
    assert_template_safe,
)
from oai2.runtime import (
    InferenceRequest,
    InferenceResponse,
    MLXHotRuntime,
    ModelSpec,
    PrefixKVCache,
    TemplateRenderError,
    validate_tool_calls,
)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = Field(default=256, ge=1, le=32_768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None


def _usage(response: InferenceResponse) -> dict[str, Any]:
    """Build OpenAI usage from what the runtime actually measured.

    ``prompt_tokens`` is the serving tokenizer's own count of the rendered
    request. It is ``None`` only when the runtime genuinely did not measure
    it — never a fabricated ``0``, which reads as "free prefill".
    ``total_tokens`` adds the real input count when there is one instead of
    silently equating total with completion.
    """
    prompt = response.prompt_tokens
    completion = response.tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": None if prompt is None else prompt + completion,
    }


def _prefix_report(
    runtime: MLXHotRuntime, response: InferenceResponse
) -> dict[str, Any] | None:
    """Prefix KV reuse evidence, or ``None`` with the reason it is absent."""
    cache = runtime.prefix_cache
    if cache is None:
        return None
    metrics = cache.metrics
    return {
        "enabled": True,
        "lookups": metrics.lookups,
        "hits": metrics.hits,
        "misses": metrics.misses,
        "restores": metrics.restores,
        "invalidations": metrics.invalidations,
        "stored_entries": metrics.stored_entries,
        "gated_on_digest": runtime.gate_digest,
        # Tokens whose prefill this request avoided. Real avoided work, not
        # a shorter payload: the same prompt still arrives in full.
        "prompt_tokens": response.prompt_tokens,
        "reused_prefix_tokens": response.prefix_matched,
        "avoided_prefill_tokens": response.prefix_matched,
    }


def _completion_payload(
    body: ChatCompletionRequest,
    response: InferenceResponse,
    prefix: PrefixSpec,
    runtime: MLXHotRuntime,
    *,
    declared_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the non-streaming chat-completion body.

    A call the client never advertised, a call whose contract does not hold, or
    a pinned function the model did not honour is **not** returned in
    ``message.tool_calls``. Clients execute that array, so putting a violating
    call there turns a contract failure into ordinary executable success. Such
    calls are reported under ``rejected_tool_calls`` — visible, auditable, and
    not runnable — and ``finish_reason`` says so.
    """
    problems = validate_tool_calls(
        response.tool_calls,
        declared=declared_tools,
        tool_choice=body.tool_choice,
    )
    message: dict[str, Any] = {"role": "assistant", "content": response.text or ""}
    finish_reason = response.finish_reason or "stop"
    rejected: list[dict[str, Any]] = []
    if problems:
        rejected = list(response.tool_calls)
        finish_reason = "tool_contract_violation"
    elif response.tool_calls:
        message["tool_calls"] = list(response.tool_calls)
    payload: dict[str, Any] = {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(response),
        "prefix_digest": prefix.digest(),
        "prefix_cache": _prefix_report(runtime, response),
        "tool_contract_problems": list(problems),
        "notes": response.notes,
        "device": response.device,
    }
    if rejected:
        payload["rejected_tool_calls"] = rejected
    return payload


def create_app(
    *,
    model_id: str,
    with_tools: bool = True,
    prefix_cache: PrefixKVCache | None = None,
    enable_prefix_cache: bool = False,
) -> FastAPI:
    spec = ModelSpec(name=model_id)
    # Without this the runtime's cache is None and the digest computed below
    # has nowhere to go — the wiring gap that made #240's reuse unreachable
    # on the only path a client actually uses.
    #
    # Reuse is OFF by default. It is wired, gated and measurable, but it is
    # not yet sound against installed mlx_lm 0.32: restoring KV state means
    # driving mlx_lm's prompt cache outside its own prefill path, and
    # ``KVCache.trim`` decrements the offset without shrinking the key/value
    # arrays, so the next model call builds its mask from a stale sequence
    # length. Measured on this endpoint, a request following an unrelated one
    # returned a continuation of the PREVIOUS prompt. Correctness outranks a
    # reuse number; see tests/test_runtime_prefix_cache_correctness.py.
    cache = (
        prefix_cache
        if prefix_cache is not None
        else (PrefixKVCache(max_entries=8) if enable_prefix_cache else None)
    )
    runtime = MLXHotRuntime(
        spec,
        model_id=model_id,
        prefix_cache=cache,
        # Serving requests carry tool schema and world state in the digest;
        # reuse across a schema change must miss.
        gate_digest=True,
    )
    # Eager load so the first request doesn't pay a 0.6s load cost.
    runtime.load()

    app = FastAPI(title="oai2-local-openai-compat", version="0.2.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": model_id,
            "load_seconds": runtime.load_seconds,
            "prefix_cache_enabled": runtime.prefix_cache is not None,
            "prefix_cache_gated_on_digest": runtime.gate_digest,
            "tools_enabled": with_tools,
        }

    @app.post("/v1/models")
    def models() -> dict[str, Any]:
        """Minimal model listing so an OpenAI-compatible client can start."""
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "oai2-local"}],
        }

    def _prepare(body: ChatCompletionRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PrefixSpec]:
        messages = [dict(m) for m in body.messages]
        tools = body.tools if with_tools else []

        # This endpoint is a transport. The caller owns its system prompt,
        # conversation history, tool schema and structured tool calls, and
        # nothing here rewrites them or injects retrieved content — a second
        # injection would duplicate whatever the host already sends (#186).
        #
        # What the host does NOT supply is prefix *identity*. Without it the
        # prefix KV cache has no key on this path, so #240's reuse never
        # engages. We therefore observe the caller's leading system block and
        # its advertised tool schema and derive a stable digest, rather than
        # imposing a prompt of our own.
        try:
            assert_template_safe(messages)
        except TemplateError as exc:
            # A system message after the first non-system turn is rejected by
            # supported chat templates (llama.cpp qwen3.8 raises outright).
            # Failing here returns an actionable 400 instead of a template
            # traceback deep inside the runtime.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return messages, tools, PrefixSpec.adopt_caller_prefix(messages, tools)

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest) -> Any:
        messages, tools, prefix = _prepare(body)

        # ``prompt`` must be non-empty; the gateway falls back to using
        # ``messages`` so we pass a placeholder.
        inference_req = InferenceRequest(
            prompt="(see messages)",
            messages=messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            tools=tools,
            tool_choice=body.tool_choice,
            # Makes #240's PrefixKVCache able to key on this path at all.
            prefix_digest=prefix.digest(),
        )
        try:
            response: InferenceResponse = runtime.generate(inference_req)
        except TemplateRenderError as exc:
            # Structured tool history the template cannot express. 400 is the
            # honest answer; flattening it would silently break the turn.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.stream:
            return StreamingResponse(
                _sse(body, response, prefix, runtime, tools),
                media_type="text/event-stream",
            )
        return _completion_payload(body, response, prefix, runtime, declared_tools=tools)

    return app


#: How this endpoint's `stream: true` actually behaves.
#:
#: BUFFERED, NOT INCREMENTAL. The runtime generates to completion internally and
#: exposes no token callback, so every SSE frame is written after generation has
#: already finished. The stream shape is OpenAI-compatible, but a client
#: receives nothing while the model is working, and client-observed first output
#: therefore equals total generation time. Nothing here should be read as
#: token-level streaming or as a TTFT improvement.
STREAM_MODE = "BUFFERED_SSE"


def _sse(
    body: ChatCompletionRequest,
    response: InferenceResponse,
    prefix: PrefixSpec,
    runtime: MLXHotRuntime,
    tools: list[dict[str, Any]],
) -> Iterator[str]:
    """Emit an OpenAI-shaped SSE stream for a completed generation.

    This is BUFFERED SSE (:data:`STREAM_MODE`), not incremental streaming: the
    runtime generates to completion first, so all three frames are emitted
    after generation has finished. A client sees no output while the model is
    working, and client-observed first output is the full generation time.

    Every frame carries ``stream_mode`` so a client can tell this apart from a
    genuinely incremental stream without having to time it. Internal timings
    remain available in the response ``notes``; they are not a substitute for
    client-observed first output and must not be reported as one.
    """

    def event(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    base = {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "model": body.model,
        "stream_mode": STREAM_MODE,
    }
    created = int(time.time())
    yield event(
        {
            **base,
            "created": created,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    if response.text:
        yield event(
            {
                **base,
                "created": created,
                "choices": [
                    {"index": 0, "delta": {"content": response.text}, "finish_reason": None}
                ],
            }
        )
    final_delta: dict[str, Any] = {}
    if response.tool_calls:
        final_delta["tool_calls"] = [
            {
                "index": i,
                "id": call["id"],
                "type": "function",
                "function": call["function"],
            }
            for i, call in enumerate(response.tool_calls)
        ]
    yield event(
        {
            **base,
            "created": created,
            "choices": [
                {
                    "index": 0,
                    "delta": final_delta,
                    "finish_reason": response.finish_reason or "stop",
                }
            ],
            "usage": _usage(response),
            "prefix_digest": prefix.digest(),
            "prefix_cache": _prefix_report(runtime, response),
            "tool_contract_problems": list(
                validate_tool_calls(response.tool_calls, declared=tools, tool_choice=body.tool_choice)
            ),
        }
    )
    yield "data: [DONE]\n\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Do not advertise tools on this endpoint (single-tenant debug).",
    )
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help=(
            "Enable prefix KV reuse (experimental; off by default because "
            "reuse is not yet sound against mlx_lm 0.32 — see "
            "tests/test_runtime_prefix_cache_correctness.py)."
        ),
    )
    args = parser.parse_args(argv)

    app = create_app(
        model_id=args.model,
        with_tools=not args.no_tools,
        enable_prefix_cache=args.prefix_cache,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
