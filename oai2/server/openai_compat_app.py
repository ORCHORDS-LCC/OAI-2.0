"""OpenAI-compatible FastAPI surface for a locally-resident MLX hot runtime.

This app exposes the same wire shape that ZCode and other OpenAI
chat-completions clients already speak, but is backed by the
:class:`oai2.runtime.MLXHotRuntime` instead of the upstream
`api.orchords.com` gateway. The point is to give the host full
control of the system prompt and tool scaffold — the upstream
gateway hard-injects its own brand identity into every request, so
the model always answers "I'm OrchardsAI" regardless of what the
host says. This local endpoint passes the host's messages through
unmodified.

Status: EXPERIMENTAL (Refs #240, sess_10cbe33c-d83b-42ce-bf2c).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from oai2.runtime import (
    InferenceRequest,
    InferenceResponse,
    MLXHotRuntime,
    ModelSpec,
)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = Field(default=256, ge=1, le=32_768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None


def create_app(*, model_id: str, with_tools: bool = True) -> FastAPI:
    spec = ModelSpec(name=model_id)
    runtime = MLXHotRuntime(spec, model_id=model_id)
    # Eager load so the first request doesn't pay a 0.6s load cost.
    runtime.load()

    app = FastAPI(title="oai2-local-openai-compat", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": model_id,
            "load_seconds": runtime.load_seconds,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest) -> dict[str, Any]:
        # Translate OpenAI messages → oai2 InferenceRequest.messages.
        # ``prompt`` must be non-empty; the gateway falls back to using
        # ``messages`` so we pass a placeholder.
        inference_req = InferenceRequest(
            prompt="(see messages)",
            messages=[dict(m) for m in body.messages],
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            tools=body.tools if with_tools else [],
            tool_choice=body.tool_choice,
        )
        response: InferenceResponse = runtime.generate(inference_req)
        # Translate oai2 InferenceResponse → OpenAI chat-completion JSON.
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text or "",
                        "tool_calls": list(response.tool_calls) if response.tool_calls else None,
                    },
                    "finish_reason": response.finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": response.tokens,
                "total_tokens": response.tokens,
            },
            "notes": response.notes,
            "device": response.device,
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--no-tools", action="store_true",
                        help="Do not advertise tools on this endpoint (single-tenant debug).")
    args = parser.parse_args(argv)

    app = create_app(model_id=args.model, with_tools=not args.no_tools)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
