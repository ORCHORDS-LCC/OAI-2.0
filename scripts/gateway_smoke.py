#!/usr/bin/env python3
"""OAI-2.0 gateway smoke probe — drives one chat completion through
api.orchords.com (or a configured alternative) and prints the result.

This is a tiny CLI on top of :class:`oai2.runtime.GatewayRuntime`. It
exists so an operator with ``OAI2_GATEWAY_API_KEY`` in a gitignored
``.env`` can prove the OAI-2.0 → public-gateway wire works end-to-end
without touching any other subsystem.

Usage::

    uv run python scripts/gateway_smoke.py
    uv run python scripts/gateway_smoke.py --prompt "ping" --max-tokens 16
    uv run python scripts/gateway_smoke.py --prompt "ping" --json

The bearer token is read from ``OAI2_GATEWAY_API_KEY`` and never echoed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oai2.runtime import (  # noqa: E402
    GatewayConfigError,
    GatewayRuntime,
    GatewayRuntimeError,
    InferenceRequest,
    load_gateway_config_from_env,
)

DEFAULT_PROMPT = "Reply with the single word: pong"
DEFAULT_MAX_TOKENS = 16


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one chat completion through the configured ORCHORDS gateway.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to send (default: %(default)r).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens to request (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: %(default)s — deterministic).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id to request. Defaults to OAI2_GATEWAY_MODEL "
            "(or oai-1.2)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object instead of a human-readable report.",
    )
    return parser


def _render_human(result: dict[str, object]) -> str:
    lines = ["gateway smoke:", f"  model:        {result['model']}", ]
    base_url = result.get("base_url", "")
    if base_url:
        lines.append(f"  base_url:     {base_url}")
    lines.extend(
        [
            f"  status_code:  {result['status_code']}",
            f"  elapsed_ms:   {result['elapsed_ms']}",
            f"  tokens:       {result['tokens']}",
            f"  text:         {result['text']}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_gateway_config_from_env()
    except Exception as exc:
        print(f"FAIL: env config error: {type(exc).__name__}: {exc}")
        return 2
    if config is None:
        print(
            "FAIL: OAI2_GATEWAY_API_KEY is not set; "
            "populate a gitignored .env before running the smoke.",
        )
        return 2
    if args.model:
        # Build a per-invocation view of the config that overrides model id.
        config = replace(config, model=args.model)

    request = InferenceRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    try:
        with GatewayRuntime(config) as runtime_ctx:
            response = runtime_ctx.generate(request)
    except GatewayConfigError as exc:
        print(f"FAIL: {exc}")
        return 2
    except GatewayRuntimeError as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "status_code": exc.status_code,
                    }
                )
            )
        else:
            print(f"FAIL: {exc}")
        return 1

    payload: dict[str, object] = {
        "ok": True,
        "model": config.model,
        "base_url": config.base_url,
        "status_code": 200,
        "elapsed_ms": round(response.elapsed_ms, 3),
        "tokens": response.tokens,
        "text": response.text,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(_render_human(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
