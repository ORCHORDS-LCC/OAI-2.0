#!/usr/bin/env python3
"""Launcher that wires the canonical :func:`create_batch_inference_app`
(``oai2.runtime.service_binding``) to :class:`MLXHotRuntime` and binds it
to 127.0.0.1. Used by the WI-PERF-003 / #240 benchmark matrix to compare
against the cold ``scripts/bench.py`` path.

Run with:    uv run python scripts/run_local_batch_inference.py --help
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from oai2.runtime import (
    AccessPolicy,
    IsolatedSessionRegistry,
    SafeBatchScheduler,
    ServiceCompatibility,
    ServiceLifecycle,
    create_batch_inference_app,
)
from oai2.runtime.mlx_hot_runtime import MLXHotRuntime
from oai2.runtime.model import ModelSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id to load (must be cached in the local HF cache).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    model_id = args.model
    spec = ModelSpec(name=model_id)
    runtime = MLXHotRuntime(spec, model_id=model_id)
    runtime.load()

    compatibility = ServiceCompatibility(
        config_digest=f"mlx-hot|{model_id}",
        model_id=model_id,
        schema_version="v1",
    )
    lifecycle = ServiceLifecycle(expected=compatibility)
    # ``create_batch_inference_app`` calls ``lifecycle.start(actual_compatibility)``
    # from the FastAPI lifespan; pre-starting here would raise.

    app = create_batch_inference_app(
        lifecycle=lifecycle,
        actual_compatibility=compatibility,
        access_policy=AccessPolicy(bind_host=args.host),
        scheduler=SafeBatchScheduler(),
        registry=IsolatedSessionRegistry(max_sessions_per_client=256),
        runtime=runtime,
        max_batch_size=args.max_batch_size,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
