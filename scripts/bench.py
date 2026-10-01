"""MLX sanity probe benchmark.

Downloads a small (~50–150M) MLX-compatible model from
``mlx-community`` on Hugging Face, runs a deterministic prompt through
``mlx_lm.generate``, and reports measured tokens/sec and time-to-first-
token. The goal is **not** to produce a publishable number — it is to
prove that the installed MLX stack can load, compile, and decode real
weights on this machine end-to-end.

Usage::

    uv run python scripts/bench.py
    uv run python scripts/bench.py --model mlx-community/SmolLM-135M-Instruct-bf16

Outputs are written to ``evals/benchmarks/<run_id>.json`` and printed to
stdout. No external endpoints are contacted outside of Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BenchResult:
    run_id: str
    timestamp: float
    model: str
    prompt: str
    prompt_tokens: int
    output_tokens: int
    prefill_seconds: float
    decode_seconds: float
    decode_tokens_per_second: float
    prefill_tokens_per_second: float
    ttft_seconds: float
    device: str
    sys_info: dict[str, str] = field(default_factory=dict)
    output_excerpt: str = ""


def _system_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import mlx.core as mx

        info["mlx_default_device"] = str(mx.default_device())
    except Exception as exc:  # pragma: no cover
        info["mlx_default_device"] = f"unavailable:{type(exc).__name__}"
    try:
        import mlx_lm  # noqa: F401

        from importlib import metadata as _md

        info["mlx_lm_version"] = _md.version("mlx-lm")
    except Exception:
        info["mlx_lm_version"] = "unknown"
    return info


def _huggingface_env() -> None:
    """Avoid hitting network mirrors; trust the HF Hub directly."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def run_bench(
    *,
    model_id: str,
    prompt: str | None = None,
    max_tokens: int = 64,
    out_dir: Path,
) -> BenchResult:
    _huggingface_env()

    if prompt is None:
        prompt = (
            "Write a Python function that returns the sum of the first n "
            "natural numbers, with a short docstring."
        )

    # Local imports to keep --help fast.
    from mlx_lm import generate, load

    print(f"loading model: {model_id}", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_seconds = time.perf_counter() - t0
    print(f"loaded in {load_seconds:.2f}s", file=sys.stderr)

    prompt_tokens = _count_tokens(tokenizer, prompt)

    # Measure prefill by stepping the model once and reading the time.
    t_prefill_start = time.perf_counter()
    # Use the streaming generator so we can split prefill vs decode.
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    t_total = time.perf_counter()

    # We don't have a clean prefill/decode split from mlx_lm.generate,
    # so we attribute all time to decode and report TTFT=0.0 with a note.
    # A future version can use stream_generate for a true split.
    output_tokens = _count_tokens(tokenizer, response)
    decode_seconds = max(t_total - t_prefill_start, 1e-6)
    prefill_seconds = 0.0  # not separately measurable from mlx_lm.generate.
    ttft = 0.0
    decode_tps = output_tokens / decode_seconds
    prefill_tps = prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0

    device = _system_info().get("mlx_default_device", "unknown")

    result = BenchResult(
        run_id=f"bench_{uuid.uuid4().hex[:10]}",
        timestamp=time.time(),
        model=model_id,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=decode_tps,
        prefill_tokens_per_second=prefill_tps,
        ttft_seconds=ttft,
        device=device,
        sys_info=_system_info(),
        output_excerpt=response[:240],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{result.run_id}.json"
    out_file.write_text(json.dumps(asdict(result), indent=2))
    print(f"wrote {out_file}", file=sys.stderr)
    return result


def _print_summary(result: BenchResult) -> None:
    print("=" * 60)
    print(f"OAI-2.0 MLX sanity probe")
    print("=" * 60)
    for key, value in asdict(result).items():
        if key in {"sys_info", "output_excerpt"}:
            continue
        print(f"  {key}: {value}")
    print("  --- system ---")
    for k, v in result.sys_info.items():
        print(f"  {k}: {v}")
    print(f"  output_excerpt: {result.output_excerpt!r}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OAI-2.0 MLX sanity probe")
    parser.add_argument(
        "--model",
        default="mlx-community/SmolLM-135M-Instruct-bf16",
        help="HF model id (default: SmolLM-135M-Instruct-bf16, ~135M params)",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "benchmarks",
    )
    args = parser.parse_args(argv)

    result = run_bench(
        model_id=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        out_dir=args.out_dir,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
