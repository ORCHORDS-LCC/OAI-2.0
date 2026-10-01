"""OAI-2.0 MLX benchmark harness — properly separated prefill / decode / TTFT.

Run with:    uv run python scripts/bench.py [args]
NOT with:    python3 scripts/bench.py   # ModuleNotFoundError: mlx_lm

This script imports :mod:`mlx_lm` (a project dependency declared in
``pyproject.toml``). It runs under the project venv created by ``uv sync``,
i.e. via ``uv run python scripts/bench.py ...``. A plain ``python3``
invocation will fail with :class:`ModuleNotFoundError` because the system
interpreter does not see project-scoped dependencies. The :func:`run_one`
guard below converts that failure into an actionable one-line error.

This harness is the source of truth for measured Apple Silicon MLX
performance on the OAI-2.0 reference machines. It supersedes v0.1.0's
``scripts/bench.py``, which incorrectly attributed prefill time to
``decode_seconds``.

Measured quantities (per run):

- ``load_seconds`` — model load + first-compile wall time.
- ``compile_seconds`` — warm-up run (compiled/uncached path).
- ``warm_run_seconds`` — second warm-up run.
- ``prompt_tokens`` — tokenized prompt size.
- ``prefill_seconds`` — wall time from prompt submitted to first token yielded.
  This is the *Time-To-First-Token* (TTFT).
- ``prefill_tokens_per_second`` — ``prompt_tokens / prefill_seconds``.
- ``decode_seconds`` — wall time from first token yielded to last token yielded.
- ``generation_tokens`` — actual number of tokens generated (excluding prefill).
- ``decode_tokens_per_second`` — ``generation_tokens / decode_seconds``.
- ``end_to_end_seconds`` — ``prefill_seconds + decode_seconds``.
- ``peak_memory_gb`` — peak unified-memory residency reported by MLX.
- ``active_memory_gb`` — active (live) MLX memory after run.
- ``cache_memory_gb`` — MLX cache memory after run.

Per-model, per-config statistics over ``--repetitions`` runs:

- ``mean``, ``median``, ``min``, ``max``, ``stddev`` for every metric.
- Generated token counts and excerpt of the final output.

If a metric cannot be measured correctly it is reported as ``null``
(JSON) / ``None`` (Python). The harness NEVER invents zeros.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RunMetrics:
    run_id: str
    timestamp: float
    model: str
    quantization: str | None
    prompt_label: str
    prompt_tokens: int
    max_tokens: int

    load_seconds: float | None
    compile_seconds: float | None
    warm_run_seconds: float | None

    prefill_seconds: float | None  # TTFT
    prefill_tokens_per_second: float | None
    decode_seconds: float | None
    decode_tokens_per_second: float | None
    end_to_end_seconds: float | None
    generation_tokens: int | None

    peak_memory_gb: float | None
    active_memory_gb: float | None
    cache_memory_gb: float | None

    device: str
    sys_info: dict[str, str] = field(default_factory=dict)
    output_text: str = ""
    output_excerpt: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Stat:
    count: int
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    stddev: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _stat(values: Iterable[float | None]) -> Stat:
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not nums:
        return Stat(count=0, mean=None, median=None, min=None, max=None, stddev=None)
    n = len(nums)
    mean = statistics.fmean(nums)
    median = statistics.median(nums)
    sd = statistics.pstdev(nums) if n > 1 else 0.0
    return Stat(
        count=n,
        mean=mean,
        median=median,
        min=min(nums),
        max=max(nums),
        stddev=sd,
    )


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
        from importlib import metadata as _md

        info["mlx_lm_version"] = _md.version("mlx-lm")
    except Exception:
        info["mlx_lm_version"] = "unknown"
    return info


def _build_prompt(target_tokens: int) -> tuple[str, int]:
    """Build a deterministic prompt of approximately ``target_tokens`` words.

    Returns (prompt, actual_word_count). Used as a controlled-length
    fill so we can hit approximate token budgets without depending on
    tokenizer-specific behavior.
    """
    if target_tokens <= 0:
        return "", 0
    base = (
        "Summarize the following Python module. Focus on public functions, "
        "their inputs and outputs, error conditions, and any side effects. "
        "Be concise and accurate. "
    )
    # 1 word ≈ 1.3 tokens for English prose; round generously.
    target_words = max(1, int(target_tokens / 1.3))
    words = (base * (target_words // len(base.split()) + 2)).split()
    text = " ".join(words[:target_words])
    return text, len(text.split())


def _measure_memory() -> tuple[float | None, float | None, float | None]:
    """Return (peak_gb, active_gb, cache_gb). Any unavailable → None."""
    try:
        import mlx.core as mx
    except Exception:
        return None, None, None
    peak = active = cache = None
    try:
        peak = mx.get_peak_memory() / 1e9
    except Exception:
        pass
    try:
        active = mx.get_active_memory() / 1e9
    except Exception:
        pass
    try:
        cache = mx.get_cache_memory() / 1e9
    except Exception:
        pass
    return peak, active, cache


def run_one(
    *,
    model_id: str,
    prompt: str,
    prompt_label: str,
    max_tokens: int,
    warm: bool,
) -> RunMetrics:
    """Load model + run a single measured generation."""
    try:
        from mlx_lm import load, stream_generate  # local import for --help speed.
    except ModuleNotFoundError as exc:
        # Python's import machinery sets `exc.name` only when the message
        # starts with "No module named ". Fall back to parsing the message
        # so the operator sees the *actual* missing module name.
        missing = exc.name or (
            exc.msg.split("'", 2)[1]
            if exc.msg.startswith("No module named '") and "'" in exc.msg[18:]
            else exc.msg
        )
        raise SystemExit(
            f"Python module {missing!r} is not installed in this interpreter. "
            f"bench.py requires project deps in the project venv. Run with "
            f"`uv run python scripts/bench.py ...`, or `uv sync` first."
        ) from exc

    run_id = f"bench_{uuid.uuid4().hex[:10]}"
    sys_info = _system_info()
    notes: list[str] = []

    # ---- Load + compile timing ----
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass

    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_seconds = time.perf_counter() - t0

    device = sys_info.get("mlx_default_device", "unknown")
    prompt_tokens = len(tokenizer.encode(prompt))

    # ---- Warm-up: forces MLX graph compile + KV cache allocation,
    #      excluded from measured runs below. ----
    compile_seconds: float | None = None
    warm_run_seconds: float | None = None

    if warm:
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
            t_compile = time.perf_counter()
            for _ in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=8,
            ):
                pass
            mx.eval(model.parameters())
            compile_seconds = time.perf_counter() - t_compile

            t_warm = time.perf_counter()
            for _ in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=8,
            ):
                pass
            mx.eval(model.parameters())
            warm_run_seconds = time.perf_counter() - t_warm
        except Exception as exc:
            notes.append(f"warmup-failed: {type(exc).__name__}: {exc}")

    # ---- Measured generation ----
    prefill_seconds: float | None = None
    prefill_tps: float | None = None
    decode_seconds: float | None = None
    decode_tps: float | None = None
    end_to_end: float | None = None
    generation_tokens: int | None = None
    output_text = ""

    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass

    try:
        t_start = time.perf_counter()
        last_response = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            last_response = response
            # First yielded response: prompt_tps / generation_tokens=1 / etc.
            if prefill_seconds is None:
                prefill_seconds = time.perf_counter() - t_start
                if prompt_tokens > 0 and prefill_seconds > 0:
                    prefill_tps = prompt_tokens / prefill_seconds
            output_text += response.text

        t_end = time.perf_counter()
        end_to_end = t_end - t_start

        if prefill_seconds is not None:
            decode_seconds = max(t_end - t_start - prefill_seconds, 0.0)

        if last_response is not None:
            generation_tokens = last_response.generation_tokens
            decode_tps = last_response.generation_tps
        else:
            notes.append("stream_generate produced no responses")
            generation_tokens = 0
            decode_tps = None
    except Exception as exc:
        notes.append(f"generate-failed: {type(exc).__name__}: {exc}")

    peak, active, cache = _measure_memory()

    return RunMetrics(
        run_id=run_id,
        timestamp=time.time(),
        model=model_id,
        quantization=_extract_quantization(model_id),
        prompt_label=prompt_label,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        load_seconds=load_seconds,
        compile_seconds=compile_seconds,
        warm_run_seconds=warm_run_seconds,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tps,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=decode_tps,
        end_to_end_seconds=end_to_end,
        generation_tokens=generation_tokens,
        peak_memory_gb=peak,
        active_memory_gb=active,
        cache_memory_gb=cache,
        device=device,
        sys_info=sys_info,
        output_text=output_text,
        output_excerpt=output_text[:240],
        notes=notes,
    )


def _extract_quantization(model_id: str) -> str | None:
    lower = model_id.lower()
    for tag in ("4bit", "3bit", "2bit", "8bit", "bf16", "fp16", "f16"):
        if tag in lower:
            return tag
    return None


def _aggregate(runs: list[RunMetrics]) -> dict[str, dict[str, float | int | None]]:
    metric_names = [
        "load_seconds",
        "compile_seconds",
        "warm_run_seconds",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_seconds",
        "decode_tokens_per_second",
        "end_to_end_seconds",
        "peak_memory_gb",
        "active_memory_gb",
        "cache_memory_gb",
    ]
    out: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = [getattr(r, name) for r in runs]
        s = _stat(values)
        out[name] = s.as_dict()
    # Token counts are integers; report mean/median/min/max only.
    counts = [r.generation_tokens for r in runs if r.generation_tokens is not None]
    if counts:
        out["generation_tokens"] = {
            "count": len(counts),
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "min": min(counts),
            "max": max(counts),
            "stddev": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        }
    else:
        out["generation_tokens"] = {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stddev": None,
        }
    return out


def _print_run_table(model_id: str, prompt_label: str, runs: list[RunMetrics]) -> None:
    print(f"\n=== {model_id}  |  prompt={prompt_label}  |  runs={len(runs)} ===")
    # Per-run numbers
    for i, r in enumerate(runs, 1):
        print(
            f"  run {i:>2}: "
            f"load={_fmt(r.load_seconds)}s "
            f"compile={_fmt(r.compile_seconds)}s "
            f"warm={_fmt(r.warm_run_seconds)}s "
            f"prefill={_fmt(r.prefill_seconds)}s "
            f"decode={_fmt(r.decode_seconds)}s "
            f"gen={r.generation_tokens}tok "
            f"decode_tps={_fmt(r.decode_tokens_per_second)} "
            f"prefill_tps={_fmt(r.prefill_tokens_per_second)} "
            f"peak={_fmt(r.peak_memory_gb)}GB"
        )
    # Aggregates
    agg = _aggregate(runs)
    for name, s in agg.items():
        print(f"  AGG {name}: {s}")


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="mlx-community/SmolLM-135M-Instruct-4bit",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[128, 1024, 4096],
        help="One or more approximate prompt token budgets to benchmark.",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up (default: warm up so measured numbers exclude compile).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "benchmarks",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional label for the run; included in the summary file name.",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or args.model.split("/")[-1]
    print(
        f"benchmark: model={args.model} max_tokens={args.max_tokens} "
        f"repetitions={args.repetitions} warmup={not args.no_warmup} tag={tag}",
        file=sys.stderr,
    )

    summary: dict[str, object] = {
        "tag": tag,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "warmup": not args.no_warmup,
        "sys_info": _system_info(),
        "configs": [],
    }

    for budget in args.prompt_tokens:
        prompt, word_count = _build_prompt(budget)
        prompt_label = f"~{budget}tokens ({word_count}words)"
        runs: list[RunMetrics] = []
        for i in range(args.repetitions):
            print(f"--- {prompt_label} rep {i + 1}/{args.repetitions} ---", file=sys.stderr)
            r = run_one(
                model_id=args.model,
                prompt=prompt,
                prompt_label=prompt_label,
                max_tokens=args.max_tokens,
                warm=not args.no_warmup,
            )
            runs.append(r)
            run_file = out_dir / f"{r.run_id}.json"
            run_file.write_text(json.dumps(asdict(r), indent=2, default=str))
        _print_run_table(args.model, prompt_label, runs)
        summary["configs"].append(
            {
                "prompt_label": prompt_label,
                "prompt_tokens": runs[0].prompt_tokens if runs else None,
                "runs": [asdict(r) for r in runs],
                "aggregate": _aggregate(runs),
            }
        )

    summary_file = out_dir / f"summary_{tag}.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote summary: {summary_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
