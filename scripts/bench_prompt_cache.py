"""WI-PERF-003 / #240 configuration C: stable-prefix prefill-reuse measurement.

Facts this script is built on (verified against the installed mlx_lm 0.32.0
source): ``generate_step`` processes the WHOLE prompt through the cache on
every call — a plain ``make_prompt_cache`` list does no token-prefix matching,
so passing one shared cache to ``stream_generate`` does NOT skip prefix
prefill. Genuine reuse therefore has to restore the cached prefix state
explicitly, which is exactly what a matching prompt-cache layer would do.

Phases:

- MISS: a fresh cache per repetition with the full prompt (stable prefix +
  per-repetition suffix). Warm full-prefill cost.
- HIT: the prefix is prefilled once, its per-layer cache state is snapshotted,
  and each repetition restores that state before prefilling only its suffix.
  Suffix-only prefill cost — the benefit a matching cache layer could deliver.

Run inside the project venv (see scripts/bench.py docstring):

    uv run python scripts/bench_prompt_cache.py --repetitions 5

Only the local mlx backend is supported; this script never talks to the
gateway.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

PREFIX_TEXT = (
    "You are the OAI-2.0 coding agent. Repository rules: main only, direct push, "
    "local verification. Always inspect current source before changing code, run the "
    "repository gate before claiming success, and post reproducible evidence. "
)


def _token_ids(tokenizer: object, text: str, budget: int) -> list[int]:
    ids = tokenizer.encode(text)
    while len(ids) < budget:
        ids = ids + ids
    return list(ids[:budget])


def _run_one(
    model: object,
    tokenizer: object,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    cache: list | None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    first_token_at: float | None = None
    tokens_generated = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        tokens_generated = response.generation_tokens
    t_end = time.perf_counter()
    prefill_seconds = (first_token_at - t0) if first_token_at is not None else (t_end - t0)
    decode_seconds = (t_end - first_token_at) if first_token_at is not None else 0.0
    return {
        "prompt_tokens": len(prompt_ids),
        "prefill_seconds": round(prefill_seconds, 6),
        "decode_seconds": round(decode_seconds, 6),
        "total_seconds": round(t_end - t0, 6),
        "generation_tokens": tokens_generated,
    }


def _agg(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    ordered = sorted(values)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "count": n,
        "mean": round(mean, 6),
        "median": round(median, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "stddev": round(variance ** 0.5, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlx-community/SmolLM-135M-Instruct-4bit")
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "benchmarks",
    )
    parser.add_argument("--tag", default="prompt-cache")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_seconds = time.perf_counter() - t0

    base_prefix = _token_ids(tokenizer, PREFIX_TEXT, args.prefix_tokens)

    # Warm-up call, discarded, so measured numbers exclude Metal compile and
    # first-allocation effects of the generation path.
    _run_one(model, tokenizer, _token_ids(tokenizer, "warm up", 8), max_tokens=1, cache=None)

    miss_runs: list[dict[str, object]] = []
    for rep in range(1, args.repetitions + 1):
        suffix = _token_ids(tokenizer, f"Exercise {rep} of this training turn. " * 8, args.suffix_tokens)
        record = _run_one(
            model,
            tokenizer,
            base_prefix + suffix,
            max_tokens=args.max_tokens,
            cache=make_prompt_cache(model),
        )
        record["rep"] = rep
        miss_runs.append(record)
        print(f"miss rep {rep}/{args.repetitions}: {record}")

    # Prefix prefill once, then snapshot the per-layer cache state.
    hit_cache = make_prompt_cache(model)
    t_prefill = time.perf_counter()
    model(mx.array([base_prefix]), cache=hit_cache)
    mx.eval(hit_cache[0].state[0])
    prefix_prefill_seconds = time.perf_counter() - t_prefill
    saved_states = [c.state for c in hit_cache]

    hit_runs: list[dict[str, object]] = []
    for rep in range(1, args.repetitions + 1):
        suffix = _token_ids(tokenizer, f"Exercise {rep} of this training turn. " * 8, args.suffix_tokens)
        for cache_layer, saved in zip(hit_cache, saved_states, strict=True):
            cache_layer.state = saved
        record = _run_one(
            model,
            tokenizer,
            suffix,
            max_tokens=args.max_tokens,
            cache=hit_cache,
        )
        record["rep"] = rep
        hit_runs.append(record)
        print(f"hit  rep {rep}/{args.repetitions}: {record}")

    miss_prefills = [float(r["prefill_seconds"]) for r in miss_runs]
    hit_prefills = [float(r["prefill_seconds"]) for r in hit_runs]
    miss_mean = sum(miss_prefills) / len(miss_prefills)
    hit_mean = sum(hit_prefills) / len(hit_prefills)
    summary = {
        "benchmark": "wi-perf-003-config-C-prompt-cache",
        "model": args.model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "load_seconds": round(load_seconds, 4),
        "prefix_tokens": len(base_prefix),
        "suffix_tokens": args.suffix_tokens,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "prefix_only_prefill_seconds": round(prefix_prefill_seconds, 6),
        "miss_runs": miss_runs,
        "miss_prefill_aggregate": _agg(miss_prefills),
        "hit_runs": hit_runs,
        "hit_prefill_aggregate": _agg(hit_prefills),
        "hit_prefill_aggregate_excluding_first": _agg(hit_prefills[1:]),
        "prefill_saving_seconds": round(miss_mean - hit_mean, 6),
        "prefill_saving_percent": round(100.0 * (miss_mean - hit_mean) / miss_mean, 2),
        "prefill_saving_seconds_excluding_first": round(miss_mean - sum(hit_prefills[1:]) / len(hit_prefills[1:]), 6),
        "prefill_saving_percent_excluding_first": round(
            100.0 * (miss_mean - sum(hit_prefills[1:]) / len(hit_prefills[1:])) / miss_mean, 2
        ),
        "note": (
            "mlx_lm 0.32.0 generate_step prefills the full prompt every call; the hit "
            "phase restores snapshotted prefix state explicitly, emulating what a "
            "matching prompt-cache layer would deliver."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"summary_prompt_cache_{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote summary: {out_path}")
    print(
        f"miss_mean={miss_mean:.4f}s hit_mean={hit_mean:.4f}s "
        f"saving={summary['prefill_saving_percent']}% ({summary['prefill_saving_seconds']}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
