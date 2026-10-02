"""WI-PERF-003 / #240 configuration G: speculative-decoding comparison.

 mlx_lm 0.32.0 supports speculative decoding through ``stream_generate``'s
``draft_model`` parameter when draft and target share a tokenizer. This
script measures raw decode throughput with and without a draft model on
the identical prompt, and compares the greedy outputs for equivalence
(speculative decoding must not change results, only speed).

Per the #240 body, speculative runs are benchmarked separately from
batching; no combined-benefit claim is made here.

Run inside the project venv:

    uv run python scripts/bench_speculative.py \
        --model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \
        --draft-model mlx-community/Qwen2.5-0.5B-Instruct-4bit

Only the local mlx backend is used; this script never talks to the
gateway.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from mlx_lm import load, stream_generate


def _run_one(
    model: object,
    tokenizer: object,
    prompt: str,
    *,
    max_tokens: int,
    draft_model: object | None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    text = ""
    tokens_generated = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
    ):
        text += response.text
        tokens_generated = response.generation_tokens
    elapsed = time.perf_counter() - t0
    return {
        "decode_seconds": round(elapsed, 6),
        "generation_tokens": tokens_generated,
        "decode_tokens_per_second": round(tokens_generated / elapsed, 2) if elapsed > 0 else 0.0,
        "text": text,
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
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
    parser.add_argument("--draft-model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "benchmarks",
    )
    parser.add_argument("--tag", default="speculative")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    target_load_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    draft_model, draft_tokenizer = load(args.draft_model)
    draft_load_seconds = time.perf_counter() - t0

    if len(tokenizer) != len(draft_tokenizer) or tokenizer.encode("probe") != draft_tokenizer.encode("probe"):
        raise SystemExit("draft and target tokenizers are incompatible")

    ids = tokenizer.encode("Write a short Python function that adds two numbers. " * 8)
    prompt = tokenizer.decode(ids[: args.prompt_tokens])

    # Warm-up calls, discarded, so measured numbers exclude Metal compile.
    _run_one(model, tokenizer, prompt, max_tokens=1, draft_model=None)
    _run_one(model, tokenizer, prompt, max_tokens=1, draft_model=draft_model)

    normal_runs: list[dict[str, object]] = []
    for rep in range(1, args.repetitions + 1):
        record = _run_one(model, tokenizer, prompt, max_tokens=args.max_tokens, draft_model=None)
        record["rep"] = rep
        normal_runs.append(record)
        print(f"normal     rep {rep}/{args.repetitions}: {record}")

    spec_runs: list[dict[str, object]] = []
    for rep in range(1, args.repetitions + 1):
        record = _run_one(model, tokenizer, prompt, max_tokens=args.max_tokens, draft_model=draft_model)
        record["rep"] = rep
        spec_runs.append(record)
        print(f"speculative rep {rep}/{args.repetitions}: {record}")

    normal_tps = [float(r["decode_tokens_per_second"]) for r in normal_runs]
    spec_tps = [float(r["decode_tokens_per_second"]) for r in spec_runs]
    text_match = all(r["text"] == normal_runs[0]["text"] for r in normal_runs + spec_runs)
    summary = {
        "benchmark": "wi-perf-003-config-G-speculative",
        "model": args.model,
        "draft_model": args.draft_model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "target_load_seconds": round(target_load_seconds, 4),
        "draft_load_seconds": round(draft_load_seconds, 4),
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "normal_runs": normal_runs,
        "speculative_runs": spec_runs,
        "normal_decode_tps_aggregate": _agg(normal_tps),
        "speculative_decode_tps_aggregate": _agg(spec_tps),
        "speedup_factor": round(
            (sum(spec_tps) / len(spec_tps)) / (sum(normal_tps) / len(normal_tps)), 3
        ),
        "greedy_text_match": text_match,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"summary_speculative_{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote summary: {out_path}")
    print(
        f"normal_tps_mean={summary['normal_decode_tps_aggregate']['mean']:.1f} "
        f"spec_tps_mean={summary['speculative_decode_tps_aggregate']['mean']:.1f} "
        f"speedup={summary['speedup_factor']}x text_match={text_match}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
