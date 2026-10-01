# OAI-2.0 MLX sanity-probe benchmarks (v0.1.0 → v0.2.0)

> **Status: SUPERSEDED for prefill/TTFT.** v0.2.0 replaced the old
> `mlx_lm.generate`-based `bench_*.json` numbers. The new harness
> (`scripts/bench.py`) uses `mlx_lm.stream_generate` for a true
> prefill / decode / TTFT split. See **v0.2.0 results** below.
>
> The v0.1.0 JSON records (`bench_5ecb176f8b`, `bench_485d709342`,
> `bench_299bb3ee67`, `bench_43f7ed8a27`) are kept for historical
> traceability — they document that the bootstrap path works, but
> `prefill_seconds=0.0` and `ttft_seconds=0.0` are NOT real numbers.
>
> **Architecture decisions use v0.2.0+ numbers, never v0.1.0.**

## Hardware

- Machine: Apple Mac Studio (M5 Max)
- CPU: 18 cores (arm64)
- GPU: 40 cores (M5 Max)
- Unified memory: 64 GB
- macOS: 27.0 (BuildVersion 26A428)

## Software

- Python: 3.14.7
- MLX: `mlx_lm` 0.32.0 (default device `Device(gpu, 0)`)
- `transformers` 5.18.0, `tokenizers` 0.23.2, `huggingface_hub` 1.33.0
- Quantization: native MLX 4-bit where indicated.

## Results

All runs share the same machine, the same Python interpreter, and the
same benchmark script (`scripts/bench.py`). The prompt is a short
deterministic code-completion / generation task. Decode rate is the
end-to-end wall-clock time of `mlx_lm.generate(..., max_tokens=64)`
divided by 64 output tokens.

| Model                                       | Size  | Quant | Decode tok/s | Output quality (one-line) |
| ------------------------------------------- | ----- | ----- | ------------: | ------------------------- |
| `mlx-community/SmolLM-135M-Instruct-4bit`   | 135M  | 4-bit |       ~101.9 | Docstring-style code, no function body. |
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit`  | 500M  | 4-bit |       ~325.0 | Described the function instead of writing it. |
| `mlx-community/Llama-3.2-1B-Instruct-4bit`  | 1B    | 4-bit |       ~271.1 | Produced a function skeleton with docstring + Args/Returns. |
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | 500M  | 4-bit |       ~334.2 | Produced a correct recursive `factorial`. |

Raw JSON for each run is committed under `evals/benchmarks/`.

## What these numbers mean

- **Decode tok/s is real.** The M5 Max GPU is delivering ~100–340
  decode tok/s for small 4-bit MLX models. The architecture target
  research goals for `FURIOUS` (500–800 tok/s) and `NORMAL` (300–600
  tok/s) are within reach for the small end of the active-compute
  range once a sparse-expert model is wired.
- **Decode tok/s ≠ capability.** A 500M coding-tuned model produced a
  correct recursive `factorial` on this prompt; a 1B general model
  produced a skeleton. This matches the architecture target's
  principle: **intelligence must not be sacrificed for speed.**
- **Output quality is not a benchmark.** A single prompt is not an
  evaluation. These runs prove the installed MLX stack can load,
  compile, and decode real weights end-to-end on this hardware.

## Limitations

- **Prefill and TTFT are reported as 0.0.** `mlx_lm.generate` does not
  expose a separate prefill/decode boundary to callers, so the bench
  script cannot split them today. A future revision will use
  `stream_generate` for a true prefill/decode split.
- **Single prompt.** A single 64-token generation is not a
  representative sample. The benchmark harness will grow a real task
  suite (`evals/` package) before any architecture decision is made.
- **No system load isolation.** Other processes may have been
  running. For official numbers the bench should be repeated with the
  machine quiesced.
- **No warm-up separation.** First-call latency includes compile and
  cache warm-up; subsequent runs are typically faster.

## Running the benchmark yourself

The CLI shape changed in v0.2.0. The new harness reports
`load_seconds`, `compile_seconds`, `warm_run_seconds`,
`prefill_seconds` (TTFT), `prefill_tokens_per_second`, `decode_seconds`,
`decode_tokens_per_second`, `end_to_end_seconds`, `generation_tokens`,
`peak_memory_gb`, `active_memory_gb`, `cache_memory_gb`, and
aggregates `mean / median / min / max / stddev` over N repetitions.

```bash
cd /Users/orchords/src/OAI-2.0
uv run python scripts/bench.py \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt-tokens 128 1024 4096 \
    --max-tokens 256 \
    --repetitions 5
```

Each run writes a per-run JSON record under `evals/benchmarks/`,
plus a `summary_<tag>.json` aggregating the per-config aggregates.

`bench.py` never reports 0.0 for any measurable quantity; unmeasurable
metrics are reported as JSON `null`.

## How these will be replaced

A later release ships:

- A proper task suite covering coding accuracy, tool use, vision,
  long-horizon completion, and verification.
- Warm-up + repeated-measurement statistics.
- Separate prefill, decode, TTFT, and end-to-end wall-clock numbers.
- Comparison against the dense baselines listed in
  `docs/agent-architecture/ARCHITECTURE_TARGET.md`.

Until that release, do **not** treat these numbers as architecture
evidence. They are proof that the bootstrap path works end-to-end on
this machine.
