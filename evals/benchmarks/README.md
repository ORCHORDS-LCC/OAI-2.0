# OAI-2.0 MLX Benchmarks

_Last reviewed: 2026-10-02._

## Which numbers are current?

Use **v0.2+ benchmark records** for prefill/TTFT and MLX-reported generation-throughput claims.

The original v0.1 `bench_*.json` files are retained as historical bootstrap evidence only. They used a short `mlx_lm.generate` wall-clock interval that included prompt work, so their ~102–334 tok/s figures must **not** be described as clean kernel-only decode results.

## Hardware for committed local records

- Apple Mac Studio, M5 Max
- 18-core CPU
- 40-core GPU
- 64 GB unified memory
- macOS 27.0 (build 26A428)

The JSON record remains the source of truth for exact software versions.

## Corrected v0.2 smoke record

Committed file: `summary_smoke.json` / `bench_b3a2faef45.json`

Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

| Metric | Value |
| --- | ---: |
| Repetitions | 1 |
| Prompt tokens | 133 |
| Generation tokens | 32 |
| Model load | ~0.663 s |
| Compile | ~0.200 s |
| Warm run | ~0.0677 s |
| Prefill / TTFT | ~0.02544 s |
| Prefill throughput | ~5,227.9 tok/s |
| Decode | ~0.16566 s |
| MLX-reported generation throughput | ~198.47 tok/s |
| End-to-end generation phase | ~0.19110 s |
| Peak MLX memory | ~0.402 GB |
| Active MLX memory | ~0.282 GB |
| Cache memory | ~0.00869 GB |

This is a **single smoke repetition**. It proves the harness records separated timing fields; it does not establish stable model throughput.

## Terminology caveat

`decode_tokens_per_second` is populated from MLX-LM's reported `generation_tps`. The local `decode_seconds` field measures the wall interval after the first yielded token. Therefore documentation calls the throughput figure **MLX-reported generation throughput**, not kernel-only or universally “pure decode” throughput.

## Harness

`scripts/bench.py` uses `mlx_lm.stream_generate` and records:

- load;
- compile;
- warm-up;
- TTFT/prefill;
- prefill tok/s;
- first-yield-to-end decode interval plus MLX-reported generation tok/s;
- end-to-end generation;
- actual generation token count;
- MLX memory;
- aggregate mean/median/min/max/stddev over requested repetitions.

Metrics that cannot be measured should be `null`, never fabricated as zero.

Example:

```bash
uv run python scripts/bench.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt-tokens 128 1024 4096 \
  --max-tokens 256 \
  --repetitions 5
```

## Required next benchmark set

Before architecture decisions:

1. repeat every baseline at least 5 times;
2. separate cold/warm behavior;
3. use multiple prompt lengths;
4. use meaningful generation lengths;
5. compare capability on machine-verifiable tasks;
6. record system load;
7. compare quantization/runtime variants;
8. test speculation/MTP only after baseline correctness.

## Research targets

OAI-2.0's architecture targets remain goals:

- FURIOUS: 500–800 tok/s;
- NORMAL: 300–600 tok/s;
- DEEP: 150–350+ tok/s/lane;
- SWARM: 600–1,500+ aggregate tok/s.

Do not rewrite these goals as measured OAI-2.0 performance.
