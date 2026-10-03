# OAI-2.0 MLX Benchmarks

_Reconciliation baseline: `c4a1f6f6134bd18668b8a621a2c0f0f89708e5e7` (source state before this documentation commit)._

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

## llama.cpp production lane (WI-PERF-003, issue #240)

Committed directory: `llamacpp_production_1ba5283/`

These are the first records in this repository measured against the **real
production serving path** rather than the local MLX path. Before commit
`1ba5283599a55f8d70364c185df0a3c1fff0758c`, `scripts/bench.py` had no
llama-server backend, so configurations B0/B1/B2 and the batch matrix could not
be run at all and no artifact for them could exist.

Every file here was produced by harness commit `1ba5283599a5` with
`dirty=false`, and each summary embeds that provenance in its own `harness`
block — a throughput number that cannot be tied to the source that produced it
is not evidence.

### Serving identity (read from `/props`, not assumed from flags)

| Field | Value |
| --- | --- |
| Endpoint | `http://127.0.0.1:8851` (NORMAL lane) |
| Model path | `/Users/orchords/models/normal/SmolLM2-1.7B-Instruct-Q4_K_M.gguf` |
| Model sha256 | `77665ea4815999596525c636fbeb56ba8b080b46ae85efef4f0d986a139834d7` |
| Model alias | `smollm2-1.7b-q4km` |
| Quantization | Q4_K_M (in the GGUF filename) |
| Backend | `llama-server` (llama.cpp) |
| Slots | 4 (`total_slots`) |
| Context per slot | 8192 |
| Host | 64 GB unified memory, macOS |

Prompt: a 21-word stable prefix plus ~128-token budget, 256 generated tokens,
5 repetitions, warm unless the configuration is `cold`.

### Results

`Decode tok/s/agent` is llama-server's own `timings.predicted_per_second`.
`Aggregate` is summed **within** one repetition and then reduced across the 5
repetitions — a flat sum over all runs multiplies by the repetition count as
well as the agent count and is not a system throughput figure.

| Mode | Agents | Runs | Decode tok/s/agent | Aggregate tok/s | TTFT ms | Cache hit | Prefill tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Cold baseline | 1 | 5 | 183.28 | 183.28 | 7.6 | 185/190 | 164.3 |
| B1 Hot | 1 | 5 | 187.90 | 187.90 | 7.9 | 185/190 | 167.2 |
| B2 Hot + prompt cache | 1 | 5 | 184.49 | 184.49 | 8.0 | 185/190 | 163.4 |
| F Hot | 2 | 10 | 144.82 | 289.64 | 9.1 | 185/190 | 137.9 |
| F Hot | 4 | 20 | 116.08 | 464.32 | 12.2 | 185/190 | 107.5 |
| F Hot | 8 | 40 | 116.35 | 927.27 | 1115.3 | 185/190 | 107.9 |

Individual per-repetition samples are preserved in each summary under
`aggregate.aggregate_decode_tokens_per_second.per_repetition`.

Host memory across the run: 64.0 GB total, 55.96 GB used, 8.04 GB free, swap
4.03 GB used of 5.0 GB.

### Confound — read this before comparing these numbers to anything else

The `:8854` STRONG lane was serving **live production traffic** and was busy
on **203 of 205** telemetry samples spanning the whole window
(`lane_contention_telemetry.json`). These figures are therefore **contended**
and are a lower bound on `:8851` capacity, not a clean measurement of it.

They must not be compared against earlier uncontended `:8851` figures (which
ran ~250–268 tok/s at 1 user) as if the difference were a property of the
configuration. It is substantially the shared GPU. One B1 repetition reached
299.03 tok/s against a 187.90 median — the spread within a single configuration
is the contention, not the configuration.

### Terminology

As with the MLX records, `decode_tokens_per_second` is the **server-reported**
generation rate. Client-observed wall-clock is recorded separately in
`ttft_seconds` and `client_decode_tokens_per_second` and is never promoted to a
decode rate. TTFT at 8 agents rises to ~1.1 s because 8 requests contend for 4
slots; that is queueing, not a regression in decode speed.

## Uncontended re-run — `llamacpp_production_1ba5283/clean_uncontended/`

Re-run after the STRONG lane went idle, to separate the throughput figures
from the contention described above. Harness `ff4956c912e9`, `dirty=false`.

`:8854` slot occupancy was sampled every second for the whole window:
**busy on 0 of 211 samples.** The GPU-contention confound from the first
run is absent here.

| Mode | Agents | Decode tok/s/agent | Aggregate tok/s | TTFT ms | Prefill tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 Cold baseline | 1 | 270.23 | 270.23 | 7.5 | 185.6 |
| B1 Hot | 1 | 245.01 | 245.01 | 7.1 | 196.6 |
| B2 Hot + prompt cache | 1 | 272.17 | 272.17 | 5.3 | 264.9 |
| F Hot | 2 | 254.93 | 509.86 | 5.8 | 234.5 |
| F Hot | 4 | 178.63 | 714.53 | 8.7 | 167.7 |
| F Hot | 8 | 179.52 | 1433.14 | 723.9 | 168.3 |

Contended → uncontended, same suite, same host, same model:

| Mode | Contended | Uncontended | Delta |
| --- | ---: | ---: | ---: |
| B0 Cold | 183.28 | 270.23 | +47% |
| B1 Hot | 187.90 | 245.01 | +30% |
| B2 Hot + prompt cache | 184.49 | 272.17 | +48% |
| N=2 | 289.64 | 509.86 | +76% |
| N=4 | 464.32 | 714.53 | +54% |
| N=8 | 927.27 | 1433.14 | +55% |

A 30–76% swing from lane contention alone. This is the reason the first
run's figures are reported as lower bounds rather than capacity.

### B0 is not the slowest configuration — a labelling caveat

B0 "cold" (270.23) is *faster* than B1 "hot" (245.01) here, which inverts
the expected ordering. B0 also has the widest spread across repetitions
(233.03–297.17) against B1's tight 243.98–265.62. The 75 s idle before B0
is not sufficient to evict llama.cpp's slot-level prefix cache, so "cold"
in this harness means "no warm-up credit from the harness", not "no
residual server state". Treat B0 as *unwarmed-harness*, not *cold server*.

### Second confound — host memory, not GPU

This run was uncontended for GPU compute but **not** for host memory. At
the end of the window the host reported 64.0 GB total, **63.41 GB used,
0.59 GB free, 3.99 GB of 5.0 GB swap in use**.

The cause is visible in process residency: the idle STRONG lane
(`Qwen3-4B-Thinking-2507-Q4_K_M`, 2 slots × 16384 ctx) holds **19.5 GB
RSS**, against 4.1 GB for the NORMAL lane. On Apple Silicon the unified
memory pool is shared, so both servers' KV caches and Metal allocations are
resident in the same physical memory regardless of lane activity — an idle
lane still costs its footprint.

So these figures answer "what does the NORMAL lane do with the GPU to
itself" — they are **not** a clean whole-host capacity baseline. A truly
clean baseline needs the STRONG lane's memory released as well as its GPU
time, which on this host means stopping that lane rather than idling it.

## Which configuration covers "Hot + KV reuse"

The #240 table lists "Hot + prompt cache" and "Hot + KV reuse" as separate
rows. They are distinct on the **MLX** path and are **not** separable on the
**llama.cpp** path. Recording the mapping so the row is not silently
missing.

| Table row | Where it is implemented | How it is measured |
| --- | --- | --- |
| Hot + prompt cache | llama.cpp slot prefix cache | `cache_hit_tokens` from `timings.cache_n`; B2 above reports 185/190 |
| Hot + KV reuse | `oai2/runtime/prefix_kv_cache.py` (`PrefixKVCache`), wired into `MLXHotRuntime` | `tests/test_runtime_prefix_kv_cache.py::test_real_model_prefix_restore_skips_prefix_prefill` — a real `mlx_lm` model load, asserting prefix restore skips prefix prefill |

`PrefixKVCache` is a longest-prefix **KV-state** store with bounded entries
and hit/miss/invalidation counters, which is what REQ-PERF-032 describes.
It is an MLX-path feature.

llama-server offers no separate KV-reuse control distinct from its prompt
cache, so a dedicated "Hot+KV reuse" cell cannot be produced on the
llama.cpp path — not "not yet measured", but **not applicable at that
layer**. Producing a number there would mean relabelling B2's
`cache_hit_tokens` as a different thing, which is how a table ends up with
two rows that quietly measure the same condition.

Verified locally: 15 tests in `tests/test_runtime_prefix_kv_cache.py` pass,
including the real-model restore test.
