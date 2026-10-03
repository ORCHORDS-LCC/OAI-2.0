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

## Workload matrix — prompt and generation length

`llamacpp_production_1ba5283/workload_matrix/`, harness `430c3aba9a49`,
`dirty=false`. Hot config, 21-word stable prefix, 5 repetitions per cell,
single agent.

> **These cells are CONTENDED.** `:8854` was busy on **257 of 293**
> telemetry samples across this window. Absolute throughput below is a
> lower bound, consistent with the earlier contended run (~93–100 tok/s)
> rather than the uncontended one (~245–272 tok/s). The *shape* of each
> sweep is still informative because contention is roughly constant across
> cells within a sweep, but the level is not capacity.

### Prompt-length sweep (generation fixed at 256)

| Prompt tokens | Prefill tok/s | TTFT p50 ms | Decode tok/s | p95 | p99 | Cache hit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 186 | 75.3 | 15.8 | 139.84 | 182.97 | 182.97 | 185 |
| 1085 | 76.3 | 17.7 | 121.72 | 133.15 | 133.15 | 1084 |
| 4166 | 76.9 | 21.9 | 101.85 | 103.88 | 103.88 | 4165 |

Decode degrades with context: **139.84 → 121.72 → 101.85 tok/s**, a **27%
drop** from the shortest to the longest prompt, while aggregate throughput
stays roughly flat. Reading a short-prompt result as a capacity number would
overstate long-context behaviour by that margin.

Prefill throughput is **flat at ~75–77 tok/s across a 22× prompt range**.
A context-independent prefill rate is not what a cache-miss path should
look like; it is consistent with the run being prefill-bound under GPU
contention rather than with a property of the configuration. Treated as an
observation to re-measure on a quiet lane, not as a finding.

### Generation-length sweep (prompt fixed at ~1K)

| Generated tokens | Decode tok/s | p95 | p99 | E2E s | E2E p95 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 108.48 | 111.74 | 111.74 | 1.188 | 1.200 |
| 512 | 108.47 | 110.87 | 110.87 | 4.730 | 4.923 |
| 1024 | 104.06 | 112.70 | 112.70 | 9.854 | 10.166 |

Decode rate is essentially invariant to generation length (108.48 / 108.47 /
104.06), and end-to-end time scales **linearly** with it (1.188 / 4.730 /
9.854 s). 1024 tokens at ~104 tok/s should take ~9.8 s, which matches —
so the rate reported is internally consistent with the wall clock, and
`client_decode_tokens_per_second` agrees.

p95 and p99 coincide throughout because each cell has 5 samples and
percentiles use nearest rank. Tail behaviour is therefore **not yet
resolved at this sample count**; 5 repetitions satisfies AC-PERF-031's
minimum but cannot separate p95 from p99. Raising repetitions is the only
way to make the tail meaningful.

### ~16K prompt: not satisfiable on the current production config

The required workload matrix includes a ~16K prompt. The NORMAL lane
serves **8192 ctx per slot**, so the server rejects it outright:

```
HTTP 400 {"error":{"code":400,
  "message":"request (9031 tokens) exceeds the available context size (8192 tokens), try increasing it",
  "type":"exceed_context_size_error","n_prompt_tokens":9031,"n_ctx":8192}}
```

Recorded artifact: `context_16k_rejected_by_server.json`. The harness
fail-closed — it recorded the 400 and the `server-published-no-timings`
note rather than reporting a silent 0 tok/s.

This is a **configuration limitation, not a measurement gap**, and it has
exactly two resolutions: raise `--ctx-size` (which increases per-slot KV
memory, and the host was already at 63.4/64 GB during the uncontended run),
or record the row as not-applicable with this reason. Silently omitting it
would leave the required table short a row.

## True cold start — `llamacpp_production_1ba5283/cold_start.json`

The `cold` label in the matrices above means *no warm-up credit from the
harness*. Config A of #240 asks for more than that: "fresh model/runtime
start; no reusable prompt/KV state". That is unobtainable against a
long-lived server, whose weights are resident and whose slot still holds the
previous request's prefix cache.

`scripts/cold_start_probe.py` measures it properly. Each sample starts a
**new** `llama-server` process on its own port, waits until `/props`
answers, issues exactly one request, and terminates the process. Nothing is
shared between samples.

Flags are copied verbatim from the production NORMAL lane except `--port`
and `--parallel 1`.

| Sample | Load s | Prefill ms | Decode tok/s | `cache_n` |
|---:|---:|---:|---:|---:|
| 0 | 0.78 | 29.0 | 160.49 | 0 |
| 1 | 0.77 | 29.6 | 159.68 | 0 |
| 2 | 0.77 | 29.2 | 164.55 | 0 |
| 3 | 0.78 | 29.5 | 157.02 | 0 |
| 4 | 0.77 | 31.3 | 163.63 | 0 |
| **median** | **0.775** | **29.5** | **160.49** | **0** |

`cache_n == 0` on every sample is the proof this is a real cold start: the
server had nothing to reuse, unlike every cell above it.

### Cold vs hot, and why the gap is smaller than expected

| | Cold (replica) | Hot (uncontended `:8851`) |
|---|---:|---:|
| Decode tok/s | 160.49 | 245.01 |
| Prefill ms | 29.5 | 7.1 |

Hot is ~1.53× the cold decode rate. Two caveats keep this from being read as
a residency win:

- **Not like-for-like.** The replica runs `--parallel 1` because a single
  sample cannot use more than one slot, and fewer slots means a smaller KV
  allocation. Production runs `--parallel 4`. Some of the gap is slot
  configuration, not residency.
- **Load time is a warm *filesystem* cache, not a warm model.** 0.775 s is
  time-to-first-serve with the GGUF already in the OS page cache. A true
  first-boot-after-reboot load would be materially longer. The figure is
  honest for the serving path and must not be quoted as cold-storage load.

`cache_n == 0` plus a 0.775 s load is the useful pair: residency and
prefix reuse are independent, and this configuration gets most of its speed
back from residency without any prefix reuse at all.

## First useful action — `llamacpp_production_1ba5283/first_useful_action.json`

The last unmeasured quantity in the #240 required table. TTFT is time to
first byte; a model can emit its first token in 16 ms and still be wrong.
Earlier work on #240 made this point about `days in a week` (23.9 ms to a
response at 13% verified-correct, implying ~183 ms to first *correct*
response) but never measured it with a harness.

`scripts/first_useful_action.py` measures it: for each case, issue attempts
until one satisfies the suite's own regex oracle, and record the cumulative
wall time at which a verified-correct output first exists.

| Case | Attempts | Time to first correct (s) | TTFT (s) |
| --- | ---: | ---: | ---: |
| days_in_a_week | 1 | 0.097 | 0.043 |
| count_letter_in_word | 1 | 0.150 | 0.018 |
| geometric_sequence | 1 | 0.060 | 0.036 |
| leap_year_length | 1 | 0.072 | 0.033 |
| arithmetic_multistep | 6 | **none** | 0.017 |
| modular_arithmetic | 6 | **none** | 0.019 |
| boolean_logic | 6 | **none** | 0.017 |
| unit_conversion | 6 | **none** | 0.015 |
| rate_word_problem | 6 | **none** | 0.014 |
| capital_city | 6 | **none** | 0.015 |
| distinct_letter_count | 6 | **none** | 0.018 |
| string_reversal | 6 | **none** | 0.018 |

**Median TTFT 16.03 ms vs median time-to-first-correct 0.101 s — TTFT
understates by ~6×** on the cases that are answered at all.

### The sharper statement: for 8 of 12 cases the understatement is unbounded

Eight cases never reach a correct answer. For those there is **no finite
time to a useful action**, so TTFT does not merely overstate by a factor —
it reports a latency for an outcome that never occurs. Any dashboard that
shows TTFT as the user-facing number is reporting success on 4 tasks in 16 ms
and silence on the other 8.

### Retrying cannot help, and the artifact proves it

Serving is deterministic (`temperature=0.0`, fixed seed), so the probe
records whether repeated attempts were byte-identical: **all of them are**.
The retry loop therefore bounds work; it does not explore a distribution.
The 8 failures are **confident, not unlucky** — the same wrong answer every
time. This matches the accuracy baseline exactly: the same 4 cases pass.

That is the useful result. Retrying a deterministic wrong answer is not a
mitigation, so the fix for those 8 is a different model or a different
serving configuration, not a retry policy.

## Tail behaviour at 20 repetitions — `llamacpp_production_1ba5283/tail_20rep/`

Every cell above uses 5 repetitions, which satisfies AC-PERF-031's stated
minimum but **cannot resolve a tail**: under nearest rank, p95 and p99 index
the same observation until N=20. `test_nearest_rank_p95_and_p99_need_twenty_samples_to_separate`
pins that boundary so the 5-rep cells are not mistaken for tail measurements.

This matrix re-runs 1/2/4/8 agents at **20 repetitions**, 128-token prompt,
256 generated tokens, hot config.

> **Contended.** `:8854` was busy on **391 of 425** telemetry samples. The
> absolute level is a lower bound, consistent with the earlier contended
> run. The *tail shape* is the finding here and is not a level claim.

| Agents | Runs | Per-agent p50 | p95 | p99 | Aggregate p50 | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms | E2E p95 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 140.69 | 161.00 | 163.31 | 140.69 | 9.2 | 18.4 | 19.2 | 2.109 |
| 2 | 40 | 115.22 | 135.98 | 141.62 | 230.38 | 14.8 | 23.4 | 24.0 | 2.510 |
| 4 | 80 | 78.57 | 83.59 | 86.14 | 314.27 | 20.8 | 39.2 | 62.5 | 3.392 |
| 8 | 160 | 78.17 | 80.76 | 83.34 | 625.14 | 1570.2 | **3344.7** | 3380.3 | 6.663 |

p95 and p99 are now **distinct at every concurrency level**.

### The 8-agent tail is the thing the 5-repetition runs hid

| Agents | TTFT p50 | p95 / p50 | p99 / p50 |
| ---: | ---: | ---: | ---: |
| 1 | 9.2 ms | 2.00× | 2.09× |
| 2 | 14.8 ms | 1.58× | 1.62× |
| 4 | 20.8 ms | 1.88× | **3.00×** |
| 8 | 1570.2 ms | **2.13×** | 2.15× |

At 8 agents the p95 latency is **2.13× the median** — over 3.3 s against a
1.57 s median. The previous 5-rep cells reported a single value for p95 and
p99, so this distribution was invisible. Any p95/p99 limit declared from
5-rep data would have been a limit on the maximum, not on the tail.

The 4-agent cell is the other thing worth noticing: p99/p50 is **3.00×**,
the sharpest relative tail in the matrix even though its absolute latency is
small. Absolute and relative tail behaviour do not pick the same cell as
worst, so a declared envelope has to say which one it bounds.

Per-agent throughput is essentially flat from 4 to 8 agents (78.57 vs 78.17)
while aggregate doubles (314.27 → 625.14). Eight agents on four slots adds
throughput and queueing, not per-agent speed — and the TTFT p50 of 1.57 s is
that queueing made visible.

## Declared NORMAL service budget — `llamacpp_production_1ba5283/normal_budget.json`

AC-PERF-035 requires promoted settings to "pass capability/verification
gates and remain inside declared p95/p99/resource limits". No limit had ever
been declared for this lane, so nothing could pass or fail one.

`scripts/declare_normal_budget.py` declares the budget in the terms of the
**existing** QoS gate (`oai2.evals.qos.WorkloadBudget` / `evaluate_budget`)
— no parallel budget system was introduced — and evaluates the measured
evidence against it. It does **not** promote anything; promotion changes
what production serves and is an owner decision.

### The budget

| Field | Value | Why |
|---|---:|---|
| `kind` | `SERVICE_BUDGET` | A research target is an aspiration; only a service budget can gate |
| `first_useful_action_p95_ms` | 1000 | 1 s to a *usable* answer is the standard interactive budget |
| `end_to_end_p95_ms` | 5000 | ~2× headroom over the measured 582 ms p95 |
| `end_to_end_p99_ms` | 10000 | ~15× headroom over the measured 679 ms p99 |
| `max_false_success_rate` | 0.02 | A confident wrong answer is the dangerous failure, not slowness |
| `min_verified_success_rate` | 0.90 | A lane right under 90% of the time is not a service |

The values are **not** reverse-engineered from the measurements. They are
stated so a reviewer can argue with them.

### Result: the current NORMAL configuration FAILS, on correctness only

| Metric | Measured | Budget | |
|---|---:|---:|---|
| `first_useful_action_p95` | 160.7 ms | ≤ 1000 | ok |
| `end_to_end_p95` | 581.7 ms | ≤ 5000 | ok |
| `end_to_end_p99` | 678.9 ms | ≤ 10000 | ok |
| `verified_success_rate` | **0.333** | ≥ 0.90 | **FAIL** |
| `false_success_rate` | **0.667** | ≤ 0.02 | **FAIL** |

**Every latency budget is met and every correctness budget is missed.** The
lane is fast and unreliable, which is the precise condition the issue's
promotion rule exists to catch: *"Promote only a configuration that improves
verified useful work, not just benchmark tok/s."* By that rule the current
configuration is not promotable, and no amount of throughput work changes
that.

`test_a_fast_but_wrong_configuration_fails_the_budget` pins this: a
configuration with every latency number comfortably inside its limit is
still rejected, so the gate is not a latency rubber stamp.

### Censored observations

Eight of twelve cases have **no finite** time to a correct answer. Rather
than dropping them — which would report a 100% verified-success rate over
whichever cases happened to be easy — each is recorded at the attempt cap as
a **lower bound**, with `deadline_missed=True`, `verified_success=False` and
`declared_success=True`, which is exactly a false success. The artifact
records the censoring explicitly, and
`test_never_solved_cases_are_recorded_as_false_success_not_dropped` pins it.

### Note on `deadline_miss_rate`

`WorkloadBudget.max_deadline_miss_rate` defaults to `1.0`, so the measured
0.667 passes. That default is permissive by construction. Any budget
intended to actually bound deadline behaviour must set it explicitly; this
one leaves it at the default and says so rather than appearing to check it.
