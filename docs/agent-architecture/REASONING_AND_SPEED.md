# Reasoning Strength and Speed

_Last reviewed: 2026-10-01._

## Modes

| Mode | Target role | Current state |
| --- | --- | --- |
| FURIOUS | minimum-compute direct action | scaffold |
| NORMAL | routine coding | scaffold |
| DEEP | bounded branching/verification | scaffold |
| SWARM | parallelizable hard tasks | scaffold |

## Research throughput targets

- FURIOUS: **500–800 tok/s**
- NORMAL: **300–600 tok/s**
- DEEP: **150–350+ tok/s/lane**
- SWARM: **600–1,500+ aggregate tok/s**

These are targets, not current OAI-2.0 model results.

## Corrected measured smoke result

The v0.2 harness measures prefill and decode separately. One committed M5 Max smoke run for `Qwen2.5-0.5B-Instruct-4bit` recorded:

| Metric | Result |
| --- | ---: |
| Prompt tokens | 133 |
| Generated tokens | 32 |
| Prefill/TTFT | ~25.44 ms |
| Prefill throughput | ~5,227.9 tok/s |
| Pure decode throughput | ~198.47 tok/s |
| Decode duration | ~165.66 ms |
| End-to-end generation phase | ~191.10 ms |
| Peak MLX memory | ~0.402 GB |

One repetition is not enough to establish a stable baseline.

## Superseded measurements

The earlier ~102–334 tok/s v0.1 values measured short end-to-end generations with prefill included in the interval. Keep them for historical bootstrap traceability only.

## Speed strategy

- sparse expert activation;
- adaptive depth/early exit;
- compact control actions;
- speculative decoding;
- multi-token prediction;
- prompt/KV reuse;
- cached repository world state;
- shared visual representations;
- compact Cloudflare evidence packages;
- selective quantization;
- kernel/runtime work only after profiling.

## Primary metric

For an agent, time-to-first-useful-action and verified task completion per unit time matter more than decorative prose throughput.
