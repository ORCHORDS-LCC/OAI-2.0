# Reasoning Strength and Speed

_Reconciliation baseline: `c4a1f6f6134bd18668b8a621a2c0f0f89708e5e7` (source state before this documentation commit)._

_Last reviewed: 2026-10-02._

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

The v0.2 harness records first-token timing separately from the subsequent generation interval and MLX-LM-reported generation throughput. One committed M5 Max smoke run for `Qwen2.5-0.5B-Instruct-4bit` recorded:

| Metric | Result |
| --- | ---: |
| Prompt tokens | 133 |
| Generated tokens | 32 |
| Prefill/TTFT | ~25.44 ms |
| Prefill throughput | ~5,227.9 tok/s |
| MLX-reported generation throughput | ~198.47 tok/s |
| Decode duration | ~165.66 ms |
| End-to-end generation phase | ~191.10 ms |
| Peak MLX memory | ~0.402 GB |

One repetition is not enough to establish a stable baseline, and the generation-throughput field is not a kernel-only decode claim.

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

## Inference/session scheduling status

Current source now includes a safe session-batching scheduler core with isolation-focused tests. This is infrastructure progress for concurrent inference; it is **not** evidence that the final OAI-2.0 model reaches the research throughput targets above.

## Primary metric

For an agent, time-to-first-useful-action and verified task completion per unit time matter more than decorative prose throughput.


## End-to-end QoS and admission control

Raw decode throughput is not the service objective. Current source now includes `oai2/evals/qos.py` for versioned workload/service budgets and `oai2/runtime/admission.py` for a deterministic admission/backpressure policy core.

An admission→scheduler bridge now exists in source so admission decisions can feed the exact-compatibility safe batching scheduler. This is still not live MLX/service proof.

The current policy surface:

- uses the canonical FURIOUS / NORMAL / DEEP / SWARM modes;
- accounts for total, used, and reserved memory;
- bounds active tasks and queue depth;
- computes SWARM aggregate memory from per-lane cost;
- supports optional deadlines;
- produces explicit ADMIT / QUEUE / REJECT outcomes with public-safe reasons;
- prefers short/deadline-sensitive work before starvation;
- promotes long-waiting work after a bounded starvation threshold;
- removes cancelled queued work immediately.

Deterministic tests cover overload, queue-full behavior, SWARM aggregate memory, deadline expiry, starvation prevention, and cancellation. Current source also includes `SafeBatchScheduler`, which batches only requests with identical model/tokenizer/prefix/tool-schema/world-state/security compatibility keys, keeps one queued request per session, bounds batch size, and isolates cancellation; since 50a029dd it is bound to the WP-39 service request surface via `oai2.runtime.service_binding`, which adds session/inference protocol routes and batch/queue/latency metrics. Admission control and the live MLX execution loop are still **not integrated** with safe batching, and neither admission nor safe batching has actual model/expert residency accounting or sustained target-hardware tail-latency testing.

Promotion now has an explicit source-level regression gate: candidate p95/p99 and deadline-miss rate are compared against the baseline and can block promotion even when mean token throughput improves. False-success/verified-work service budgets remain separate mandatory gates.
