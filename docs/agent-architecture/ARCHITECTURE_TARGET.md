# Architecture target

> **Status: PROPOSED.** This document is the source of truth for the
> OAI-2.0 model architecture and routing strategy. Earlier notes that
> described the system as a "compact ~4–8B equivalent" are superseded by
> this document.

## Design principle

> **LARGE TOTAL INTELLIGENCE CAPACITY, SMALL DYNAMIC INSTANTANEOUS COMPUTE.**

OAI-2.0 must not become a permanently tiny model plus retrieval. It must
become a **large-capacity, dynamically activated, multimodal coding
intelligence** whose active compute scales with problem difficulty.

## Long-term parameter targets

These are **research ranges**, not fixed final counts.

```text
Total specialist capacity:      ~10–30B+ eventually
FURIOUS active compute:         ~500M–1B
NORMAL active compute:          ~1–2B
DEEP   active compute:          ~2–4B
SWARM  aggregate:               multiple independent ~1–2B+ lanes
```

A model that is fast but loses serious reasoning ability is a failure.
Every optimization is evaluated against **verified capability** *and*
**performance**, and rejected if it materially harms:

- coding accuracy;
- reasoning;
- architecture understanding;
- tool selection;
- unseen tool-schema generalization;
- visual grounding;
- UI debugging;
- long-horizon task completion;
- verification;
- correct stopping behavior.

## Dynamic expert routing

The model exposes a large pool of specialists and routes to the relevant
subset per state and per token. Only the relevant experts execute;
non-relevant experts are not computed for that token.

Initial expert families include:

- shared reasoning
- code
- debugging
- architecture
- build systems
- tool use
- planning
- verification
- agent orchestration
- vision
- UI
- UX / accessibility
- source / render mapping

Routing is not "one expert per token"; it is a small subset, and the
subset changes across tokens and across reasoning modes.

## Adaptive depth

Easy operations may terminate after fewer layers; hard reasoning
continues through deeper layers. Conceptual layout:

```text
easy:    layers 1 → 8/12   → exit
normal:  layers 1 → 20/24
hard:    layers 1 → full depth
```

Exact thresholds are empirical and are not hard-coded before
experiments. The runtime must support both early-exit and full-depth
forward passes for the same model.

## Reasoning-mode routing

| Mode    | Active compute target | Latency target (research) | Use |
| ------- | --------------------- | ------------------------- | --- |
| FURIOUS | ~500M–1B              | 500–800 tok/s             | Compact tool decisions, direct action. |
| NORMAL  | ~1–2B                 | 300–600 tok/s             | Routine coding, reading, editing. |
| DEEP    | ~2–4B                 | 150–350+ tok/s            | Hard reasoning, branching, verification. |
| SWARM   | multiple ~1–2B+ lanes | 600–1,500+ aggregate tok/s | Long-horizon / parallelizable problems. |

These are **goals**, not guarantees. Real measured numbers override them.

## Dense baselines

Architecture experiments must be benchmarked against dense reference
models of comparable sizes. At minimum:

- ~0.5–1B dense
- ~1–2B dense
- ~3–4B dense
- a strong available coding baseline

Comparisons keep equal or clearly recorded:

- task sets;
- contexts;
- quantization;
- time budgets;
- hardware;
- verification criteria.

Tracked metrics:

```text
verified success / task
latency
TTFT
decode tok/s
prefill tok/s
memory
tool calls
reasoning-mode choice
vision success
regression rate
```

## Speed strategy (does not replace capacity)

The following continue to be pursued — but **only** as long as they
preserve verified capability:

- sparse expert activation;
- adaptive depth (early-exit);
- compact action tokens;
- multi-token prediction;
- speculative decoding;
- prompt / KV cache reuse;
- cached repository world state;
- shared visual embeddings;
- Cloudflare knowledge retrieval;
- selective quantization;
- runtime / kernel optimization **only after profiling**.

Architecture intelligence takes priority over arbitrary raw tok/s.

## Cloudflare knowledge — split of responsibilities

Model weights primarily carry:

- abstraction;
- language;
- coding;
- reasoning;
- planning;
- vision;
- tool behavior;
- agent orchestration;
- verification.

Cloudflare / pipeline provides:

- durable knowledge;
- provenance;
- project memory;
- verified lessons;
- documentation;
- reusable evidence;
- retrieved technical facts.

External knowledge does **not** replace reasoning capacity.

## Final rule

OAI-2.0 must not become "a tiny model with RAG." It must become "a
large-capacity, dynamically activated, multimodal coding intelligence
system whose compute scales with problem difficulty."
