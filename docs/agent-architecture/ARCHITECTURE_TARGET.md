# Architecture Target

_Reconciliation baseline: `8520cdb5d8d3e49bc9e12dfd79ccf59e7422e44d` (source state before this documentation commit)._

_Last reviewed: 2026-10-02._

> **Final architecture status: PROPOSED.**  
> **Implementation scaffolding status: EXPERIMENTAL.**

This document supersedes earlier compact-4–8B-only concepts.

## Design principle

> **LARGE TOTAL INTELLIGENCE CAPACITY, SMALL DYNAMIC INSTANTANEOUS COMPUTE.**

OAI-2.0 should preserve substantial reasoning capacity while avoiding unnecessary parameter/layer execution on easy work.

## Research ranges

```text
Total specialist capacity:      ~10–30B+ eventually
FURIOUS active compute:         ~500M–1B
NORMAL active compute:          ~1–2B
DEEP active compute:            ~2–4B
SWARM:                          multiple independent ~1–2B+ lanes
```

These are targets, not a statement about weights currently trained in this repository.

## Dynamic expert routing

Candidate expert families:

- shared reasoning;
- code;
- debugging;
- architecture;
- build systems;
- tools;
- planning;
- verification;
- orchestration;
- vision;
- UI;
- UX/accessibility;
- source/render mapping.

Routing should select a small relevant subset by state/token rather than execute every specialist.

## Adaptive depth

Easy operations may exit after fewer layers; harder reasoning uses deeper computation. Exact exit points must be learned/measured, not hard-coded from diagrams.

## Reasoning modes

| Mode | Target active compute | Research throughput target | Current implementation |
| --- | ---: | ---: | --- |
| FURIOUS | ~500M–1B | 500–800 tok/s | controller scaffold |
| NORMAL | ~1–2B | 300–600 tok/s | controller scaffold |
| DEEP | ~2–4B | 150–350+ tok/s/lane | branch/select scaffold |
| SWARM | multiple ~1–2B+ lanes | 600–1,500+ aggregate tok/s | orchestration scaffold |

Throughput targets are not guarantees. Capability regressions invalidate a speed optimization.

## Capability gate

Reject an optimization if it materially harms coding, reasoning, architecture understanding, dynamic tool use, unseen-schema generalization, visual grounding, UI debugging, long-horizon completion, verification, or correct stopping.

## Current implementation boundary

The final sparse/adaptive model is still **PROPOSED**. Supporting truth/evidence infrastructure has advanced with a versioned claim-evidence policy, but that is not a trained reasoning model or proof of hallucination resistance. Current `main` contains supporting experimental infrastructure—runner-free verification, controller/tool/evidence scaffolds, knowledge/GC/QoS primitives, and a safe session-batching scheduler core—but no trained 10–30B+ specialist-capacity OAI-2.0 model.

## Numerical-safety foundation

WP-71 now has a versioned reference precision/sentinel policy, a reference-vs-optimized tolerance/fallback matrix, and a canonical backend/export/kernel promotion gate. The gate carries the applicable tolerance profile and artifact identity and refuses speed-only promotion. The MLX smoke probe consumes the sentinel path, but real backend/quantized/custom-kernel numerical equivalence, overhead and fallback measurements remain open.

## Current evidence

The corrected MLX harness has one committed v0.2 smoke result on the M5 Max for a 0.5B 4-bit Qwen model: ~198.47 tok/s **MLX-reported generation throughput** with a 133-token prompt and 32 output tokens. This is not a claim of kernel-only decode throughput. This is infrastructure evidence, not architecture proof.

See [benchmark notes](../../evals/benchmarks/README.md).

## Knowledge split

**Weights:** abstraction, language, coding, reasoning, planning, vision, tool behavior, orchestration, verification.

**External knowledge:** durable facts, project memory, verified lessons, documentation, provenance, reusable evidence.

See [CLOUDFLARE_KNOWLEDGE.md](CLOUDFLARE_KNOWLEDGE.md).

## Final rule

OAI-2.0 must not become a permanently tiny model plus RAG. It should become a dynamically activated multimodal coding intelligence system whose compute scales with problem difficulty.
