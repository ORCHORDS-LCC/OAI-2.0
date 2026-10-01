# System Architecture

_Reconciliation baseline: `c4a1f6f6134bd18668b8a621a2c0f0f89708e5e7` (source state before this documentation commit)._

_Last reviewed: 2026-10-02._

## Goal

Build a portable multimodal coding intelligence that treats code, language, UI state, screenshots, tools, test/runtime evidence, external knowledge, and subagents as one evolving software world.

## Current versus target

```mermaid
flowchart TB
    H[IDE / agent host] --> P[Host protocol adapters - PROPOSED]
    P --> C[OAI-2.0 core model - PROPOSED]
    C --> R[Reasoning controllers - EXPERIMENTAL scaffold]
    C --> V[Vision encoder - PROPOSED]
    C --> T[Tool policy/dispatch - EXPERIMENTAL]
    C --> A[Agent orchestration - EXPERIMENTAL scaffold]
    C --> E[Evidence graph - EXPERIMENTAL]
    C --> K[AsyncCloudflareKnowledgeRuntime + source adapters - EXPERIMENTAL]
    K --> CF[Authenticated/deployed Cloudflare Worker path - NOT YET VERIFIED]
    C --> QOS[QoS/admission + safe batching + admission/scheduler bridge - EXPERIMENTAL]
```

## Assembled source-level knowledge runtime

The knowledge path now has source-level async orchestration across D1 authority, R2 bodies, Vectorize semantic matches, and best-effort KV cache, plus an authenticated Worker transport handler/component factory/public-safe Python Worker entrypoint. This materially advances WP-02, but it remains a source/test implementation until authenticated Worker deployment and private end-to-end Cloudflare evidence exist.

## Implemented scaffold

Current source contains typed protocol/tool models, reasoning controllers, agent specs/orchestrator interfaces, evidence graphs, vision-state abstractions, knowledge-store abstractions, source-level D1/R2/KV/Vectorize wrappers plus GC lease/delete primitives, corrected benchmark tooling, capability evals, QoS/admission logic, and a safe session-batching scheduler core.

It does **not** contain the trained final multimodal model.

## Current implementation caveat

Source-level contracts, wrappers, focused tests, or scheduler cores are not equivalent to a production deployment. Live Cloudflare transport, final model inference, real vision grounding, host integrations, and production swarm execution remain unverified end-to-end.

## World state

The target world state covers:

- repository/symbol graph;
- build/test state;
- runtime/device state;
- UI/visual state;
- evidence/claim graph;
- external knowledge references;
- active agents and budgets;
- change/commit state.

Large shared artifacts should be referenced rather than repeatedly pasted into every agent context.

## Model objective

The architecture should learn more than next-token generation:

```text
(current world, goal, source, visual state, evidence, history)
 -> next action
 -> expected result
 -> verification method
 -> confidence/evidence status
```

See [ARCHITECTURE_TARGET.md](ARCHITECTURE_TARGET.md) and [CLOUDFLARE_KNOWLEDGE.md](CLOUDFLARE_KNOWLEDGE.md).
