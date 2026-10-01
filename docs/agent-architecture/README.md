# OAI-2.0 Agent Architecture

_Last reviewed: 2026-10-02._

This folder is the public architecture source for OAI-2.0. It now documents both the **PROPOSED final architecture** and the **EXPERIMENTAL scaffold that exists on current main**.

## Status language

| Label | Meaning |
| --- | --- |
| **IMPLEMENTED** | Code plus repository evidence exists. |
| **EXPERIMENTAL** | Prototype/scaffold exists but is not production-supported. |
| **PROPOSED** | Design target only. |
| **BLOCKED** | Waiting on a named dependency or missing proof. |

## Current implementation snapshot

Experimental source currently includes:

- typed core/protocol models;
- dynamic tool definitions and a six-gate dispatcher;
- FURIOUS/NORMAL/DEEP/SWARM controller scaffolds;
- multi-agent orchestration interfaces;
- evidence/claim state models;
- four-view vision abstractions;
- corrected MLX benchmark harness;
- offline capability-evaluation scaffolds;
- application-level Cloudflare knowledge contract + deterministic mocks;
- versioned public-safe Cloudflare transport/schema models plus source-level async D1 reader/writer and R2/KV/Vectorize binding wrappers;
- reference-safe R2 reconciliation and conservative sweep core;
- deterministic GC lease state model plus versioned D1 lease schema, async binding-facing adapter, durable D1 knowledge-index/corpus-revision writer, and async D1/R2 delete boundary;
- QoS workload/tail/useful-work/deadline metrics, deterministic admission/backpressure policy core, exact-compatibility safe batching scheduler, and p95/p99/deadline promotion regression gate;
- strict q-pipe verified-import policy.

Not yet implemented as production capabilities:

- the custom 10–30B+ specialist model;
- trained dynamic expert router/adaptive depth;
- real multimodal vision encoder;
- real parallel swarm inference;
- full live Cloudflare Worker transport (D1 metadata/revision writer plus R2/KV/Vectorize binding wrappers and a D1/R2 GC delete boundary exist, but the authenticated Worker endpoint and end-to-end transport are not verified);
- production Vectorize semantic retrieval;
- Android Studio/Hermes/OpenCode integrations.

## Read order

1. [Architecture target](ARCHITECTURE_TARGET.md)
2. [System architecture](SYSTEM_ARCHITECTURE.md)
3. [Cloudflare knowledge](CLOUDFLARE_KNOWLEDGE.md)
4. [Tool calling](TOOL_CALLING.md)
5. [Reasoning and speed](REASONING_AND_SPEED.md)
6. [Vision and UI](VISION_AND_UI.md)
7. [Multi-agent orchestration](MULTI_AGENT.md)
8. [Verification and evidence](VERIFICATION_AND_EVIDENCE.md)
9. [Training and evaluation](TRAINING_AND_EVALUATION.md)
10. [Host compatibility](HOST_COMPATIBILITY.md)
11. [Security and sandboxing](SECURITY_AND_SANDBOXING.md)
12. [Roadmap](ROADMAP.md)
13. [Sources](SOURCES.md)
14. [Engineering issue/traceability standard](ENGINEERING_ISSUE_STANDARD.md)

## Design principle

> **LARGE TOTAL INTELLIGENCE CAPACITY, SMALL DYNAMIC INSTANTANEOUS COMPUTE.**

The goal is not a tiny model with retrieval. The target is a large-capacity multimodal coding intelligence whose active experts/depth scale with task difficulty.


## Implementation issue hierarchy

- **Master implementation map:** [Issue #1](https://github.com/ORCHORDS-LCC/OAI-2.0/issues/1)
- Detailed work packages and work items use `WP-*`, `WI-*`, `REQ-*`, `AC-*`, `VER-*`, `RISK-*`, `DEP-*`, and `EVID-*` identifiers.
- The issue hierarchy is canonical for dependency/acceptance tracking; Markdown describes the architecture and current public-safe state.
- OAI-2.0 remains **runner-free**: no GitHub-hosted or self-hosted runner is an acceptance mechanism.
