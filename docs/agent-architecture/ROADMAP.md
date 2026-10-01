# Implementation Roadmap

_Last reviewed: 2026-10-01._

Stages are evidence gates, not delivery promises. The detailed dependency-ordered implementation map lives in [GitHub Issue #1](https://github.com/ORCHORDS-LCC/OAI-2.0/issues/1).

OAI-2.0 uses **runner-free local verification**. Do not make GitHub Actions runners part of an acceptance gate.

| Stage | Scope | Current status |
| --- | --- | --- |
| 0 | Public architecture/contracts | **IMPLEMENTED docs** |
| 1 | Host-neutral protocol/tool types | **EXPERIMENTAL scaffold** |
| 2 | Fast single-agent controllers | **EXPERIMENTAL scaffold** |
| 3 | Vision/UI grounding | **EXPERIMENTAL data model; real grounding PROPOSED** |
| 4 | Evidence verification | **EXPERIMENTAL scaffold** |
| 5 | Native multi-agent | **EXPERIMENTAL orchestration scaffold; real swarm PROPOSED** |
| 6 | Adaptive routing/depth/custom model | **PROPOSED** |
| 7 | Host adapters | **PROPOSED** |
| 8 | Production hardening | **IN PROGRESS foundations; production PROPOSED** |

## Knowledge track

| Knowledge milestone | Status |
| --- | --- |
| KnowledgeStore abstraction | **EXPERIMENTAL** |
| In-memory deterministic store | **EXPERIMENTAL** |
| Application-level Cloudflare contract | **EXPERIMENTAL contract / PROPOSED live** |
| Cache revision invalidation | **EXPERIMENTAL** |
| Strict q-pipe verified import gate | **EXPERIMENTAL** |
| Synthetic 50-row round trip | **IMPLEMENTED test** |
| Real q-pipe corpus migration | **PROPOSED** |
| Live Worker + D1/R2/KV | **PROPOSED** |
| Production Vectorize semantic retrieval | **PROPOSED** |

## Immediate next evidence gates

1. run full corrected test/lint/type suite after adapter changes;
2. implement and test a live Worker transport without committing secrets;
3. deploy public-safe D1/R2/Vectorize/KV schemas/bindings privately;
4. perform a small real q-pipe promoted/verified migration;
5. add live round-trip retrieval/evidence tests;
6. collect repeated M5 Max baseline benchmark statistics;
7. prototype routing/adaptive-depth behavior at small model scale;
8. only then scale architecture experiments.

A capability is promoted only when code, tests, held-out evaluation, safety checks, and limitations are recorded.
