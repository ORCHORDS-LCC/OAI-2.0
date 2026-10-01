<p align="center">
  <img src="./assets/branding/orchords-banner.jpg" width="1080" alt="ORCHORDS — BUILD DIFFERENT.">
</p>

# OAI-2.0

[![License: Non-Commercial](https://img.shields.io/badge/license-non--commercial-red.svg)](./LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)](./docs/agent-architecture/README.md)

> ⭐ If OAI-2.0 is useful or interesting, consider starring the repository.

**ORCHORDS — BUILD DIFFERENT.**

OAI-2.0 is a public next-generation coding-agent research project focused on **large intelligence capacity with dynamically small active compute**: code + vision reasoning, native dynamic tools, evidence-driven verification, adaptive reasoning depth, and conditional multi-agent execution.

## Current state — 2026-10-01

OAI-2.0 is no longer documentation-only. The repository contains an **EXPERIMENTAL implementation scaffold**, but the final custom model remains **PROPOSED**.

The master implementation map is [Issue #1](https://github.com/ORCHORDS-LCC/OAI-2.0/issues/1). This project intentionally uses **runner-free local verification**; GitHub Actions runners are not part of the acceptance workflow.

| Area | Current status |
| --- | --- |
| Python package, typed core/protocols | **EXPERIMENTAL** |
| Six-gate tool dispatcher | **EXPERIMENTAL** |
| FURIOUS / NORMAL / DEEP / SWARM controllers | **EXPERIMENTAL scaffold** |
| Evidence graph / claim state | **EXPERIMENTAL** |
| Four-view vision data model | **EXPERIMENTAL abstraction** — no trained vision encoder yet |
| Multi-agent orchestrator interfaces | **EXPERIMENTAL scaffold** — no production swarm runtime yet |
| Capability-eval harness | **EXPERIMENTAL** |
| MLX benchmark harness | **EXPERIMENTAL**, corrected prefill/decode split |
| Cloudflare knowledge contract + mocks | **PROPOSED live / tested mock contract** |
| q-pipe import gate | **EXPERIMENTAL** and aligned to verified export rules |
| Live Cloudflare Worker transport | **PROPOSED** |
| Vectorize-backed semantic retrieval | **PROPOSED** |
| Custom 10–30B+ specialist model | **PROPOSED** |
| Android Studio / Hermes / OpenCode adapters | **PROPOSED** |

## Architecture target

The current source of truth is [ARCHITECTURE_TARGET.md](docs/agent-architecture/ARCHITECTURE_TARGET.md).

```text
Total specialist capacity:  ~10–30B+ eventually
FURIOUS active compute:     ~500M–1B
NORMAL active compute:      ~1–2B
DEEP active compute:        ~2–4B
SWARM:                      multiple ~1–2B+ lanes
```

These are research ranges, not a claim about a trained model currently in this repository.

## Current measured MLX smoke evidence

The corrected v0.2 benchmark harness has one committed smoke result for `mlx-community/Qwen2.5-0.5B-Instruct-4bit` on the M5 Max:

- prompt: 133 tokens;
- generated: 32 tokens;
- prefill / TTFT: ~25.44 ms;
- prefill throughput: ~5,227.9 tok/s;
- pure decode throughput: ~198.47 tok/s;
- end-to-end generation phase: ~191.10 ms;
- peak MLX memory: ~0.402 GB.

This is **one smoke repetition**, not an architecture benchmark. See [evals/benchmarks/README.md](evals/benchmarks/README.md).

The earlier ~102–334 tok/s bootstrap numbers used an older end-to-end measurement method and remain historical only.

## Knowledge architecture

OAI-2.0 keeps long-tail factual/project knowledge outside the final model weights.

```mermaid
flowchart TD
    A[OAI-2.0 agent] --> K[KnowledgeStore]
    K --> W[Cloudflare Worker transport - proposed]
    W --> D[D1 metadata/index]
    W --> R[R2 content-addressed bodies]
    W --> V[Vectorize semantic index]
    W --> C[KV query cache]
    Q[q-pipe verified learning] --> I[Strict import gate]
    I --> K
```

Current source implements the application-level contract, deterministic mock bindings, cache revisioning, and a strict q-pipe importer. It does **not** yet make live Cloudflare network calls.

The committed 50-row pilot is **synthetic test data**, not a real q-pipe corpus migration.

See [CLOUDFLARE_KNOWLEDGE.md](docs/agent-architecture/CLOUDFLARE_KNOWLEDGE.md).

## q-pipe import safety

Default imports now mirror q-pipe's Cloudflare export rules:

- default sources: `scenario-forge`, `terminal-bench-2.1`;
- `android-curriculum-oss` requires explicit opt-in;
- promoted rows only by default;
- positive independent verification;
- success count must cover verification and exceed failure count;
- bounded fingerprint and structured guidance;
- unsafe guidance is rejected;
- accepted guidance receives a deterministic content hash.

## Documentation map

| Topic | Document |
| --- | --- |
| Architecture index | [docs/agent-architecture/README.md](docs/agent-architecture/README.md) |
| Architecture target | [ARCHITECTURE_TARGET.md](docs/agent-architecture/ARCHITECTURE_TARGET.md) |
| System architecture | [SYSTEM_ARCHITECTURE.md](docs/agent-architecture/SYSTEM_ARCHITECTURE.md) |
| Cloudflare knowledge | [CLOUDFLARE_KNOWLEDGE.md](docs/agent-architecture/CLOUDFLARE_KNOWLEDGE.md) |
| Tool calling | [TOOL_CALLING.md](docs/agent-architecture/TOOL_CALLING.md) |
| Vision/UI | [VISION_AND_UI.md](docs/agent-architecture/VISION_AND_UI.md) |
| Multi-agent | [MULTI_AGENT.md](docs/agent-architecture/MULTI_AGENT.md) |
| Reasoning/speed | [REASONING_AND_SPEED.md](docs/agent-architecture/REASONING_AND_SPEED.md) |
| Training/evaluation | [TRAINING_AND_EVALUATION.md](docs/agent-architecture/TRAINING_AND_EVALUATION.md) |
| Verification | [VERIFICATION_AND_EVIDENCE.md](docs/agent-architecture/VERIFICATION_AND_EVIDENCE.md) |
| Host compatibility | [HOST_COMPATIBILITY.md](docs/agent-architecture/HOST_COMPATIBILITY.md) |
| Security/sandboxing | [SECURITY_AND_SANDBOXING.md](docs/agent-architecture/SECURITY_AND_SANDBOXING.md) |
| Roadmap | [ROADMAP.md](docs/agent-architecture/ROADMAP.md) |
| Public sources | [SOURCES.md](docs/agent-architecture/SOURCES.md) |
| Issue/traceability standard | [ENGINEERING_ISSUE_STANDARD.md](docs/agent-architecture/ENGINEERING_ISSUE_STANDARD.md) |

## ⭐ Star OAI-2.0

⭐ https://github.com/ORCHORDS-LCC/OAI-2.0

Stars are voluntary and do not provide access, commercial rights, or control over project decisions.

## Donations & sponsorship

To support ORCHORDS public engineering, documentation, or research, contact **crm@orchords.com**.

Donations/sponsorship do not grant commercial rights, roadmap control, guaranteed implementation, or private access. Commercial licensing is separate and must be agreed in writing.

## License

**Non-commercial use only.** See [LICENSE](LICENSE).

## Contact

Public contact: **crm@orchords.com**

## Branding

See [BRANDING.md](BRANDING.md) and [assets/branding/](assets/branding/README.md).

**ORCHORDS — BUILD DIFFERENT.**
