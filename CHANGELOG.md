# Changelog

All notable public changes to OAI-2.0 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Runner-free local preflight at `scripts/verify.py` covering Ruff, MyPy, Pytest, public-safety and Markdown-link checks.
- Standards-aligned master issue hierarchy with work packages/work items through core architecture, data/training, inference, observability, resource control, supply-chain and configuration domains.

- Experimental OAI-2.0 Python package scaffold for protocols, reasoning, tools, agents, verification, vision abstractions, knowledge, runtime, and evaluation.
- Corrected MLX benchmark harness with separate load, compile/warm-up, prefill/TTFT, decode, end-to-end, and memory metrics.
- Offline capability-evaluation scaffolds for coding, tool use, bug diagnosis, reasoning, verification, vision, and orchestration.
- Runner-free local verification is the canonical lint/test/type/public-safety workflow.
- Application-level Cloudflare knowledge contract using logical D1/R2/Vectorize/KV roles plus deterministic mock bindings.
- Strict q-pipe import policy aligned to q-pipe's Cloudflare export gate.
- Synthetic 50-row q-pipe import round-trip test.
- Public architecture, branding, support, security, contribution, stars, and donation/sponsorship documentation.

### Changed

- Removed GitHub Actions runner-backed workflows; OAI-2.0 acceptance is local/manual-first and runner-free.
- Added master implementation map in GitHub Issue #1.

- Architecture target supersedes the original small-model-only concept: OAI-2.0 targets **10–30B+ total specialist capacity** with difficulty-dependent active compute.
- q-pipe knowledge imports now default to **promoted, independently verified, quality-gated rows only**.
- Android curriculum imports now require explicit opt-in.
- Cloudflare query caches are revision-keyed so writes invalidate earlier logical cache entries.
- Cloudflare documentation now distinguishes application-level contracts from Cloudflare's asynchronous Worker binding APIs.
- Benchmark v0.1 numbers are historical bootstrap evidence, not pure decode evidence.

### Fixed

- Repaired the local preflight hard-coded-secret regex and excluded generated environments/caches/build trees from repository scans.

- Removed stale claim that the mock Cloudflare methods map 1:1 to live Worker APIs.
- Removed mock-internal assumptions from `KnowledgeStore.all()`.
- Fixed q-pipe dedupe ordering so an ineligible row cannot suppress a later eligible row with the same identity.
- Fixed q-pipe guidance hashing to match the verified exported guidance shape.
