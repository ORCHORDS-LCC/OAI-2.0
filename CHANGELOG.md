# Changelog

All notable public changes to OAI-2.0 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Async D1 knowledge reader/query contracts and focused tests for the evolving live Cloudflare adapter surface.
- Safe session-batching scheduler core with exact-compatibility isolation tests for concurrent inference groundwork.

- Repository-wide Markdown reconciliation against current source/issues on 2026-10-02, including corrected closed/open status, Cloudflare progress, benchmark terminology, runner-free verification policy, and WP-76 mapping.
- Async R2 and KV binding wrappers plus D1 knowledge-table/schema export coverage for the evolving live Cloudflare adapter surface.

- Expanded standards-aligned work-package map through WP-76, including claim-level evidence enforcement/hallucination resistance and end-to-end service-quality budgets.
- D1-authoritative GC deletion-lease state contract with expected-revision acquisition, writer exclusion, expired-owner takeover/fencing, retryable failure, idempotent finalize/release, deleted-body tombstones and verified restore semantics.
- Reference-safe, non-destructive R2 liveness reconciliation with shared-body grouping, missing/orphan classification, byte/age metrics, resumable pagination and authoritative-reference fingerprinting.
- Conservative R2 orphan sweep decision core with grace windows, immediate pre-delete D1 recheck, dry-run default, explicit authorization/recovery gates, idempotent absence handling, bounded failure resume and integrity-checked checkpoints.
- Versioned public-safe Cloudflare knowledge transport schemas for request/response, normalized auth context, explicit errors, D1 metadata, R2 body descriptors, Vectorize metadata and KV cache envelopes.
- Regression tests for transport contract versioning, operation requirements, response invariants, content-addressed R2 integrity, Vectorize provenance and revisioned KV envelopes.
- q-pipe compatibility source revision/blob markers tied to the exact public q-pipe source used for importer-policy verification.
- Runner-free local preflight at `scripts/verify.py` covering Ruff, MyPy, Pytest, public-safety and Markdown-link checks.
- Standards-aligned master issue hierarchy and engineering issue/traceability nomenclature.
- Experimental OAI-2.0 Python package scaffold for protocols, reasoning, tools, agents, verification, vision abstractions, knowledge, runtime, and evaluation.
- Corrected MLX benchmark harness with separate load, compile/warm-up, prefill/TTFT, decode, end-to-end, and memory metrics.
- Offline capability-evaluation scaffolds for coding, tool use, bug diagnosis, reasoning, verification, vision, and orchestration.
- Application-level Cloudflare knowledge contract using logical D1/R2/Vectorize/KV roles plus deterministic mock bindings.
- Strict q-pipe import policy aligned to q-pipe's Cloudflare export gate.
- Synthetic 50-row q-pipe import round-trip test.
- Public architecture, branding, support, security, contribution, stars, and donation/sponsorship documentation.

### Changed

- WP-01 verification baseline and WI-VV-001/WI-VV-002 are now completed/closed; q-pipe compatibility WI-MIG-001 is also closed, while the real migration pilot remains open.
- R2 liveness reconciliation WI-GC-001 (#214) is closed; destructive sweep (#215) and D1 deletion-lease/live-concurrency work (#233) remain open.
- Benchmark documentation now calls the v0.2 decode-rate field **MLX-reported generation throughput** rather than overstating it as kernel-level “pure decode.”

- Master Issue #1 is the canonical global map through WP-76 and issue boundary #233, with parent WP issues owning detailed work-item registries.
- Runner-free preflight reports explicit Apple-Silicon PASS/SKIP state and verifies a real sibling/configured q-pipe checkout against pinned compatibility revision/blob hashes when available.
- Removed GitHub Actions runner-backed workflows; OAI-2.0 acceptance is local/manual-first and runner-free.
- Architecture target supersedes the original small-model-only concept: OAI-2.0 targets **10–30B+ total specialist capacity** with difficulty-dependent active compute.
- q-pipe knowledge imports default to **promoted, independently verified, quality-gated rows only**.
- Android curriculum imports require explicit opt-in.
- Cloudflare query caches are revision-keyed so writes invalidate earlier logical cache entries.
- Cloudflare documentation distinguishes application-level contracts from asynchronous live Worker binding APIs.
- Benchmark v0.1 numbers are historical bootstrap evidence, not pure decode evidence.

### Fixed

- Closed R2 sweep race windows by adding a second authoritative reference check immediately before deletion and retiring re-referenced candidates until a fresh dry-run/grace cycle.
- Added a D1 claim/writer-exclusion contract so live integration can fence the remaining cross-service D1-reference/R2-delete race rather than treating one pre-delete lookup as atomic.
- Sweep runtime rejects non-boolean dependency results instead of coercing them into destructive decisions.
- Sweep checkpoints fingerprint full state, including cursor/history/retired candidates, so tampered resume state is rejected.
- Dry-run GC page ingestion is atomic; snapshot booleans/cursors/inventory values are strictly validated; resumed scans can be checked against the current authoritative reference set.
- Repaired the local preflight hard-coded-secret regex and excluded generated environments/caches/build trees from repository scans.
- Removed stale claim that mock Cloudflare methods map 1:1 to live Worker APIs.
- Removed mock-internal assumptions from `KnowledgeStore.all()`.
- Fixed q-pipe dedupe ordering so an ineligible row cannot suppress a later eligible row with the same identity.
- Fixed q-pipe guidance hashing to match the verified exported guidance shape.
