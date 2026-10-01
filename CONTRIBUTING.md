# Contributing to OAI-2.0

_Last reviewed: 2026-10-01._

OAI-2.0 is a public, non-commercial research and engineering repository. Read current source before treating any architecture document as implemented behavior.

## Status vocabulary

- **IMPLEMENTED** — code and repository evidence exist.
- **EXPERIMENTAL** — prototype/scaffold exists but is not production-supported.
- **PROPOSED** — design target only.
- **BLOCKED** — explicitly waiting on a dependency or proof.

## Before changing anything

1. Fetch and inspect current `main`.
2. Read [ARCHITECTURE_TARGET.md](docs/agent-architecture/ARCHITECTURE_TARGET.md).
3. Preserve the non-commercial [LICENSE](LICENSE).
4. Prefer current primary/vendor documentation for external technical claims.
5. Never publish credentials, private endpoints, account IDs, private topology, sensitive datasets, private provider arrangements, customer data, or non-public training sources.
6. Use only `crm@orchords.com` where a public email is required.
7. Do not claim tests, benchmarks, Cloudflare connectivity, model capability, or host compatibility that was not actually demonstrated.
8. OAI-2.0 is runner-free: do not add GitHub-hosted or self-hosted Actions runners as an acceptance dependency.

## Engineering workflow

```text
INSPECT -> CURRENT SOURCE -> OFFICIAL RESEARCH -> VERIFY GAP
-> IMPLEMENT -> TEST -> BENCHMARK WHEN RELEVANT
-> RECHECK -> COMMIT MAIN -> VERIFY RESULT
```

Authorized maintainers may work directly on `main` where repository rules permit it.

The dependency-ordered master work map is [Issue #1](https://github.com/ORCHORDS-LCC/OAI-2.0/issues/1). Commit messages should describe code/docs changes only; do not embed runner names, workflow IDs, runner logs, or operational run records.

## Current verification commands

Typical local checks:

```bash
uv sync --extra dev
uv run ruff check oai2 tests scripts
uv run mypy --ignore-missing-imports oai2
uv run pytest -W error
```

MLX tests require compatible Apple Silicon/macOS. Run them locally on supported hardware; do not substitute a runner-backed CI result.

## Knowledge/Cloudflare changes

Read [CLOUDFLARE_KNOWLEDGE.md](docs/agent-architecture/CLOUDFLARE_KNOWLEDGE.md).

The current repository has a tested **application-level contract and mock bindings**, not a verified live Cloudflare Worker deployment. Any live adapter must map correctly to current asynchronous D1, R2, Vectorize, and KV APIs.

q-pipe imports default to q-pipe's verified Cloudflare export rules: promoted rows, independent verification, quality gates, approved source allow-list, and explicit Android-curriculum opt-in.

## Documentation changes

- update indexes when adding/moving docs;
- use Mermaid where flow visualization helps;
- keep sensitive implementation details out of diagrams;
- update stale status labels;
- cite current primary sources in [SOURCES.md](docs/agent-architecture/SOURCES.md);
- keep benchmark limitations next to benchmark numbers.

## Security

Security-sensitive findings belong in [SECURITY.md](SECURITY.md), not public issues.
