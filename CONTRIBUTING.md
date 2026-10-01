# Contributing to OAI-2.0

_Reconciliation baseline: `8520cdb5d8d3e49bc9e12dfd79ccf59e7422e44d` (source state before this documentation commit)._

_Last reviewed: 2026-10-02._

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

## Parallel-agent synchronization

Multiple AIs may work on `main` concurrently. While actively editing, re-fetch remote `main` approximately every 3 minutes and **always** immediately before a commit. If remote `main` moved, inspect the new commits/diffs and discard or reconcile duplicated local work instead of overwriting it.

## Engineering workflow

```text
INSPECT -> CURRENT SOURCE -> OFFICIAL RESEARCH -> VERIFY GAP
-> IMPLEMENT -> TEST -> BENCHMARK WHEN RELEVANT
-> RECHECK REMOTE MAIN -> RECONCILE CONCURRENT WORK -> COMMIT MAIN -> VERIFY REMOTE RESULT
```

Authorized maintainers may work directly on `main` where repository rules permit it.

The dependency-ordered master work map is [Issue #1](https://github.com/ORCHORDS-LCC/OAI-2.0/issues/1). Detailed work packages shall follow [ENGINEERING_ISSUE_STANDARD.md](docs/agent-architecture/ENGINEERING_ISSUE_STANDARD.md). Commit messages should describe code/docs changes only; do not embed runner names, workflow IDs, runner logs, or operational run records.

## Current verification command

Run the single local preflight from the repository root:

```bash
uv run python scripts/verify.py
```

It runs dependency sync, Ruff, MyPy, pytest, a public-safety scan, and a Markdown relative-link scan. It fails fast on any mandatory failure and does not invoke GitHub Actions. MLX tests require compatible Apple Silicon/macOS; platform-specific skips must be reported separately from passes.

The MLX benchmark harness has the same environment requirement: run it as `uv run python scripts/bench.py --help`. `--help` works with a bare interpreter, but an actual run needs `mlx` / `mlx-lm` from the uv environment and exits with an actionable error instead of a `ModuleNotFoundError` traceback when they are missing.

## Local pre-commit gate (Refs #236)

The repo is runner-free and direct-push-on-main. To catch the
cumulative-state failure mode (where each commit's gate sees the prior
state, not the cumulative state — see #236), every commit runs a local
pre-commit hook:

```bash
pip install pre-commit      # one-time
pre-commit install          # one-time, installs the .git/hooks/pre-commit
```

The `.pre-commit-config.yaml` runs:

1. `ruff check --fix` — catches F401 (unused import), F811 (redefinition), I001 (unsorted imports), W292 (missing newline).
2. `ruff format` — applies the project's chosen formatter.
3. `pytest --co -q` — collection-only smoke check; surfaces F401 import cycles, syntax errors, and missing dependencies without running tests.

Run the same checks on demand against the whole tree:

```bash
pre-commit run --all-files
```

Emergency bypass (NOT recommended; documents the gate skip):

```bash
git commit --no-verify
```

## Knowledge/Cloudflare changes

Read [CLOUDFLARE_KNOWLEDGE.md](docs/agent-architecture/CLOUDFLARE_KNOWLEDGE.md).

The current repository has tested application-level contracts/mocks plus async D1 reader/writer, R2/KV/Vectorize wrappers, an assembled knowledge runtime, authenticated transport handler, component factory, and public-safe Python Worker entrypoint/template. It still does **not** have a fully verified private Cloudflare deployment/end-to-end network proof. Any live adapter must map correctly to current asynchronous D1, R2, Vectorize, and KV APIs.

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
