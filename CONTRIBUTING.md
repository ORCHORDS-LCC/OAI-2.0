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

## Lint gate (Refs #236, #264)

The repo is runner-free and direct-push-on-main, with no hosted CI gate. To
catch the cumulative-state failure mode (where each commit's gate sees the
prior state, not the cumulative state — see #236), the lint gate is enforced
**by the test suite**:

```bash
uv run pytest -W error
```

`tests/test_lint_gate.py` runs the exact `ruff check` command that
`scripts/verify.py` declares and fails the suite when it reports errors. The
command is parsed out of `verify.py` rather than duplicated, so the test and
the script cannot drift apart. If that check is ever removed from
`verify.py`, the test fails and asks to be updated — it does not quietly stop
checking anything.

This is deliberate. `pre-commit` was previously declared in
`.pre-commit-config.yaml` but never installed, and its pinned rev was not the
version the project actually lints with. A real `NameError` once survived in
the tree behind exactly that gap (#264): the F821 rule that can see the
defect existed in configuration and was never executed. A gate that exists
only in configuration is not a gate.

The gate deliberately does not skip when `ruff` cannot be invoked. A check
that disappears when its tooling is missing is the same failure wearing a
different hat, so a missing tool is a failure with an actionable message.

### Pre-commit as an optional convenience

If `pre-commit` is installed, the hooks give faster per-commit feedback:

```bash
pip install pre-commit      # one-time
pre-commit install          # one-time, installs the .git/hooks/pre-commit
pre-commit run --all-files
```

They run `ruff check --fix` and a `pytest --co -q` collection smoke check.
This is a convenience layer only; correctness does not depend on it.

`ruff-format` is **not** enabled (decision recorded in #264). At the time it
was considered, `ruff format --check` reported 121 files needing changes, and
formatter output differs substantially between the hook's pinned rev and the
0.16.9 the project lints with. Reformatting repo-wide would bury real fixes
under version-fragile churn in a repository that takes commits continuously
from parallel agents. Revisit deliberately, repo-wide and in one commit, if
the project ever standardises on a formatter.

Emergency bypass (NOT recommended; documents the gate skip):

```bash
git commit --no-verify
```

Note that `--no-verify` bypasses the pre-commit convenience layer only. It
does **not** bypass the suite-enforced gate: `tests/test_lint_gate.py` still
runs inside `pytest` and still fails.

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
