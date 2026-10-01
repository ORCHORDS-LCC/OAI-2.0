# Contributing to OAI-2.0

## Before changing anything

1. Read the existing architecture docs first.
2. Preserve the distinction between **IMPLEMENTED**, **EXPERIMENTAL**, **PROPOSED**, and **BLOCKED**.
3. Prefer current primary/vendor documentation for protocol and host claims.
4. Keep public docs free of credentials, private endpoints, private topology, sensitive datasets, private provider arrangements, customer data, and internal identifiers.
5. Keep diagrams provider-neutral and architecture-focused.
6. Do not claim tests, benchmarks or host compatibility that were not actually run.
7. Update related indexes when adding or moving docs.

## Workflow

Authorized maintainers work directly on `main` when repository policy permits.

For engineering work:

```text
INSPECT -> SOURCE -> RESEARCH WHEN NEEDED -> VERIFY GAP
-> IMPLEMENT -> TEST -> RECHECK -> COMMIT -> VERIFY RESULT
```

For documentation work:

- use Mermaid when a flow benefits from visualization;
- validate links;
- cite public primary sources;
- mark speculative work as PROPOSED;
- do not publish sensitive implementation details.

## Architecture docs

Public architecture work lives under [docs/agent-architecture/](docs/agent-architecture/README.md).

## Security

Do not publish vulnerabilities or secrets in issues. Follow [SECURITY.md](SECURITY.md).
