# Security and sandboxing

> **Status: PROPOSED requirements.**

Agentic coding combines untrusted repository content, tool output, network content, model output and potentially multiple agents. Prompts are not a security boundary.

```mermaid
flowchart LR
    M[Agent request] --> P[Policy engine]
    P --> S[Schema validation]
    S --> C[Scope validation]
    C --> A{Allowed?}
    A -->|yes| X[Sandboxed execution]
    A -->|no| D[Structured denial]
    X --> L[Audit result]
    L --> M
    D --> M
```

## Prompt-injection boundary

```mermaid
flowchart TD
    I[Repository / web / tool content] --> U[Untrusted data]
    U --> M[Model reasoning]
    M --> A[Requested action]
    A --> P[Independent policy engine]
    P -->|allowed| E[Execute]
    P -->|denied| N[Do not execute]
```

Required controls include explicit workspace roots, path traversal defense, process/time/memory limits, network restrictions where appropriate, secret redaction, scoped credentials, per-agent capability isolation, approval policy for high-impact actions and auditable tool calls.

Test prompt injection, poisoned tool descriptions, path escape, shell injection, exfiltration attempts, privilege escalation, cross-agent leakage, resource exhaustion and unsafe retries.
