# Portable native tool calling

> **Status: PROPOSED.**

The host supplies tool names, descriptions, argument schemas, permissions and results at runtime. The model learns tool meaning from those definitions rather than memorizing one product's tool names.

## Lifecycle

```mermaid
sequenceDiagram
    participant H as Host
    participant M as Agent
    participant P as Policy
    participant T as Tool
    participant V as Verifier

    H->>M: Goal + context + tool schemas
    M->>P: Typed tool call
    P->>P: Validate schema + scope + permission
    alt Allowed
      P->>T: Execute
      T-->>M: Structured result
      M->>V: Expected vs observed state
      V-->>M: Verified / conflict / insufficient
    else Denied
      P-->>M: Structured denial
    end
    M-->>H: Next action or final output
```

## Semantic generalization

```mermaid
flowchart LR
    A[read_file path] --> S[workspace.read]
    B[file_get location] --> S
    C[workspace_open uri] --> S
    D[fetch_source target] --> S
    S --> M[Reason from schema + description]
```

## Validation

Before execution:

1. tool exists;
2. arguments validate;
3. capability is allowed;
4. resource is in scope;
5. budget is respected;
6. high-impact actions satisfy host policy.

## Failure behavior

```mermaid
flowchart TD
    C[Tool call] --> R{Result}
    R -->|success| O[Observe]
    R -->|schema error| F[Repair arguments]
    R -->|denied| P[Choose allowed alternative]
    R -->|timeout| B[Retry within budget]
    R -->|unavailable| U[Re-plan]
    O --> V[Verify state]
    F --> C
    B --> C
```

A tool error is never converted into a fabricated success.
