# Native multi-agent orchestration

> **Status: PROPOSED.**

The same core model may act as primary agent or as scoped subagents with different goals, tools and budgets.

## Adaptive modes

```mermaid
flowchart TD
    T[Task] --> C{Cheap direct solution?}
    C -->|yes| F[FURIOUS]
    C -->|no| P{Useful independent subproblems?}
    P -->|no| D[DEEP]
    P -->|yes| S[SWARM]
    F --> V[Verify]
    D --> V
    S --> J[Join findings]
    J --> I[Integrator]
    I --> V
    V --> X{Acceptance proven?}
    X -->|yes| Z[Stop]
    X -->|no| R[Re-plan within budget]
    R --> C
```

Native orchestration semantics include spawn, delegate, message, inspect, join, cancel and merge.

## Example swarm

```mermaid
flowchart TB
    O[Orchestrator] --> A[Source investigator]
    O --> B[Visual/runtime investigator]
    O --> C[Test/log investigator]
    O --> D[Documentation investigator]
    A --> E[Shared evidence]
    B --> E
    C --> E
    D --> E
    E --> I[Integrator]
    I --> T[Build + tests + runtime verification]
    T --> O
```

## Parallel research, serial integration

```mermaid
flowchart LR
    A1[Proposal A] --> I[Integrator]
    A2[Proposal B] --> I
    A3[Evidence C] --> I
    A4[Test result D] --> I
    I --> W[Single coherent write path]
    W --> V[Deterministic verification]
```

Controls include maximum depth/count, per-agent budgets, no child privilege escalation, duplicate-work cancellation and measured benefit before raising parallelism.
