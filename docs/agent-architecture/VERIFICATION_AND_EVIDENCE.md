# Verification and evidence

> **Status: PROPOSED.**

The goal is to sharply reduce unsupported output through explicit evidence and execution checks. It does not promise literal zero hallucination.

```mermaid
flowchart TD
    Q[Task] --> P[Proposed action / claim]
    P --> S{Support needed?}
    S -->|source fact| SRC[Read source/history]
    S -->|external fact| WEB[Authoritative reference]
    S -->|runtime behavior| RUN[Execute/inspect]
    S -->|visual behavior| VIS[Capture/inspect UI]
    S -->|code correctness| TEST[Build/tests/analysis]
    SRC --> E[Evidence graph]
    WEB --> E
    RUN --> E
    VIS --> E
    TEST --> E
    E --> C{Supported?}
    C -->|yes| O[Supported conclusion]
    C -->|conflict| R[Research/test again]
    C -->|insufficient| U[Unverified / blocked]
    R --> E
```

Evidence classes include user intent, repository source, change history, deterministic execution, runtime observation, visual observation, authoritative external sources and model hypotheses.

```text
"looks fixed"
 < patch parses
 < targeted test passes
 < relevant suite/build passes
 < requested runtime behavior is reproduced
 < regression checks and acceptance criteria pass
```

A blocked result is valid. Fabricated completion is not.
