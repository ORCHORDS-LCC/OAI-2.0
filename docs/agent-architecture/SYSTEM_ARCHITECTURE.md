# System architecture

> **Status: PROPOSED.**

OAI-2.0 targets a portable coding foundation agent that treats text, code, screenshots, UI structure, tools, execution results and subagents as one software environment.

```mermaid
flowchart TB
    subgraph Hosts
      IDE[IDE]
      SHELL[Agent shell]
      API[Compatible API host]
    end

    subgraph Protocol
      AD[Host adapter]
      TD[Dynamic tool definitions]
      MM[Multimodal messages]
      AO[Agent operations]
    end

    subgraph Core
      LC[Language + code encoder]
      VI[Vision encoder]
      UI[UI structure encoder]
      SH[Shared multimodal core]
      RC[Reasoning controller]
      TH[Tool head]
      AH[Agent-control head]
      VH[Verification head]
      OH[Output head]
    end

    subgraph Runtime
      CTX[Shared context/cache]
      EV[Evidence store]
      EX[Execution adapters]
      BENCH[Evaluation hooks]
    end

    IDE --> AD
    SHELL --> AD
    API --> AD
    AD --> MM
    TD --> MM
    AO --> MM
    MM --> LC
    MM --> VI
    MM --> UI
    LC --> SH
    VI --> SH
    UI --> SH
    SH --> RC
    SH --> TH
    SH --> AH
    SH --> VH
    SH --> OH
    TH --> EX
    AH --> EX
    EX --> EV
    EV --> VH
    SH --> CTX
    SH --> BENCH
```

## World state

```mermaid
flowchart TD
    W[Software world state] --> R[Repository graph]
    W --> U[UI state graph]
    W --> B[Build/test state]
    W --> D[Runtime/device state]
    W --> G[Change state]
    W --> E[Evidence graph]
    W --> A[Active-agent graph]
    W --> P[Permissions + budgets]
```

The main learning target is not only next-token prediction:

```text
(current state, goal, source, screen, history)
 -> next action
 -> expected result
 -> verification method
 -> evidence status
```

## Non-goals

- claiming literal zero hallucinations;
- forcing every task into multi-agent mode;
- replacing compilers/tests with model opinion;
- hard-coding one IDE's tool names;
- exposing private infrastructure;
- claiming frontier capability without benchmark evidence.
