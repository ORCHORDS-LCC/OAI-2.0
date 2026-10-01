# Training and evaluation strategy

> **Status: PROPOSED.** Public documentation is provider-neutral and excludes private datasets, providers, infrastructure and sensitive training sources.

## Capability curriculum

```mermaid
flowchart TD
    B[Language + code foundations] --> T[Single-tool use]
    T --> D[Dynamic/unseen tool schemas]
    D --> C[Real coding tasks]
    C --> V[Vision + UI grounding]
    V --> X[Code-to-render diagnosis]
    X --> R[Verification + abstention]
    R --> M[Multi-agent orchestration]
    M --> L[Long-horizon projects]
    L --> A[Adversarial evaluation]
```

## Training records

Use auditable task artifacts: goals, acceptance criteria, available tools, tool calls/results, screenshots/UI state, patches, compiler/test/runtime outcomes, evidence references and success/failure labels.

Do not require reproducing any model's private hidden reasoning.

## Reference evaluation

External reference systems may create scenarios, critiques or preference labels, but deterministic execution should dominate when available.

```mermaid
flowchart LR
    TASK[Task + environment] --> A[Agent attempt]
    A --> RUN[Real tools / tests / runtime]
    RUN --> OBS[Observed outcome]
    OBS --> DET[Deterministic score]
    OBS --> REF[Reference evaluation]
    DET --> DATA[Reviewed record]
    REF --> DATA
    DATA --> TRAIN[Train / tune]
    TRAIN --> HOLD[Held-out evaluation]
```

Tool schemas should be randomized. Held-out tests must include unseen names and shapes.

Negative examples should include hallucinated tools, fake test claims, bad screenshots, regressions, unnecessary swarms, unsafe actions and wrong stopping behavior.
