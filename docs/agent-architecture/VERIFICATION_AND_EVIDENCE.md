# Verification and Evidence

_Last reviewed: 2026-10-02._

> **Current status:** typed evidence/claim structures and evidence-graph behavior are **EXPERIMENTAL**. WP-75 (#226–#228) now explicitly tracks claim-level evidence binding, contradiction/freshness handling, and adversarial false-success/hallucination gates; full automatic enforcement in final generation remains **NOT YET PROVEN**.

## Principle

Model confidence is not proof.

```mermaid
flowchart TD
    Q[Task] --> H[Hypothesis/action]
    H --> S{Evidence needed}
    S --> SRC[Repository/source]
    S --> RUN[Runtime/tool]
    S --> TEST[Build/tests]
    S --> VIS[Visual observation]
    S --> EXT[Authoritative external source]
    SRC --> E[Evidence graph]
    RUN --> E
    TEST --> E
    VIS --> E
    EXT --> E
    E --> C{Supported?}
    C -->|yes| O[Supported conclusion]
    C -->|conflict| R[Recheck]
    C -->|insufficient| U[UNVERIFIED / BLOCKED]
```

## Evidence classes

User intent, repository source, history, deterministic execution, runtime/device state, visual state, primary external references, secondary references, and model hypotheses should remain distinguishable.

## Coding proof hierarchy

```text
plausible explanation
 < patch parses
 < targeted test
 < relevant build/suite
 < requested runtime state reproduced
 < regression/acceptance checks
```

## False-success boundary

A generated claim is not verified merely because the model is confident, a tool returned text, or an earlier issue comment said “fixed.” Current-source evidence, freshness, contradiction state, and the owning acceptance criteria must determine support. Unsupported current-state claims should remain UNVERIFIED / NEED_MORE_EVIDENCE / BLOCKED.

## Knowledge evidence

External knowledge objects carry source URI, content hash, retrieval time, authority, lifecycle status, and future artifact/vector references.

q-pipe knowledge does not become active shared knowledge simply because it exists; it must pass the strict import gate.
