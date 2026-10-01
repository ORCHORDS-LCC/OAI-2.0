# Implementation roadmap

> **Status: PROPOSED.** Stages are validation gates, not delivery promises.

```mermaid
flowchart LR
    S0[0 Public contracts] --> S1[1 Host-neutral tool protocol]
    S1 --> S2[2 Fast single-agent controller]
    S2 --> S3[3 Vision/UI grounding]
    S3 --> S4[4 Evidence verification]
    S4 --> S5[5 Native multi-agent]
    S5 --> S6[6 Adaptive reasoning]
    S6 --> S7[7 Host adapters]
    S7 --> S8[8 Hardening]
```

## Stage 0
Architecture, security boundaries, eval requirements, diagrams and contribution rules.

## Stage 1
Normalize multimodal messages, dynamic tools, results, cancellation, permissions and agent operations.

## Stage 2
Build the smallest useful single-agent controller and measure correct actions/stopping.

## Stage 3
Add screenshots, UI structure, localization and source-render mapping.

## Stage 4
Add explicit evidence and expected-vs-observed verification.

## Stage 5
Add spawn/delegate/message/join/cancel/merge with scoped capabilities and one integration owner.

## Stage 6
Train/evaluate FURIOUS, DEEP and SWARM routing under fixed budgets.

## Stage 7
Add host adapters only after inspecting and testing each public contract.

## Stage 8
Security, stress, recovery, cancellation, redaction and compatibility hardening.

A capability moves from PROPOSED only when code, tests and held-out evidence exist.
