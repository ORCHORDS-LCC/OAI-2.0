# Vision, UI testing, aesthetics and UX

> **Status: PROPOSED.**

Vision is a first-class coding capability because UI defects frequently exist only in rendered behavior.

## Four views

- pixels;
- UI/accessibility structure;
- source-to-render mapping;
- temporal interaction frames.

```mermaid
flowchart TB
    APP[Running app] --> PX[Pixels]
    APP --> ST[UI structure]
    APP --> TM[Interaction frames]
    SRC[Source graph] --> MAP[Source-render mapping]
    PX --> W[Unified UI state]
    ST --> W
    TM --> W
    MAP --> W
    W --> D[Diagnose]
    D --> P[Patch]
    P --> B[Build + launch]
    B --> APP
```

## Visual debugging loop

```mermaid
flowchart TD
    G[Goal / bug] --> C[Capture baseline]
    C --> O[Detect discrepancy]
    O --> L[Localize region + element]
    L --> M[Map to source candidates]
    M --> I[Inspect source/history]
    I --> P[Smallest patch]
    P --> T[Build/test]
    T --> R[Run target state]
    R --> C2[Capture result]
    C2 --> V{Expected vs observed}
    V -->|match| E[Record proof]
    V -->|mismatch| O
```

## Objective checks

Clipping, overlap, contrast, interaction target size, insets, text overflow, state transitions, blank frames, crashes, accessibility mismatches and approved screenshot regressions.

## Preference checks

Hierarchy, balance, typography, spacing, color coherence, density, perceived polish, discoverability, feedback clarity and task friction.

## Shared visual encoding

```mermaid
flowchart LR
    IMG[One capture] --> ENC[One visual encoding]
    ENC --> UX[UX critic]
    ENC --> BUG[Bug localizer]
    ENC --> ACC[Accessibility reviewer]
    ENC --> CODE[Source-mapping agent]
```

Prefer structured device/IDE actions over blind coordinate clicking. Always observe the resulting state.
