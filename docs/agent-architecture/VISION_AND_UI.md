# Vision, UI Testing, Aesthetics and UX

_Reconciliation baseline: `c4a1f6f6134bd18668b8a621a2c0f0f89708e5e7` (source state before this documentation commit)._

_Last reviewed: 2026-10-02._

> **Current status:** four-view data abstractions are **EXPERIMENTAL**. A trained vision encoder, real device pipeline, temporal visual reasoning, and automated aesthetics/UX model are still **PROPOSED**.

## Four target views

1. pixels/screenshots;
2. semantic/accessibility/UI structure;
3. source-to-render mapping;
4. temporal interaction frames.

Current source can represent these views and decode image bytes for basic tests. That is not equivalent to visual intelligence.

## Target debugging loop

```mermaid
flowchart TD
    G[Goal / visual bug] --> C[Capture baseline]
    C --> O[Detect discrepancy]
    O --> L[Localize element/region]
    L --> M[Map to source]
    M --> P[Patch]
    P --> T[Build/test]
    T --> R[Run target state]
    R --> C2[Capture result]
    C2 --> V{Expected = observed?}
    V -->|yes| E[Record proof]
    V -->|no| O
```

## Accessibility and semantics roadmap

WP-66 (#197–#202) tracks canonical accessibility semantics, objective checks, source mapping, and post-fix verification. These are mapped requirements, not current production vision capability. WI-VIS-003 / #229 separately tracks aesthetics/design-system/human-preference calibration.

## Objective checks

Future deterministic/visual checks should cover clipping, overlap, contrast, tap targets, insets, overflow, navigation state, blank frames, crashes, accessibility mismatches, and approved visual regressions.

## Preference checks

Aesthetics/UX preference scoring should remain separate from objective correctness. Project-specific taste needs human/reference calibration.

## Performance

Visual encodings should be cached and shared across subagents; vision should not run on code-only turns.
