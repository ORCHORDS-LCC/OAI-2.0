"""Reasoning-mode enum and routing heuristics."""

from __future__ import annotations

from enum import Enum


class ReasoningMode(str, Enum):
    """The four public reasoning modes from ``ARCHITECTURE_TARGET.md``.

    Each mode names an *active-compute* profile, not a fixed model size:

    - ``FURIOUS`` — smallest active subset, very low latency, direct
      tool/action behavior. Long-term target: ~500M–1B active.
    - ``NORMAL`` — broader coding/reasoning subset for routine work.
      Long-term target: ~1–2B active.
    - ``DEEP`` — significantly more active capacity plus deeper layers,
      hypothesis branching, and verification. Long-term target: ~2–4B active.
    - ``SWARM`` — multiple independent reasoning instances running in
      parallel, sharing immutable context/evidence with one integration
      authority. Long-term target: multiple ~1–2B+ lanes.
    """

    FURIOUS = "FURIOUS"   # small active subset, near-instant tools.
    NORMAL = "NORMAL"     # broader coding/reasoning subset.
    DEEP = "DEEP"         # more active capacity, deeper, branching+verify.
    SWARM = "SWARM"       # multi-agent orchestration, high budget.


def choose_mode(
    *,
    task_size: int,
    evidence_pressure: float,
    budget_remaining: float,
) -> ReasoningMode:
    """Pick a reasoning mode from cheap, observable signals.

    These thresholds are placeholders — a tuned controller will replace
    them once we have real benchmarks. The four-way routing is what
    matters: easy routing goes to FURIOUS, broader coding goes to
    NORMAL, hard reasoning goes to DEEP, and parallelizable work goes
    to SWARM.
    """
    if budget_remaining <= 0.0:
        return ReasoningMode.FURIOUS  # last-resort cheap mode.
    if evidence_pressure >= 0.85 or task_size >= 16:
        return ReasoningMode.DEEP
    if task_size >= 8 and budget_remaining >= 0.6:
        return ReasoningMode.SWARM
    if task_size >= 3 or evidence_pressure >= 0.3:
        return ReasoningMode.NORMAL
    return ReasoningMode.FURIOUS


__all__ = ["ReasoningMode", "choose_mode"]
