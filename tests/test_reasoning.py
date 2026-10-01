"""Reasoning-mode + FURIOUS controller tests."""

from __future__ import annotations

from oai2.core import Status
from oai2.reasoning import (
    DeepContext,
    DeepController,
    FuriousContext,
    FuriousController,
    FuriousDecision,
    FuriousState,
    NormalContext,
    NormalController,
    ReasoningMode,
    choose_mode,
)
from oai2.reasoning.swarm import SwarmContext, SwarmController, SwarmRole


def test_choose_mode_low_pressure_is_furious() -> None:
    assert (
        choose_mode(task_size=2, evidence_pressure=0.1, budget_remaining=1.0)
        is ReasoningMode.FURIOUS
    )


def test_choose_mode_mid_pressure_is_normal() -> None:
    assert (
        choose_mode(task_size=4, evidence_pressure=0.4, budget_remaining=0.8)
        is ReasoningMode.NORMAL
    )


def test_choose_mode_high_pressure_is_deep() -> None:
    assert (
        choose_mode(task_size=8, evidence_pressure=0.9, budget_remaining=0.5)
        is ReasoningMode.DEEP
    )


def test_choose_mode_large_task_with_budget_is_swarm() -> None:
    assert (
        choose_mode(task_size=10, evidence_pressure=0.5, budget_remaining=0.9)
        is ReasoningMode.SWARM
    )


def test_choose_mode_no_budget_is_furious() -> None:
    assert (
        choose_mode(task_size=2, evidence_pressure=0.0, budget_remaining=0.0)
        is ReasoningMode.FURIOUS
    )


def test_furious_state_machine_idle_to_planning() -> None:
    decision: FuriousDecision = FuriousController.decide(
        FuriousContext(status=Status.PROPOSED), current=FuriousState.IDLE
    )
    assert decision.next_state is FuriousState.PLANNING


def test_furious_escalates_after_three_failures() -> None:
    decision = FuriousController.decide(
        FuriousContext(consecutive_failures=3, status=Status.PROPOSED),
        current=FuriousState.OBSERVING,
    )
    assert decision.next_state is FuriousState.ESCALATE
    assert decision.escalate_to == "DEEP"


def test_furious_dones_on_zero_budget() -> None:
    decision = FuriousController.decide(
        FuriousContext(budget_remaining=0.0, status=Status.PROPOSED),
        current=FuriousState.PLANNING,
    )
    assert decision.next_state is FuriousState.DONE


def test_deep_controller_picks_highest_above_threshold() -> None:
    from oai2.reasoning.deep import Candidate

    ctx = DeepContext(evidence_threshold=0.5, status=Status.PROPOSED)
    picked = DeepController.select(
        [
            Candidate("a", 0.2),
            Candidate("b", 0.9),
            Candidate("c", 0.7),
        ],
        ctx,
    )
    assert picked is not None and picked.action == "b"


def test_deep_controller_returns_none_when_nothing_above_threshold() -> None:
    from oai2.reasoning.deep import Candidate

    ctx = DeepContext(evidence_threshold=0.8, status=Status.PROPOSED)
    assert (
        DeepController.select([Candidate("a", 0.1), Candidate("b", 0.3)], ctx) is None
    )


def test_swarm_critic_pick_needs_threshold() -> None:
    ctrl = SwarmController(
        SwarmContext(consensus_threshold=0.7, status=Status.PROPOSED)
    )
    assert ctrl.critic_pick({"a": 0.9}) == "a"
    assert ctrl.critic_pick({"a": 0.4}) is None


def test_swarm_role_enum_members() -> None:
    assert SwarmRole.PLANNER.value == "planner"
    assert SwarmRole.IMPLEMENTER.value == "implementer"
    assert SwarmRole.CRITIC.value == "critic"
    assert SwarmRole.VERIFIER.value == "verifier"


def test_normal_controller_picks_speculative_mtp_when_both_on() -> None:
    ctx = NormalContext(speculative=True, multi_token_prediction=True)
    assert NormalController.choose_decode_strategy(ctx) == "speculative_mtp"


def test_normal_controller_picks_vanilla_when_nothing_on() -> None:
    ctx = NormalContext(speculative=False, multi_token_prediction=False)
    assert NormalController.choose_decode_strategy(ctx) == "vanilla"


def test_reasoning_mode_has_four_modes() -> None:
    expected = {"FURIOUS", "NORMAL", "DEEP", "SWARM"}
    assert {m.name for m in ReasoningMode} == expected
