"""Reasoning controllers: FURIOUS / DEEP / SWARM.

Status: PROPOSED. The controller interfaces and state machines are
defined; live LLM-driven decisions require an IMPLEMENTED
:class:`~oai2.runtime.InferenceRuntime`.
"""

from __future__ import annotations

from .deep import Candidate, DeepContext, DeepController
from .furious import (
    FuriousContext,
    FuriousController,
    FuriousDecision,
    FuriousState,
)
from .modes import ReasoningMode, choose_mode
from .normal import NormalContext, NormalController
from .swarm import SwarmContext, SwarmController, SwarmRole

__all__ = [
    "Candidate",
    "DeepContext",
    "DeepController",
    "FuriousContext",
    "FuriousController",
    "FuriousDecision",
    "FuriousState",
    "NormalContext",
    "NormalController",
    "ReasoningMode",
    "choose_mode",
    "SwarmContext",
    "SwarmController",
    "SwarmRole",
]
