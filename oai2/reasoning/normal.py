"""NORMAL controller scaffold.

NORMAL is the broader coding/reasoning profile between FURIOUS and DEEP.
It activates a wider expert subset than FURIOUS but does not branch or
verify as aggressively as DEEP. Used for routine coding, reading,
editing, and tool-driven multi-step work.

Long-term active compute target: ~1–2B parameters.

Status: PROPOSED.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status


class NormalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Width of the active expert subset for this mode.
    active_experts: int = Field(default=8, ge=1, le=64)
    # Whether speculative decoding is enabled for this run.
    speculative: bool = True
    # Whether multi-token prediction is enabled for this run.
    multi_token_prediction: bool = True
    status: Status = Status.PROPOSED


class NormalController:
    STATUS = Status.PROPOSED

    @staticmethod
    def choose_decode_strategy(ctx: NormalContext) -> str:
        if ctx.speculative and ctx.multi_token_prediction:
            return "speculative_mtp"
        if ctx.speculative:
            return "speculative"
        if ctx.multi_token_prediction:
            return "multi_token_prediction"
        return "vanilla"


__all__ = ["NormalContext", "NormalController"]
