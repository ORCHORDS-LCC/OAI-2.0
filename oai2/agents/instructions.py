"""Instruction precedence, trust classes, and conflict resolution.

Implements the model defined in
``docs/agent-architecture/INSTRUCTION_PRECEDENCE.md`` for #185 / `WI-PROMPT-001`.

Requirements covered:

- **REQ-PROMPT-011** — explicit precedence/trust classes for runtime/system,
  repository policy, user goal, tool metadata, and untrusted content.
- **REQ-PROMPT-012** — lower-trust repository/web/tool output cannot override
  higher-priority policy.
- **REQ-PROMPT-013** — same-level incompatible instructions surface a conflict
  and get a deterministic tie-break.
- **REQ-PROMPT-014** — every precedence/trust decision is traceable.
- **REQ-PROMPT-015** — a tool description is semantic input, never a grant.
- **REQ-PROMPT-016** — the instruction model/schema is versioned.

This module is **pure**: no model calls, no I/O, no clock, no global state. The
same instruction list always produces the same resolution, which is what makes
REQ-PROMPT-014 satisfiable. It is a host-side control that makes precedence
explicit and auditable — it is not, and is not intended to be, a security
boundary. Tool access remains governed by the host-side tool policy.

Design notes worth keeping in mind when editing:

* Conflict detection is **structural**, never semantic. Two instructions
  conflict only when they share a precedence, share a ``directive``, and assert
  different ``value``\\ s. Instructions without a ``directive`` never conflict.
* Override detection is **structural**, never textual. Nothing here scans for
  "ignore previous instructions"; phrase matching is **evadable and incomplete**
  (it misses paraphrase, non-English text, and instructions that simply do not
  announce themselves), so it cannot be relied on as the control. Deterministic
  phrase matching is entirely possible — it is just not a sound boundary.
* ``asserted_precedence`` is a **claim to evaluate**, never a grant. It can only
  cause a rejection. Actual trust is assigned by the trusted ingestion layer
  through :func:`trusted_instruction`, never by the content itself.
* Malicious content that supplies *no* metadata at all is expected, not
  exceptional. It stays ``UNTRUSTED`` and cannot bind a directive already held
  by a stronger class; see the fence tests in ``tests/test_instructions.py``.
* Nothing is dropped silently. Rejections land in the
  :class:`ResolutionTrace` so a later reader can distinguish "the resolver
  rejected this" from "the resolver never saw it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Version of the instruction model and the rules that implement it
#: (REQ-PROMPT-016). Bump when the class set, the rejection rules, or the
#: meaning of a recorded field changes. Carried onto every resolution so a
#: stored trace identifies the rules that produced it.
INSTRUCTION_SCHEMA_VERSION = "1.0"


class Precedence(StrEnum):
    """Ordered instruction precedence. Lower rank wins.

    The order is a trust gradient, not a flat demotion: ``USER`` outranks
    ``REPOSITORY`` so that a hostile checkout cannot permanently override the
    operator, while repository content still outranks tool metadata and web
    output (REQ-PROMPT-011, REQ-PROMPT-012).
    """

    SYSTEM = "system"
    USER = "user"
    REPOSITORY = "repository"
    TOOL_METADATA = "tool_metadata"
    UNTRUSTED = "untrusted"

    @property
    def rank(self) -> int:
        """Precedence rank; 0 is strongest."""
        return _PRECEDENCE_RANK[self]

    def outranks(self, other: Precedence) -> bool:
        """True when this class strictly outranks ``other``."""
        return self.rank < other.rank


_PRECEDENCE_RANK: dict[Precedence, int] = {
    Precedence.SYSTEM: 0,
    Precedence.USER: 1,
    Precedence.REPOSITORY: 2,
    Precedence.TOOL_METADATA: 3,
    Precedence.UNTRUSTED: 4,
}


class RejectionReason(StrEnum):
    """Why an instruction was not accepted into the effective set."""

    #: Content claimed a precedence stronger than the one it was delivered at.
    PRIVILEGE_ESCALATION = "privilege_escalation"
    #: Content re-asserted a directive already bound by a stronger class.
    OVERRIDE_ATTEMPT = "override_attempt"
    #: A tool description claimed to grant a permission (REQ-PROMPT-015).
    PERMISSION_GRANT_CLAIMED = "permission_grant_claimed"
    #: Same precedence, same directive, different value; lost the tie-break.
    SUPERSEDED_BY_SAME_LEVEL = "superseded_by_same_level"
    #: An unauthenticated source tried to revise an instruction it does not own.
    REVISION_NOT_AUTHORISED = "revision_not_authorised"


#: Classes whose content is authored by the operator and may therefore revise
#: an earlier instruction from the same class. Everything outside this set is
#: evidence: it may inform, never correct.
REVISION_AUTHORISED_CLASSES = frozenset({Precedence.SYSTEM, Precedence.USER, Precedence.REPOSITORY})


@dataclass(frozen=True, slots=True)
class Instruction:
    """One instruction bound for a single resolution.

    Args:
        text: The instruction content. Never parsed by this module.
        precedence: The class this instruction was delivered at.
        source: Where the content came from, for the trace
            (``"runtime"``, ``"AGENTS.md"``, ``"web"``, a tool name, ...).
        directive: Optional normalised key this instruction binds, e.g.
            ``"network_access"``. Instructions without one never conflict.
        value: Optional value for ``directive``, e.g. ``"allow"``/``"deny"``.
        asserted_precedence: The precedence the *content claims for itself*.
            Defaults to ``precedence``. A **claim to evaluate, never a grant**:
            it can only cause this instruction to be rejected, never promoted.
            Actual trust is assigned by :func:`trusted_instruction`.
        grants_permission: Whether the content claims to authorise an action.
            Only meaningful for :attr:`Precedence.TOOL_METADATA`; a tool
            description asserting this is rejected (REQ-PROMPT-015).
        revises: Directive this instruction explicitly corrects. An authorised
            same-class revision replaces the earlier binding and is recorded as
            a revision rather than a conflict. An unauthenticated source may
            not revise (REQ-PROMPT-013).
        authenticated: Whether the host positively established this
            instruction's origin. Only an authenticated instruction from an
            operator-authored class may revise. Untrusted content is
            authenticated ``False`` by construction.
    """

    text: str
    precedence: Precedence
    source: str = ""
    directive: str | None = None
    value: str | None = None
    asserted_precedence: Precedence | None = None
    grants_permission: bool = False
    revises: str | None = None
    authenticated: bool = False

    @property
    def claims(self) -> Precedence:
        """The precedence this content asserts for itself."""
        return self.asserted_precedence or self.precedence

    @property
    def binds_directive(self) -> bool:
        """True when this instruction constrains a named directive."""
        return bool(self.directive) and self.value is not None

    @property
    def may_revise(self) -> bool:
        """True when this instruction is allowed to correct an earlier one."""
        return (
            self.authenticated
            and self.revises is not None
            and self.precedence in REVISION_AUTHORISED_CLASSES
        )


@dataclass(frozen=True, slots=True)
class Rejection:
    """An instruction that did not make it into the effective set."""

    reason: RejectionReason
    source: str
    precedence: Precedence
    detail: str
    instruction: Instruction = field(repr=False)

    def describe(self) -> str:
        """One-line human-readable form for logs and evidence."""
        return f"{self.reason}: {self.source} ({self.precedence}) — {self.detail}"


@dataclass(frozen=True, slots=True)
class Conflict:
    """A same-precedence disagreement on one directive.

    ``kept``/``dropped`` are the two sides of the tie-break. The winner is the
    one that arrived first, so the outcome does not depend on model judgment
    (REQ-PROMPT-013).
    """

    directive: str
    precedence: Precedence
    kept_source: str
    dropped_source: str
    kept_value: str
    dropped_value: str

    def describe(self) -> str:
        return (
            f"directive={self.directive} at {self.precedence}: kept "
            f"{self.kept_source}={self.kept_value}, dropped "
            f"{self.dropped_source}={self.dropped_value}"
        )


@dataclass(frozen=True, slots=True)
class ResolutionTrace:
    """Everything the resolver decided, and why (REQ-PROMPT-014)."""

    schema_version: str
    accepted: tuple[str, ...]
    rejections: tuple[Rejection, ...]
    conflicts: tuple[Conflict, ...]
    revisions: tuple[Conflict, ...] = ()

    @property
    def accepted_sources(self) -> tuple[str, ...]:
        """Sources that made it into the effective set, in effective order."""
        return self.accepted

    @property
    def was_rejected(self) -> bool:
        """True when at least one instruction was refused.

        Convenience for REQ-PROMPT-012 assertions: a rejection is always
        recorded, never silent.
        """
        return bool(self.rejections)

    def reasons(self) -> tuple[RejectionReason, ...]:
        """The distinct reasons present in the trace."""
        seen: list[RejectionReason] = []
        for rejection in self.rejections:
            if rejection.reason not in seen:
                seen.append(rejection.reason)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for debug mode and evidence records."""
        return {
            "schema_version": self.schema_version,
            "accepted": list(self.accepted),
            "rejections": [
                {
                    "reason": str(r.reason),
                    "source": r.source,
                    "precedence": str(r.precedence),
                    "detail": r.detail,
                }
                for r in self.rejections
            ],
            "conflicts": [
                {
                    "directive": c.directive,
                    "precedence": str(c.precedence),
                    "kept_source": c.kept_source,
                    "dropped_source": c.dropped_source,
                    "kept_value": c.kept_value,
                    "dropped_value": c.dropped_value,
                }
                for c in self.conflicts
            ],
            "revisions": [
                {
                    "directive": c.directive,
                    "precedence": str(c.precedence),
                    "kept_source": c.kept_source,
                    "dropped_source": c.dropped_source,
                    "kept_value": c.kept_value,
                    "dropped_value": c.dropped_value,
                }
                for c in self.revisions
            ],
        }


@dataclass(frozen=True, slots=True)
class ResolvedInstructions:
    """The effective instruction set plus its audit trail."""

    effective: tuple[Instruction, ...]
    trace: ResolutionTrace

    @property
    def effective_text(self) -> tuple[str, ...]:
        """Texts of the accepted instructions, strongest precedence first."""
        return tuple(i.text for i in self.effective)

    def directives(self) -> dict[str, str]:
        """Directives bound by the effective set, strongest precedence first."""
        bound: dict[str, str] = {}
        for instruction in self.effective:
            if not instruction.binds_directive:
                continue
            assert instruction.directive is not None
            assert instruction.value is not None
            bound[instruction.directive] = instruction.value
        return bound


def resolve(instructions: list[Instruction] | tuple[Instruction, ...]) -> ResolvedInstructions:
    """Order instructions by precedence and refuse lower-trust overrides.

    The function is pure and deterministic: the same input sequence always
    yields the same :class:`ResolutionTrace`.

    Rejection order is fixed so a trace is stable regardless of how the input
    was assembled:

    1. :attr:`RejectionReason.PRIVILEGE_ESCALATION` — ``asserted_precedence``
       stronger than the delivered ``precedence``.
    2. :attr:`RejectionReason.PERMISSION_GRANT_CLAIMED` — tool metadata
       asserting ``grants_permission`` (REQ-PROMPT-015).
    3. :attr:`RejectionReason.OVERRIDE_ATTEMPT` — a ``directive`` already bound
       by a strictly stronger class.
    4. :attr:`RejectionReason.SUPERSEDED_BY_SAME_LEVEL` — same-precedence
       conflict lost the arrival-order tie-break (REQ-PROMPT-013).

    Args:
        instructions: The instructions to resolve, in delivery order. Delivery
            order is the tie-break of last resort, so it must be stable.

    Returns:
        The accepted instructions strongest-precedence-first, plus the trace of
        every decision.
    """
    rejections: list[Rejection] = []
    conflicts: list[Conflict] = []

    # Pre-pass: the three structural refusals, evaluated in the fixed order
    # above. Keeping these separate from the tie-break pass means a rejected
    # instruction can never occupy a directive slot and block a legitimate one.
    survivors: list[Instruction] = []
    for instruction in instructions:
        reason = _structural_rejection(instruction)
        if reason is not None:
            rejections.append(reason)
            continue
        survivors.append(instruction)

    # Pass 2: walk strongest-first, binding directives. Anything that re-asserts
    # a directive already held by a stronger class is an override attempt; a
    # same-class disagreement is a recorded conflict resolved by arrival order.
    # Arrival position is carried explicitly in the sort key rather than stored
    # on the frozen dataclass, so the module keeps no state between calls.
    effective: list[Instruction] = []
    # directive -> (position in `effective`, the winning instruction)
    bound: dict[str, tuple[int, Instruction]] = {}
    revisions: list[Conflict] = []
    ordered = sorted(enumerate(survivors), key=lambda pair: (pair[1].precedence.rank, pair[0]))
    for _, instruction in ordered:
        # `binds_directive` already requires both fields to be present; the
        # locals give the type checker something it can narrow on.
        directive = instruction.directive
        value = instruction.value
        if not instruction.binds_directive or directive is None or value is None:
            effective.append(instruction)
            continue

        entry = bound.get(directive)
        if entry is None:
            bound[directive] = (len(effective), instruction)
            effective.append(instruction)
            continue

        held_index, held = entry
        if held.precedence.outranks(instruction.precedence):
            rejections.append(
                Rejection(
                    reason=RejectionReason.OVERRIDE_ATTEMPT,
                    source=instruction.source,
                    precedence=instruction.precedence,
                    detail=(
                        f"directive={instruction.directive} already bound by "
                        f"{held.source} at stronger precedence {held.precedence}"
                    ),
                    instruction=instruction,
                )
            )
            continue

        # Same precedence and same value: the two instructions agree, so this is
        # redundancy rather than a conflict. Both stay effective.
        if held.value == value:
            effective.append(instruction)
            continue

        # An explicit, authorised revision replaces the earlier binding instead
        # of losing the tie-break. "Use repository B instead of A" and "do not
        # push; inspect only" must not be silently discarded merely because an
        # earlier instruction arrived first.
        if instruction.may_revise and instruction.revises == directive:
            effective[held_index] = instruction
            bound[directive] = (held_index, instruction)
            revisions.append(
                Conflict(
                    directive=directive,
                    precedence=instruction.precedence,
                    kept_source=instruction.source,
                    dropped_source=held.source,
                    kept_value=str(value),
                    dropped_value=str(held.value),
                )
            )
            continue

        # Same precedence, different values, no revision claim: REQ-PROMPT-013
        # requires the conflict to surface and the tie-break to be
        # deterministic. Earliest arrival wins.
        conflicts.append(
            Conflict(
                directive=str(instruction.directive),
                precedence=instruction.precedence,
                kept_source=held.source,
                dropped_source=instruction.source,
                kept_value=str(held.value),
                dropped_value=str(instruction.value),
            )
        )
        rejections.append(
            Rejection(
                reason=RejectionReason.SUPERSEDED_BY_SAME_LEVEL,
                source=instruction.source,
                precedence=instruction.precedence,
                detail=(
                    f"directive={instruction.directive} already bound by "
                    f"{held.source} at the same precedence; earliest arrival kept"
                ),
                instruction=instruction,
            )
        )

    trace = ResolutionTrace(
        schema_version=INSTRUCTION_SCHEMA_VERSION,
        accepted=tuple(i.source for i in effective),
        rejections=tuple(rejections),
        conflicts=tuple(conflicts),
        revisions=tuple(revisions),
    )
    return ResolvedInstructions(effective=tuple(effective), trace=trace)


def trusted_instruction(
    text: str,
    precedence: Precedence,
    *,
    source: str,
    authenticated: bool,
    directive: str | None = None,
    value: str | None = None,
    revises: str | None = None,
) -> Instruction:
    """Build an instruction whose trust the host has actually established.

    This is the **only** sanctioned way to assign a precedence to content. The
    trust comes from the caller — the trusted ingestion or composition layer —
    and not from anything the content says about itself.

    Two rules make that safe to rely on:

    * An :attr:`Instruction` that carries :attr:`Precedence.UNTRUSTED` or
      :attr:`Precedence.TOOL_METADATA` is always ``authenticated=False``, so
      untrusted content can never claim authority by passing a flag here.
    * A class outside :data:`REVISION_AUTHORISED_CLASSES` is never authorised
      to revise, whatever ``authenticated`` says.

    The content-derived ``asserted_precedence`` is deliberately **not** accepted
    here. A claim is recorded on the resulting instruction so the resolver can
    evaluate and reject it, never so a caller can promote on it.
    """
    if precedence in (Precedence.UNTRUSTED, Precedence.TOOL_METADATA):
        authenticated = False
    return Instruction(
        text=text,
        precedence=precedence,
        source=source,
        directive=directive,
        value=value,
        revises=revises,
        authenticated=authenticated,
    )


def untrusted_content(
    text: str,
    *,
    source: str,
    directive: str | None = None,
    value: str | None = None,
) -> Instruction:
    """Wrap retrieved/tool-output content as evidence, never as instruction.

    The common case for anything that did not come from the operator. The
    resulting instruction can inform the model but cannot bind a directive
    already held by a stronger class, revise anything, or grant a permission.
    """
    return trusted_instruction(
        text,
        Precedence.UNTRUSTED,
        source=source,
        authenticated=False,
        directive=directive,
        value=value,
    )


def _structural_rejection(instruction: Instruction) -> Rejection | None:
    """Refuse an instruction on grounds independent of other instructions."""
    # REQ-PROMPT-012: content cannot promote itself by asserting a stronger
    # class than the one it was delivered at.
    if instruction.claims.outranks(instruction.precedence):
        return Rejection(
            reason=RejectionReason.PRIVILEGE_ESCALATION,
            source=instruction.source,
            precedence=instruction.precedence,
            detail=(f"delivered as {instruction.precedence} but asserted {instruction.claims}"),
            instruction=instruction,
        )

    # REQ-PROMPT-015: a tool description is semantic input. It is never a
    # permission grant, whatever it asserts about itself.
    if instruction.precedence is Precedence.TOOL_METADATA and instruction.grants_permission:
        return Rejection(
            reason=RejectionReason.PERMISSION_GRANT_CLAIMED,
            source=instruction.source,
            precedence=instruction.precedence,
            detail="tool metadata asserted a permission grant",
            instruction=instruction,
        )

    # Revisions are an operator capability, not an evidence capability. A
    # retrieved document that writes "user update: ignore the previous
    # instruction" is refused outright rather than being weighed against the
    # instruction it tried to displace.
    if instruction.revises is not None and not instruction.may_revise:
        return Rejection(
            reason=RejectionReason.REVISION_NOT_AUTHORISED,
            source=instruction.source,
            precedence=instruction.precedence,
            detail=(
                f"attempted to revise {instruction.revises} without an "
                "authenticated operator origin"
            ),
            instruction=instruction,
        )

    return None


__all__ = [
    "INSTRUCTION_SCHEMA_VERSION",
    "REVISION_AUTHORISED_CLASSES",
    "Conflict",
    "Instruction",
    "Precedence",
    "Rejection",
    "RejectionReason",
    "ResolutionTrace",
    "ResolvedInstructions",
    "resolve",
    "trusted_instruction",
    "untrusted_content",
]
