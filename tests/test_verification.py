"""Verification graph tests."""

from __future__ import annotations

from oai2.core import EvidenceId
from oai2.verification import (
    ClaimState,
    ClaimStatus,
    Evidence,
    EvidenceClass,
    EvidenceGraph,
    EvidenceNode,
    EvidenceStatus,
    SupportNeed,
    VerificationContext,
)


def _evidence(eid: str, cls: EvidenceClass = EvidenceClass.REPO_SOURCE) -> Evidence:
    return Evidence(id=EvidenceId(eid), cls=cls, summary=f"summary {eid}")


def test_evidence_graph_upsert_combines_support() -> None:
    g = EvidenceGraph()
    e1 = _evidence("e1")
    e2 = _evidence("e2", EvidenceClass.RUNTIME_OBS)
    g.add("c1", e1, supports=True)
    g.add("c1", e2, supports=True)
    node = g.nodes["c1"]
    assert len(node.supporting) == 2
    # Two supporting evidences without any refuting evidence: the verdict is
    # settled, so the node must be VERIFIED, not the dead-default UNVERIFIED.
    assert g.status_for("c1") is EvidenceStatus.VERIFIED
    assert node.status is EvidenceStatus.VERIFIED


def test_evidence_graph_dedupes_by_id() -> None:
    g = EvidenceGraph()
    e = _evidence("dup")
    g.add("c1", e, supports=True)
    g.add("c1", e, supports=True)
    assert len(g.nodes["c1"].supporting) == 1


def test_empty_node_starts_unverified() -> None:
    g = EvidenceGraph()
    assert g.status_for("unknown_claim") is EvidenceStatus.UNVERIFIED
    node = EvidenceNode(claim_id="c_empty")
    assert node.status is EvidenceStatus.UNVERIFIED


def test_supporting_only_is_verified() -> None:
    g = EvidenceGraph()
    g.add("c", _evidence("s1"), supports=True)
    assert g.status_for("c") is EvidenceStatus.VERIFIED


def test_refuting_only_is_verified() -> None:
    g = EvidenceGraph()
    g.add("c", _evidence("r1"), supports=False)
    # Refuting evidence is settled (the claim is refuted) — only the joint
    # presence of supporting AND refuting makes the verdict conflicting.
    assert g.status_for("c") is EvidenceStatus.VERIFIED


def test_supporting_and_refuting_is_conflicting() -> None:
    g = EvidenceGraph()
    g.add("c", _evidence("s1"), supports=True)
    g.add("c", _evidence("r1"), supports=False)
    assert g.status_for("c") is EvidenceStatus.CONFLICTING


def test_evidence_graph_merge_recomputes_status() -> None:
    # Two nodes that both carry supporting evidence for the same claim must
    # merge into a VERIFIED node, not preserve the dead UNVERIFIED default.
    n1 = EvidenceNode(claim_id="c1", supporting=(_evidence("e1"),))
    n2 = EvidenceNode(claim_id="c1", supporting=(_evidence("e2"),))
    merged = n1.merge(n2)
    assert merged.status is EvidenceStatus.VERIFIED
    assert len(merged.supporting) == 2


def test_evidence_graph_merge_preserves_conflicting_status() -> None:
    n1 = EvidenceNode(
        claim_id="c1",
        supporting=(_evidence("s1"),),
        refuting=(_evidence("r1"),),
    )
    n2 = EvidenceNode(claim_id="c1", supporting=(_evidence("s2"),))
    merged = n1.merge(n2)
    assert merged.status is EvidenceStatus.CONFLICTING


def test_claim_state_machine_open_filter() -> None:
    ctx = VerificationContext()
    ctx.add(
        ClaimState(
            id="c_open",
            text="x",
            support_need=SupportNeed.SOURCE_FACT,
            status=ClaimStatus.UNVERIFIED,
        )
    )
    ctx.add(
        ClaimState(
            id="c_done",
            text="y",
            support_need=SupportNeed.SOURCE_FACT,
            status=ClaimStatus.SUPPORTED,
        )
    )
    assert [c.id for c in ctx.open()] == ["c_open"]


# --- EvidenceNode status invariant (Refs #239) ---
#
# `EvidenceNode.status` is documented as derived from supporting/refuting
# contents (UNVERIFIED / VERIFIED / CONFLICTING). Every construction path —
# direct constructor, model_validate — must end with the derived value.
# The following tests pin that invariant.


def test_evidence_node_status_default_empty_is_unverified() -> None:
    n = EvidenceNode(claim_id="c_empty_default")
    assert n.status is EvidenceStatus.UNVERIFIED
    assert n.supporting == ()
    assert n.refuting == ()


def test_evidence_node_lying_verified_with_empty_buckets_coerced() -> None:
    # Lying status=VERIFIED with empty supporting/refuting used to produce a
    # node whose status lied about its contents. The validator must coerce
    # to the derived UNVERIFIED value.
    n = EvidenceNode(
        claim_id="c_lying_verified",
        supporting=(),
        refuting=(),
        status=EvidenceStatus.VERIFIED,
    )
    assert n.status is EvidenceStatus.UNVERIFIED


def test_evidence_node_lying_conflicting_with_empty_buckets_coerced() -> None:
    # CONFLICTING with both lists empty is impossible by construction;
    # the validator must coerce to UNVERIFIED.
    n = EvidenceNode(
        claim_id="c_lying_conflicting",
        supporting=(),
        refuting=(),
        status=EvidenceStatus.CONFLICTING,
    )
    assert n.status is EvidenceStatus.UNVERIFIED


def test_evidence_node_lying_verified_with_full_buckets_coerced() -> None:
    # status=VERIFIED with both supporting and refuting populated is
    # actually CONFLICTING — the validator must coerce.
    n = EvidenceNode(
        claim_id="c_lying_verified_full",
        supporting=(_evidence("s1"),),
        refuting=(_evidence("r1"),),
        status=EvidenceStatus.VERIFIED,
    )
    assert n.status is EvidenceStatus.CONFLICTING


def test_evidence_node_truthful_status_unchanged() -> None:
    # Truthful inputs are unchanged — the validator is a guard, not a
    # transformer of correct state.
    n_truthful = EvidenceNode(
        claim_id="c_truthful_verified",
        supporting=(_evidence("e1"),),
        refuting=(),
        status=EvidenceStatus.VERIFIED,
    )
    assert n_truthful.status is EvidenceStatus.VERIFIED


def test_evidence_node_model_validate_lying_status_coerced() -> None:
    # Same invariant must hold for model_validate (the deserialization path
    # used by JSON round-trips).
    n = EvidenceNode.model_validate(
        {
            "claim_id": "c_via_validate",
            "supporting": [],
            "refuting": [],
            "status": "verified",
        }
    )
    assert n.status is EvidenceStatus.UNVERIFIED


def test_evidence_node_lying_status_does_not_survive_truthful_construction() -> None:
    # When the caller passes truthful status alongside truthful contents
    # (the normal EvidenceGraph.add/merge path), the validator is a no-op.
    n = EvidenceNode(
        claim_id="c_truthful_default",
        supporting=(_evidence("e1"),),
        refuting=(_evidence("e2"),),
        status=EvidenceStatus.CONFLICTING,
    )
    assert n.status is EvidenceStatus.CONFLICTING
    assert n.net_count == 0


def test_evidence_node_model_copy_rederives_status() -> None:
    node = EvidenceNode(
        claim_id="c_copy",
        supporting=(_evidence("copy_support"),),
        status=EvidenceStatus.VERIFIED,
    )
    copied = node.model_copy(
        update={"supporting": (), "refuting": (), "status": EvidenceStatus.VERIFIED}
    )
    assert copied.status is EvidenceStatus.UNVERIFIED
    assert copied.supporting == ()
    assert copied.refuting == ()


def test_evidence_node_assignment_rederives_status() -> None:
    node = EvidenceNode(
        claim_id="c_assignment",
        supporting=(_evidence("assignment_support"),),
        status=EvidenceStatus.VERIFIED,
    )
    node.supporting = ()
    assert node.status is EvidenceStatus.UNVERIFIED
    node.status = EvidenceStatus.VERIFIED
    assert node.status is EvidenceStatus.UNVERIFIED
