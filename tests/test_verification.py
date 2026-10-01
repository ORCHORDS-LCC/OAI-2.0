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
