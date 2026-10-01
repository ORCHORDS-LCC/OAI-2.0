"""Verification graph tests."""

from __future__ import annotations

from oai2.core import EvidenceId
from oai2.verification import (
    ClaimState,
    ClaimStatus,
    Evidence,
    EvidenceClass,
    EvidenceGraph,
    EvidenceStatus,
    SupportNeed,
    VerificationContext,
)


def test_evidence_graph_upsert_combines_support() -> None:
    g = EvidenceGraph()
    e1 = Evidence(
        id=EvidenceId("e1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="repo confirms foo",
    )
    e2 = Evidence(
        id=EvidenceId("e2"),
        cls=EvidenceClass.RUNTIME_OBS,
        summary="runtime confirms foo",
    )
    g.add("c1", e1, supports=True)
    g.add("c1", e2, supports=True)
    node = g.nodes["c1"]
    assert len(node.supporting) == 2
    assert g.status_for("c1") is EvidenceStatus.UNVERIFIED


def test_evidence_graph_dedupes_by_id() -> None:
    g = EvidenceGraph()
    e = Evidence(
        id=EvidenceId("dup"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="once",
    )
    g.add("c1", e, supports=True)
    g.add("c1", e, supports=True)
    assert len(g.nodes["c1"].supporting) == 1


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
