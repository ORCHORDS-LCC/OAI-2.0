"""q-pipe -> OAI-2.0 knowledge contract: temporal, trust and isolation.

Covers the fields OAI-2.0 #200/#207/#185 need on the canonical record, and the
gates that decide what may enter a store that serves every caller.

The decisive property is that claim identity and content version are separate
values. A claim whose recommendation was flipped keeps its claim_key and gains
a new content_version; if those were conflated, two revisions of one claim
would be indistinguishable from two unrelated claims, and supersession and
conflict reconciliation would be impossible.
"""

from __future__ import annotations

import json

import pytest

from oai2.core import Status
from oai2.knowledge.qpipe_import import (
    DEFAULT_SOURCES,
    EXPORTABLE_SCOPES,
    OPT_IN_SOURCES,
    ImportPolicy,
    import_qpipe_rows,
)
from oai2.knowledge.transport import (
    D1KnowledgeIndexRecord,
    R2BodyDescriptor,
    VectorizeMetadata,
)


def _row(
    *,
    ext: str,
    rid: int,
    steps: list[str],
    risks: list[str] | None = None,
    files: list[str] | None = None,
    claim_key: str = "ck-deploy",
    source_version: str = "v1",
    scope: str = "generic",
    source: str = "recipe_candidates",
    provenance: str = "verified",
    superseded_by: int | None = None,
) -> dict[str, object]:
    return {
        "id": rid,
        "source": source,
        "external_id": ext,
        "scope": scope,
        "fingerprint": "deploy staging verification rollback release",
        "body_json": json.dumps(
            {"guidance": {"steps": steps, "risks": risks or [], "files": files or []}}
        ),
        "capture_count": 4,
        "success_count": 3,
        "failure_count": 1,
        "verified_count": 2,
        "status": "promoted",
        "matcher_version": "v1",
        "source_updated_at": 1_700_000_000,
        "imported_at": 1_700_000_100,
        "promoted_at": 1_700_000_200,
        "promoter": "campaign",
        "last_used_at": None,
        "claim_key": claim_key,
        "source_version": source_version,
        "domain": "ops",
        "provenance_state": provenance,
        "superseded_by": superseded_by,
        "superseded_at": 1_700_009_000 if superseded_by is not None else None,
    }


STEP_A = "Verify the staged deployment health check before promoting a release."
STEP_B = "Roll back as soon as the error budget is half exhausted."


def _policy() -> ImportPolicy:
    return ImportPolicy(allow_recipe_candidates=True)


class TestClaimIdentityIsCarriedAndDistinct:
    def test_revisions_of_one_claim_share_identity_and_differ_in_version(self):
        rows = [
            _row(ext="v1", rid=1, steps=[STEP_A], source_version="v1"),
            _row(ext="v2", rid=2, steps=[STEP_A, STEP_B], source_version="v2"),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 2
        first, second = rep.imported
        assert first.claim_key == second.claim_key == "ck-deploy"
        assert first.content_version == "v1"
        assert second.content_version == "v2"
        # ...and each revision still has its own exact content digest.
        assert first.content_hash != second.content_hash


class TestSupersessionIsLinkedNotRetired:
    def test_superseded_record_stays_traceable_and_is_not_default(self):
        rows = [
            _row(ext="old", rid=1, steps=[STEP_A], source_version="v1",
                 superseded_by=2),
            _row(ext="new", rid=2, steps=[STEP_A, STEP_B], source_version="v2"),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        by_ext = {o.source_uri.rsplit("/", 1)[-1]: o for o in rep.imported}
        old, new = by_ext["old"], by_ext["new"]

        assert old.superseded_by == "qpipe:recipe_candidates:new"
        assert old.superseded_at == 1_700_009_000
        # REQ-TEMP-004/013: the loser is retained, linked, and identifiable as
        # historical rather than silently removed.
        assert old.status is Status.EXPERIMENTAL
        assert new.status is Status.IMPLEMENTED
        assert new.superseded_by is None

    def test_unresolvable_supersession_is_reported_not_faked(self):
        """A target outside the import set must not become a plausible link."""
        rows = [_row(ext="orphan", rid=7, steps=[STEP_A], superseded_by=999)]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 1
        assert rep.imported[0].superseded_by is None
        assert rep.unresolved_supersession == [("orphan", 999)]


class TestIsolationGatesOnAStoreThatServesEveryone:
    def test_private_scope_is_refused(self):
        rep = import_qpipe_rows(
            [_row(ext="p", rid=1, steps=[STEP_A], scope="private")], policy=_policy()
        )
        assert rep.imported == []
        assert rep.skipped_non_exportable_scope == ["p"]

    @pytest.mark.parametrize(
        "scope",
        ["client:acme", "client:acme:agent", "session:s1", "project:oai",
         "unrecognised-scope"],
    )
    def test_identity_bound_and_unknown_scopes_are_refused(self, scope: str) -> None:
        rep = import_qpipe_rows(
            [_row(ext="s", rid=1, steps=[STEP_A], scope=scope)], policy=_policy()
        )
        assert rep.imported == [], f"{scope} must not enter a shared store"

    def test_absolute_paths_are_refused(self):
        rep = import_qpipe_rows(
            [_row(ext="h", rid=1, steps=[STEP_A],
                  files=["/Users/someone/src/secret/file.py"])],
            policy=_policy(),
        )
        assert rep.imported == []
        assert rep.skipped_unverified_or_low_quality == ["h"]

    def test_unattributable_provenance_is_refused(self):
        rep = import_qpipe_rows(
            [_row(ext="u", rid=1, steps=[STEP_A], provenance="unattributed")],
            policy=_policy(),
        )
        assert rep.imported == []
        assert rep.skipped_unattributable == ["u"]

    def test_exportable_scopes_are_the_allow_list(self):
        assert EXPORTABLE_SCOPES == {"generic", "global", "local", "router"}
        # Allow-list, not deny-list: anything unclassified is refused.
        assert "private" not in EXPORTABLE_SCOPES


class TestSourceOptInIsNotSilentlyWidened:
    def test_recipe_candidates_requires_the_flag(self):
        row = _row(ext="r", rid=1, steps=[STEP_A])
        assert import_qpipe_rows([row]).imported == []
        assert len(import_qpipe_rows([row], policy=_policy()).imported) == 1

    def test_recipe_candidates_is_not_a_default_source(self):
        assert "recipe-candidates" not in DEFAULT_SOURCES
        assert "recipe-candidates" in OPT_IN_SOURCES

    def test_legacy_source_spelling_is_accepted(self):
        """The live store carries the pre-rename spelling."""
        rows = [_row(ext="r", rid=1, steps=[STEP_A], source="recipe_candidates")]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 1
        assert rep.imported[0].topic.startswith("qpipe:recipe-candidates:")

    def test_unknown_source_is_still_refused(self):
        rows = [_row(ext="x", rid=1, steps=[STEP_A], source="who-knows")]
        assert import_qpipe_rows(rows, policy=_policy()).imported == []


class TestTransportCarriesTheNewFields:
    def _object(self):
        rep = import_qpipe_rows([_row(ext="a", rid=1, steps=[STEP_A, STEP_B])],
                                policy=_policy())
        return rep.imported[0]

    def test_d1_record_carries_temporal_and_trust_fields(self):
        obj = self._object()
        d1 = D1KnowledgeIndexRecord.from_knowledge(
            obj, corpus_revision=3, r2_blob_key="oai2-blobs/x", vectorize_id="v-1"
        )
        assert d1.claim_key == "ck-deploy"
        assert d1.content_version == "v1"
        assert d1.effective_at == 1_700_000_000
        assert d1.trust_class == "retrieved_evidence"
        assert d1.scope_class == "global"
        assert d1.is_active

    def test_d1_is_active_reflects_supersession(self):
        rows = [
            _row(ext="old", rid=1, steps=[STEP_A], superseded_by=2),
            _row(ext="new", rid=2, steps=[STEP_A, STEP_B]),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        actives = [
            D1KnowledgeIndexRecord.from_knowledge(
                o, corpus_revision=1, r2_blob_key=None, vectorize_id=None
            ).is_active
            for o in rep.imported
        ]
        assert actives == [False, True]

    def test_r2_body_hash_is_verified(self):
        obj = self._object()
        desc = R2BodyDescriptor.from_knowledge(obj)
        assert desc.object_key == f"oai2-blobs/{obj.content_hash}"
        assert desc.size_bytes == len(obj.content.encode("utf-8"))

    def test_vectorize_metadata_carries_stable_id_and_version(self):
        obj = self._object()
        meta = VectorizeMetadata.from_knowledge(obj, embedding_version="embed-v1")
        assert meta.knowledge_id == obj.knowledge_id
        assert meta.content_hash == obj.content_hash
        assert meta.embedding_version == "embed-v1"

    def test_retrieved_evidence_is_never_marked_as_an_instruction(self):
        obj = self._object()
        assert obj.trust_class == "retrieved_evidence"
        assert obj.status in {Status.IMPLEMENTED, Status.EXPERIMENTAL}
