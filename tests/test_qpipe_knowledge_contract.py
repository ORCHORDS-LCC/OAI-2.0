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
from oai2.knowledge.abstraction import (
    MAX_SUPERSEDED_BY_LENGTH,
    InMemoryKnowledgeStore,
    RetrievalRequest,
)
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

        # Canonical, hyphenated source -- the same namespace the topic uses.
        # Previously this read "qpipe:recipe_candidates:new", a namespace
        # the store never emits, which made the link unresolvable.
        assert old.superseded_by == "qpipe:recipe-candidates:new"
        assert old.superseded_at == 1_700_009_000
        # REQ-TEMP-004/013: the loser is retained, linked, and identifiable as
        # historical rather than silently removed.
        assert old.status is Status.EXPERIMENTAL
        assert new.status is Status.IMPLEMENTED
        assert new.superseded_by is None

    def test_supersession_ref_uses_the_canonical_source_namespace(self):
        """The link must point at a namespace the store actually writes.

        The topic is written as `qpipe:{_canonical_source(source)}:{scope}`,
        so a ref built from the RAW source names `qpipe:recipe_candidates:new`
        while every topic reads `qpipe:recipe-candidates:...`. The link would
        then be unresolvable, defeating the REQ-TEMP-004 traceability it
        exists to provide.
        """
        rows = [
            _row(ext="old", rid=1, steps=[STEP_A], superseded_by=2),
            _row(ext="new", rid=2, steps=[STEP_A, STEP_B]),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        loser = next(o for o in rep.imported if o.superseded_by is not None)
        assert loser.superseded_by == "qpipe:recipe-candidates:new"
        ref_namespace = ":".join(loser.superseded_by.split(":")[:2])
        topic_namespace = ":".join(loser.topic.split(":")[:2])
        assert ref_namespace == topic_namespace

    def test_ref_namespace_matches_for_every_importable_source(self):
        """Opposite-direction guard: canonicalisation must not corrupt a
        source that is already canonical.

        `_canonical_source` is identity for sources outside the alias set, so
        a non-aliased source must produce the same ref it always did. This
        is the direction that a naive "always rewrite the namespace" fix
        would break.
        """
        for source in ("scenario-forge", "terminal-bench-2.1"):
            policy = ImportPolicy()
            rows = [
                _row(ext="old", rid=1, steps=[STEP_A], source=source,
                     superseded_by=2),
                _row(ext="new", rid=2, steps=[STEP_A, STEP_B], source=source),
            ]
            rep = import_qpipe_rows(rows, policy=policy)
            loser = next(o for o in rep.imported if o.superseded_by is not None)
            assert loser.superseded_by == f"qpipe:{source}:new"
            assert ":".join(loser.superseded_by.split(":")[:2]) == ":".join(
                loser.topic.split(":")[:2]
            )

    def test_supersession_never_crosses_a_source_namespace(self):
        """A same-numbered id in another source is not this row's target.

        ``recipe_id`` is unique per source, not globally, so two sources may
        both contain id 1. Keying the winner map on the bare integer let the
        last source win, and a row was linked to another source's external_id
        inside its own namespace -- ``qpipe:scenario-forge:TB`` for a record
        that does not exist, with ``unresolved_supersession`` left empty so the
        corruption was reported nowhere. That is precisely the "link that
        names the wrong or a nonexistent record" the module says it will not
        render.
        """
        rows = [
            _row(ext="SF", rid=1, steps=[STEP_A], source="scenario-forge"),
            _row(ext="TB", rid=1, steps=[STEP_B], source="terminal-bench-2.1"),
            _row(
                ext="OLD",
                rid=2,
                steps=[STEP_A, STEP_B],
                source="scenario-forge",
                superseded_by=1,
            ),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 3
        linked = [o for o in rep.imported if o.superseded_by is not None]
        assert len(linked) == 1
        # The winner is the one in the losser's OWN source.
        assert linked[0].superseded_by == "qpipe:scenario-forge:SF"
        assert "TB" not in linked[0].superseded_by
        # The target is genuinely in the import, so nothing is unresolved.
        assert rep.unresolved_supersession == []

    def test_supersession_resolves_across_source_alias_spellings(self):
        """Opposite-direction guard: scoping by source must not over-refuse.

        The winner map is keyed on the CANONICAL source, because the ref is
        built from the canonical namespace too. If the key used the raw
        spelling, a winner stored as ``recipe_candidates`` would not match a
        loser stored as ``recipe-candidates`` -- the same source under two
        spellings -- and a real, resolvable link would be wrongly reported as
        unresolved. This is the aliasing ``_canonical_source`` exists to
        absorb, and scoping the key must not throw it away.
        """
        rows = [
            _row(ext="new", rid=1, steps=[STEP_A], source="recipe_candidates"),
            _row(
                ext="old",
                rid=2,
                steps=[STEP_A, STEP_B],
                source="recipe-candidates",
                superseded_by=1,
            ),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 2
        assert rep.unresolved_supersession == []
        linked = [o for o in rep.imported if o.superseded_by is not None]
        assert len(linked) == 1
        assert linked[0].superseded_by == "qpipe:recipe-candidates:new"

    def test_unresolvable_supersession_is_reported_not_faked(self):
        """A target outside the import set must not become a plausible link."""
        rows = [_row(ext="orphan", rid=7, steps=[STEP_A], superseded_by=999)]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 1
        assert rep.imported[0].superseded_by is None
        assert rep.unresolved_supersession == [("orphan", 999)]

    def test_link_at_the_exact_field_bound_is_still_resolved(self):
        """Opposite-direction guard: the fix must not refuse a representable link.

        ``MAX_SUPERSEDED_BY_LENGTH`` is 128 and the
        ``qpipe:recipe_candidates:`` prefix is 24 chars, so a winner
        external_id of exactly 104 produces a ref of exactly 128. That is
        representable and must still be linked; only 105 must be refused.
        """
        winner_ext = "w" * (MAX_SUPERSEDED_BY_LENGTH - len("qpipe:recipe-candidates:"))
        assert len(f"qpipe:recipe-candidates:{winner_ext}") == MAX_SUPERSEDED_BY_LENGTH
        rows = [
            _row(ext=winner_ext, rid=1, steps=[STEP_A]),
            _row(ext="old", rid=2, steps=[STEP_A, STEP_B], superseded_by=1),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 2
        assert rep.unresolved_supersession == []
        linked = [o for o in rep.imported if o.superseded_by is not None]
        assert len(linked) == 1
        assert linked[0].superseded_by == f"qpipe:recipe-candidates:{winner_ext}"
        assert len(linked[0].superseded_by) == MAX_SUPERSEDED_BY_LENGTH

    def test_unrepresentable_link_is_reported_and_the_rest_of_the_batch_survives(self):
        """An over-long derived link must cost one link, not the whole import.

        ``_validate_row`` admits an external_id of up to 160 chars, so a legal
        winner can make a ``superseded_by`` ref longer than the model field
        allows. Before this was gated, that single row raised out of
        ``import_qpipe_rows`` and discarded every other row in the batch.
        """
        prefix = "qpipe:recipe-candidates:"
        winner_ext = "w" * (MAX_SUPERSEDED_BY_LENGTH - len(prefix) + 1)
        rows = [
            _row(ext=winner_ext, rid=1, steps=[STEP_A]),
            _row(ext="old", rid=2, steps=[STEP_A, STEP_B], superseded_by=1),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        # Both rows still import; only the unlinkable link is reported.
        assert len(rep.imported) == 2
        assert rep.unresolved_supersession == [("old", 1)]
        loser = next(o for o in rep.imported if o.source_uri.endswith("/old"))
        assert loser.superseded_by is None
        # The refusal must not silently invent a truncated link either.
        assert not any(
            o.superseded_by and "w" * 40 in o.superseded_by for o in rep.imported
        )

    def test_one_unrepresentable_link_does_not_discard_unrelated_rows(self):
        """Blast radius: unrelated good rows must survive a poison supersession row."""
        prefix = "qpipe:recipe-candidates:"
        winner_ext = "w" * (MAX_SUPERSEDED_BY_LENGTH - len(prefix) + 1)
        rows = [
            _row(ext=f"good-{i}", rid=100 + i, steps=[STEP_A]) for i in range(8)
        ]
        rows += [
            _row(ext=winner_ext, rid=1, steps=[STEP_A]),
            _row(ext="old", rid=2, steps=[STEP_A, STEP_B], superseded_by=1),
        ]
        rep = import_qpipe_rows(rows, policy=_policy())
        assert len(rep.imported) == 10
        assert rep.unresolved_supersession == [("old", 1)]
        # total_seen counts the 10 imported rows plus the 1 reported link.
        assert rep.total_seen == 11

    def test_qpipe_row_external_id_bound_does_not_imply_a_representable_link(self):
        """Document the mismatch that made this reachable.

        The importer admits external_id up to 160 chars while the derived
        superseded_by ref is capped at 128, so the two bounds disagree by
        construction. This test exists so the gap stays visible if either
        bound is changed in isolation.
        """
        assert 160 > MAX_SUPERSEDED_BY_LENGTH


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


class TestSupersededEvidenceIsNotRetrievalDefault:
    """REQ-TEMP-014/025: excluded by default, still reachable on request.

    The mitigation `row_to_knowledge_object` relies on is a status demotion
    to EXPERIMENTAL, on the stated basis that it is then "not
    retrieval-default". That never held: `RetrievalRequest.include_status`
    defaults to `(IMPLEMENTED, EXPERIMENTAL)`, so the demotion excluded the
    row from nothing. These tests pin the field-level gate instead.
    """

    def _store(self) -> InMemoryKnowledgeStore:
        rep = import_qpipe_rows(
            [
                _row(ext="old", rid=1, steps=[STEP_A], superseded_by=2),
                _row(ext="new", rid=2, steps=[STEP_A, STEP_B]),
            ],
            policy=_policy(),
        )
        store = InMemoryKnowledgeStore()
        for obj in rep.imported:
            store.put(obj)
        return store

    def test_a_superseded_record_is_not_returned_by_a_default_retrieve(self):
        store = self._store()
        topic = "qpipe:recipe-candidates:generic"
        result = store.retrieve(RetrievalRequest(topic=topic))
        returned = {o.source_uri.rsplit("/", 1)[-1] for o in result.objects}
        # REQ-TEMP-014: the superseded record must not be retrieval-default.
        assert returned == {"new"}

    def test_a_historical_question_can_still_retrieve_the_superseded_record(self):
        """Opposite-direction guard: the fix must not simply hide it forever.

        REQ-TEMP-025 requires historical retrieval to keep working, so
        exclusion is a default rather than a permanent deletion.
        """
        store = self._store()
        topic = "qpipe:recipe-candidates:generic"
        result = store.retrieve(
            RetrievalRequest(topic=topic, include_superseded=True)
        )
        returned = {o.source_uri.rsplit("/", 1)[-1] for o in result.objects}
        assert returned == {"old", "new"}

    def test_exclusion_does_not_depend_on_the_status_demotion(self):
        """The gate reads the field, not the status.

        If this regressed to a status-based filter it would break the moment
        a superseded row carried IMPLEMENTED, because IMPLEMENTED is in every
        default include set.
        """
        store = self._store()
        topic = "qpipe:recipe-candidates:generic"
        strict = store.retrieve(
            RetrievalRequest(
                topic=topic,
                include_status=(Status.IMPLEMENTED, Status.EXPERIMENTAL),
            )
        )
        assert all(o.superseded_by is None for o in strict.objects)

    def test_every_retrieval_gate_routes_through_the_one_definition(self):
        """The rule must have exactly one owner.

        It was previously written out at each gate. A fourth retrieval path
        added later would have had no way to know the rule existed, and the
        drift would be invisible until a superseded record leaked. This
        pins the three gates AND the record-level accessor to the shared
        predicate, so a future inlining at any one of them fails here.
        """
        import inspect

        from oai2.knowledge import abstraction as abstraction_module
        from oai2.knowledge import cloudflare, cloudflare_runtime, transport

        # The predicate itself is the only place the rule is written.
        assert "superseded_by is None" in inspect.getsource(abstraction_module)

        # The in-memory gate calls it.
        assert "is_active_evidence(" in inspect.getsource(abstraction_module)
        # The semantic D1 gate calls it.
        assert "is_active_evidence(" in inspect.getsource(cloudflare_runtime)
        # The record-level accessor is defined in terms of it.
        assert "is_active_evidence(" in inspect.getsource(transport)
        # And no retrieval gate re-inlines the raw comparison.
        for module in (abstraction_module, cloudflare, cloudflare_runtime):
            body = inspect.getsource(module)
            assert "superseded_by is not None and not request.include_superseded" not in body

    def test_is_active_accessor_agrees_with_the_gate(self):
        """`is_active` promised exclusion; now the gate implements it."""
        store = self._store()
        topic = "qpipe:recipe-candidates:generic"
        active = {o.source_uri.rsplit("/", 1)[-1] for o in
                  store.retrieve(RetrievalRequest(topic=topic)).objects}
        everything = {
            o.source_uri.rsplit("/", 1)[-1]
            for o in store.all()
            if o.superseded_by is None
        }
        assert active == everything
