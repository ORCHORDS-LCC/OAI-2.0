"""Tests for the strict q-pipe -> OAI-2.0 knowledge import contract."""

from __future__ import annotations

import json

from oai2.knowledge import (
    CloudflareKnowledgeStore,
    ImportPolicy,
    MockCloudflareBindings,
    QPipeSource,
    QPipeStatus,
    derive_authority,
    import_qpipe_rows,
)


def _safe_guidance() -> dict[str, object]:
    return {
        "guidance": {
            "steps": ["inspect current source", "apply the smallest verified change"],
            "files": ["relative/module.py"],
            "verification": ["run targeted tests"],
            "risks": ["avoid unrelated changes"],
        }
    }


def _row(
    *,
    sid: str,
    ext: str,
    source: str = "scenario-forge",
    scope: str = "global",
    status: str = "promoted",
    body: dict[str, object] | None = None,
    captures: int = 4,
    successes: int = 3,
    failures: int = 1,
    verified: int = 1,
    matcher: str = "v1",
) -> dict[str, object]:
    numeric_id = abs(hash((sid, ext))) % 1_000_000 + 1
    return {
        "id": numeric_id,
        "source": source,
        "external_id": ext,
        "scope": scope,
        "fingerprint": f"repair verified workflow {sid}",
        "body_json": json.dumps(body or _safe_guidance()),
        "capture_count": captures,
        "success_count": successes,
        "failure_count": failures,
        "verified_count": verified,
        "status": status,
        "matcher_version": matcher,
        "source_updated_at": 1_700_000_000,
        "imported_at": 1_700_000_100,
        "promoted_at": 1_700_000_200 if status == "promoted" else None,
        "promoter": "test" if status == "promoted" else None,
        "last_used_at": 1_700_000_300,
    }


def test_rejects_secret_bearing_guidance() -> None:
    body = _safe_guidance()
    body["guidance"]["steps"] = ["use token=abc"]  # type: ignore[index]
    rep = import_qpipe_rows([_row(sid="a", ext="x", body=body)])
    assert rep.imported == []
    assert rep.skipped_unverified_or_low_quality == ["x"]


def test_rejects_absolute_paths_in_public_guidance() -> None:
    body = _safe_guidance()
    body["guidance"]["files"] = ["/etc/secrets.conf"]  # type: ignore[index]
    rep = import_qpipe_rows([_row(sid="a", ext="x", body=body)])
    assert rep.imported == []
    assert rep.skipped_unverified_or_low_quality == ["x"]


def test_authority_is_smoothed_success_rate_capped_at_0_95() -> None:
    assert derive_authority(100, 99) == 0.95
    assert derive_authority(0, 0) == 0.5
    assert 0.20 <= derive_authority(2, 0) <= 0.30
    assert 0.30 <= derive_authority(1, 0) <= 0.36


def test_import_dedupes_only_after_row_is_eligible() -> None:
    rows = [
        _row(sid="bad", ext="e1", status="candidate"),
        _row(sid="good", ext="e1", status="promoted"),
        _row(sid="dup", ext="e1", status="promoted"),
    ]
    rep = import_qpipe_rows(rows)
    assert len(rep.imported) == 1
    assert rep.skipped_not_promoted == ["e1"]
    assert rep.skipped_duplicate == ["e1"]


def test_import_skips_disallowed_source() -> None:
    rows = [
        _row(sid="a", ext="x", source="rogue-corpus"),
        _row(sid="b", ext="y", source=QPipeSource.TERMINAL_BENCH.value),
    ]
    rep = import_qpipe_rows(rows)
    assert rep.skipped_disallowed_source == ["x"]
    assert len(rep.imported) == 1


def test_android_curriculum_requires_explicit_opt_in() -> None:
    row = _row(
        sid="a",
        ext="android-1",
        source=QPipeSource.ANDROID_CURRICULUM.value,
    )
    blocked = import_qpipe_rows([row])
    assert blocked.imported == []
    assert blocked.skipped_requires_opt_in == ["android-1"]

    allowed = import_qpipe_rows(
        [row],
        policy=ImportPolicy(allow_android_curriculum=True),
    )
    assert len(allowed.imported) == 1


def test_import_skips_rejected_rows() -> None:
    rows = [
        _row(sid="a", ext="x", status="rejected"),
        _row(sid="b", ext="y", status="promoted"),
    ]
    rep = import_qpipe_rows(rows)
    assert rep.skipped_rejected == ["x"]
    assert len(rep.imported) == 1
    assert rep.imported[0].status.value == "IMPLEMENTED"


def test_candidates_are_not_cloud_knowledge_by_default() -> None:
    rep = import_qpipe_rows([_row(sid="a", ext="x", status="candidate")])
    assert rep.imported == []
    assert rep.skipped_not_promoted == ["x"]


def test_promoted_row_must_be_independently_verified() -> None:
    rep = import_qpipe_rows(
        [_row(sid="a", ext="x", status="promoted", verified=0)]
    )
    assert rep.imported == []
    assert rep.skipped_unverified_or_low_quality == ["x"]


def test_success_must_exceed_failure_and_cover_verified_count() -> None:
    bad_ratio = import_qpipe_rows(
        [_row(sid="a", ext="x", successes=1, failures=1, verified=1)]
    )
    assert bad_ratio.imported == []

    impossible_verified = import_qpipe_rows(
        [_row(sid="b", ext="y", successes=1, failures=0, verified=2)]
    )
    assert impossible_verified.imported == []


def test_import_rejects_malformed_status() -> None:
    rep = import_qpipe_rows([_row(sid="a", ext="x", status="frobnicated")])
    assert rep.imported == []
    assert rep.rejected_malformed and rep.rejected_malformed[0][0] == "x"


def test_content_hash_matches_qpipe_cloudflare_guidance_hash_shape() -> None:
    row = _row(sid="a", ext="x")
    rep = import_qpipe_rows([row])
    obj = rep.imported[0]
    guidance = _safe_guidance()["guidance"]
    exact_json = json.dumps(guidance, ensure_ascii=False, separators=(",", ":"))

    from oai2.knowledge import sha256_hex

    assert obj.content == exact_json
    assert obj.content_hash == sha256_hex(exact_json)


def test_import_maps_topic_authority_and_source_uri() -> None:
    rows = [_row(sid="a", ext="x", source="scenario-forge", scope="router")]
    rep = import_qpipe_rows(rows)
    obj = rep.imported[0]
    assert obj.topic == "qpipe:scenario-forge:router"
    assert 0.0 < obj.authority < 1.0
    assert obj.source_uri is not None
    assert obj.source_uri.startswith("qpipe://scenario-forge/")


def test_synthetic_50_row_pilot_round_trip_through_cloudflare_contract() -> None:
    rows: list[dict[str, object]] = []
    sources = [
        QPipeSource.SCENARIO_FORGE.value,
        QPipeSource.TERMINAL_BENCH.value,
    ]
    for i in range(50):
        rows.append(
            _row(
                sid=str(i),
                ext=f"recipe-{i}",
                source=sources[i % len(sources)],
                scope=("global" if i % 2 == 0 else "router"),
                captures=10 + i,
                successes=5 + (i % 6),
                failures=1,
                verified=1,
            )
        )

    rep = import_qpipe_rows(rows)
    assert rep.total_seen == 50
    assert len(rep.imported) == 50

    bindings = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(bindings, embedding_digest="test-embed-digest-v1")
    for obj in rep.imported:
        store.put(obj)

    from oai2.knowledge import RetrievalRequest

    res = store.retrieve(
        RetrievalRequest(topic="qpipe:", limit=200, min_authority=0.0)
    )
    assert len(res.objects) == 50
    for fetched in res.objects:
        json.loads(fetched.content)


def test_qpipe_symbols_are_public() -> None:
    from oai2.knowledge import ImportReport, QPipeRow

    assert QPipeSource.SCENARIO_FORGE.value == "scenario-forge"
    assert QPipeStatus.PROMOTED.value == "promoted"
    assert QPipeRow.__name__ == "QPipeRow"
    assert ImportReport.__name__ == "ImportReport"


def test_learning_recipes_public_shape_with_parsed_body_is_accepted() -> None:
    row = _row(sid="parsed", ext="parsed-1")
    row["body"] = json.loads(str(row.pop("body_json")))
    rep = import_qpipe_rows([row])
    assert len(rep.imported) == 1


def test_missing_or_zero_recipe_id_is_rejected() -> None:
    row = _row(sid="id", ext="id-1")
    row["id"] = 0
    rep = import_qpipe_rows([row])
    assert rep.imported == []
    assert rep.rejected_malformed and rep.rejected_malformed[0][0] == "id-1"


def test_absolute_path_detection_rejects_unix_and_windows_paths() -> None:
    unix = _safe_guidance()
    unix["guidance"]["files"] = ["/private/tmp/file.txt"]  # type: ignore[index]
    windows = _safe_guidance()
    windows["guidance"]["files"] = ["C:\\Users\\Example\\secret.txt"]  # type: ignore[index]

    assert import_qpipe_rows([_row(sid="u", ext="unix", body=unix)]).imported == []
    assert import_qpipe_rows([_row(sid="w", ext="win", body=windows)]).imported == []


def test_qpipe_compatibility_revision_is_pinned() -> None:
    from oai2.knowledge.qpipe_import import (
        QPIPE_COMPATIBILITY_SOURCE_BLOBS,
        QPIPE_COMPATIBILITY_SOURCE_REVISION,
    )

    assert QPIPE_COMPATIBILITY_SOURCE_REVISION == "39f7e791aa38e8fd9e006aee9c15686ec51a3cf2"
    assert QPIPE_COMPATIBILITY_SOURCE_BLOBS == {
        "qpipe/cloudflare_learning.py": "3cf94523564fdd919bec177f71740ff7e979f805",
        "qpipe/memory.py": "f7188d3128e2c647ece9f457b053dcb3d7b705a8",
    }
