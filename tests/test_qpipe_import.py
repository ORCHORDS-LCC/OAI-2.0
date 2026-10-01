"""Tests for the q-pipe knowledge importer.

Also exercises a small (50-row) synthetic pilot that mimics what a
real q-pipe ``learning_recipes`` SELECT would return. The pilot is
deterministic and contains a mix of allowed / disallowed / rejected /
duplicate rows so dedupe and sanitization are both covered.
"""

from __future__ import annotations

import json

from oai2.knowledge import (
    CloudflareKnowledgeStore,
    MockCloudflareBindings,
    QPipeSource,
    QPipeStatus,
    derive_authority,
    import_qpipe_rows,
)


def _row(
    *,
    sid: str,
    ext: str,
    source: str = "scenario-forge",
    scope: str = "global",
    status: str = "candidate",
    body: dict[str, object] | None = None,
    captures: int = 4,
    successes: int = 3,
    failures: int = 1,
    verified: int = 0,
    matcher: str = "v1",
) -> dict[str, object]:
    body_json = json.dumps(body or {"actions": [{"tool": "echo", "arguments": {"msg": "hi"}}]})
    return {
        "source": source,
        "external_id": ext,
        "scope": scope,
        "fingerprint": f"fp_{sid}",
        "body_json": body_json,
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


# ----- Sanitization -----


def test_sanitize_redacts_secrets_in_action_arguments() -> None:
    from oai2.knowledge.qpipe_import import _sanitize_body

    body = {
        "actions": [
            {
                "tool": "http",
                "arguments": {
                    "Authorization": "Bearer abc.def.ghi",
                    "url": "https://x.example/api?api_key=zzz",
                },
            }
        ]
    }
    sanitized = _sanitize_body(json.dumps(body))
    args = sanitized["actions"][0]["arguments"]  # type: ignore[index]
    assert args["Authorization"] == "<redacted>"  # type: ignore[index]
    assert "<redacted>" in args["url"]  # type: ignore[index]


def test_sanitize_redacts_absolute_paths() -> None:
    from oai2.knowledge.qpipe_import import _sanitize_body

    body = json.dumps({"body": "see /etc/secrets.conf for token=xyz"})
    out = _sanitize_body(body)
    g = out["guidance"]  # type: ignore[index]
    assert "<path>" in g
    assert "<redacted>" in g
    assert "/etc/secrets.conf" not in g


# ----- Authority mapping -----


def test_authority_is_smoothed_success_rate_capped_at_0_95() -> None:
    # 99 successes out of 100 -> smoothed 100/102 ≈ 0.980, capped at 0.95.
    assert derive_authority(100, 99) == 0.95
    # 0 captures -> neutral prior.
    assert derive_authority(0, 0) == 0.5
    # 2 captures, 0 successes -> Laplace (0+1)/(2+2) = 0.25.
    assert 0.20 <= derive_authority(2, 0) <= 0.30
    # 1 capture, 0 successes -> (0+1)/(1+2) = 0.333.
    assert 0.30 <= derive_authority(1, 0) <= 0.36


# ----- Dedupe & source/status rules -----


def test_import_dedupes_on_source_external_id() -> None:
    rows = [
        _row(sid="a", ext="e1"),
        _row(sid="b", ext="e1"),  # duplicate of e1
        _row(sid="c", ext="e2"),
    ]
    rep = import_qpipe_rows(rows)
    assert len(rep.imported) == 2
    assert rep.skipped_duplicate == ["e1"]


def test_import_skips_disallowed_source() -> None:
    rows = [
        _row(sid="a", ext="x", source="rogue-corpus"),
        _row(sid="b", ext="y", source=QPipeSource.TERMINAL_BENCH.value),
    ]
    rep = import_qpipe_rows(rows)
    assert rep.skipped_disallowed_source == ["x"]
    assert len(rep.imported) == 1


def test_import_skips_rejected_rows() -> None:
    rows = [
        _row(sid="a", ext="x", status="rejected"),
        _row(sid="b", ext="y", status="promoted"),
    ]
    rep = import_qpipe_rows(rows)
    assert rep.skipped_rejected == ["x"]
    assert len(rep.imported) == 1
    assert rep.imported[0].status.value == "IMPLEMENTED"


def test_import_rejects_malformed_status() -> None:
    rows = [
        _row(sid="a", ext="x", status="frobnicated"),
    ]
    rep = import_qpipe_rows(rows)
    assert len(rep.imported) == 0
    assert rep.rejected_malformed and rep.rejected_malformed[0][0] == "x"


# ----- Topic + authority preservation -----


def test_import_maps_topic_authority_and_source_uri() -> None:
    rows = [_row(sid="a", ext="x", source="scenario-forge", scope="router")]
    rep = import_qpipe_rows(rows)
    obj = rep.imported[0]
    assert obj.topic == "qpipe:scenario-forge:router"
    assert 0.0 < obj.authority < 1.0
    assert obj.source_uri is not None
    assert obj.source_uri.startswith("qpipe://scenario-forge/")


# ----- Round-trip through Cloudflare adapter -----


def test_pilot_50_rows_round_trip_through_cloudflare_adapter() -> None:
    """50-row synthetic pilot: every imported row survives a put/get/retrieve."""
    rows: list[dict[str, object]] = []
    # Cycle across the 3 allow-listed sources, 4 statuses, and dedupe pattern.
    sources = [s.value for s in QPipeSource]
    statuses = ["candidate", "promoted", "rejected"]
    for i in range(50):
        src = sources[i % 3]
        ext = f"recipe-{i // 3}"  # intentional duplicates across sources
        rows.append(
            _row(
                sid=str(i),
                ext=ext,
                source=src,
                status=statuses[i % 3],
                scope=("global" if i % 2 == 0 else "router"),
                captures=10 + i,
                successes=5 + (i % 6),
            )
        )
    # Plus a disallowed-source row to exercise the filter.
    rows.append(_row(sid="99", ext="bad", source="rogue-corpus"))

    rep = import_qpipe_rows(rows)
    assert rep.total_seen == len(rows)
    assert rep.skipped_disallowed_source == ["bad"]
    # Dedupe: only first occurrence of each (source, ext) survives.
    assert len(rep.imported) <= len(rows)

    # Round-trip through the Cloudflare adapter.
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b)
    for obj in rep.imported:
        store.put(obj)

    from oai2.knowledge import RetrievalRequest

    res = store.retrieve(
        RetrievalRequest(topic="qpipe:", limit=200, min_authority=0.0)
    )
    assert len(res.objects) == len(rep.imported)
    # Each fetched object retains its sanitized body (JSON-able).
    for fetched in res.objects:
        json.loads(fetched.content)


# ----- Importable by package -----


def test_qpipe_symbols_are_public() -> None:
    from oai2.knowledge import (
        ImportReport,
        QPipeRow,
        QPipeSource,
    )

    assert QPipeSource.SCENARIO_FORGE.value == "scenario-forge"
    assert QPipeStatus.PROMOTED.value == "promoted"
    assert QPipeRow.__name__ == "QPipeRow"
    assert ImportReport.__name__ == "ImportReport"
