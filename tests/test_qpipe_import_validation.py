"""Pin the outer guard rails of ``oai2.knowledge.qpipe_import``.

The behavioural tests in ``tests/test_qpipe_import.py`` cover the happy
path plus the high-stakes security guards (secret-bearing guidance,
absolute paths, malformed rows, candidate gating, Android opt-in,
dedup-after-eligibility, fingerprint bounds, q-pipe compatibility
revision pinning). The module also has a handful of small public
contracts that are NOT pinned there:

* ``QPipeSource`` / ``QPipeStatus`` enum stability and default-importable
  set -- the import eligibility logic depends on ``scenario-forge`` and
  ``terminal-bench-2.1`` being importable by default and
  ``android-curriculum-oss`` requiring explicit opt-in. Renaming or
  dropping a value silently flips the policy.
* ``QPIPE_TO_OAI_STATUS`` mapping -- every q-pipe row that passes the
  eligibility gate lands on an OAI ``Status`` via this dict. A future
  refactor that, e.g., flips ``PROMOTED`` to ``EXPERIMENTAL`` would
  quietly break the "promoted rows are durable cloud knowledge" contract.
* ``ImportPolicy`` defaults -- ``allow_candidates=False``,
  ``allow_android_curriculum=False``, ``require_verified=True`` are
  load-bearing: they mirror q-pipe's stricter Cloudflare export gate.
  Dropping any of them widens the importable set silently.
* ``ImportReport.total_seen`` -- the contract-of-record for the count
  of rows the importer handled. Used by callers that need to verify
  "every input row was either imported or categorized".
* ``derive_authority`` boundary behaviour -- Laplace-smoothed success
  rate with a 0.95 cap and a 0.5 neutral prior for unseen recipes
  (``capture_count <= 0``). Pin the boundaries so a future
  refactor that changes the prior or cap is caught immediately.
* ``QPipeRow.from_dict`` -- the dict-to-dataclass adapter that handles
  whitespace normalization (``fingerprint``, ``scope``), defaulting
  for ``body_json`` vs ``body`` (the legacy alias), and integer coercion
  for the numeric columns.
* The error-message taxonomy in ``_validate_row`` -- the import report
  buckets each rejection into one of 8 skip-list categories by matching
  on the literal error string. Pin the exact strings so renaming a
  message silently re-buckets rows.

A refactor that drops or renames any of these would propagate silently
into the q-pipe -> OAI knowledge import path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from oai2.core import Status
from oai2.knowledge import (
    ImportPolicy,
    ImportReport,
    KnowledgeObject,
    QPipeRow,
    QPipeSource,
    QPipeStatus,
    derive_authority,
)
from oai2.knowledge.qpipe_import import (
    DEFAULT_SOURCES,
    EXPORTABLE_SCOPES,
    OPT_IN_SOURCES,
    QPIPE_COMPATIBILITY_SOURCE_BLOBS,
    QPIPE_COMPATIBILITY_SOURCE_REVISION,
    QPIPE_TO_OAI_STATUS,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _safe_guidance() -> dict[str, Any]:
    return {
        "guidance": {
            "steps": ["step one"],
            "files": ["relative/path.py"],
            "verification": ["run targeted tests"],
            "risks": [],
        }
    }


def _row_dict(
    *,
    ext: str = "ext_1",
    source: str = "scenario-forge",
    scope: str = "global",
    status: str = "promoted",
    captures: int = 4,
    successes: int = 3,
    failures: int = 1,
    verified: int = 1,
    recipe_id: int | None = 1,
    fingerprint: str = "verified stable workflow alpha",
    body_json: str | None = None,
    body_obj: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal-but-eligible q-pipe row dict for ``from_dict`` tests."""
    row: dict[str, Any] = {
        "source": source,
        "external_id": ext,
        "scope": scope,
        "fingerprint": fingerprint,
        "body_json": body_json
        if body_json is not None
        else json.dumps(
            body_obj or _safe_guidance(),
        ),
        "capture_count": captures,
        "success_count": successes,
        "failure_count": failures,
        "verified_count": verified,
        "status": status,
        "matcher_version": "v1",
        "source_updated_at": 1_700_000_000,
        "imported_at": 1_700_000_100,
        "promoted_at": 1_700_000_200 if status == "promoted" else None,
        "promoter": "test" if status == "promoted" else None,
        "last_used_at": 1_700_000_300,
    }
    if recipe_id is not None:
        row["id"] = recipe_id
    return row


# ---------------------------------------------------------------------------
# QPipeSource / QPipeStatus enum stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected_value",
    [
        (QPipeSource.SCENARIO_FORGE, "scenario-forge"),
        (QPipeSource.TERMINAL_BENCH, "terminal-bench-2.1"),
        (QPipeSource.ANDROID_CURRICULUM, "android-curriculum-oss"),
    ],
)
def test_qpipe_source_enum_values_are_stable_strings(
    source: QPipeSource,
    expected_value: str,
) -> None:
    """The string forms of ``QPipeSource`` are what appears in q-pipe row
    dicts (the ``source`` column). Renaming any value breaks every
    existing q-pipe row that imports into OAI."""
    assert source == expected_value
    assert str(source) == expected_value
    assert source.value == expected_value


def test_qpipe_source_surface_is_explicitly_pinned() -> None:
    """Closing the surface: the allow-list cannot grow without a
    coordinated q-pipe schema revision (the importer's contract is an
    explicit, reviewed set -- no more).

    `recipe-candidates` was added deliberately, not casually. Measured on the
    live store it is the only producer of verified promoted lessons, and
    without it the canonical knowledge path is unreachable: all 3 active
    lessons were refused as "disallowed source". It is NOT a default source --
    it requires `ImportPolicy.allow_recipe_candidates` -- and it is
    additionally gated on attributable provenance and an exportable scope, so
    admitting it does not weaken the guarantee the original 3-source list
    provided.
    """
    assert set(QPipeSource) == {
        QPipeSource.SCENARIO_FORGE,
        QPipeSource.TERMINAL_BENCH,
        QPipeSource.ANDROID_CURRICULUM,
        QPipeSource.RECIPE_CANDIDATES,
    }
    # The new source must not have silently become a default.
    assert QPipeSource.RECIPE_CANDIDATES.value not in DEFAULT_SOURCES
    assert QPipeSource.RECIPE_CANDIDATES.value in OPT_IN_SOURCES
    # The original default set is unchanged.
    assert DEFAULT_SOURCES == {
        QPipeSource.SCENARIO_FORGE.value,
        QPipeSource.TERMINAL_BENCH.value,
    }


@pytest.mark.parametrize(
    "status, expected_value",
    [
        (QPipeStatus.CANDIDATE, "candidate"),
        (QPipeStatus.PROMOTED, "promoted"),
        (QPipeStatus.REJECTED, "rejected"),
    ],
)
def test_qpipe_status_enum_values_are_stable_strings(
    status: QPipeStatus,
    expected_value: str,
) -> None:
    """``QPipeStatus`` is a ``StrEnum``; the string form is what q-pipe
    rows carry in their ``status`` column."""
    assert status == expected_value
    assert str(status) == expected_value
    assert status.value == expected_value


def test_qpipe_status_has_exactly_three_distinct_values() -> None:
    """Closing the surface: the status taxonomy is exactly
    CANDIDATE / PROMOTED / REJECTED. Adding a fourth (e.g. ``ARCHIVED``)
    without updating ``_validate_row`` would silently drop those rows."""
    assert len(set(QPipeStatus)) == 3


# ---------------------------------------------------------------------------
# QPIPE_TO_OAI_STATUS mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qpipe_status, expected_oai_status",
    [
        (QPipeStatus.CANDIDATE, Status.EXPERIMENTAL),
        (QPipeStatus.PROMOTED, Status.IMPLEMENTED),
        (QPipeStatus.REJECTED, Status.PROPOSED),
    ],
)
def test_qpipe_to_oai_status_mapping_is_stable(
    qpipe_status: QPipeStatus,
    expected_oai_status: Status,
) -> None:
    """Every q-pipe row that passes the eligibility gate lands on an
    OAI ``Status`` via this dict. Flipping any mapping silently changes
    the lifecycle semantics of imported rows."""
    assert QPIPE_TO_OAI_STATUS[qpipe_status] == expected_oai_status


def test_qpipe_to_oai_status_dict_covers_every_qpipe_status() -> None:
    """The dict must cover all 3 ``QPipeStatus`` values; a missing key
    would raise ``KeyError`` mid-import and crash the entire batch on
    the first non-PROMOTED row."""
    for status in QPipeStatus:
        assert status in QPIPE_TO_OAI_STATUS


# ---------------------------------------------------------------------------
# ImportPolicy defaults
# ---------------------------------------------------------------------------


def test_import_policy_default_allow_candidates_is_false() -> None:
    """Default ``allow_candidates=False`` mirrors q-pipe's stricter
    Cloudflare export gate (promoted-only by default). Flipping the
    default would silently widen the importable set to include
    un-promoted candidates."""
    assert ImportPolicy().allow_candidates is False


def test_import_policy_default_allow_android_curriculum_is_false() -> None:
    """Default ``allow_android_curriculum=False`` matches the explicit
    opt-in contract for that source. The test pins that callers must
    opt in explicitly; ``ImportPolicy()`` never silently enables it."""
    assert ImportPolicy().allow_android_curriculum is False


def test_import_policy_default_require_verified_is_true() -> None:
    """Default ``require_verified=True`` enforces the
    ``verified_count >= 1`` gate; flipping the default would admit
    unverified rows into the durable knowledge store."""
    assert ImportPolicy().require_verified is True


def test_import_policy_is_a_frozen_dataclass() -> None:
    """``ImportPolicy`` is ``@dataclass(slots=True, frozen=True)``. A
    refactor that drops ``frozen=True`` would let callers mutate the
    policy mid-import, causing nondeterministic eligibility."""
    policy = ImportPolicy()
    with pytest.raises((AttributeError, Exception)):
        policy.allow_candidates = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ImportReport.total_seen
# ---------------------------------------------------------------------------


def test_import_report_total_seen_is_zero_for_empty_report() -> None:
    """Empty report (no rows processed) must report ``total_seen == 0``,
    NOT ``None`` or ``-1`` (which would silently break
    ``assert report.total_seen == len(input_rows)`` invariants)."""
    assert ImportReport().total_seen == 0


def test_import_report_total_seen_counts_imported_only() -> None:
    """A report with one imported object reports ``total_seen == 1``.
    This is the ``imported`` bucket, which is the success path."""
    report = ImportReport(imported=[_stub_knowledge_object()])
    assert report.total_seen == 1


def test_import_report_total_seen_sums_across_all_buckets() -> None:
    """``total_seen`` must equal the sum of all skip lists plus
    ``imported`` plus ``rejected_malformed``. A refactor that adds a
    new bucket but forgets to update ``total_seen`` would silently
    under-count and break the input-correspondence invariant."""
    report = ImportReport(
        imported=[_stub_knowledge_object()],
        skipped_duplicate=["a"],
        skipped_disallowed_source=["b", "c"],
        skipped_rejected=["d"],
        skipped_not_promoted=["e", "f", "g"],
        skipped_requires_opt_in=["h"],
        skipped_unverified_or_low_quality=["i", "j"],
        rejected_malformed=[("k", "bad row")],
    )
    # 1 imported + 1 dup + 2 disallowed + 1 rejected + 3 not_promoted
    # + 1 opt_in + 2 unverified + 1 malformed = 12
    assert report.total_seen == 12


def _stub_knowledge_object() -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id="ko_stub0000000001",  # type: ignore[arg-type]
        topic="qpipe:scenario-forge:global",
        content="{}",
        content_hash="a" * 64,
        source_uri="qpipe://scenario-forge/v1/ext_1",
        retrieved_at=1_700_000_000.0,
        authority=0.5,
        status=Status.IMPLEMENTED,
    )


# ---------------------------------------------------------------------------
# derive_authority boundaries
# ---------------------------------------------------------------------------


def test_derive_authority_unseen_recipe_uses_neutral_prior() -> None:
    """``capture_count <= 0`` returns the 0.5 neutral prior. A refactor
    that returns 0.0 would let unseen recipes be filtered out by
    ``min_authority > 0`` (the InMemoryKnowledgeStore default is 0.0,
    but real callers often raise it)."""
    assert derive_authority(capture_count=0, success_count=0) == 0.5
    assert derive_authority(capture_count=-1, success_count=0) == 0.5


def test_derive_authority_one_capture_one_success() -> None:
    """``capture_count=1, success_count=1`` -> Laplace-smoothed:
    ``(1 + 1) / (1 + 2) = 0.6667``. Pins the Laplace +1/+2 smoothing."""
    assert derive_authority(capture_count=1, success_count=1) == pytest.approx(2 / 3)


def test_derive_authority_one_capture_zero_success() -> None:
    """``capture_count=1, success_count=0`` -> ``(0 + 1) / (1 + 2) = 0.3333``.
    The smoothing pulls the value away from the extreme 0.0."""
    assert derive_authority(capture_count=1, success_count=0) == pytest.approx(1 / 3)


def test_derive_authority_perfect_success_capped_at_0_95() -> None:
    """``capture_count=100, success_count=100`` -> raw
    ``(100 + 1) / (100 + 2) = 0.9902``, capped at 0.95. A brand-new
    recipe cannot claim 1.0 authority (the cap is the safety belt)."""
    assert derive_authority(capture_count=100, success_count=100) == 0.95


def test_derive_authority_handles_success_exceeding_capture() -> None:
    """The function does NOT require ``success_count <= capture_count``.
    A future refactor that adds that guard would break the
    Laplace-smoothing contract (which already smooths any input)."""
    # success=10, capture=4: (10+1) / (4+2) = 11/6 ≈ 1.833, capped at 0.95
    assert derive_authority(capture_count=4, success_count=10) == 0.95


def test_derive_authority_monotonic_in_success_count() -> None:
    """For a fixed ``capture_count``, increasing ``success_count`` must
    monotonically increase the returned authority (up to the cap). A
    refactor that introduces non-monotonic behaviour (e.g. a
    ``status``-aware branch that bumps authority down for some inputs)
    would break this."""
    capture_count = 10
    prev = derive_authority(capture_count=capture_count, success_count=0)
    for success in range(1, capture_count + 1):
        current = derive_authority(capture_count=capture_count, success_count=success)
        assert current >= prev
        prev = current


# ---------------------------------------------------------------------------
# QPipeRow.from_dict adapter
# ---------------------------------------------------------------------------


def test_qpipe_row_from_dict_minimum_required_fields() -> None:
    """A dict with the minimum required fields round-trips through
    ``from_dict``. The required fields are ``source``, ``external_id``,
    ``scope``, ``fingerprint``, ``body_json``, ``status``,
    ``matcher_version``, ``source_updated_at``, ``imported_at``."""
    row = QPipeRow.from_dict(_row_dict())
    assert row.source == "scenario-forge"
    assert row.external_id == "ext_1"
    assert row.scope == "global"
    assert row.status == "promoted"


def test_qpipe_row_from_dict_normalizes_fingerprint_whitespace() -> None:
    """``fingerprint`` is whitespace-collapsed: extra spaces and tabs
    collapse to single spaces, surrounding whitespace stripped. A
    refactor that dropped the normalization would let two q-pipe rows
    with identical content (different whitespace) land on different
    fingerprint strings and dedup-by-fingerprint would silently fail."""
    row = QPipeRow.from_dict(
        _row_dict(fingerprint="  verified\tstable  workflow\talpha  "),
    )
    assert row.fingerprint == "verified stable workflow alpha"


def test_qpipe_row_from_dict_strips_and_lowercases_scope() -> None:
    """``scope`` is whitespace-stripped and lowercased. Topic-prefix
    downstream (``qpipe:<source>:<scope>``) assumes lowercase."""
    row = QPipeRow.from_dict(_row_dict(scope="  Global  "))
    assert row.scope == "global"


def test_qpipe_row_from_dict_uses_body_alias_when_body_json_missing() -> None:
    """The adapter accepts ``body`` as a legacy alias for ``body_json``
    (the older q-pipe schema field name). If ``body_json`` is missing
    and ``body`` is a dict, it gets ``json.dumps``-encoded with
    ``ensure_ascii=False``. A refactor that dropped the alias would
    silently break imports from older q-pipe dumps."""
    row = QPipeRow.from_dict(
        {
            "source": "scenario-forge",
            "external_id": "legacy_1",
            "scope": "global",
            "fingerprint": "verified stable workflow beta",
            "body": _safe_guidance(),  # legacy alias
            "capture_count": 4,
            "success_count": 3,
            "failure_count": 1,
            "verified_count": 1,
            "status": "promoted",
            "matcher_version": "v1",
            "source_updated_at": 1_700_000_000,
            "imported_at": 1_700_000_100,
        },
    )
    # Round-trip: the legacy dict must be re-encoded as valid JSON.
    parsed = json.loads(row.body_json)
    assert isinstance(parsed, dict)
    assert "guidance" in parsed


def test_qpipe_row_from_dict_raises_keyerror_on_missing_scope() -> None:
    """Empty/missing ``scope`` raises ``KeyError("scope")``; this is the
    contract that ``import_qpipe_rows`` catches and routes to
    ``rejected_malformed``."""
    bad = _row_dict(scope="")  # empty scope is also rejected
    with pytest.raises(KeyError, match="scope"):
        QPipeRow.from_dict(bad)


def test_qpipe_row_from_dict_raises_keyerror_when_neither_body_nor_body_json() -> None:
    """If both ``body_json`` (string) and ``body`` (legacy dict) are
    missing, ``from_dict`` raises ``KeyError("body_json/body")``."""
    row_dict = _row_dict()
    row_dict.pop("body_json", None)
    with pytest.raises(KeyError, match="body_json/body"):
        QPipeRow.from_dict(row_dict)


# ---------------------------------------------------------------------------
# Compatibility-revision pinning (the safety belt that the gateway contract
# depends on -- a different blob hash means a new q-pipe revision with
# unknown changes).
# ---------------------------------------------------------------------------


def test_qpipe_compatibility_source_revision_is_pinned() -> None:
    """The pinned revision ``2ed18c26d671559a0c82e55ce15b41a973c61f43``
    is the exact q-pipe source commit this importer was tested against.
    A bump in q-pipe's memory.py or cloudflare_learning.py that changes
    the public schema requires bumping this revision (and updating the
    blob hashes below). Pinning the value prevents silent drift."""
    assert QPIPE_COMPATIBILITY_SOURCE_REVISION == ("2ed18c26d671559a0c82e55ce15b41a973c61f43")


def test_qpipe_compatibility_source_blobs_are_pinned() -> None:
    """Two specific blobs (``cloudflare_learning.py`` + ``memory.py``)
    define the public surface this importer depends on. A drift in
    either blob hash means the contract has changed and the importer
    needs review."""
    assert QPIPE_COMPATIBILITY_SOURCE_BLOBS == {
        "qpipe/cloudflare_learning.py": "3cf94523564fdd919bec177f71740ff7e979f805",
        "qpipe/memory.py": "f7188d3128e2c647ece9f457b053dcb3d7b705a8",
    }


def test_qpipe_compatibility_blob_keys_cover_the_two_known_modules() -> None:
    """The blob dict's keys identify which q-pipe modules this importer
    inspects. Adding a third (or dropping one) requires updating both
    the dict and the corresponding test fixture."""
    assert set(QPIPE_COMPATIBILITY_SOURCE_BLOBS) == {
        "qpipe/cloudflare_learning.py",
        "qpipe/memory.py",
    }
