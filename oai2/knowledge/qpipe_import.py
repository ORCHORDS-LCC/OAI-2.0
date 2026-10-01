"""q-pipe knowledge importer (logical mapping only).

Translates q-pipe ``learning_recipes`` rows into OAI-2.0
:class:`KnowledgeObject` records. The importer does NOT open a q-pipe
SQLite connection in this release — it operates on in-memory row dicts
matching the q-pipe schema, so the mapping is fully testable in CI
without the q-pipe runtime on disk.

Source schema (q-pipe, see ``qpipe/memory.py``):

    id, source, external_id, scope, fingerprint,
    body_json, capture_count, success_count, failure_count,
    verified_count, status, matcher_version,
    source_updated_at, imported_at, promoted_at, promoter, last_used_at

Default-importable q-pipe sources: ``scenario-forge`` and
``terminal-bench-2.1``. ``android-curriculum-oss`` requires explicit
operator opt-in, matching q-pipe's Cloudflare export gate.

Mapping summary:

    source            -> topic prefix  (``qpipe:<source>:<scope>``)
    external_id       -> knowledge_id suffix
    fingerprint       -> retained as an eligibility signal
    verified guidance -> content + content_hash
    status            -> OAI Status (promoted by default only)
    capture_count     -> authority (0.0–1.0, capped)
    matcher_version   -> carried as ``source_uri`` for traceability
    source_updated_at -> retrieved_at

Status: PROPOSED. The default gate mirrors q-pipe's stricter Cloudflare
export contract: promoted rows only, positive independent verification,
success greater than failure, bounded fingerprint/guidance, and explicit
Android-curriculum opt-in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from ..core import KnowledgeId, Status
from .abstraction import KnowledgeObject, now_epoch, sha256_hex

# Exact public q-pipe source revision used to verify this compatibility gate.
# Update this marker only after re-reading q-pipe's export/memory contracts and
# updating the compatibility fixtures in tests/test_qpipe_import.py.
QPIPE_COMPATIBILITY_SOURCE_REVISION = "2ed18c26d671559a0c82e55ce15b41a973c61f43"
QPIPE_COMPATIBILITY_SOURCE_BLOBS = {
    "qpipe/cloudflare_learning.py": "3cf94523564fdd919bec177f71740ff7e979f805",
    "qpipe/memory.py": "f7188d3128e2c647ece9f457b053dcb3d7b705a8",
}


class QPipeSource(StrEnum):
    """Allow-listed q-pipe sources for OAI knowledge imports."""

    SCENARIO_FORGE = "scenario-forge"
    TERMINAL_BENCH = "terminal-bench-2.1"
    ANDROID_CURRICULUM = "android-curriculum-oss"


class QPipeStatus(StrEnum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"


# Status mapping: q-pipe candidate/promoted/rejected -> OAI lifecycle.
QPIPE_TO_OAI_STATUS: dict[QPipeStatus, Status] = {
    QPipeStatus.CANDIDATE: Status.EXPERIMENTAL,
    QPipeStatus.PROMOTED: Status.IMPLEMENTED,
    QPipeStatus.REJECTED: Status.PROPOSED,  # carried for audit, not active.
}


# q-pipe's Cloudflare export gate accepts only bounded, structured guidance.
_SECRET_MARKERS = (
    "authorization:",
    "authorization=",
    "api_key:",
    "api_key=",
    "api-key:",
    "api-key=",
    "token:",
    "token=",
    "secret:",
    "secret=",
    "password:",
    "password=",
)
_ABS_PATH_RE = re.compile(r"(?:^|\s)(?:[A-Za-z]:[\\/]|~[\\/]|/)\S+")


def _strings(value: object, *, count: int, length: int) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > count:
        return None
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if not text or len(text) > length:
            return None
        lowered = text.casefold()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            return None
        if _ABS_PATH_RE.search(text):
            return None
        output.append(text)
    return output


def _guidance_from_body_json(body_json: str) -> dict[str, list[str]] | None:
    """Return the same bounded guidance shape q-pipe exports to Cloudflare."""
    try:
        row = json.loads(body_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict):
        return None
    guidance = row.get("guidance")
    if not isinstance(guidance, dict):
        return None
    steps = _strings(guidance.get("steps"), count=32, length=500)
    files = _strings(guidance.get("files"), count=64, length=300)
    verification = _strings(guidance.get("verification"), count=32, length=300)
    risks = _strings(guidance.get("risks"), count=32, length=500)
    if any(item is None for item in (steps, files, verification, risks)):
        return None
    if not steps and not verification:
        return None
    return {
        "steps": steps or [],
        "files": files or [],
        "verification": verification or [],
        "risks": risks or [],
    }


def derive_authority(capture_count: int, success_count: int) -> float:
    """Convert q-pipe counters to an OAI authority in [0, 1].

    Uses Laplace-smoothed success rate, capped at 0.95 so a brand-new
    recipe cannot claim 1.0 authority.
    """
    if capture_count <= 0:
        return 0.5  # neutral prior for unseen recipes
    smooth = (success_count + 1) / (capture_count + 2)
    return min(0.95, max(0.0, smooth))


@dataclass(slots=True, frozen=True)
class QPipeRow:
    """In-memory shape of one q-pipe ``learning_recipes`` row."""

    recipe_id: int
    source: str
    external_id: str
    scope: str
    fingerprint: str
    body_json: str
    capture_count: int
    success_count: int
    failure_count: int
    verified_count: int
    status: str
    matcher_version: str
    source_updated_at: int
    imported_at: int
    promoted_at: int | None = None
    promoter: str | None = None
    last_used_at: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> QPipeRow:
        def _i(key: str) -> int:
            return int(d[key])  # type: ignore[call-overload]

        def _i_or(key: str, default: int) -> int:
            val = d.get(key, default)
            return int(val) if val is not None else default  # type: ignore[call-overload]

        def _i_opt(key: str) -> int | None:
            val = d.get(key)
            return int(val) if val is not None else None  # type: ignore[call-overload]

        def _s_opt(key: str) -> str | None:
            val = d.get(key)
            return str(val) if val is not None else None

        raw_body_json = d.get("body_json")
        if raw_body_json is None:
            raw_body = d.get("body")
            if raw_body is None:
                raise KeyError("body_json/body")
            raw_body_json = json.dumps(raw_body, ensure_ascii=False)

        raw_scope = d.get("scope")
        if not isinstance(raw_scope, str) or not raw_scope.strip():
            raise KeyError("scope")
        scope = raw_scope.strip().lower()

        return cls(
            recipe_id=_i_or("id", 0),
            source=str(d["source"]).strip().lower(),
            external_id=str(d["external_id"]).strip(),
            scope=scope,
            fingerprint=" ".join(str(d["fingerprint"]).strip().lower().split()),
            body_json=str(raw_body_json),
            capture_count=_i_or("capture_count", 0),
            success_count=_i_or("success_count", 0),
            failure_count=_i_or("failure_count", 0),
            verified_count=_i_or("verified_count", 0),
            status=str(d["status"]),
            matcher_version=str(d["matcher_version"]),
            source_updated_at=_i("source_updated_at"),
            imported_at=_i("imported_at"),
            promoted_at=_i_opt("promoted_at"),
            promoter=_s_opt("promoter"),
            last_used_at=_i_opt("last_used_at"),
        )


@dataclass(slots=True, frozen=True)
class ImportPolicy:
    """Explicit import policy matching q-pipe's default Cloudflare export."""

    allow_candidates: bool = False
    allow_android_curriculum: bool = False
    require_verified: bool = True


@dataclass(slots=True)
class ImportReport:
    """Outcome of one :func:`import_qpipe_rows` call."""

    imported: list[KnowledgeObject] = field(default_factory=list)
    skipped_duplicate: list[str] = field(default_factory=list)
    skipped_disallowed_source: list[str] = field(default_factory=list)
    skipped_rejected: list[str] = field(default_factory=list)
    skipped_not_promoted: list[str] = field(default_factory=list)
    skipped_requires_opt_in: list[str] = field(default_factory=list)
    skipped_unverified_or_low_quality: list[str] = field(default_factory=list)
    rejected_malformed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        return (
            len(self.imported)
            + len(self.skipped_duplicate)
            + len(self.skipped_disallowed_source)
            + len(self.skipped_rejected)
            + len(self.skipped_not_promoted)
            + len(self.skipped_requires_opt_in)
            + len(self.skipped_unverified_or_low_quality)
            + len(self.rejected_malformed)
        )


def _validate_row(row: QPipeRow, policy: ImportPolicy) -> str | None:
    """Return an eligibility error, or None when the row may import."""
    default_sources = {
        QPipeSource.SCENARIO_FORGE.value,
        QPipeSource.TERMINAL_BENCH.value,
    }
    if row.source == QPipeSource.ANDROID_CURRICULUM.value:
        if not policy.allow_android_curriculum:
            return "android curriculum requires opt-in"
    elif row.source not in default_sources:
        return f"disallowed source: {row.source}"

    if row.recipe_id < 1:
        return "invalid recipe id"
    if row.status not in {s.value for s in QPipeStatus}:
        return f"invalid status: {row.status}"
    if row.capture_count < 0 or row.success_count < 0 or row.failure_count < 0:
        return "negative counters"
    if not row.external_id or len(row.external_id) > 160:
        return "invalid external_id"

    fingerprint = " ".join(row.fingerprint.strip().lower().split())
    if len(fingerprint.split()) < 3 or len(fingerprint) > 1000:
        return "fingerprint outside q-pipe export bounds"

    if row.status == QPipeStatus.REJECTED.value:
        return "rejected"
    if row.status != QPipeStatus.PROMOTED.value and not policy.allow_candidates:
        return "not promoted"

    if policy.require_verified:
        if row.verified_count < 1:
            return "unverified"
        if row.success_count < row.verified_count:
            return "verified_count exceeds success_count"
        if row.success_count <= row.failure_count:
            return "success_count must exceed failure_count"

    if _guidance_from_body_json(row.body_json) is None:
        return "invalid or unsafe guidance"
    return None


def row_to_knowledge_object(
    row: QPipeRow,
    *,
    imported_at_epoch: float | None = None,
) -> KnowledgeObject:
    """Translate one q-pipe row into an OAI-2.0 :class:`KnowledgeObject`.

    Stable :attr:`KnowledgeObject.knowledge_id` derived from
    ``source + external_id`` so dedupe works on re-imports.
    """
    digest = sha256_hex(f"{row.source}|{row.external_id}")
    guidance = _guidance_from_body_json(row.body_json)
    if guidance is None:
        raise ValueError("row guidance failed q-pipe export validation")
    body_text = json.dumps(
        guidance,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    topic = f"qpipe:{row.source}:{row.scope}"
    authority = derive_authority(row.capture_count, row.success_count)
    qpipe_status = QPipeStatus(row.status)
    oai_status = QPIPE_TO_OAI_STATUS[qpipe_status]

    return KnowledgeObject(
        knowledge_id=KnowledgeId(f"ko_{digest[:12]}"),
        topic=topic,
        content=body_text,
        content_hash=sha256_hex(body_text),
        source_uri=f"qpipe://{row.source}/{row.matcher_version}/{row.external_id}",
        retrieved_at=(
            float(row.source_updated_at)
            if imported_at_epoch is None
            else imported_at_epoch
        ),
        authority=authority,
        status=oai_status,
    )


def import_qpipe_rows(
    rows: Iterable[dict[str, object] | QPipeRow],
    *,
    policy: ImportPolicy | None = None,
) -> ImportReport:
    """Translate q-pipe rows using the verified Cloudflare-export gate."""
    policy = policy or ImportPolicy()
    report = ImportReport()
    seen: set[tuple[str, str]] = set()

    for raw in rows:
        try:
            row = raw if isinstance(raw, QPipeRow) else QPipeRow.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            external_id = (
                str(raw.get("external_id", "<unknown>"))
                if isinstance(raw, dict)
                else "<unknown>"
            )
            report.rejected_malformed.append(
                (external_id, f"malformed row: {exc}")
            )
            continue

        err = _validate_row(row, policy)
        if err is not None:
            if err.startswith("disallowed source"):
                report.skipped_disallowed_source.append(row.external_id)
            elif err == "android curriculum requires opt-in":
                report.skipped_requires_opt_in.append(row.external_id)
            elif err == "rejected":
                report.skipped_rejected.append(row.external_id)
            elif err == "not promoted":
                report.skipped_not_promoted.append(row.external_id)
            elif err in {
                "unverified",
                "verified_count exceeds success_count",
                "success_count must exceed failure_count",
                "invalid or unsafe guidance",
                "fingerprint outside q-pipe export bounds",
            }:
                report.skipped_unverified_or_low_quality.append(row.external_id)
            else:
                report.rejected_malformed.append((row.external_id, err))
            continue

        key = (row.source, row.external_id)
        if key in seen:
            report.skipped_duplicate.append(row.external_id)
            continue
        seen.add(key)
        report.imported.append(row_to_knowledge_object(row))

    return report


__all__ = [
    "QPIPE_COMPATIBILITY_SOURCE_REVISION",
    "QPIPE_COMPATIBILITY_SOURCE_BLOBS",
    "QPipeSource",
    "QPipeStatus",
    "QPIPE_TO_OAI_STATUS",
    "QPipeRow",
    "ImportPolicy",
    "ImportReport",
    "derive_authority",
    "row_to_knowledge_object",
    "import_qpipe_rows",
    "now_epoch",
]
