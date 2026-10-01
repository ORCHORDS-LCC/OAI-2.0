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

Allowed q-pipe sources: ``scenario-forge``, ``terminal-bench-2.1``,
``android-curriculum-oss`` (opt-in).

Mapping summary:

    source            -> topic prefix  (``qpipe:<source>:<scope>``)
    external_id       -> knowledge_id suffix
    fingerprint       -> content_hash (server-of-record)
    body_json         -> content (after sanitization)
    status            -> OAI Status (promoted/candidate/rejected)
    capture_count     -> authority (0.0–1.0, capped)
    matcher_version   -> carried as ``source_uri`` for traceability
    source_updated_at -> retrieved_at

Status: PROPOSED. Sanitization matches q-pipe's
``_sanitize_argument`` / ``_sanitize_learning_body`` rules — secrets,
tokens, and absolute paths are redacted before the body lands in OAI.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from ..core import KnowledgeId, Status
from .abstraction import KnowledgeObject, now_epoch, sha256_hex


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


# Patterns mirrored from q-pipe._sanitize_argument so we don't import
# q-pipe into the OAI repo (which must remain public-safe).
_REDACT_KEY_PATTERNS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
)
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/|~[\\/])\S+")
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b(token|secret|password|api[_-]?key)\s*[=:]\s*\S+"
)
_HEX_HASH_RE = re.compile(r"\b[0-9a-f]{32,}\b")


def _redact_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _REDACT_KEY_PATTERNS)


def _sanitize_scalar(key: str, value: object) -> object:
    if _redact_key(str(key)):
        return "<redacted>"
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return "<number>"
    text = str(value)
    text = _INLINE_SECRET_RE.sub(
        lambda m: m.group(1) + "=<redacted>", text
    )
    if re.match(r"^(?:[A-Za-z]:[\\/]|/|~[\\/])", text):
        return "<path>"
    if _HEX_HASH_RE.search(text):
        return _HEX_HASH_RE.sub("<id>", text)
    return text[:200]


def _sanitize_arguments(args: object) -> object:
    if isinstance(args, dict):
        return {
            str(k)[:100]: _sanitize_arguments(v)
            if isinstance(v, (dict, list))
            else _sanitize_scalar(str(k), v)
            for k, v in list(args.items())[:32]
        }
    if isinstance(args, list):
        return [_sanitize_arguments(item) for item in args[:20]]
    return _sanitize_scalar("", args)


def _sanitize_actions(raw: object) -> list[dict[str, object]]:
    """Mimic q-pipe._sanitize_actions: keep tool name + sanitized args."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for action in raw[:8]:
        if isinstance(action, str):
            out.append(
                {"tool": action.split("(", 1)[0][:100], "arguments": {}}
            )
            continue
        if not isinstance(action, dict):
            continue
        tool = str(action.get("tool") or action.get("name") or "")[:100].strip()
        if not tool:
            continue
        args = action.get("arguments")
        out.append(
            {
                "tool": tool,
                "arguments": _sanitize_arguments(args if isinstance(args, (dict, list)) else {}),
            }
        )
    return out


def _sanitize_body(body_json: str) -> dict[str, object]:
    """Mirror q-pipe._sanitize_learning_body for the body_json field."""
    try:
        row = json.loads(body_json)
    except json.JSONDecodeError:
        # Fall back to a guidance string with inline secret redaction.
        body = body_json[:4000]
        body = _INLINE_SECRET_RE.sub(
            lambda m: m.group(1) + "=<redacted>", body
        )
        body = _ABS_PATH_RE.sub("<path>", body)
        return {"guidance": body}
    actions = _sanitize_actions(row.get("actions") if isinstance(row, dict) else None)
    if actions:
        return {"actions": actions}
    if isinstance(row, dict) and isinstance(row.get("body"), str):
        body = row["body"][:4000]
        body = _INLINE_SECRET_RE.sub(
            lambda m: m.group(1) + "=<redacted>", body
        )
        body = _ABS_PATH_RE.sub("<path>", body)
        return {"guidance": body}
    return {"guidance": ""}


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

        return cls(
            source=str(d["source"]),
            external_id=str(d["external_id"]),
            scope=str(d.get("scope", "global")),
            fingerprint=str(d["fingerprint"]),
            body_json=str(d["body_json"]),
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


@dataclass(slots=True)
class ImportReport:
    """Outcome of one :func:`import_qpipe_rows` call."""

    imported: list[KnowledgeObject] = field(default_factory=list)
    skipped_duplicate: list[str] = field(default_factory=list)
    skipped_disallowed_source: list[str] = field(default_factory=list)
    skipped_rejected: list[str] = field(default_factory=list)
    rejected_malformed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        return (
            len(self.imported)
            + len(self.skipped_duplicate)
            + len(self.skipped_disallowed_source)
            + len(self.skipped_rejected)
            + len(self.rejected_malformed)
        )


def _validate_row(row: QPipeRow) -> str | None:
    """Return an error message if the row is malformed, else None."""
    if row.source not in {s.value for s in QPipeSource}:
        return f"disallowed source: {row.source}"
    if row.status not in {s.value for s in QPipeStatus}:
        return f"invalid status: {row.status}"
    if row.capture_count < 0 or row.success_count < 0:
        return "negative counters"
    if not row.fingerprint:
        return "missing fingerprint"
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
    sanitized = _sanitize_body(row.body_json)
    body_text = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
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
) -> ImportReport:
    """Translate a batch of q-pipe rows into KnowledgeObjects.

    The function is side-effect free: it does not write anywhere.
    Dedupe is in-memory, keyed on (source, external_id).
    """
    report = ImportReport()
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        row = raw if isinstance(raw, QPipeRow) else QPipeRow.from_dict(raw)
        key = (row.source, row.external_id)
        if key in seen:
            report.skipped_duplicate.append(row.external_id)
            continue
        seen.add(key)
        err = _validate_row(row)
        if err is not None:
            if err.startswith("disallowed source"):
                report.skipped_disallowed_source.append(row.external_id)
            elif err.startswith("invalid status"):
                report.rejected_malformed.append((row.external_id, err))
            else:
                report.rejected_malformed.append((row.external_id, err))
            continue
        if row.status == QPipeStatus.REJECTED.value:
            report.skipped_rejected.append(row.external_id)
            continue
        report.imported.append(row_to_knowledge_object(row))
    return report


__all__ = [
    "QPipeSource",
    "QPipeStatus",
    "QPIPE_TO_OAI_STATUS",
    "QPipeRow",
    "ImportReport",
    "derive_authority",
    "row_to_knowledge_object",
    "import_qpipe_rows",
    "now_epoch",
]
