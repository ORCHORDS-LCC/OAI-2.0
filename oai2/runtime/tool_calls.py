"""Parsing of model-emitted tool calls into OpenAI wire shape.

Chat templates teach the model to emit calls as an XML-tagged JSON object
(``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` for the Qwen
family). ``MLXHotRuntime`` renders the template, so the runtime — not the
host — is the component that has to turn that text back into the structured
``tool_calls`` array an OpenAI-compatible client expects.

This module deliberately imports nothing from ``mlx`` and nothing from the
model: it is pure text handling so it can be unit-tested on a machine with no
weights present.

``model.py`` remains the single sanctioned MLX importer in the runtime
package; nothing here violates that.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

#: Tag emitted by Qwen-family templates. Templates sometimes contain an
#: invisible character inside the tag (some ship a zero-width space between
#: ``<`` and ``tool_call``), so the pattern tolerates non-word filler
#: between ``<`` and the tag name rather than assuming a byte-exact tag.
_TOOL_CALL_RE = re.compile(r"<\W*tool_call>(.*?)<\W*tool_call>", re.DOTALL)

#: Fenced block the model used instead of the tag. Small instruct models
#: routinely answer a "call a tool" request with ```json {...} ``` rather
#: than the tag the template asked for.
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(.*?)```", re.DOTALL)


def _balanced_json_objects(text: str) -> list[str]:
    """Every top-level balanced ``{...}`` region in ``text``, in order.

    A regex cannot find a JSON object that contains nested objects, and a
    tool call almost always does (``arguments`` is itself an object), so
    the scan tracks brace depth and string state instead.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            if depth > 0:
                in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out


def _candidate_blocks(text: str, *, allow_bare_json: bool) -> list[str]:
    """Candidate tool-call payloads, most-trusted first."""
    tagged = [m.group(1) for m in _TOOL_CALL_RE.finditer(text)]
    if tagged:
        return tagged
    if not allow_bare_json:
        return []
    fenced = [b.strip() for b in _FENCE_RE.findall(text)]
    fenced = [b for b in fenced if b]
    # Fenced blocks win over a raw scan: if the model chose to fence its
    # output, that fence is the whole payload, not something to be picked
    # out of surrounding prose.
    return fenced or _balanced_json_objects(text)

TOOL_CALLS_ENABLED = True


class ToolCallParseError(ValueError):
    """A tool-call block was present but could not be turned into a call."""


def _coerce_arguments(raw: Any) -> str:
    """Return ``arguments`` as a JSON string, per the OpenAI wire shape.

    Only an object or an already-encoded JSON string is a valid ``arguments``
    payload. Accepting, say, an integer would turn an unrelated JSON record
    that happens to have ``name`` and ``arguments`` keys into a fabricated
    tool call.
    """
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise ToolCallParseError(
            f"'arguments' must be a JSON object or string, got {type(raw).__name__}"
        )
    try:
        return json.dumps(raw, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ToolCallParseError(f"arguments are not JSON-serialisable: {exc}") from exc


def _normalise_arguments(block: dict[str, Any]) -> str:
    """Accept both ``{"a": 1}`` and ``"[1, 2]"`` argument encodings."""
    if "arguments" in block:
        return _coerce_arguments(block["arguments"])
    if "parameters" in block:  # some templates use the schema field name
        return _coerce_arguments(block["parameters"])
    # No arguments key at all: a zero-argument call is legitimate.
    return "{}"


def _build_call(name: str, arguments: str, index: int) -> dict[str, Any]:
    return {
        "id": f"call_{index}_{abs(hash((name, arguments))) % 10**8:08d}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def parse_tool_calls(
    text: str,
    *,
    allow_bare_json: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Extract tool calls from raw model text.

    Returns an empty tuple when the model wrote prose instead of a call —
    that is a normal outcome, not an error. Raises
    :class:`ToolCallParseError` only when a candidate block exists but
    carries no usable function name, because silently dropping it would let
    a caller believe the model declined to call a tool when it actually
    tried and failed.

    ``allow_bare_json`` additionally accepts a fenced or bare JSON object
    carrying ``name`` and ``arguments``. It is off by default: on a code
    model that writes JSON in ordinary prose, an untagged object is far more
    likely to be an example than a real call. Callers that know the request
    advertised tools — and therefore that the model was asked for a call —
    should switch it on, because small instruct models routinely answer a
    tool request with ```json {...}``` instead of the tag their template
    named.
    """
    if not text:
        return ()

    blocks = _candidate_blocks(text, allow_bare_json=allow_bare_json)
    if not blocks:
        return ()

    calls: list[dict[str, Any]] = []
    for raw in blocks:
        candidate = raw.strip()
        if not candidate:
            continue
        parsed: Any = None
        for attempt in (candidate, re.sub(r"^```(?:json)?|```$", "", candidate, flags=re.M).strip()):
            try:
                parsed = json.loads(attempt)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            # A block that is not JSON at all (prose inside a fence, say) is
            # simply not a tool call; only fail when it *looks* like one.
            if allow_bare_json and not _looks_like_call(candidate):
                continue
            raise ToolCallParseError("tool_call block is not valid JSON")
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            if allow_bare_json and not _looks_like_call(candidate):
                continue
            raise ToolCallParseError("tool_call block has no usable 'name'")
        try:
            calls.append(_build_call(name, _normalise_arguments(parsed), len(calls)))
        except ToolCallParseError:
            if not allow_bare_json:
                raise
            # Untagged mode cannot distinguish "the model meant to call a
            # tool and got the payload wrong" from "this JSON is prose", so
            # an unusable payload is skipped rather than raised. The tag
            # path above still reports the failure loudly.
            continue
    return tuple(calls)


def _looks_like_call(candidate: str) -> bool:
    """Whether a block claims to be a function call at all."""
    return "name" in candidate and ("arguments" in candidate or "parameters" in candidate)


def tool_names(calls: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Names of the functions in an OpenAI-shaped ``tool_calls`` array."""
    names: list[str] = []
    for call in calls:
        fn = call.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
    return tuple(names)


def render_tool_result(call_id: str, name: str, content: str) -> dict[str, Any]:
    """Build the ``tool`` message that answers ``call_id``.

    The OpenAI contract requires the result message to carry the *same*
    ``tool_call_id`` as the call it answers; a mismatched id is silently
    dropped by most clients, so this is centralised rather than spelled out
    at each call site.
    """
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def normalise_tool_choice(tool_choice: str | dict[str, Any] | None) -> str:
    """Reduce the several spellings of ``tool_choice`` to one mode.

    Returns ``"auto"``, ``"none"``, ``"required"`` or ``"forbidden"``. A
    dict selecting one function collapses to ``"auto"`` here; the forced
    function name is carried separately by :func:`forced_tool_name`.
    """
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        low = tool_choice.strip().lower()
        if low in {"none", "no_tool", "no-tools"}:
            return "none"
        if low in {"required", "any", "force"}:
            return "required"
        return "auto"
    if isinstance(tool_choice, dict):
        return "auto"
    return "auto"


def forced_tool_name(tool_choice: str | dict[str, Any] | None) -> str | None:
    """The function name ``tool_choice`` pins, if it pins one."""
    if not isinstance(tool_choice, dict):
        return None
    fn = tool_choice.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    if isinstance(tool_choice.get("name"), str):
        return tool_choice["name"]
    return None


def validate_tool_calls(
    calls: Sequence[dict[str, Any]],
    *,
    declared: Sequence[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
) -> tuple[str, ...]:
    """Return human-readable reasons ``calls`` fail the request's contract.

    An empty tuple means the calls are well-formed. This exists because a
    client that advertises a tool set has a right to know when the model
    invented a function, and a silent pass would hide that.
    """
    problems: list[str] = []
    mode = normalise_tool_choice(tool_choice)
    available = {
        t.get("function", {}).get("name")
        for t in declared
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    available.discard(None)
    for call in calls:
        emitted = tool_names([call])
        call_name = emitted[0] if emitted else ""
        if not call_name:
            problems.append("tool call has no function name")
            continue
        if available and call_name not in available:
            problems.append(f"model called undeclared function {call_name!r}")
    if mode == "required" and not calls:
        problems.append("tool_choice=required but the model produced no tool call")
    pinned = forced_tool_name(tool_choice)
    if pinned and calls:
        names = tool_names(calls)
        if pinned not in names:
            problems.append(f"tool_choice pinned {pinned!r} but the model called {list(names)!r}")
    return tuple(problems)


__all__ = [
    "TOOL_CALLS_ENABLED",
    "ToolCallParseError",
    "forced_tool_name",
    "normalise_tool_choice",
    "parse_tool_calls",
    "render_tool_result",
    "tool_names",
    "validate_tool_calls",
]
