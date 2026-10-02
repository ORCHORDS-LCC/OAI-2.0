"""Tests for the real tool-call path (Refs #186, #240).

Covers the two halves separately because they fail differently:

* :mod:`oai2.runtime.tool_calls` is pure text handling, so every parsing
  edge is exercised directly with no weights present.
* :class:`~oai2.runtime.mlx_hot_runtime.MLXHotRuntime`'s prompt rendering
  is driven through a fake tokenizer, so we can assert that declared tools
  actually reach ``apply_chat_template`` — the gap that made the advertised
  tool set invisible to the model — without paying a model load.
"""

from __future__ import annotations

import pytest

from oai2.agents.composer import PrefixSpec
from oai2.runtime.inference import InferenceRequest, TemplateRenderError
from oai2.runtime.mlx_hot_runtime import MLXHotRuntime, _has_structured_tool_history
from oai2.runtime.model import ModelSpec
from oai2.runtime.tool_calls import (
    ToolCallParseError,
    forced_tool_name,
    normalise_tool_choice,
    parse_tool_calls,
    render_tool_result,
    tool_names,
    validate_tool_calls,
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

TAGGED = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
FENCED = '```json\n{\n  "name": "list_dir",\n  "arguments": {\n    "path": "src"\n  }\n}\n```'


def test_tagged_call_is_parsed() -> None:
    calls = parse_tool_calls(TAGGED)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == '{"path":"a.py"}'
    assert calls[0]["id"]


def test_fenced_json_is_ignored_unless_the_caller_opted_in() -> None:
    """Strict parsing must not read an untagged object as a call by default."""
    assert parse_tool_calls(FENCED) == ()
    calls = parse_tool_calls(FENCED, allow_bare_json=True)
    assert tool_names(calls) == ("list_dir",)


def test_nested_arguments_survive_extraction() -> None:
    """A regex cannot find an object containing an object; the scan must."""
    text = '```json\n{"name": "f", "arguments": {"a": {"b": [1, 2]}}}\n```'
    calls = parse_tool_calls(text, allow_bare_json=True)
    assert calls[0]["function"]["arguments"] == '{"a":{"b":[1,2]}}'


def test_prose_is_not_a_tool_call() -> None:
    assert parse_tool_calls("I will list the directory for you.") == ()
    assert parse_tool_calls("") == ()


def test_unrelated_json_in_prose_is_not_a_call() -> None:
    text = 'Example: x = {"name": "bob", "arguments": 1} is a person record.'
    # "arguments": 1 is not a valid arguments payload, so this is prose.
    assert parse_tool_calls(text, allow_bare_json=True) == ()


def test_multiple_calls_are_all_returned_in_order() -> None:
    text = '```json\n{"name":"a","arguments":{}}\n```\nor\n```json\n{"name":"b","arguments":{"x":2}}\n```'
    assert tool_names(parse_tool_calls(text, allow_bare_json=True)) == ("a", "b")


def test_malformed_tagged_block_raises_rather_than_vanishing() -> None:
    with pytest.raises(ToolCallParseError):
        parse_tool_calls("<tool_call>not json</tool_call>")


def test_call_without_arguments_is_legal() -> None:
    calls = parse_tool_calls('<tool_call>{"name":"ping"}</tool_call>')
    assert calls[0]["function"]["arguments"] == "{}"


def test_tool_result_carries_the_matching_call_id() -> None:
    call = parse_tool_calls(TAGGED)[0]
    msg = render_tool_result(call["id"], "read_file", "contents")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == call["id"]
    assert msg["name"] == "read_file"


# ---------------------------------------------------------------------------
# tool_choice contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        ("any", "required"),
        ({"type": "function", "function": {"name": "x"}}, "auto"),
    ],
)
def test_tool_choice_normalises(given, expected) -> None:
    assert normalise_tool_choice(given) == expected


def test_forced_tool_name_is_extracted_from_the_pinned_form() -> None:
    assert forced_tool_name({"type": "function", "function": {"name": "ls"}}) == "ls"
    assert forced_tool_name({"name": "ls"}) == "ls"
    assert forced_tool_name("auto") is None


DECLARED = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]


def test_undeclared_function_is_reported() -> None:
    """A call the client never advertised must be surfaced, not passed off."""
    other = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
    problems = validate_tool_calls(parse_tool_calls(TAGGED), declared=other, tool_choice="auto")
    assert any("undeclared" in p for p in problems)


def test_declared_call_has_no_problems() -> None:
    assert validate_tool_calls(parse_tool_calls(TAGGED), declared=DECLARED, tool_choice="auto") == ()


def test_required_with_no_call_is_reported() -> None:
    problems = validate_tool_calls((), declared=DECLARED, tool_choice="required")
    assert any("no tool call" in p for p in problems)


def test_pinned_function_mismatch_is_reported() -> None:
    problems = validate_tool_calls(
        parse_tool_calls(TAGGED),
        declared=DECLARED,
        tool_choice={"type": "function", "function": {"name": "list_dir"}},
    )
    assert any("pinned" in p for p in problems)


# ---------------------------------------------------------------------------
# Prompt rendering: do declared tools actually reach the model?
# ---------------------------------------------------------------------------


class _RecordingTokenizer:
    """Minimal tokenizer double that records what the template was given."""

    def __init__(self, *, fail: bool = False) -> None:
        self.kwargs: dict[str, object] = {}
        self.fail = fail

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        self.kwargs = dict(kwargs)
        if self.fail:
            raise RuntimeError("template exploded")
        return "RENDERED"


def _runtime() -> MLXHotRuntime:
    return MLXHotRuntime(ModelSpec(name="stub"), model_id="stub")


TOOLS = [{"type": "function", "function": {"name": "ls", "parameters": {}}}]


def test_declared_tools_are_handed_to_the_chat_template() -> None:
    tok = _RecordingTokenizer()
    prompt, fallback = _runtime()._render_prompt(  # noqa: SLF001
        tok,  # type: ignore[arg-type]
        InferenceRequest(prompt="x", messages=[{"role": "user", "content": "hi"}], tools=TOOLS),
    )
    assert prompt == "RENDERED"
    assert fallback is False
    assert tok.kwargs["tools"] == TOOLS


def test_tool_choice_none_withholds_the_tool_signatures() -> None:
    tok = _RecordingTokenizer()
    _runtime()._render_prompt(  # noqa: SLF001
        tok,  # type: ignore[arg-type]
        InferenceRequest(
            prompt="x", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, tool_choice="none"
        ),
    )
    assert "tools" not in tok.kwargs


def test_no_messages_falls_back_to_the_raw_prompt() -> None:
    prompt, fallback = _runtime()._render_prompt(  # noqa: SLF001
        _RecordingTokenizer(),  # type: ignore[arg-type]
        InferenceRequest(prompt="raw text"),
    )
    assert prompt == "raw text"
    assert fallback is False


def test_template_failure_while_tools_are_advertised_raises() -> None:
    with pytest.raises(TemplateRenderError):
        _runtime()._render_prompt(  # noqa: SLF001
            _RecordingTokenizer(fail=True),  # type: ignore[arg-type]
            InferenceRequest(
                prompt="x", messages=[{"role": "user", "content": "hi"}], tools=TOOLS
            ),
        )


def test_template_failure_never_silently_flattens_tool_history() -> None:
    messages = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "ls"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "ls", "content": "a.py"},
    ]
    with pytest.raises(TemplateRenderError):
        _runtime()._render_prompt(  # noqa: SLF001
            _RecordingTokenizer(fail=True),  # type: ignore[arg-type]
            InferenceRequest(prompt="x", messages=messages),
        )


def test_template_failure_on_plain_messages_falls_back_but_says_so() -> None:
    prompt, fallback = _runtime()._render_prompt(  # noqa: SLF001
        _RecordingTokenizer(fail=True),  # type: ignore[arg-type]
        InferenceRequest(prompt="x", messages=[{"role": "user", "content": "hi"}]),
    )
    assert fallback is True
    assert "hi" in prompt


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"role": "user", "content": "hi"}, False),
        ({"role": "tool", "tool_call_id": "c1", "content": "x"}, True),
        ({"role": "assistant", "tool_calls": [{"id": "c1"}]}, True),
        ({"role": "assistant", "tool_call_id": "c1"}, True),
    ],
)
def test_structured_tool_history_detection(message, expected) -> None:
    assert _has_structured_tool_history([message]) is expected


# ---------------------------------------------------------------------------
# The rendered prefix must change when the tool schema changes
# ---------------------------------------------------------------------------


def test_tool_schema_is_part_of_the_prefix_identity() -> None:
    """Tools are rendered into the prompt, so they belong in the digest."""
    messages = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "go"}]
    one = PrefixSpec.adopt_caller_prefix(messages, TOOLS)
    two = PrefixSpec.adopt_caller_prefix(
        messages, [*TOOLS, {"type": "function", "function": {"name": "write", "parameters": {}}}]
    )
    assert one.digest() != two.digest()
