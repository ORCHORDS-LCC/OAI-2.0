"""Tool schema and action-token parser tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai2.core import ToolId
from oai2.protocols import (
    ImageToken,
    ReadToken,
    TestToken,
    TokenKind,
    ToolArgument,
    ToolCall,
    ToolDefinition,
    parse_action_token,
)


def _td() -> ToolDefinition:
    return ToolDefinition(
        id=ToolId("read"),
        name="read",
        description="read a file",
        arguments=(ToolArgument(name="path", type="path"),),
        capability="fs.read",
        scoped=True,
    )


def test_tool_definition_signature_is_stable() -> None:
    td = _td()
    assert td.signature() == "read(path:path)"


def test_tool_argument_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ToolArgument(name="x", type="decimal-foo")


def test_tool_call_defaults_are_safe() -> None:
    tc = ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "/x"})
    assert tc.policy.budget_calls == 64
    assert tc.policy.budget_seconds == 60.0
    assert tc.policy.high_impact_approved is False


def test_action_token_parser_basic() -> None:
    text = "I will <READ F:foo.py> and <TEST T:tests/test_foo.py> and <IMAGE I:cap-1>."
    tokens = parse_action_token(text)
    assert tokens == [
        ReadToken(TokenKind.READ, "foo.py"),
        TestToken(TokenKind.TEST, "tests/test_foo.py"),
        ImageToken(TokenKind.IMAGE, "cap-1"),
    ]


def test_action_token_parser_ignores_malformed() -> None:
    assert parse_action_token("nothing here") == []
    assert parse_action_token("<READ foo>") == []  # no colon
    assert parse_action_token("<BOGUS k:v>") == []  # unknown kind


def test_protocols_module_exports_from_protocols_package() -> None:
    from oai2.protocols import ActionToken as ExportedActionToken
    from oai2.protocols import ImageToken as ExportedImageToken
    from oai2.protocols import ReadToken as ExportedReadToken
    from oai2.protocols import TestToken as ExportedTestToken
    from oai2.protocols import TokenKind as ExportedTokenKind
    from oai2.protocols import (
        ToolArgument as ExportedToolArgument,
    )
    from oai2.protocols import ToolCall as ExportedToolCall
    from oai2.protocols import (
        ToolDefinition as ExportedToolDefinition,
    )
    from oai2.protocols import ToolPolicy as ExportedToolPolicy
    from oai2.protocols import ToolResult as ExportedToolResult
    from oai2.protocols import (
        parse_action_token as ExportedParseActionToken,
    )
    from oai2.protocols.tokens import (
        ActionToken,
        ImageToken,
        ReadToken,
        TestToken,
        TokenKind,
        parse_action_token,
    )
    from oai2.protocols.tools import (
        ToolArgument,
        ToolCall,
        ToolDefinition,
        ToolPolicy,
        ToolResult,
    )

    assert ExportedActionToken is ActionToken
    assert ExportedImageToken is ImageToken
    assert ExportedReadToken is ReadToken
    assert ExportedTestToken is TestToken
    assert ExportedTokenKind is TokenKind
    assert ExportedParseActionToken is parse_action_token
    assert ExportedToolArgument is ToolArgument
    assert ExportedToolCall is ToolCall
    assert ExportedToolDefinition is ToolDefinition
    assert ExportedToolPolicy is ToolPolicy
    assert ExportedToolResult is ToolResult
