"""Tests for the canonical tool registry and local execution."""

from __future__ import annotations

import json
from pathlib import Path

from oai2.core import ToolId
from oai2.protocols import ToolCall
from oai2.tools.registry import (
    default_tool_definitions,
    execute_tool,
    to_openai_wire,
)


def test_default_tool_definitions_has_six_tools() -> None:
    tools = default_tool_definitions()
    names = sorted(td.name for td in tools)
    assert names == ["Bash", "Edit", "Glob", "Grep", "Read", "Write"]


def test_to_openai_wire_shape() -> None:
    wire = to_openai_wire(default_tool_definitions())
    assert len(wire) == 6
    for entry in wire:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
    names = {entry["function"]["name"] for entry in wire}
    assert "Read" in names


def test_to_openai_wire_marks_required_args() -> None:
    wire = to_openai_wire(default_tool_definitions())
    by_name = {entry["function"]["name"]: entry for entry in wire}
    read_required = by_name["Read"]["function"]["parameters"].get("required", [])
    assert "path" in read_required
    bash_required = by_name["Bash"]["function"]["parameters"].get("required", [])
    assert "command" in bash_required


def test_execute_read_returns_file_contents(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")
    call = ToolCall(
        id="c1", tool_id=ToolId("read"), arguments={"path": str(f)}
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert result.output == "hello\nworld\n"
    assert result.call_id == "c1"


def test_execute_read_missing_file(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("read"),
        arguments={"path": "nope.txt"},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "not found" in (result.error or "")


def test_execute_edit_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("alpha beta gamma\n", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("edit"),
        arguments={
            "path": str(f),
            "old_string": "beta",
            "new_string": "BETA",
        },
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert f.read_text(encoding="utf-8") == "alpha BETA gamma\n"


def test_execute_edit_rejects_ambiguous_string(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("a a a\n", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("edit"),
        arguments={
            "path": str(f),
            "old_string": "a",
            "new_string": "b",
        },
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "unique" in (result.error or "").lower()


def test_execute_write_creates_file(tmp_path: Path) -> None:
    f = tmp_path / "sub" / "x.txt"
    call = ToolCall(
        id="c1",
        tool_id=ToolId("write"),
        arguments={"path": str(f), "content": "hi"},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert f.read_text(encoding="utf-8") == "hi"


def test_execute_bash_runs_command(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("bash"),
        arguments={"command": "echo hello-from-bash", "timeout_seconds": 5},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert "hello-from-bash" in result.output


def test_execute_bash_surfaces_failure(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("bash"),
        arguments={"command": "exit 7", "timeout_seconds": 5},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "7" in (result.error or "")


def test_execute_glob_lists_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("glob"),
        arguments={"pattern": "*.py"},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert "c.txt" not in result.output


def test_execute_grep_finds_pattern(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\nbeta hit\ngamma\n", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("grep"),
        arguments={"pattern": r"hit", "path": str(tmp_path)},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert "hit" in result.output
    assert "a.txt:2" in result.output


def test_execute_unknown_tool_returns_error(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("nope"),
        arguments={},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "unsupported" in (result.error or "")


def test_to_openai_wire_is_json_serialisable() -> None:
    """The wire shape must be JSON-serialisable for the gateway POST body."""
    wire = to_openai_wire(default_tool_definitions())
    json.dumps(wire)  # must not raise
