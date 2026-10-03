"""Controlled acceptance tests for the incremental repository graph."""

from __future__ import annotations

import subprocess
from pathlib import Path

from oai2.context import (
    ParserStatus,
    build_repository_graph,
    update_repository_graph,
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_graph_indexes_symbols_references_dependencies_and_affected_tests(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/math_utils.py",
        "def add(left, right):\n    return left + right\n",
    )
    _write(
        tmp_path, "app.py", "from pkg.math_utils import add\n\ndef main():\n    return add(1, 2)\n"
    )
    _write(tmp_path, "tests/test_app.py", "from app import main\n\nassert main() == 3\n")
    _write(tmp_path, "README.txt", "not a Python source file\n")

    graph = build_repository_graph(tmp_path)

    assert graph.files["pkg/math_utils.py"].parser_status is ParserStatus.PARSED
    definitions = graph.definitions("add", path="pkg/math_utils.py")
    assert len(definitions) == 1
    symbol = definitions[0]
    assert symbol.kind == "function"
    assert graph.references_to(symbol.symbol_id)
    assert graph.dependencies_of("app.py") == ("pkg/math_utils.py",)
    assert graph.dependents_of("app.py") == ("tests/test_app.py",)
    assert graph.likely_affected_tests(["pkg/math_utils.py"]) == ("tests/test_app.py",)
    assert graph.files["README.txt"].parser_status is ParserStatus.UNSUPPORTED
    assert graph.files["README.txt"].symbols == ()


def test_malformed_python_uses_conservative_import_fallback_without_symbols(tmp_path: Path) -> None:
    _write(tmp_path, "pkg.py", "VALUE = 1\n")
    _write(tmp_path, "broken.py", "def broken(:\n    import pkg\n")
    _write(tmp_path, "notes.rb", "def fake_symbol; end\n")

    graph = build_repository_graph(tmp_path)

    broken = graph.files["broken.py"]
    assert broken.parser_status is ParserStatus.FALLBACK
    assert broken.symbols == ()
    assert broken.imports
    assert graph.dependencies_of("broken.py") == ("pkg.py",)
    assert graph.files["notes.rb"].parser_status is ParserStatus.UNSUPPORTED
    assert graph.files["notes.rb"].symbols == ()


def test_incremental_update_matches_clean_rebuild_and_preserves_ids(tmp_path: Path) -> None:
    _write(tmp_path, "lib.py", "def keep():\n    return 1\n\ndef change():\n    return 2\n")
    _write(tmp_path, "consumer.py", "from lib import keep\n")
    _write(tmp_path, "tests/test_lib.py", "from lib import keep\n")
    initial = build_repository_graph(tmp_path)
    keep_id = initial.definitions("keep", path="lib.py")[0].symbol_id

    _write(
        tmp_path,
        "lib.py",
        "def keep():\n    return 1\n\ndef change():\n    return 3\n\ndef added():\n    return 4\n",
    )
    (tmp_path / "consumer.py").unlink()
    _write(tmp_path, "new_module.py", "from lib import added\n")

    incremental = update_repository_graph(
        initial,
        changed_paths=["lib.py", "consumer.py", "new_module.py"],
    )
    clean = build_repository_graph(tmp_path)

    assert incremental.fingerprint == clean.fingerprint
    assert incremental.canonical() == clean.canonical()
    assert incremental.stats.removed_files == 1
    assert incremental.stats.parsed_files == 2
    assert incremental.stats.reused_files == 1
    assert incremental.definitions("keep", path="lib.py")[0].symbol_id == keep_id
    assert "consumer.py" not in incremental.files


def test_dirty_worktree_changes_snapshot_identity_even_when_head_is_unchanged(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "module.py", "VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=test@example.test",
            "-c",
            "user.name=test",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    clean = build_repository_graph(tmp_path)
    _write(tmp_path, "module.py", "VALUE = 2\n")
    dirty = build_repository_graph(tmp_path)

    assert clean.identity.git_revision == dirty.identity.git_revision
    assert clean.identity.snapshot_id != dirty.identity.snapshot_id
    assert clean.identity.source_digest != dirty.identity.source_digest
    assert dirty.identity.dirty is True
