"""Structural / validation tests for ``oai2.protocols.tokens``.

Pin the action-token vocabulary's public surface: the
:class:`TokenKind` :class:`enum.StrEnum` wire strings, the three
frozen-slotted token dataclasses (:class:`ReadToken`, :class:`TestToken`,
:class:`ImageToken`), the :data:`ActionToken` union alias, and the
tolerant :func:`parse_action_token` parser (no exceptions, returns a
list, recognises the documented kind / key combinations and silently
ignores the rest).

Behavioural end-to-end coverage lives in ``tests/test_protocols.py``;
this file pins the *shape* of the API and the invariants the runtime
relies on.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Union, get_args, get_origin

import pytest

from oai2.protocols import (
    ActionToken,
    ImageToken,
    ReadToken,
    TestToken,
    TokenKind,
    parse_action_token,
)
from oai2.protocols.tokens import (
    ActionToken as ActionTokenFromModule,
)
from oai2.protocols.tokens import (
    ImageToken as ImageTokenFromModule,
)
from oai2.protocols.tokens import (
    ReadToken as ReadTokenFromModule,
)
from oai2.protocols.tokens import (
    TestToken as TestTokenFromModule,
)
from oai2.protocols.tokens import (
    TokenKind as TokenKindFromModule,
)
from oai2.protocols.tokens import (
    parse_action_token as parse_action_token_from_module,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "protocols" / "tokens.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Module docstring + import surface
# ---------------------------------------------------------------------------


def test_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "protocols/tokens.py is unexpectedly empty"


def test_module_has_docstring() -> None:
    """The module ships an overview of the action-token vocabulary."""

    assert _MODULE_SOURCE.startswith('"""')
    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert first_para, "module docstring is empty"


def test_module_docstring_mentions_action_token_and_compact() -> None:
    """The overview mentions both "action token" and the "compact" framing."""

    first_para = _MODULE_SOURCE.split('"""', 2)[1].lower()
    assert "action token" in first_para
    assert "compact" in first_para


def test_module_uses_future_annotations() -> None:
    """``from __future__ import annotations`` is present (UP006-clean)."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_module_uses_only_stdlib_imports() -> None:
    """Tokens is a leaf module — it does not import from ``oai2`` or any third-party package."""

    import_lines = [
        line.strip()
        for line in _MODULE_SOURCE.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for line in import_lines:
        # No `import oai2…` (must be a leaf module)
        assert not re.match(r"^import\s+oai2(\.|\s|$)", line), f"forbidden import: {line!r}"
        # No `from oai2…`
        assert not re.match(r"^from\s+oai2(\.|\s)", line), f"forbidden import: {line!r}"
        # No third-party packages; only stdlib (incl. `from __future__`).
        assert (
            line.startswith("import re")
            or line.startswith("from dataclasses")
            or line.startswith("from enum")
            or line.startswith("from __future__")
        ), f"unexpected import line: {line!r}"


def test_module_has_no_wildcard_imports() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


# ---------------------------------------------------------------------------
# __all__ completeness + package re-export
# ---------------------------------------------------------------------------


def test_dunder_all_lists_exactly_six_public_names() -> None:
    """The module's public surface is exactly 6 names, no more, no less."""

    import oai2.protocols.tokens as mod

    assert isinstance(mod.__all__, list)
    assert set(mod.__all__) == {
        "ActionToken",
        "ImageToken",
        "ReadToken",
        "TestToken",
        "TokenKind",
        "parse_action_token",
    }
    assert len(mod.__all__) == 6


def test_each_all_name_is_importable_from_module() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    import oai2.protocols.tokens as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ name missing: {name}"


def test_public_names_are_reexported_from_protocols_package() -> None:
    """Top-level ``oai2.protocols`` re-exports the 6 token public names."""

    import oai2.protocols as pkg

    for name in (
        "ActionToken",
        "ImageToken",
        "ReadToken",
        "TestToken",
        "TokenKind",
        "parse_action_token",
    ):
        assert name in pkg.__all__, f"package re-export missing: {name}"


def test_package_re_export_preserves_identity() -> None:
    """Package re-exports point at the *same* objects as the module."""

    import oai2.protocols as pkg
    import oai2.protocols.tokens as mod

    assert pkg.ActionToken is mod.ActionToken
    assert pkg.ImageToken is mod.ImageToken
    assert pkg.ReadToken is mod.ReadToken
    assert pkg.TestToken is mod.TestToken
    assert pkg.TokenKind is mod.TokenKind
    assert pkg.parse_action_token is mod.parse_action_token


def test_module_imports_match_top_level() -> None:
    """``from oai2.protocols.tokens import X`` matches the top-level re-export."""

    assert ActionToken is ActionTokenFromModule
    assert ImageToken is ImageTokenFromModule
    assert ReadToken is ReadTokenFromModule
    assert TestToken is TestTokenFromModule
    assert TokenKind is TokenKindFromModule
    assert parse_action_token is parse_action_token_from_module


# ---------------------------------------------------------------------------
# TokenKind StrEnum
# ---------------------------------------------------------------------------


def test_token_kind_is_a_strenum() -> None:
    """``TokenKind`` subclasses :class:`enum.StrEnum` so wire strings
    round-trip through JSON without a custom serializer."""

    assert issubclass(TokenKind, StrEnum)
    assert issubclass(TokenKind, str)


def test_token_kind_has_nine_members() -> None:
    """``TokenKind`` has exactly 9 members — no more, no less."""

    assert len(TokenKind) == 9


def test_token_kind_members_and_wire_strings() -> None:
    """Every member's wire string is the uppercase member name."""

    expected = {
        "READ": "READ",
        "WRITE": "WRITE",
        "EDIT": "EDIT",
        "TEST": "TEST",
        "RUN": "RUN",
        "IMAGE": "IMAGE",
        "UI": "UI",
        "KNOW": "KNOW",
        "ARTIFACT": "ARTIFACT",
    }
    actual = {member.name: str(member.value) for member in TokenKind}
    assert actual == expected


def test_token_kind_member_names_are_unique() -> None:
    """No duplicate member names — ``TokenKind`` is a valid enum."""

    names = [member.name for member in TokenKind]
    assert len(names) == len(set(names))


def test_token_kind_wire_strings_are_unique() -> None:
    """No duplicate wire-string values — the enum is a true bijection."""

    values = [str(member.value) for member in TokenKind]
    assert len(values) == len(set(values))


def test_token_kind_lookup_by_wire_string() -> None:
    """``TokenKind("READ")`` resolves to ``READ`` (and likewise for each member)."""

    for member in TokenKind:
        assert TokenKind(str(member.value)) is member


def test_token_kind_rejects_unknown_wire_string() -> None:
    """An unknown wire string raises ``ValueError`` — no silent fallback."""

    with pytest.raises(ValueError):
        TokenKind("BOGUS")


def test_token_kind_string_membership() -> None:
    """``"READ" in {"READ": 1}`` style — token kind survives plain ``str`` round-trip."""

    assert str(TokenKind.READ) == "READ"
    assert TokenKind.READ == "READ"  # StrEnum equality with str
    assert "READ" == TokenKind.READ


# ---------------------------------------------------------------------------
# ActionToken type alias
# ---------------------------------------------------------------------------


def test_action_token_is_a_union_of_three_classes() -> None:
    """``ActionToken`` is a union of ``ReadToken | TestToken | ImageToken``."""

    # At runtime, ``ActionToken`` is the union object produced by the ``|`` operator.
    # It must flatten to exactly three concrete classes.
    args = set(get_args(ActionToken))
    assert args == {ReadToken, TestToken, ImageToken}


def test_action_token_accepts_each_variant() -> None:
    """Each token class is a member of the ``ActionToken`` union (isinstance)."""

    r = ReadToken(TokenKind.READ, "foo.py")
    t = TestToken(TokenKind.TEST, "tests/test_foo.py")
    i = ImageToken(TokenKind.IMAGE, "cap-1")
    assert isinstance(r, get_args(ActionToken))
    assert isinstance(t, get_args(ActionToken))
    assert isinstance(i, get_args(ActionToken))


def test_action_token_runtime_value_is_a_class_union() -> None:
    """``ActionToken`` is a Union type — not a concrete class itself."""

    assert get_origin(ActionToken) is Union


# ---------------------------------------------------------------------------
# ReadToken / TestToken / ImageToken dataclasses
# ---------------------------------------------------------------------------


def test_read_token_is_a_dataclass() -> None:
    """``ReadToken`` is a :func:`dataclasses.dataclass`."""

    assert is_dataclass(ReadToken)


def test_read_token_is_frozen() -> None:
    """``ReadToken`` is frozen — mutating an instance raises ``FrozenInstanceError``."""

    token = ReadToken(TokenKind.READ, "foo.py")
    with pytest.raises(FrozenInstanceError):
        token.path = "bar.py"  # type: ignore[misc]


def test_read_token_is_slotted() -> None:
    """``ReadToken`` is slotted — ``__slots__`` is declared on the class."""

    assert "__slots__" in ReadToken.__dict__


def test_read_token_fields() -> None:
    """``ReadToken`` has exactly ``(kind, path)`` fields."""

    assert [f.name for f in fields(ReadToken)] == ["kind", "path"]


def test_read_token_equality_and_hash() -> None:
    """Two ``ReadToken`` instances with equal fields compare equal and share a hash."""

    a = ReadToken(TokenKind.READ, "foo.py")
    b = ReadToken(TokenKind.READ, "foo.py")
    assert a == b
    assert hash(a) == hash(b)
    # … and inequality on a different path is honored.
    c = ReadToken(TokenKind.READ, "bar.py")
    assert a != c


def test_test_token_is_a_dataclass() -> None:
    """``TestToken`` is a :func:`dataclasses.dataclass`."""

    assert is_dataclass(TestToken)


def test_test_token_is_frozen() -> None:
    """``TestToken`` is frozen — mutating an instance raises ``FrozenInstanceError``."""

    token = TestToken(TokenKind.TEST, "tests/test_foo.py")
    with pytest.raises(FrozenInstanceError):
        token.selector = "tests/test_bar.py"  # type: ignore[misc]


def test_test_token_is_slotted() -> None:
    """``TestToken`` is slotted — ``__slots__`` is declared on the class."""

    assert "__slots__" in TestToken.__dict__


def test_test_token_fields() -> None:
    """``TestToken`` has exactly ``(kind, selector)`` fields."""

    assert [f.name for f in fields(TestToken)] == ["kind", "selector"]


def test_test_token_opted_out_of_pytest_collection() -> None:
    """``TestToken.__test__ is False`` — pytest must not collect it as a test class.

    Pytest's collection looks at the ``__test__`` attribute on classes whose
    name starts with ``Test``. Setting it to ``False`` opts the class out
    even though it is a valid test-shaped class. The setting is load-bearing
    for the test suite itself; this test pins it.
    """

    assert TestToken.__test__ is False


def test_image_token_is_a_dataclass() -> None:
    """``ImageToken`` is a :func:`dataclasses.dataclass`."""

    assert is_dataclass(ImageToken)


def test_image_token_is_frozen() -> None:
    """``ImageToken`` is frozen — mutating an instance raises ``FrozenInstanceError``."""

    token = ImageToken(TokenKind.IMAGE, "cap-1")
    with pytest.raises(FrozenInstanceError):
        token.image_id = "cap-2"  # type: ignore[misc]


def test_image_token_is_slotted() -> None:
    """``ImageToken`` is slotted — ``__slots__`` is declared on the class."""

    assert "__slots__" in ImageToken.__dict__


def test_image_token_fields() -> None:
    """``ImageToken`` has exactly ``(kind, image_id)`` fields."""

    assert [f.name for f in fields(ImageToken)] == ["kind", "image_id"]


def test_token_classes_are_distinct() -> None:
    """``ReadToken`` / ``TestToken`` / ``ImageToken`` are three distinct classes."""

    assert ReadToken is not TestToken
    assert ReadToken is not ImageToken
    assert TestToken is not ImageToken
    # A ``ReadToken`` is not a ``TestToken`` and not an ``ImageToken``.
    r = ReadToken(TokenKind.READ, "foo.py")
    assert not isinstance(r, TestToken)
    assert not isinstance(r, ImageToken)


def test_token_kind_value_is_a_token_kind() -> None:
    """Each token's ``kind`` attribute is a ``TokenKind`` enum member."""

    r = ReadToken(TokenKind.READ, "foo.py")
    t = TestToken(TokenKind.TEST, "tests/test_foo.py")
    i = ImageToken(TokenKind.IMAGE, "cap-1")
    assert isinstance(r.kind, TokenKind)
    assert isinstance(t.kind, TokenKind)
    assert isinstance(i.kind, TokenKind)


def test_token_dataclass_decorator_signature() -> None:
    """Every token class is decorated with ``@dataclass(slots=True, frozen=True)``."""

    for cls in (ReadToken, TestToken, ImageToken):
        # The decorator's __wrapped__ chain ends in dataclass(...). We can
        # also check that the class is frozen via ``__dataclass_params__``.
        params = getattr(cls, "__dataclass_params__", None)
        assert params is not None, f"{cls.__name__} is not a dataclass"
        assert params.frozen is True, f"{cls.__name__} is not frozen"
        assert params.slots is True, f"{cls.__name__} is not slotted"


# ---------------------------------------------------------------------------
# parse_action_token function
# ---------------------------------------------------------------------------


def test_parse_action_token_returns_list() -> None:
    """``parse_action_token`` always returns a list — never raises on benign text."""

    result = parse_action_token("")
    assert isinstance(result, list)
    assert result == []


def test_parse_action_token_empty_text() -> None:
    """An empty string yields an empty list."""

    assert parse_action_token("") == []


def test_parse_action_token_no_match() -> None:
    """Free-form text without action tokens yields an empty list."""

    assert parse_action_token("hello world, no tokens here") == []


def test_parse_action_token_basic_read() -> None:
    """The basic ``<READ F:foo.py>`` form is parsed into a ``ReadToken``."""

    tokens = parse_action_token("<READ F:foo.py>")
    assert tokens == [ReadToken(TokenKind.READ, "foo.py")]


def test_parse_action_token_basic_test() -> None:
    """The basic ``<TEST T:tests/test_x.py>`` form is parsed into a ``TestToken``."""

    tokens = parse_action_token("<TEST T:tests/test_x.py>")
    assert tokens == [TestToken(TokenKind.TEST, "tests/test_x.py")]


def test_parse_action_token_basic_image() -> None:
    """The basic ``<IMAGE I:cap-1>`` form is parsed into an ``ImageToken``."""

    tokens = parse_action_token("<IMAGE I:cap-1>")
    assert tokens == [ImageToken(TokenKind.IMAGE, "cap-1")]


def test_parse_action_token_read_aliases() -> None:
    """``READ`` accepts the ``F`` / ``FILE`` / ``PATH`` keys (any case)."""

    for key in ("F", "FILE", "PATH", "f", "file", "path", "File", "Path"):
        tokens = parse_action_token(f"<READ {key}:value>")
        assert tokens == [ReadToken(TokenKind.READ, "value")], f"failed for key={key!r}"


def test_parse_action_token_test_aliases() -> None:
    """``TEST`` accepts the ``T`` / ``TEST`` / ``SEL`` keys (any case)."""

    for key in ("T", "TEST", "SEL", "t", "test", "sel", "Test", "Sel"):
        tokens = parse_action_token(f"<TEST {key}:value>")
        assert tokens == [TestToken(TokenKind.TEST, "value")], f"failed for key={key!r}"


def test_parse_action_token_image_aliases() -> None:
    """``IMAGE`` accepts the ``I`` / ``IMG`` / ``IMAGE`` keys (any case)."""

    for key in ("I", "IMG", "IMAGE", "i", "img", "image", "Img", "Image"):
        tokens = parse_action_token(f"<IMAGE {key}:value>")
        assert tokens == [ImageToken(TokenKind.IMAGE, "value")], f"failed for key={key!r}"


def test_parse_action_token_strips_whitespace() -> None:
    """The parser strips whitespace around key and value."""

    tokens = parse_action_token("<READ   F :   foo.py   >")
    assert tokens == [ReadToken(TokenKind.READ, "foo.py")]


def test_parse_action_token_ignores_malformed_no_colon() -> None:
    """A token without a colon separator is silently ignored."""

    assert parse_action_token("<READ foo>") == []


def test_parse_action_token_ignores_unknown_key() -> None:
    """A token with an unknown key for a known kind is silently ignored."""

    # READ with key ``BOGUS`` does not match {F, FILE, PATH}.
    assert parse_action_token("<READ BOGUS:value>") == []


def test_parse_action_token_unrecognized_kind_ignored() -> None:
    """Kinds outside the parser's typed set (``WRITE`` / ``EDIT`` / ``RUN`` /
    ``UI`` / ``KNOW`` / ``ARTIFACT``) are recognised by the regex but produce
    no ``ActionToken`` because the source has no ``elif`` branch for them.

    The parser is tolerant — it skips these silently and returns whatever
    other tokens it found.
    """

    assert parse_action_token("<WRITE F:foo.py>") == []
    assert parse_action_token("<EDIT F:foo.py>") == []
    assert parse_action_token("<RUN T:tests/test_x.py>") == []
    assert parse_action_token("<UI I:cap-1>") == []
    assert parse_action_token("<KNOW F:doc>") == []
    assert parse_action_token("<ARTIFACT I:cap-2>") == []


def test_parse_action_token_preserves_order() -> None:
    """Multiple tokens are returned in the order they appear in the text."""

    text = "I will <READ F:foo.py> and <TEST T:tests/test_foo.py> and <IMAGE I:cap-1>."
    assert parse_action_token(text) == [
        ReadToken(TokenKind.READ, "foo.py"),
        TestToken(TokenKind.TEST, "tests/test_foo.py"),
        ImageToken(TokenKind.IMAGE, "cap-1"),
    ]


def test_parse_action_token_ignores_surrounding_prose() -> None:
    """The parser only extracts token-shaped substrings; surrounding text is ignored."""

    text = (
        "Plan: read <READ F:foo.py>, then test <TEST T:tests/test_foo.py>.\n"
        "Notes: please don't crash, this is line two."
    )
    assert parse_action_token(text) == [
        ReadToken(TokenKind.READ, "foo.py"),
        TestToken(TokenKind.TEST, "tests/test_foo.py"),
    ]


def test_parse_action_token_value_can_contain_colon() -> None:
    """A single ``:`` separates key and value; additional ``:`` in the value is preserved."""

    tokens = parse_action_token("<READ PATH:foo:bar:baz.py>")
    assert tokens == [ReadToken(TokenKind.READ, "foo:bar:baz.py")]


def test_parse_action_token_duplicate_tokens_are_all_returned() -> None:
    """Two tokens with the same kind and value are returned as two list entries."""

    text = "<READ F:foo.py> and again <READ F:foo.py>"
    assert parse_action_token(text) == [
        ReadToken(TokenKind.READ, "foo.py"),
        ReadToken(TokenKind.READ, "foo.py"),
    ]


def test_parse_action_token_does_not_raise_on_benign_text() -> None:
    """The parser never raises on benign input — it returns ``[]`` instead.

    Pinned here so a future refactor that turns a tolerated case into a
    hard error shows up in this test first.
    """

    benign_inputs = [
        "",
        "no tokens at all",
        "<READ foo>",  # no colon
        "<BOGUS k:v>",  # unknown kind
        "<READ BOGUS:value>",  # unknown key
        "<WRITE F:foo.py>",  # recognized kind, not typed
        "<<<>>>",  # malformed brackets
    ]
    for text in benign_inputs:
        result = parse_action_token(text)
        assert isinstance(result, list), f"non-list returned for {text!r}"


def test_parse_action_token_signature() -> None:
    """``parse_action_token`` is a regular function (not a class) with a single
    string parameter and a list return annotation."""

    assert callable(parse_action_token)
    sig = inspect.signature(parse_action_token)
    params = list(sig.parameters.values())
    assert len(params) == 1
    assert params[0].name == "text"
    # ``from __future__ import annotations`` makes annotations lazy strings,
    # so we accept either the resolved type or its string form.
    annotation = params[0].annotation
    assert annotation is str or annotation == "str"


# ---------------------------------------------------------------------------
# Module source — public-safety boundary
# ---------------------------------------------------------------------------


def test_module_source_has_no_cloud_sdk_reference() -> None:
    """The parser must not pull any cloud SDK or vendor helper."""

    forbidden = (
        "boto3",
        "azure",
        "google.cloud",
        "gcp",
        "aws_access_key",
        "kubernetes",
        "docker",
    )
    for needle in forbidden:
        assert needle not in _MODULE_SOURCE, f"forbidden cloud reference: {needle}"


def test_module_source_has_no_hardcoded_api_key() -> None:
    """No long alphanumeric secret is hardcoded in the module source."""

    code_only = "\n".join(
        line for line in _MODULE_SOURCE.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"api_key\s*=\s*[\"']sk-[A-Za-z0-9]{16,}", code_only)
    assert not re.search(r"[\"']sk-[A-Za-z0-9]{16,}[\"']", code_only)
    assert "BEGIN PRIVATE KEY" not in code_only


def test_module_source_has_no_print_or_pprint() -> None:
    """The parser is silent — no ``print`` / ``pprint``."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_module_source_has_no_subprocess_or_os_system() -> None:
    """No subprocess / os.system — the module is pure-Python parsing."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_module_source_has_no_direct_network_imports() -> None:
    """No direct ``requests`` / ``urllib`` / ``httpx`` imports."""

    assert "import requests" not in _MODULE_SOURCE
    assert "from urllib" not in _MODULE_SOURCE
    assert "import urllib" not in _MODULE_SOURCE
    assert "import httpx" not in _MODULE_SOURCE
    assert "from httpx" not in _MODULE_SOURCE


def test_module_source_has_no_eval_or_exec() -> None:
    """No ``eval`` / ``exec`` — the parser's grammar is static."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_module_source_has_no_wildcard_imports_in_source() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


def test_module_source_does_not_read_environment_directly() -> None:
    """The parser does not read ``os.environ`` — it is purely a string function."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_module_source_has_no_outstanding_todo_markers() -> None:
    """No outstanding ``TODO`` / ``FIXME`` / ``XXX`` — the file is shipped
    as production code."""

    for marker in (r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b"):
        assert not re.search(marker, _MODULE_SOURCE), f"forbidden marker in source: {marker}"


def test_module_source_uses_strenum_for_token_kind() -> None:
    """``TokenKind`` is a ``StrEnum`` so wire strings survive JSON round-trips."""

    assert "StrEnum" in _MODULE_SOURCE
    assert "class TokenKind(StrEnum)" in _MODULE_SOURCE


def test_module_source_uses_frozen_slotted_dataclasses() -> None:
    """Every token dataclass declares ``slots=True, frozen=True``."""

    assert _MODULE_SOURCE.count("slots=True, frozen=True") == 3
    assert _MODULE_SOURCE.count("class ReadToken") == 1
    assert _MODULE_SOURCE.count("class TestToken") == 1
    assert _MODULE_SOURCE.count("class ImageToken") == 1


def test_module_source_has_no_pydantic_import() -> None:
    """Tokens is dataclass-based — no Pydantic dependency."""

    assert "pydantic" not in _MODULE_SOURCE
    assert "BaseModel" not in _MODULE_SOURCE
    assert "ConfigDict" not in _MODULE_SOURCE


def test_module_source_uses_compile_token_regex() -> None:
    """The parser is backed by a compiled regex — the grammar is a single source of truth."""

    assert "_TOKEN_RE" in _MODULE_SOURCE
    assert "re.compile" in _MODULE_SOURCE
    # The regex alternation must list every recognised kind.
    for kind in ("READ", "WRITE", "EDIT", "TEST", "RUN", "IMAGE", "UI", "KNOW", "ARTIFACT"):
        assert kind in _MODULE_SOURCE, f"missing kind in regex alternation: {kind}"


def test_module_source_exposes_action_token_union() -> None:
    """``ActionToken`` is the union of the three concrete token classes."""

    assert "ActionToken = ReadToken | TestToken | ImageToken" in _MODULE_SOURCE


def test_module_source_does_not_use_unittest_mock() -> None:
    """Production code does not import a test-time helper."""

    assert "unittest.mock" not in _MODULE_SOURCE
    assert "import unittest" not in _MODULE_SOURCE
