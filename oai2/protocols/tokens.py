"""Compact action-token vocabulary.

The agent emits and consumes a small set of well-known action tokens
rather than free-form tool names. This gives the model a stable surface
to plan over while the host retains freedom to back the token with any
underlying tool. Mapping ``action token → host tool`` lives in the host's
runtime config.

Reference: ``docs/agent-architecture/SYSTEM_ARCHITECTURE.md`` —
"compact action tokens" section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class TokenKind(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    EDIT = "EDIT"
    TEST = "TEST"
    RUN = "RUN"
    IMAGE = "IMAGE"
    UI = "UI"
    KNOW = "KNOW"
    ARTIFACT = "ARTIFACT"


@dataclass(slots=True, frozen=True)
class ReadToken:
    kind: TokenKind
    path: str


@dataclass(slots=True, frozen=True)
class TestToken:
    # Not a pytest test class — pytest auto-discovers classes prefixed with
    # ``Test``; this attribute opts out of collection.
    __test__ = False

    kind: TokenKind
    selector: str


@dataclass(slots=True, frozen=True)
class ImageToken:
    kind: TokenKind
    image_id: str


ActionToken = ReadToken | TestToken | ImageToken

# <KIND K:VALUE> — used in agent output and host parsing.
_TOKEN_RE = re.compile(r"<(READ|WRITE|EDIT|TEST|RUN|IMAGE|UI|KNOW|ARTIFACT)\s+([^>]+)>")


def parse_action_token(text: str) -> list[ActionToken]:
    """Parse all compact action tokens from a piece of agent output.

    The grammar is intentionally minimal. Anything more complex belongs in
    the tool-call channel (:mod:`oai2.protocols.tools`).
    """
    tokens: list[ActionToken] = []
    for match in _TOKEN_RE.finditer(text):
        kind = TokenKind(match.group(1))
        raw = match.group(2).strip()
        # Each kind has its own ``KEY:VALUE`` shape.
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if kind is TokenKind.READ and key.upper() in {"F", "FILE", "PATH"}:
            tokens.append(ReadToken(kind, value))
        elif kind is TokenKind.TEST and key.upper() in {"T", "TEST", "SEL"}:
            tokens.append(TestToken(kind, value))
        elif kind is TokenKind.IMAGE and key.upper() in {"I", "IMG", "IMAGE"}:
            tokens.append(ImageToken(kind, value))
        # Other kinds are recognized but not yet typed into Pydantic shapes.
    return tokens


__all__ = [
    "ActionToken",
    "ImageToken",
    "ReadToken",
    "TestToken",
    "TokenKind",
    "parse_action_token",
]
