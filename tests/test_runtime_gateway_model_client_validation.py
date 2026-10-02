"""Structural / validation tests for ``oai2.runtime.gateway_model_client``.

Pin the module's public surface, the :class:`ChatReply` dataclass
contract, and the :class:`GatewayModelClient` adapter's structural
semantics. Behavioural end-to-end coverage with a live HTTP transport
lives in ``tests/test_gateway_model_client.py``; the q-pipe cross-repo
seam lives in ``tests/test_gateway_model_client_protocol.py``. This file
focuses on the *shape* of the API and the contract that downstream
callers (including q-pipe's ``HeldOutRunner``) rely on.

The adapter is a thin composition over :class:`GatewayRuntime`; for
these tests we substitute a tiny ``_RecordingRuntime`` that records
``generate(request)`` and ``close()`` calls and returns a configurable
:class:`InferenceResponse`. No HTTP is involved — the transport path is
exercised in the existing test files via ``httpx.MockTransport``.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from oai2.core import Status
from oai2.runtime import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ChatReply,
    GatewayModelClient,
)
from oai2.runtime.gateway_model_client import (
    _coerce_max_tokens,
    _coerce_temperature,
)
from oai2.runtime.gateway_runtime import GatewayRuntimeError
from oai2.runtime.inference import InferenceRequest, InferenceResponse

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "oai2" / "runtime" / "gateway_model_client.py"
)
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Recording runtime — minimal stand-in for GatewayRuntime
# ---------------------------------------------------------------------------


class _RecordingRuntime:
    """Stand-in for :class:`GatewayRuntime` recording ``generate`` / ``close``.

    The adapter only touches ``generate(request)`` and ``close()``; the
    rest of ``GatewayRuntime`` is irrelevant to the structural contract
    pinned here. We deliberately do not subclass ``GatewayRuntime`` —
    the point is to exercise the adapter against the surface it actually
    uses, not the static type.
    """

    def __init__(self, response: InferenceResponse) -> None:
        self._response = response
        self.generate_calls: list[InferenceRequest] = []
        self.close_calls: int = 0

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.generate_calls.append(request)
        return self._response

    def close(self) -> None:
        self.close_calls += 1


def _runtime(
    *,
    text: str = "hello",
    finish_reason: str | None = "stop",
    tool_calls: tuple[dict[str, Any], ...] = (),
    elapsed_ms: float = 12.5,
    device: str = "fake-device",
    status: Status = Status.EXPERIMENTAL,
) -> _RecordingRuntime:
    """Build a recording runtime that returns a fixed InferenceResponse."""

    return _RecordingRuntime(
        InferenceResponse(
            text=text,
            tokens=1,
            elapsed_ms=elapsed_ms,
            device=device,
            status=status,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
    )


def _client(
    runtime: _RecordingRuntime | None = None,
) -> tuple[GatewayModelClient, _RecordingRuntime]:
    """Pair an adapter with its recording runtime for assertions."""

    rt = runtime if runtime is not None else _runtime()
    return GatewayModelClient(rt), rt  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module docstring + import surface
# ---------------------------------------------------------------------------


def test_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "gateway_model_client.py is unexpectedly empty"


def test_module_has_docstring() -> None:
    """The module ships a one-paragraph overview of the adapter role."""

    assert _MODULE_SOURCE.startswith('"""')
    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    # The role summary must name the seam and the runner that depends on it
    assert "ModelClient" in first_para
    assert "q-pipe" in first_para


def test_module_docstring_names_the_public_endpoint() -> None:
    """The overview names the public endpoint the adapter targets."""

    first_para = _MODULE_SOURCE.split('"""', 2)[1]
    assert "api.orchords.com" in first_para


def test_module_uses_future_annotations() -> None:
    """``from __future__ import annotations`` keeps the surface clean."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_module_imports_use_relative_from_imports() -> None:
    """No absolute ``import gateway_runtime`` and no wildcard imports."""

    assert "import *" not in _MODULE_SOURCE
    # Required relative imports for runtime / inference seam
    assert "from .gateway_runtime import GatewayRuntime" in _MODULE_SOURCE
    assert "from .inference import InferenceRequest" in _MODULE_SOURCE
    # No accidental absolute imports of the same modules
    assert not re.search(
        r"^import (?:gateway_runtime|inference)\b",
        _MODULE_SOURCE,
        re.MULTILINE,
    )


# ---------------------------------------------------------------------------
# __all__ completeness + package re-export
# ---------------------------------------------------------------------------


def test_dunder_all_lists_exactly_four_public_names() -> None:
    """The module's public surface is exactly 4 names, no more, no less."""

    import oai2.runtime.gateway_model_client as mod

    assert isinstance(mod.__all__, list)
    assert set(mod.__all__) == {
        "ChatReply",
        "DEFAULT_MAX_TOKENS",
        "DEFAULT_TEMPERATURE",
        "GatewayModelClient",
    }
    assert len(mod.__all__) == 4


def test_each_all_name_is_importable_from_module() -> None:
    """Every name in ``__all__`` resolves to a real attribute on the module."""

    import oai2.runtime.gateway_model_client as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ name missing: {name}"


def test_private_helpers_are_not_in_all() -> None:
    """The private flatten / coerce helpers are not part of the public API."""

    import oai2.runtime.gateway_model_client as mod

    for private in (
        "_flatten_messages",
        "_coerce_max_tokens",
        "_coerce_temperature",
    ):
        assert private not in mod.__all__
        assert private.startswith("_")


def test_public_names_are_reexported_from_package() -> None:
    """Top-level ``oai2.runtime`` re-exports the four public names."""

    import oai2.runtime as pkg

    for name in (
        "ChatReply",
        "DEFAULT_MAX_TOKENS",
        "DEFAULT_TEMPERATURE",
        "GatewayModelClient",
    ):
        assert name in pkg.__all__, f"package re-export missing: {name}"


# ---------------------------------------------------------------------------
# DEFAULT_TEMPERATURE / DEFAULT_MAX_TOKENS + coerce helpers
# ---------------------------------------------------------------------------


def test_default_temperature_is_float_zero_point_seven() -> None:
    """``DEFAULT_TEMPERATURE`` is the documented 0.7 fallback value."""

    assert DEFAULT_TEMPERATURE == pytest.approx(0.7)
    assert isinstance(DEFAULT_TEMPERATURE, float)


def test_default_max_tokens_is_int_two_fifty_six() -> None:
    """``DEFAULT_MAX_TOKENS`` is the documented 256 fallback value."""

    assert DEFAULT_MAX_TOKENS == 256
    assert isinstance(DEFAULT_MAX_TOKENS, int)


def test_default_constants_are_not_bools() -> None:
    """Booleans subclass ints — the constants must not be bools accidentally."""

    assert not isinstance(DEFAULT_TEMPERATURE, bool)
    assert not isinstance(DEFAULT_MAX_TOKENS, bool)


def test_coerce_max_tokens_returns_default_when_none() -> None:
    """``_coerce_max_tokens(None)`` returns ``DEFAULT_MAX_TOKENS``."""

    assert _coerce_max_tokens(None) == DEFAULT_MAX_TOKENS


def test_coerce_temperature_returns_default_when_none() -> None:
    """``_coerce_temperature(None)`` returns ``DEFAULT_TEMPERATURE``."""

    assert _coerce_temperature(None) == pytest.approx(DEFAULT_TEMPERATURE)


def test_coerce_max_tokens_casts_provided_value_to_int() -> None:
    """``_coerce_max_tokens(x)`` returns ``int(x)`` when x is not None."""

    assert _coerce_max_tokens(64) == 64
    assert _coerce_max_tokens(64.0) == 64  # type: ignore[arg-type]
    assert _coerce_max_tokens("128") == 128  # type: ignore[arg-type]


def test_coerce_temperature_casts_provided_value_to_float() -> None:
    """``_coerce_temperature(x)`` returns ``float(x)`` when x is not None."""

    assert _coerce_temperature(0.25) == pytest.approx(0.25)
    assert _coerce_temperature(0) == pytest.approx(0.0)
    assert _coerce_temperature(1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ChatReply dataclass contract
# ---------------------------------------------------------------------------


def test_chat_reply_is_a_dataclass() -> None:
    """``ChatReply`` is a real dataclass (class and instance)."""

    assert is_dataclass(ChatReply)
    assert is_dataclass(ChatReply(content="x"))


def test_chat_reply_is_frozen() -> None:
    """Frozen instances raise on attribute assignment."""

    reply = ChatReply(content="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.content = "y"  # type: ignore[misc]


def test_chat_reply_is_slotted() -> None:
    """``slots=True`` removes the per-instance ``__dict__``."""

    reply = ChatReply(content="x")
    assert hasattr(reply, "__slots__")
    assert not hasattr(reply, "__dict__")


def test_chat_reply_slots_match_field_names() -> None:
    """The ``__slots__`` tuple lists every dataclass field."""

    reply = ChatReply(content="x")
    assert set(reply.__slots__) == {
        "content",
        "finish_reason",
        "reasoning",
        "tool_calls",
        "raw",
    }


def test_chat_reply_field_names_and_order() -> None:
    """Dataclass fields are exactly: content, finish_reason, reasoning,
    tool_calls, raw — in that declaration order."""

    field_names = [f.name for f in fields(ChatReply)]
    assert field_names == [
        "content",
        "finish_reason",
        "reasoning",
        "tool_calls",
        "raw",
    ]


def test_chat_reply_field_annotations() -> None:
    """Field annotations match the documented types verbatim."""

    by_name = {f.name: f for f in fields(ChatReply)}
    assert by_name["content"].type == "str"
    assert by_name["finish_reason"].type == "str | None"
    assert by_name["reasoning"].type == "str"
    assert by_name["tool_calls"].type == "tuple[dict[str, Any], ...]"
    assert by_name["raw"].type == "dict[str, Any] | None"


def test_chat_reply_field_defaults() -> None:
    """``content`` is required; the other four carry explicit defaults."""

    by_name = {f.name: f for f in fields(ChatReply)}
    assert by_name["content"].default is dataclasses.MISSING
    assert by_name["finish_reason"].default == "stop"
    assert by_name["reasoning"].default == ""
    assert by_name["tool_calls"].default == ()
    assert by_name["raw"].default is None


def test_chat_reply_minimal_construction() -> None:
    """Only ``content`` is required; the rest default sensibly."""

    reply = ChatReply(content="hi")
    assert reply.content == "hi"
    assert reply.finish_reason == "stop"
    assert reply.reasoning == ""
    assert reply.tool_calls == ()
    assert reply.raw is None


def test_chat_reply_positional_construction() -> None:
    """Fields can be passed positionally in declaration order."""

    reply = ChatReply("hi", "length", "because", ({"id": 1},), {"k": "v"})
    assert reply.content == "hi"
    assert reply.finish_reason == "length"
    assert reply.reasoning == "because"
    assert reply.tool_calls == ({"id": 1},)
    assert reply.raw == {"k": "v"}


def test_chat_reply_finish_reason_accepts_none() -> None:
    """``finish_reason`` is ``str | None`` — the dataclass accepts ``None``."""

    reply = ChatReply(content="x", finish_reason=None)
    assert reply.finish_reason is None


def test_chat_reply_equality_and_hash() -> None:
    """Frozen dataclasses compare by value and are hashable."""

    a = ChatReply(content="x", finish_reason="stop")
    b = ChatReply(content="x", finish_reason="stop")
    c = ChatReply(content="y", finish_reason="stop")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # And usable in a set without raising
    assert {a, b, c} == {a, c}


def test_chat_reply_repr_includes_class_name() -> None:
    """The auto-generated ``repr`` names ``ChatReply`` for log readability."""

    assert "ChatReply" in repr(ChatReply(content="x"))


# ---------------------------------------------------------------------------
# GatewayModelClient — constructor + chat() signature
# ---------------------------------------------------------------------------


def test_init_stores_runtime_on_private_attribute() -> None:
    """The adapter owns the runtime on a private slot named ``_runtime``."""

    runtime = _runtime()
    client = GatewayModelClient(runtime)  # type: ignore[arg-type]
    assert client._runtime is runtime


def test_runtime_property_exposes_underlying_runtime() -> None:
    """The public ``runtime`` property returns the same object passed in."""

    runtime = _runtime()
    client = GatewayModelClient(runtime)  # type: ignore[arg-type]
    assert client.runtime is runtime


def test_runtime_property_is_read_only() -> None:
    """``runtime`` has no setter — assignment raises ``AttributeError``."""

    client, _ = _client()
    with pytest.raises(AttributeError):
        client.runtime = None  # type: ignore[misc, assignment]


def test_init_rejects_extra_kwargs() -> None:
    """``__init__`` only accepts the single positional ``runtime``."""

    runtime = _runtime()
    with pytest.raises(TypeError):
        GatewayModelClient(runtime, extra=1)  # type: ignore[arg-type, call-arg]


def test_chat_signature_layout() -> None:
    """``chat(self, messages, *, max_tokens, temperature)`` — exact layout."""

    sig = inspect.signature(GatewayModelClient.chat)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ["self", "messages", "max_tokens", "temperature"]
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[3].kind is inspect.Parameter.KEYWORD_ONLY


def test_chat_default_max_tokens_is_none() -> None:
    """``max_tokens`` defaults to ``None`` — the adapter substitutes the
    module default at call time."""

    sig = inspect.signature(GatewayModelClient.chat)
    assert sig.parameters["max_tokens"].default is None


def test_chat_default_temperature_is_none() -> None:
    """``temperature`` defaults to ``None`` — the adapter substitutes the
    module default at call time."""

    sig = inspect.signature(GatewayModelClient.chat)
    assert sig.parameters["temperature"].default is None


def test_chat_rejects_positional_max_tokens() -> None:
    """Positional second arg after ``messages`` raises ``TypeError``
    (max_tokens is keyword-only)."""

    client, _ = _client()
    with pytest.raises(TypeError):
        client.chat([{"role": "user", "content": "x"}], 64)  # type: ignore[call-arg]


def test_chat_rejects_positional_temperature() -> None:
    """Positional third arg after ``messages`` raises ``TypeError``
    (temperature is keyword-only)."""

    client, _ = _client()
    with pytest.raises(TypeError):
        client.chat(
            [{"role": "user", "content": "x"}],
            64,  # type: ignore[call-arg]
            0.5,
        )


# ---------------------------------------------------------------------------
# chat() — flattening & forwarding
# ---------------------------------------------------------------------------


def test_chat_invokes_runtime_generate_exactly_once() -> None:
    """Each ``chat`` call delegates to ``runtime.generate`` exactly once."""

    client, runtime = _client()
    assert runtime.generate_calls == []
    client.chat([{"role": "user", "content": "x"}])
    assert len(runtime.generate_calls) == 1
    client.chat([{"role": "user", "content": "y"}])
    assert len(runtime.generate_calls) == 2


def test_chat_flattens_multi_message_transcript() -> None:
    """Multi-message transcripts are rendered as ``role: content`` lines."""

    client, runtime = _client()
    client.chat(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
    )
    request = runtime.generate_calls[0]
    assert request.prompt == ("system: be terse\nuser: hello\nassistant: hi\nuser: bye\n")


def test_chat_with_no_messages_uses_user_marker() -> None:
    """An empty message list flattens to the documented ``"user: \\n"``
    sentinel — the adapter must not let ``InferenceRequest`` reject an
    empty prompt."""

    client, runtime = _client()
    client.chat([])
    assert runtime.generate_calls[0].prompt == "user: \n"


def test_chat_skips_empty_or_none_message_content() -> None:
    """Messages with empty-string or ``None`` content are dropped silently."""

    client, runtime = _client()
    client.chat(
        [
            {"role": "user", "content": "real"},
            {"role": "user", "content": ""},
            {"role": "user", "content": None},
        ]
    )
    assert runtime.generate_calls[0].prompt == "user: real\n"


def test_chat_defaults_to_module_max_tokens_when_none() -> None:
    """``chat(max_tokens=None)`` forwards ``DEFAULT_MAX_TOKENS`` into the
    request — the coerce helper is exercised on the None branch."""

    client, runtime = _client()
    client.chat([{"role": "user", "content": "x"}])
    assert runtime.generate_calls[0].max_tokens == DEFAULT_MAX_TOKENS


def test_chat_defaults_to_module_temperature_when_none() -> None:
    """``chat(temperature=None)`` forwards ``DEFAULT_TEMPERATURE`` into the
    request — the coerce helper is exercised on the None branch."""

    client, runtime = _client()
    client.chat([{"role": "user", "content": "x"}])
    assert runtime.generate_calls[0].temperature == pytest.approx(DEFAULT_TEMPERATURE)


def test_chat_forwards_explicit_max_tokens_int() -> None:
    """Explicit int ``max_tokens`` reach the request unchanged."""

    client, runtime = _client()
    client.chat([{"role": "user", "content": "x"}], max_tokens=77)
    assert runtime.generate_calls[0].max_tokens == 77


def test_chat_forwards_explicit_temperature_float() -> None:
    """Explicit float ``temperature`` reaches the request unchanged."""

    client, runtime = _client()
    client.chat([{"role": "user", "content": "x"}], temperature=0.25)
    assert runtime.generate_calls[0].temperature == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# chat() — ChatReply construction
# ---------------------------------------------------------------------------


def test_chat_returns_chat_reply_instance() -> None:
    """``chat`` returns a ``ChatReply`` — not a tuple, dict, or ``None``."""

    client, _ = _client()
    reply = client.chat([{"role": "user", "content": "x"}])
    assert isinstance(reply, ChatReply)


def test_chat_propagates_response_text_into_content() -> None:
    """``reply.content`` is ``response.text`` verbatim."""

    client, _ = _client(_runtime(text="reply-text"))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.content == "reply-text"


def test_chat_propagates_finish_reason_string() -> None:
    """``reply.finish_reason`` mirrors ``response.finish_reason`` for strings."""

    client, _ = _client(_runtime(finish_reason="length"))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.finish_reason == "length"


def test_chat_propagates_finish_reason_none() -> None:
    """``reply.finish_reason`` is ``None`` when the runtime reports no reason."""

    client, _ = _client(_runtime(finish_reason=None))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.finish_reason is None


def test_chat_always_sets_reasoning_to_empty_string() -> None:
    """``reasoning`` is always ``""`` — the adapter does not surface a
    reasoning channel even when the runtime eventually grows one."""

    client, _ = _client()
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.reasoning == ""


def test_chat_propagates_tool_calls_tuple() -> None:
    """``reply.tool_calls`` mirrors ``response.tool_calls``."""

    calls = ({"id": "call_1", "name": "echo", "args": {"x": 1}},)
    client, _ = _client(_runtime(tool_calls=calls))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.tool_calls == calls


def test_chat_default_tool_calls_is_empty_tuple() -> None:
    """When the response has no ``tool_calls``, the reply exposes ``()``."""

    client, _ = _client()
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.tool_calls == ()


def test_chat_raw_has_exactly_three_keys() -> None:
    """``reply.raw`` is a dict with exactly ``elapsed_ms``, ``device``,
    and ``status`` — the documented sidecar shape."""

    client, _ = _client(_runtime(elapsed_ms=42.0, device="mlx:0"))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.raw is not None
    assert set(reply.raw.keys()) == {"elapsed_ms", "device", "status"}


def test_chat_raw_status_is_status_enum_name_string() -> None:
    """``reply.raw["status"]`` is the ``.name`` of the status enum, not
    the enum value itself."""

    client, _ = _client(_runtime(status=Status.EXPERIMENTAL))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.raw is not None
    assert reply.raw["status"] == "EXPERIMENTAL"
    assert isinstance(reply.raw["status"], str)


def test_chat_raw_propagates_elapsed_ms_and_device() -> None:
    """``raw['elapsed_ms']`` and ``raw['device']`` come from the response."""

    client, _ = _client(_runtime(elapsed_ms=99.5, device="mlx:unit-test"))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.raw is not None
    assert reply.raw["elapsed_ms"] == pytest.approx(99.5)
    assert reply.raw["device"] == "mlx:unit-test"


# ---------------------------------------------------------------------------
# chat() — error propagation
# ---------------------------------------------------------------------------


class _ExplodingRuntime:
    """Runtime that raises a configured exception from ``generate``."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.generate_calls: int = 0
        self.close_calls: int = 0

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.generate_calls += 1
        raise self._exc

    def close(self) -> None:
        self.close_calls += 1


def test_chat_propagates_gateway_runtime_error_unchanged() -> None:
    """``GatewayRuntimeError`` from the runtime is re-raised unwrapped —
    the q-pipe runner's exception branch must keep working."""

    err = GatewayRuntimeError(message="upstream 503", status_code=503)
    runtime = _ExplodingRuntime(err)
    client = GatewayModelClient(runtime)  # type: ignore[arg-type]
    with pytest.raises(GatewayRuntimeError) as excinfo:
        client.chat([{"role": "user", "content": "x"}])
    assert excinfo.value is err
    assert excinfo.value.status_code == 503


def test_chat_propagates_unrelated_exceptions_unchanged() -> None:
    """Any exception from ``runtime.generate`` is re-raised without
    being wrapped in a different type."""

    class Boom(RuntimeError):
        pass

    runtime = _ExplodingRuntime(Boom("kapow"))
    client = GatewayModelClient(runtime)  # type: ignore[arg-type]
    with pytest.raises(Boom) as excinfo:
        client.chat([{"role": "user", "content": "x"}])
    assert str(excinfo.value) == "kapow"


# ---------------------------------------------------------------------------
# Lifecycle — close, context manager
# ---------------------------------------------------------------------------


def test_close_calls_runtime_close() -> None:
    """``close()`` delegates to the underlying runtime's ``close()``."""

    client, runtime = _client()
    assert runtime.close_calls == 0
    client.close()
    assert runtime.close_calls == 1


def test_close_is_idempotent() -> None:
    """``close()`` can be called repeatedly — every call reaches the runtime."""

    client, runtime = _client()
    client.close()
    client.close()
    client.close()
    assert runtime.close_calls == 3


def test_enter_returns_the_adapter() -> None:
    """``__enter__`` returns the :class:`GatewayModelClient`, not the runtime."""

    client, _ = _client()
    with client as entered:
        assert entered is client


def test_exit_invokes_close() -> None:
    """``__exit__`` calls ``close()`` — runtime close count is 1 after the block."""

    client, runtime = _client()
    with client:
        pass
    assert runtime.close_calls == 1


def test_exit_does_not_suppress_exceptions() -> None:
    """``__exit__`` returns falsy — exceptions raised inside the ``with``
    block still propagate to the caller, even though ``__exit__`` ran."""

    runtime = _ExplodingRuntime(RuntimeError("upstream-boom"))
    client = GatewayModelClient(runtime)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="upstream-boom"):
        with client:
            # The exception is raised inside the `with` block, so
            # __exit__ is invoked and must propagate the exception
            # because its return value is None (falsy).
            client.chat([{"role": "user", "content": "x"}])


def test_exit_accepts_arbitrary_args() -> None:
    """``__exit__(exc_type, exc_value, tb)`` ignores its arguments and
    never raises on a clean exit."""

    client, _ = _client()
    # The method is declared ``-> None``; calling it without using its
    # return value is the canonical context-manager-exit shape.
    client.__exit__(None, None, None)
    client.__exit__(ValueError, ValueError("x"), None)


# ---------------------------------------------------------------------------
# Module source — public-safety boundary
# ---------------------------------------------------------------------------


def test_module_source_has_no_cloud_sdk_reference() -> None:
    """The adapter must not import any cloud SDK or vendor credential helper."""

    forbidden = ("boto3", "azure", "google.cloud", "gcp", "aws_access_key")
    for needle in forbidden:
        assert needle not in _MODULE_SOURCE, f"forbidden cloud reference: {needle}"


def test_module_source_has_no_hardcoded_api_key() -> None:
    """No long alphanumeric secret is hardcoded in the module source."""

    # Strip comment lines so docstring examples don't false-positive
    code_only = "\n".join(
        line for line in _MODULE_SOURCE.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"api_key\s*=\s*[\"']sk-[A-Za-z0-9]{16,}", code_only)
    assert not re.search(r"[\"']sk-[A-Za-z0-9]{16,}[\"']", code_only)


def test_module_source_has_no_print_or_pprint() -> None:
    """The adapter is silent — no ``print`` / ``pprint`` at module scope
    or inside methods."""

    assert "print(" not in _MODULE_SOURCE
    assert "pprint(" not in _MODULE_SOURCE


def test_module_source_has_no_subprocess_or_os_system() -> None:
    """No subprocess / os.system — the adapter is pure-Python orchestration."""

    assert "subprocess" not in _MODULE_SOURCE
    assert "os.system" not in _MODULE_SOURCE


def test_module_source_has_no_direct_requests_or_urllib() -> None:
    """All HTTP goes through :class:`GatewayRuntime`; no direct
    ``requests`` / ``urllib`` imports."""

    assert "import requests" not in _MODULE_SOURCE
    assert "from urllib" not in _MODULE_SOURCE
    assert "import urllib" not in _MODULE_SOURCE


def test_module_source_has_no_eval_or_exec() -> None:
    """No ``eval`` / ``exec`` — runtime config is static."""

    assert "eval(" not in _MODULE_SOURCE
    assert "exec(" not in _MODULE_SOURCE


def test_module_source_has_no_wildcard_imports() -> None:
    """No ``from X import *`` — the public surface is enumerated in ``__all__``."""

    assert "import *" not in _MODULE_SOURCE


def test_module_source_does_not_read_environment_directly() -> None:
    """The adapter does not read ``os.environ`` — :class:`GatewayRuntime`
    owns environment access."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_module_source_has_no_outstanding_todo_markers() -> None:
    """No outstanding ``TODO`` / ``FIXME`` / ``XXX`` — the file is shipped
    as production code."""

    for marker in (r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b"):
        assert not re.search(marker, _MODULE_SOURCE), f"forbidden marker in source: {marker}"


def test_module_source_exposes_all_via_dunder_all() -> None:
    """The source declares ``__all__`` at module level — the public
    surface is enumerable."""

    assert "__all__" in _MODULE_SOURCE
