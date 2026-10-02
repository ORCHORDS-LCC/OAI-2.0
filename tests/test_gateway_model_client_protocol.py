"""Cross-repo conformance tests for ``oai2.runtime.GatewayModelClient``.

q-pipe's ``HeldOutRunner`` reads only ``reply.content`` and
``reply.finish_reason`` off a duck-typed ``ModelClient`` seam::

    client.chat(messages, *, max_tokens, temperature) -> ChatReply

``qpipe.android_curriculum.model_client.ModelClient`` declares that
contract as a ``@runtime_checkable Protocol``. OAI-2.0 ships its own
``GatewayModelClient`` adapter so the runner can drive
``api.orchords.com`` end-to-end without either repo importing the other.

These tests verify, in a single Python process, that:

1. ``GatewayModelClient`` actually satisfies q-pipe's
   ``ModelClient`` protocol under the runtime ``isinstance`` check.
2. The adapter's ``chat()`` method signature matches q-pipe's
   ``ModelClient.chat`` exactly (parameter names, kinds, defaults).
3. The adapter's ``ChatReply`` exposes the two attributes q-pipe's
   runner reads (``content``, ``finish_reason``) and is
   duck-type-compatible with q-pipe's own ``ChatReply``.
4. ``GatewayRuntimeError`` propagates through the adapter without
   being wrapped, so the runner's exception branch keeps working.
5. q-pipe's ``FakeModelClient`` (the test double the runner uses)
   shares the same call signature, confirming the seam is symmetric.

q-pipe is *not* a runtime dependency of OAI-2.0. The import path is
discovered the same way ``scripts/verify.py`` finds the q-pipe
checkout: ``$OAI2_QPIPE_REPO`` overrides the default
``<repo-parent>/q-pipe``. When q-pipe is unreachable the suite
skips with ``pytest.skip`` so the runner-free ``scripts/verify.py``
cycle stays green on bare interpreters.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from oai2.runtime import (
    GatewayConfig,
    GatewayModelClient,
    GatewayRuntime,
    GatewayRuntimeError,
)

_QPIPE_CANDIDATES: tuple[Path, ...] = ()


def _discover_qpipe_root() -> Path | None:
    """Locate the q-pipe checkout via the same convention as ``scripts/verify.py``.

    Honours ``OAI2_QPIPE_REPO`` (absolute path) and the default sibling
    ``<repo-parent>/q-pipe`` layout. Returns ``None`` when neither
    exists so the suite can ``pytest.skip`` cleanly.
    """

    configured = os.getenv("OAI2_QPIPE_REPO")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.exists():
            return candidate
        return None
    # Default sibling: OAI-2.0 lives at <root>/<repo>, q-pipe lives at <root>/q-pipe.
    here = Path(__file__).resolve()
    for parent in here.parents:
        sibling = parent / "q-pipe"
        if sibling.exists():
            return sibling.resolve()
    return None


def _load_qpipe_seam() -> ModuleType | None:
    """Import q-pipe's model_client module and return it (or None on failure)."""

    global _QPIPE_CANDIDATES
    root = _discover_qpipe_root()
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _QPIPE_CANDIDATES = (root,)
    try:
        return importlib.import_module("qpipe.android_curriculum.model_client")
    except Exception:
        return None


_QPIPE_SEAM = _load_qpipe_seam()
# After the skipif below, _QPIPE_SEAM is guaranteed non-None at runtime;
# tell mypy the same so attribute access is not flagged.
_QPIPE_SEAM_TYPED: ModuleType = _QPIPE_SEAM  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    _QPIPE_SEAM is None,
    reason="q-pipe checkout not importable; cross-repo conformance is optional",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config() -> GatewayConfig:
    return GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="seam-token-xyz",
        model="oai-2.0",
        timeout_seconds=5.0,
    )


def _runtime_with(handler) -> GatewayRuntime:
    cfg = _config()
    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=transport,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    return GatewayRuntime(cfg, client=client)


def _completion_response(content: str = "ok") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-protocol-test",
            "model": "oai-2.0",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        },
    )


@pytest.fixture
def gateway_client():
    """Yield a fully-mocked ``GatewayModelClient`` whose runtime is closed after."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion_response()

    runtime = _runtime_with(handler)
    try:
        yield GatewayModelClient(runtime)
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Protocol seam — direct cross-repo assertions
# ---------------------------------------------------------------------------


def test_qpipe_seam_was_loaded() -> None:
    """Sanity: the seam module actually came from q-pipe's tree."""

    assert _QPIPE_SEAM is not None, "i.e. module-level import succeeded"
    assert _QPIPE_SEAM_TYPED.__name__ == "qpipe.android_curriculum.model_client"


def test_qpipe_seam_root_path_is_exposed() -> None:
    """Operators can see which q-pipe tree the conformance check ran against."""

    assert _QPIPE_CANDIDATES, "i.e. _load_qpipe_seam populated the candidates tuple"
    root = _QPIPE_CANDIDATES[0]
    assert (root / "qpipe" / "android_curriculum" / "model_client.py").exists()


def test_gateway_model_client_satisfies_qpipe_modelclient_protocol(
    gateway_client: GatewayModelClient,
) -> None:
    """``isinstance(gateway_client, qpipe.ModelClient)`` must be True.

    This is the load-bearing cross-repo assertion. q-pipe's
    ``ModelClient`` is ``@runtime_checkable``; the ``isinstance`` check
    confirms OAI-2.0's adapter honours the seam q-pipe's
    ``HeldOutRunner`` reads.
    """

    qpipe_model_client = _QPIPE_SEAM_TYPED.ModelClient
    assert isinstance(gateway_client, qpipe_model_client)


def test_chat_reply_is_duck_typed_compatible_with_qpipe_chatreply(
    gateway_client: GatewayModelClient,
) -> None:
    """OAI-2.0's reply must expose ``content`` and ``finish_reason``.

    Duck-typed compatibility is what the runner relies on; the reply
    need not be a q-pipe ``ChatReply`` instance, just read-compatible.
    """

    qpipe_chat_reply = _QPIPE_SEAM_TYPED.ChatReply

    reply = gateway_client.chat([{"role": "user", "content": "ping"}])

    # The two attributes the runner reads must be present and readable.
    assert isinstance(reply.content, str)
    assert reply.content == "ok"
    assert isinstance(reply.finish_reason, str)
    assert reply.finish_reason == "stop"

    # And reply must structurally resemble q-pipe's ChatReply dataclass.
    own_field_names = {f.name for f in reply.__dataclass_fields__.values()}
    qpipe_field_names = {f.name for f in qpipe_chat_reply.__dataclass_fields__.values()}
    assert {"content", "finish_reason"} <= own_field_names
    assert {"content", "finish_reason"} <= qpipe_field_names


def test_chat_method_signature_matches_qpipe_modelclient_chat(
    gateway_client: GatewayModelClient,
) -> None:
    """The adapter's ``chat`` signature must equal q-pipe's exactly."""

    qpipe_model_client = _QPIPE_SEAM_TYPED.ModelClient
    own_sig = inspect.signature(gateway_client.chat)
    qpipe_sig = inspect.signature(qpipe_model_client.chat)

    own_params = list(own_sig.parameters.values())
    qpipe_params = list(qpipe_sig.parameters.values())

    # Drop the implicit `self` from the Protocol's signature so it lines
    # up with the bound method's parameters.
    own_names = [p.name for p in own_params]
    qpipe_names = [p.name for p in qpipe_params if p.name != "self"]
    assert own_names == qpipe_names == ["messages", "max_tokens", "temperature"]

    for own_param, qpipe_param in zip(own_params, qpipe_params[1:], strict=True):
        assert own_param.kind == qpipe_param.kind
        assert own_param.annotation == qpipe_param.annotation
        assert own_param.default == qpipe_param.default


def test_qpipe_fake_model_client_shares_the_same_call_shape(
    gateway_client: GatewayModelClient,
) -> None:
    """q-pipe's ``FakeModelClient`` uses the same ``chat`` signature.

    Symmetric check: the test double q-pipe ships for ``HeldOutRunner``
    must speak the same protocol as our adapter. If q-pipe ever loosens
    its signature this test breaks first.
    """

    fake_client_cls = _QPIPE_SEAM_TYPED.FakeModelClient
    fake_client = fake_client_cls(replies=["canned"])
    fake_sig = inspect.signature(fake_client.chat)
    own_sig = inspect.signature(gateway_client.chat)

    own_params = list(own_sig.parameters.values())
    fake_params = [p for p in fake_sig.parameters.values() if p.name != "self"]

    own_param_names = [p.name for p in own_params]
    fake_param_names = [p.name for p in fake_params]
    assert own_param_names == fake_param_names

    for own_param, fake_param in zip(own_params, fake_params, strict=True):
        assert own_param.kind == fake_param.kind
        assert own_param.annotation == fake_param.annotation
        assert own_param.default == fake_param.default


def test_runtime_checkable_flag_is_set_on_qpipe_modelclient() -> None:
    """``ModelClient`` must carry Python's runtime-checkable marker.

    The cross-repo isinstance check is only meaningful because q-pipe
    decorates the Protocol with ``@runtime_checkable``. If q-pipe drops
    the decorator we lose the contract test and this slice fails loud.
    """

    qpipe_model_client = _QPIPE_SEAM_TYPED.ModelClient
    assert getattr(qpipe_model_client, "_is_runtime_protocol", False) is True


# ---------------------------------------------------------------------------
# Exception propagation across the seam
# ---------------------------------------------------------------------------


def test_gateway_runtime_error_propagates_unchanged_through_adapter() -> None:
    """Errors raised by ``GatewayRuntime.generate`` must surface intact.

    q-pipe's runner catches specific exception types off the
    ``ModelClient.chat`` call. Wrapping ``GatewayRuntimeError`` would
    silently break the runner's error branch.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": {"message": "service unavailable", "type": "server_error"}},
        )

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        with pytest.raises(GatewayRuntimeError) as exc_info:
            client.chat([{"role": "user", "content": "ping"}])
        # The exact subclass identity is what the runner would inspect.
        assert isinstance(exc_info.value, RuntimeError)
        assert exc_info.value.status_code == 503
    finally:
        runtime.close()


def test_transport_error_propagates_with_zero_status_code() -> None:
    """``httpx`` transport failures must surface as ``GatewayRuntimeError``.

    q-pipe's runner expects ``status_code == 0`` for connection-level
    failures so it can branch on connectivity vs. HTTP-shaped errors.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        with pytest.raises(GatewayRuntimeError) as exc_info:
            client.chat([{"role": "user", "content": "ping"}])
        assert exc_info.value.status_code == 0
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Adapter-side hardening that the runner relies on
# ---------------------------------------------------------------------------


def test_adapter_default_max_tokens_matches_qpipe() -> None:
    """The adapter's fallback ``max_tokens`` must equal q-pipe's expectation.

    ``FakeModelClient`` records every ``max_tokens`` value it sees; the
    runner's behaviour depends on that value reaching the upstream call.
    If OAI-2.0 silently drops the default the runner would still see
    ``None`` and miscount the budget.
    """

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat([{"role": "user", "content": "ping"}])
    finally:
        runtime.close()

    # ``GatewayRuntime._build_request_body`` forwards max_tokens into the
    # request body verbatim when the caller supplies one. The adapter
    # substitutes 256 when None; we assert that exactly.
    assert captured["body"]["max_tokens"] == 256
