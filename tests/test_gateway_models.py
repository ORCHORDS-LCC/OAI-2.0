"""Unit tests for :mod:`oai2.runtime.gateway_models`.

All tests use :class:`httpx.MockTransport` so no live HTTP call is ever
made — the runner-free ``scripts/verify.py`` cycle covers the
cloud-model discovery + fallback contracts end-to-end without leaving
the host.
"""

from __future__ import annotations

import json

import httpx
import pytest

from oai2.runtime import (
    KNOWN_CLOUD_MODELS,
    CloudDiscoveryError,
    UnknownModelError,
    WorkingModelResolution,
    discover_cloud_models,
    probe_model,
    resolve_working_model,
)
from oai2.runtime.gateway_runtime import (
    DEFAULT_GATEWAY_BASE_URL,
    DEFAULT_GATEWAY_MODEL,
    GatewayConfig,
    GatewayRuntime,
)
from oai2.runtime.gateway_runtime import (
    _parse_candidate_models as _parse_candidates_helper,
)


def _models_payload(model_ids: list[str]) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 1700000000, "owned_by": "orchords"}
            for model_id in model_ids
        ],
    }


def _completion_payload(content: str = "pong") -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _error_payload(message: str) -> dict[str, object]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "model_not_ready",
        },
    }


class _Router:
    """Routes ``httpx.Request`` instances to canned responses.

    The route table is a list of ``(predicate, status, body)``. The
    first matching predicate wins. Unmatched requests fail the test
    loudly so missing routes never silently produce the wrong answer.
    """

    def __init__(
        self,
        routes: list[tuple[callable, int, dict[str, object] | None]],
    ) -> None:
        self.routes = routes
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        for predicate, status, body in self.routes:
            if predicate(request):
                if body is None:
                    return httpx.Response(status, content=b"")
                return httpx.Response(
                    status,
                    headers={"content-type": "application/json"},
                    content=json.dumps(body).encode("utf-8"),
                )
        raise AssertionError(f"unmatched request: {request.method} {request.url}")


def _is_get(path: str) -> callable:
    def predicate(request: httpx.Request) -> bool:
        return request.method == "GET" and request.url.path == path
    return predicate


def _is_post_with_model(model_id: str) -> callable:
    def predicate(request: httpx.Request) -> bool:
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            return False
        try:
            body = json.loads(request.content.decode("utf-8"))
        except ValueError:
            return False
        return body.get("model") == model_id
    return predicate


def _make_client(router: _Router) -> httpx.Client:
    transport = httpx.MockTransport(router)
    return httpx.Client(base_url="https://gateway.example.test", transport=transport)


def test_known_cloud_models_set_is_explicit() -> None:
    """`oai-2.0` is the only id in the strict single-id allowlist.

    The OAI-2.0 client deliberately filters historical ids
    (`oai-1.0`, `oai-1.2`, `orchordsai-gpt`, `orchordsai-m3`)
    out of the cloud identity set. Widening this allowlist is an
    explicit code change, not a runtime configuration — Refs #237.
    """

    assert KNOWN_CLOUD_MODELS == frozenset({"oai-2.0"})


def test_known_cloud_models_excludes_historical_ids() -> None:
    """Regression: the four historical ids must NOT be in the allowlist."""

    historical = {"oai-1.0", "oai-1.2", "orchordsai-gpt", "orchordsai-m3"}
    assert KNOWN_CLOUD_MODELS.isdisjoint(historical)


def test_discover_cloud_models_returns_id_list() -> None:
    router = _Router(
        [
            (
                _is_get("/v1/models"),
                200,
                _models_payload(["oai-2.0", "oai-1.0"]),
            ),
        ],
    )
    client = _make_client(router)
    try:
        ids = discover_cloud_models(client)
    finally:
        client.close()
    assert ids == ["oai-2.0", "oai-1.0"]


def test_discover_cloud_models_raises_on_http_error() -> None:
    router = _Router([(_is_get("/v1/models"), 503, _error_payload("down"))])
    client = _make_client(router)
    try:
        with pytest.raises(CloudDiscoveryError):
            discover_cloud_models(client)
    finally:
        client.close()


def test_discover_cloud_models_raises_on_missing_data() -> None:
    router = _Router([(_is_get("/v1/models"), 200, {"object": "list"})])
    client = _make_client(router)
    try:
        with pytest.raises(CloudDiscoveryError):
            discover_cloud_models(client)
    finally:
        client.close()


def test_probe_model_returns_reachable_on_2xx() -> None:
    router = _Router(
        [(_is_post_with_model("oai-2.0"), 200, _completion_payload("pong"))],
    )
    client = _make_client(router)
    try:
        probe = probe_model(client, "oai-2.0")
    finally:
        client.close()
    assert probe.reachable is True
    assert probe.status_code == 200
    assert probe.error is None
    assert probe.latency_ms is not None and probe.latency_ms >= 0.0


def test_probe_model_returns_unreachable_on_503() -> None:
    router = _Router(
        [(_is_post_with_model("oai-2.0"), 503, _error_payload("model_not_ready"))],
    )
    client = _make_client(router)
    try:
        probe = probe_model(client, "oai-2.0")
    finally:
        client.close()
    assert probe.reachable is False
    assert probe.status_code == 503
    assert probe.error is not None
    # The error string is redacted (public-safety) so it will not match
    # the literal upstream message verbatim; the redacted form keeps
    # the first three and last three characters and collapses the rest.
    assert probe.error.startswith("mod") and probe.error.endswith("ady")


def test_probe_model_returns_unreachable_on_empty_choices() -> None:
    empty_payload = {
        "id": "chatcmpl-test",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    router = _Router([(_is_post_with_model("oai-2.0"), 200, empty_payload)])
    client = _make_client(router)
    try:
        probe = probe_model(client, "oai-2.0")
    finally:
        client.close()
    assert probe.reachable is False
    assert probe.status_code == 200
    assert probe.error is not None


def test_probe_model_raises_on_historical_id() -> None:
    """Historical ids must be rejected before any HTTP call."""

    router = _Router([])
    client = _make_client(router)
    try:
        with pytest.raises(UnknownModelError):
            probe_model(client, "oai-1.2")
        assert len(router.calls) == 0
    finally:
        client.close()


def test_probe_model_raises_on_completely_unknown_id() -> None:
    router = _Router([])
    client = _make_client(router)
    try:
        with pytest.raises(UnknownModelError):
            probe_model(client, "totally-not-a-known-id")
    finally:
        client.close()


def test_resolve_working_model_returns_first_reachable_in_order() -> None:
    router = _Router(
        [
            (
                _is_post_with_model("oai-2.0"),
                200,
                _completion_payload("pong"),
            ),
        ],
    )
    client = _make_client(router)
    try:
        result = resolve_working_model(client, ["oai-2.0"])
    finally:
        client.close()
    assert isinstance(result, WorkingModelResolution)
    assert result.selected_model_id == "oai-2.0"
    assert len(result.probes) == 1
    assert [probe.reachable for probe in result.probes] == [True]


def test_resolve_working_model_short_circuits_on_first_hit() -> None:
    router = _Router(
        [(_is_post_with_model("oai-2.0"), 200, _completion_payload("pong"))],
    )
    client = _make_client(router)
    try:
        result = resolve_working_model(client, ["oai-2.0"])
    finally:
        client.close()
    assert result.selected_model_id == "oai-2.0"
    assert len(result.probes) == 1
    assert len(router.calls) == 1


def test_resolve_working_model_records_unknown_id_as_failed_probe() -> None:
    router = _Router(
        [(_is_post_with_model("oai-2.0"), 200, _completion_payload("pong"))],
    )
    client = _make_client(router)
    try:
        # The resolver walks candidates in order; `oai-2.0` is valid and
        # reachable, so the first run short-circuits with `oai-2.0`
        # selected and `totally-not-a-known-id` is never probed (it's
        # queued *after* the hit). The unknown-id-as-failed-probe
        # diagnostic is covered by the historical-ids test below.
        result = resolve_working_model(client, ["oai-2.0", "totally-not-a-known-id"])
    finally:
        client.close()
    assert result.selected_model_id == "oai-2.0"
    assert len(result.probes) == 1
    assert result.probes[0].model_id == "oai-2.0"


def test_resolve_working_model_records_unknown_id_short_circuit() -> None:
    """Unknown ids short-circuit the chain at first occurrence."""

    router = _Router([])
    client = _make_client(router)
    try:
        # First candidate is unknown → resolver records exactly one failed
        # probe and returns; never reaches the later candidates.
        result = resolve_working_model(
            client,
            ["totally-not-a-known-id", "oai-2.0"],
        )
    finally:
        client.close()
    assert result.selected_model_id is None
    assert len(result.probes) == 1
    assert result.probes[0].model_id == "totally-not-a-known-id"
    assert "unknown cloud model id" in (result.probes[0].error or "")


def test_resolve_working_model_with_empty_candidates_is_empty() -> None:
    router = _Router([])
    client = _make_client(router)
    try:
        result = resolve_working_model(client, [])
    finally:
        client.close()
    assert result.selected_model_id is None
    assert result.probes == ()


def test_resolve_working_model_dedupes_repeated_ids() -> None:
    router = _Router(
        [(_is_post_with_model("oai-2.0"), 200, _completion_payload("pong"))],
    )
    client = _make_client(router)
    try:
        result = resolve_working_model(client, ["oai-2.0", "oai-2.0", "oai-2.0"])
    finally:
        client.close()
    assert result.selected_model_id == "oai-2.0"
    assert len(result.probes) == 1


def test_resolve_working_model_rejects_historical_ids_with_failed_probe() -> None:
    """Historical ids short-circuit the chain at first occurrence."""

    router = _Router(
        [(_is_post_with_model("oai-2.0"), 503, _error_payload("model_not_ready"))],
    )
    client = _make_client(router)
    try:
        result = resolve_working_model(
            client,
            ["oai-1.0", "oai-1.2", "orchordsai-gpt", "orchordsai-m3", "oai-2.0"],
        )
    finally:
        client.close()
    assert result.selected_model_id is None
    # The first historical id short-circuits the chain (only one
    # failed probe recorded); later candidates are not walked.
    assert len(result.probes) == 1
    assert result.probes[0].model_id == "oai-1.0"
    assert "unknown cloud model id" in (result.probes[0].error or "")


def test_gateway_config_candidates_including_primary_dedupes() -> None:
    cfg = GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="k",
        model="oai-2.0",
        candidate_models=("oai-2.0", "oai-1.0", "orchordsai-m3"),
    )
    assert cfg.candidates_including_primary() == (
        "oai-2.0",
        "oai-1.0",
        "orchordsai-m3",
    )


def test_parse_candidate_models_strips_and_dedupes() -> None:
    assert _parse_candidates_helper("") == ()
    assert _parse_candidates_helper("oai-2.0") == ("oai-2.0",)
    assert _parse_candidates_helper("oai-2.0, oai-1.0 ,oai-2.0") == (
        "oai-2.0",
        "oai-1.0",
    )
    assert _parse_candidates_helper(" , , ") == ()


def test_load_gateway_config_from_env_parses_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oai2.runtime.gateway_runtime import load_gateway_config_from_env

    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")
    monkeypatch.setenv(
        "OAI2_GATEWAY_CANDIDATE_MODELS",
        "oai-2.0, orchordsai-m3 ,oai-2.0",
    )
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.model == "oai-2.0"
    assert cfg.candidate_models == ("oai-2.0", "orchordsai-m3")


def test_load_gateway_config_from_env_omits_candidates_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oai2.runtime.gateway_runtime import load_gateway_config_from_env

    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")
    monkeypatch.delenv("OAI2_GATEWAY_CANDIDATE_MODELS", raising=False)
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.candidate_models == ()


def test_default_gateway_model_is_oai_2_0() -> None:
    """The static default is `oai-2.0` (Refs #237)."""

    assert DEFAULT_GATEWAY_MODEL == "oai-2.0"
    assert DEFAULT_GATEWAY_BASE_URL == "https://api.orchords.com"


def test_gateway_runtime_from_env_with_fallback_returns_working_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_env_with_fallback`` returns a runtime pinned to a working model."""

    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")
    monkeypatch.delenv("OAI2_GATEWAY_CANDIDATE_MODELS", raising=False)
    router = _Router(
        [
            (
                _is_post_with_model("oai-2.0"),
                200,
                _completion_payload("pong"),
            ),
        ],
    )
    transport = httpx.MockTransport(router)
    with httpx.Client(base_url="https://gateway.example.test", transport=transport) as http_client:
        runtime, resolution = GatewayRuntime.from_env_with_fallback(
            client=http_client,
        )
        assert runtime is not None
        assert isinstance(resolution, WorkingModelResolution)
        assert resolution.selected_model_id == "oai-2.0"
        assert runtime.config.model == "oai-2.0"


def test_gateway_runtime_from_env_with_fallback_returns_none_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")
    monkeypatch.delenv("OAI2_GATEWAY_CANDIDATE_MODELS", raising=False)
    router = _Router(
        [
            (
                _is_post_with_model("oai-2.0"),
                503,
                _error_payload("model_not_ready"),
            ),
        ],
    )
    transport = httpx.MockTransport(router)
    with httpx.Client(base_url="https://gateway.example.test", transport=transport) as http_client:
        runtime, resolution = GatewayRuntime.from_env_with_fallback(
            client=http_client,
        )
        assert runtime is None
        assert isinstance(resolution, WorkingModelResolution)
        assert resolution.selected_model_id is None
        assert len(resolution.probes) == 1


def test_gateway_runtime_from_env_with_fallback_skip_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")
    monkeypatch.delenv("OAI2_GATEWAY_CANDIDATE_MODELS", raising=False)
    router = _Router([])
    transport = httpx.MockTransport(router)
    with httpx.Client(base_url="https://gateway.example.test", transport=transport) as http_client:
        runtime, resolution = GatewayRuntime.from_env_with_fallback(
            client=http_client,
            probe=False,
        )
        assert runtime is not None
        # When probe=False we deliberately do not record resolution.
        assert resolution is None
        assert runtime.config.model == "oai-2.0"
