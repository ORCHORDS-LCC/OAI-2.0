# OAI-2.0 → public gateway wiring

This document describes the live wire between the OAI-2.0 runtime and
the public ORCHORDS inference gateway (`api.orchords.com`). It is the
source of truth for the env-var contract and the operational shape of
the path. No credentials or live tokens appear in this file.

## Data path

```
+------------------+      HTTP (HTTPS)      +-------------------+      +--------+
| OAI-2.0 runtime  |  ───────────────────▶  | api.orchords.com  |  ──▶ | oai-1.2 |
| GatewayRuntime   |  POST /v1/chat/        | ORCHORDS gateway  |      +--------+
| (InferenceRuntime|       completions       | (q-pipe)          |
|  subclass)       |  Authorization: Bearer |                   |
+------------------+                        +-------------------+
```

The runtime is :class:`oai2.runtime.gateway_runtime.GatewayRuntime`,
an :class:`oai2.runtime.InferenceRuntime` subclass that translates
:class:`oai2.runtime.InferenceRequest` into an OpenAI-compatible chat
completions request, POSTs it to the configured base URL, and parses
the response back into an :class:`oai2.runtime.InferenceResponse`.

The HTTP transport is `httpx`. For tests the transport is replaced
with `httpx.MockTransport` so the runner-free `scripts/verify.py`
cycle covers the wire path end-to-end with zero live calls.

## Configuration

The gateway runtime reads four environment variables. The first is
required; the others have documented defaults.

| Variable                      | Required | Default                     | Purpose                                         |
|-------------------------------|----------|------------------------------|--------------------------------------------------|
| `OAI2_GATEWAY_API_KEY`        | yes      | (empty)                      | Bearer token sent on every request.             |
| `OAI2_GATEWAY_BASE_URL`       | no       | `https://api.orchords.com`   | Base URL of the gateway.                        |
| `OAI2_GATEWAY_MODEL`          | no       | `oai-1.2`                    | Default model id sent in the request body.      |
| `OAI2_GATEWAY_TIMEOUT_SECONDS`| no       | `60`                         | Per-request HTTP timeout.                       |

The populated values must live in a gitignored `.env`. `.env.example`
documents the contract with empty values; real tokens are NEVER
committed. The `scripts/verify.py` `public-safety` scan enforces
this — it flags any committed string that looks like a hard-coded
`api_key=...` of 16+ characters.

## Operating the wire

A single chat completion probe (operator-side, after populating
`.env`):

```
uv run python scripts/gateway_smoke.py --prompt "ping"
uv run python scripts/gateway_smoke.py --prompt "ping" --json
```

A multi-prompt sweep with token/latency breakdown (planned,
see issue tracker): not in scope for this revision.

## Local gate

`scripts/verify.py` runs four mandatory checks (sync / ruff / mypy /
pytest) plus three opt-in ones. The `gateway-reach` sub-check is
opt-in — it executes only when `OAI2_GATEWAY_API_KEY` is set, and
exits cleanly (SKIP) otherwise. When it runs, it sends one
`GET /v1/models` request and confirms the configured
`OAI2_GATEWAY_MODEL` is in the response.

```
PASS gateway-reach: https://api.orchords.com exposes oai-1.2 (of 1 models)
```

The check is intentionally live (no `MockTransport`) — it is the
single source of truth that the gateway is actually reachable from
the local host. Disable by leaving the env var unset.

## Errors

The runtime raises :class:`GatewayRuntimeError` on non-2xx
responses or transport failures. The error message is redacted
through :func:`_redact` so a bearer token that accidentally
appears in an upstream error body is not echoed to logs. The
HTTP status code is preserved on `exc.status_code`.

| Code | Meaning                                                    |
|------|-------------------------------------------------------------|
| 0    | Transport error (connect refused, DNS failure, timeout).    |
| 200  | Upstream returned 200 with malformed payload.               |
| 401  | Missing or invalid bearer token.                            |
| 5xx  | Upstream gateway fault.                                     |

## Status of the wire

| Layer                           | Status      | Notes                                                  |
|----------------------------------|-------------|--------------------------------------------------------|
| `oai2.runtime.GatewayRuntime`   | IMPLEMENTED | 15/15 unit tests, mock-transport only.                 |
| `oai2.runtime.GatewayConfig`    | IMPLEMENTED | `repr()` redacts the API key.                          |
| `scripts/gateway_smoke.py`      | IMPLEMENTED | Single-shot end-to-end probe.                          |
| `scripts/verify.py gateway-reach` | IMPLEMENTED | SKIP by default, live when key is set.             |
| End-to-end from P50             | OPEN        | Not yet exercised — pending operator-supplied token.    |
| api.orchords.com model list      | OPEN        | Cloud currently exposes 4 models; local qpipe exposes 1. |
| Knowledge transport worker       | PROPOSED    | Cloudflare Worker entrypoint in design phase.          |

## Status of the related q-pipe cloud

The local q-pipe checkout (`../q-pipe`) restricts `models_payload()`
to the single public model id `oai-1.2`. The deployed
`api.orchords.com` instance still exposes four models — that change
must be re-applied on the cloud deployment. Tracking is in the
issue tracker.